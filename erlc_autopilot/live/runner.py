"""The real application loop: capture a frame (real screen or simulator),
run it through the perception/control pipeline, and actuate the result
(real keyboard or a dry-run log) -- at a fixed rate, independent of
whichever UI (wizard/overlay) happens to be on screen. This is the piece
that makes the project an actual self-driving *app* rather than just the
in-browser dashboard demo: swap the frame source and actuator and nothing
else changes, exactly the design pipeline.py describes.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import cv2

from ..config import AutopilotConfig
from ..control.imitation import build_imitation_model_for_tier
from ..pipeline import AutopilotPipeline
from ..perception.speed_reader import SpeedReader
from .input_driver import KeyboardActuator
from .kill_switch import KillSwitch

TICK_HZ = 20
DT = 1.0 / TICK_HZ


@dataclass
class RunnerState:
    frame_jpeg: bytes = b""
    telemetry: dict = field(default_factory=dict)


class LiveRunner:
    """Owns the perception/control pipeline and drives it against
    whichever frame source / actuator it's given, exposing a thread-safe
    snapshot for the overlay UI to poll -- the live-app equivalent of
    `dashboard.engine.SimulationEngine`, but pluggable instead of hardwired
    to the simulator.
    """

    def __init__(self, frame_source, actuator: KeyboardActuator,
                 config: Optional[AutopilotConfig] = None,
                 speed_reader: Optional[SpeedReader] = None,
                 ground_truth_speed: Optional[Callable[[], float]] = None,
                 arm_kill_switch: bool = True,
                 advance_sim: Optional[Callable[["object", float], None]] = None):
        """`ground_truth_speed`, if given, is a zero-arg callable returning
        a real speed value -- only ever wired up in sim mode, where the
        simulator happens to know its own ground truth. In live mode this
        stays None and speed comes from `speed_reader` (OCR) instead,
        exactly like the real game would have to be driven.

        `advance_sim`, if given, is called as `advance_sim(command, dt)`
        once per tick -- this is how Sim Mode actually steps the bundled
        simulator's physics forward using the command the pipeline just
        produced (a real screen capture doesn't need this, the real game
        simulates itself; a `SimFrameSource` does, since nothing else
        would ever move the fake car).
        """
        self.frame_source = frame_source
        self.actuator = actuator
        self.config = config or AutopilotConfig()
        self.pipeline = AutopilotPipeline(self.config)
        self.pipeline.object_detector = build_detector_for_tier(self.config.model_tier)
        self.pipeline.policy.imitation_model = build_imitation_model_for_tier(self.config.model_tier)
        self.speed_reader = speed_reader
        self._ground_truth_speed = ground_truth_speed
        self._advance_sim = advance_sim
        self.state = RunnerState()
        self.lock = threading.Lock()
        self._running = False
        self._engaged = True
        self._thread: Optional[threading.Thread] = None
        self._last_speed_mps = 0.0
        self.kill_switch: Optional[KillSwitch] = None
        if arm_kill_switch:
            self.kill_switch = KillSwitch(on_trigger=self.emergency_stop)
            self.kill_switch.arm()  # best-effort; silently no-ops if unavailable

    def start(self) -> None:
        self.frame_source.start()
        self.actuator.start()
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="autopilot-runner")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self.kill_switch:
            self.kill_switch.disarm()
        self.actuator.stop()
        self.frame_source.stop()

    def set_engaged(self, engaged: bool) -> None:
        with self.lock:
            self._engaged = engaged
        self.pipeline.set_engaged(engaged)

    def toggle_engaged(self) -> bool:
        with self.lock:
            self._engaged = not self._engaged
            engaged = self._engaged
        self.pipeline.set_engaged(engaged)
        return engaged

    def emergency_stop(self) -> None:
        """The kill switch: instantly disengage and release every key,
        callable from any thread (e.g. the global F9 hotkey listener)."""
        self.set_engaged(False)
        self.actuator.emergency_stop()

    def is_alive(self) -> bool:
        """False if the background driving thread has died outright (should
        no longer happen -- `_run` now catches every per-tick exception
        itself -- but kept as a last-resort check so the console/overlay
        can say something useful instead of just going quiet forever)."""
        return bool(self._thread and self._thread.is_alive())

    def honk(self) -> None:
        self.actuator.honk()

    def set_hazards(self, on: bool) -> None:
        self.actuator.set_hazards(on)

    def _measure_speed(self, frame, dt: float, throttle: float, brake: float):
        if self._ground_truth_speed is not None:
            return self._ground_truth_speed(), True
        if self.speed_reader is not None:
            return self.speed_reader.estimate_mps(frame, dt, throttle, brake)
        return self._last_speed_mps, False

    def _run(self) -> None:
        import traceback

        last = time.time()
        last_throttle, last_brake = 0.0, 0.0
        consecutive_errors = 0
        while self._running:
            now = time.time()
            elapsed = now - last
            if elapsed < DT:
                time.sleep(DT - elapsed)
                now = time.time()
            dt = max(1e-3, now - last)
            last = now

            try:
                frame = self.frame_source.get_frame()
                if frame is None:
                    # no frame yet (capture just started, or game window
                    # minimized/occluded) -- hold last actuation rather than
                    # spin hot or feed garbage into perception
                    time.sleep(0.01)
                    continue

                speed_mps, speed_is_real = self._measure_speed(frame, dt, last_throttle, last_brake)
                self._last_speed_mps = speed_mps

                result = self.pipeline.tick(frame, speed_mps, dt)
                cmd = result.command
                last_throttle, last_brake = cmd.throttle, cmd.brake

                with self.lock:
                    engaged = self._engaged
                cmd.engaged = cmd.engaged and engaged
                self.actuator.apply(cmd)
                if self._advance_sim is not None:
                    self._advance_sim(cmd, dt)

                ok, buf = cv2.imencode(".jpg", result.debug_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                telemetry = {
                    "speed_mph": round(speed_mps * 2.23694, 1),
                    "speed_is_real": speed_is_real,
                    "target_speed_mph": round(cmd.target_speed * 2.23694, 1),
                    "status": cmd.status,
                    "engaged": cmd.engaged,
                    "turn_signal": cmd.turn_signal,
                    "steer": round(cmd.steer, 3),
                    "throttle": round(cmd.throttle, 2),
                    "brake": round(cmd.brake, 2),
                    "fps": round(result.fps, 1),
                    "model_tier": self.config.model_tier,
                    "emergency_vehicles": [
                        {"distance_m": round(e.distance_m, 1), "flashing": e.flashing,
                         "confidence": round(e.confidence, 2)}
                        for e in result.emergency_vehicles
                    ],
                    "events": list(cmd.events),
                    "error": None,
                }
                with self.lock:
                    if ok:
                        self.state.frame_jpeg = buf.tobytes()
                    self.state.telemetry = telemetry
                consecutive_errors = 0
            except Exception as exc:
                # A perception/control/actuation hiccup must never be able
                # to silently kill the whole driving thread again (this is
                # exactly what happened when a missing Tesseract install
                # threw out of the speed reader uncaught) -- release the
                # keys to a safe neutral state, surface the error in
                # telemetry/console, and keep the loop alive so a transient
                # problem doesn't permanently end the drive.
                consecutive_errors += 1
                try:
                    self.actuator.release_all()
                except Exception:
                    pass
                print(f"\n[autopilot] tick error ({exc.__class__.__name__}: {exc}) -- "
                      f"releasing keys and continuing.")
                if consecutive_errors <= 3 or consecutive_errors % 50 == 0:
                    traceback.print_exc()
                with self.lock:
                    self.state.telemetry = {
                        **self.state.telemetry,
                        "status": "error",
                        "error": f"{exc.__class__.__name__}: {exc}",
                    }
                time.sleep(0.2)

    def get_frame(self) -> bytes:
        with self.lock:
            return self.state.frame_jpeg

    def get_telemetry(self) -> dict:
        with self.lock:
            return dict(self.state.telemetry)


def build_detector_for_tier(tier: str, model_dir: str = "models"):
    """Lite forces the dependency-light heuristic detector even if real
    MobileNet-SSD weights are present, trading a bit of accuracy for raw
    speed on lower-end PCs; Standard/Pro use the best backend available."""
    from ..perception.object_detection import HeuristicDetector, build_default_detector

    if tier == "lite":
        return HeuristicDetector()
    return build_default_detector(model_dir)

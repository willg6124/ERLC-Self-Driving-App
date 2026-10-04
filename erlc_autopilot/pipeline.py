"""The full perceive -> decide -> act loop, decoupled from where the frame
came from (simulator renderer or a real screen capture of ER:LC) and
decoupled from where the command goes (simulator physics or real keyboard
input). This is the piece that makes the whole project reusable for the
actual game: swap `FrameSource` / `ActuatorSink` and everything else is
identical."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .config import AutopilotConfig
from .control.driving_policy import Command, DrivingPolicy
from .perception.emergency_vehicle import EmergencyVehicleDetector
from .perception.lane_detection import LaneDetector
from .perception.object_detection import Detection, build_default_detector
from .perception.traffic_light import TrafficLightDetector


@dataclass
class TickResult:
    command: Command
    debug_frame: "np.ndarray"
    lane: "object"
    detections: list
    light: Optional[object]
    emergency_vehicles: list
    fps: float


class AutopilotPipeline:
    def __init__(self, config: Optional[AutopilotConfig] = None, model_dir: str = "models"):
        self.config = config or AutopilotConfig()
        self.lane_detector = LaneDetector()
        self.object_detector = build_default_detector(model_dir)
        self.light_detector = TrafficLightDetector()
        self.ev_detector = EmergencyVehicleDetector()
        self.policy = DrivingPolicy(self.config)
        self._last_tick_time = None
        self._fps_ema = 0.0

    def set_engaged(self, engaged: bool):
        self.policy.set_engaged(engaged)

    def apply_config(self, config: AutopilotConfig):
        self.config = config
        self.policy.apply_config(config)

    def perceive_only(self, frame: "np.ndarray"):
        """Runs just the perception stack (lane/object/light/EV detection)
        without the driving policy -- used by the recorder (`erlc_autopilot
        /live/record_runner.py`) while you drive manually, so the features
        it logs come from exactly the same detectors (same instances, same
        internal smoothing/memory state) that would be driving the car,
        without needing to also compute (and discard) a Command every tick.
        Returns (lane, detections, light, ev_signals)."""
        lane = self.lane_detector.process(frame)
        detections = self.object_detector.detect(frame)
        light = self.light_detector.detect(frame)
        ev_signals = self.ev_detector.detect(frame, detections)
        return lane, detections, light, ev_signals

    def tick(self, frame: "np.ndarray", speed_mps: float, dt: float) -> TickResult:
        now = time.time()
        lane = self.lane_detector.process(frame)
        detections = self.object_detector.detect(frame)
        light = self.light_detector.detect(frame)
        ev_signals = self.ev_detector.detect(frame, detections)

        command = self.policy.step(lane, detections, light, speed_mps, dt, ev_signals)

        debug = lane.debug
        for d in detections:
            x1, y1, x2, y2 = d.bbox
            color = (0, 200, 255) if d.cls == "vehicle" else (0, 0, 255)
            cv2.rectangle(debug, (x1, y1), (x2, y2), color, 2)
            cv2.putText(debug, f"{d.cls} {d.distance_m:.1f}m", (x1, max(12, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
        if light is not None:
            x1, y1, x2, y2 = light.bbox
            lcolor = {"red": (0, 0, 255), "yellow": (0, 220, 255), "green": (0, 200, 0)}.get(light.state, (200, 200, 200))
            cv2.rectangle(debug, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), lcolor, 2)
            cv2.putText(debug, f"{light.state} {light.distance_m:.1f}m", (x1 - 4, max(12, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, lcolor, 1, cv2.LINE_AA)
        for ev in ev_signals:
            x1, y1, x2, y2 = ev.bbox
            ecolor = (255, 0, 255) if ev.flashing else (120, 0, 120)
            cv2.rectangle(debug, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), ecolor, 2)
            tag = "EMERGENCY VEHICLE" if ev.flashing else "possible lights"
            cv2.putText(debug, f"{tag} {ev.distance_m:.1f}m", (x1 - 3, max(12, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, ecolor, 1, cv2.LINE_AA)

        status_color = {
            "cruising": (0, 220, 0), "following": (0, 200, 220), "braking": (0, 140, 255),
            "stopped_at_light": (0, 140, 255), "emergency_braking": (0, 0, 255),
            "yielding": (255, 0, 255),
        }.get(command.status, (255, 255, 255))
        cv2.putText(debug, f"AUTOPILOT: {command.status.upper()}", (10, debug.shape[0] - 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2, cv2.LINE_AA)
        cv2.putText(debug, f"steer={command.steer:+.2f} throttle={command.throttle:.2f} brake={command.brake:.2f}",
                    (10, debug.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        if self._last_tick_time is not None:
            inst_fps = 1.0 / max(1e-6, now - self._last_tick_time)
            self._fps_ema = inst_fps if self._fps_ema == 0 else 0.9 * self._fps_ema + 0.1 * inst_fps
        self._last_tick_time = now

        return TickResult(command=command, debug_frame=debug, lane=lane, detections=detections,
                           light=light, emergency_vehicles=ev_signals, fps=self._fps_ema)


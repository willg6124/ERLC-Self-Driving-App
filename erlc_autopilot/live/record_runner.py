"""Recording mode: watches your screen and logs perception features paired
with the keys you're actually pressing, tick by tick, to a CSV -- *without*
sending any synthetic input of its own (you're the one driving). Run
`train_model.py` on the resulting file(s) afterwards to produce a small
model `run_live.py` can blend in at the Pro tier (see
`erlc_autopilot/control/imitation.py`). This is the "trainable from your
own recorded driving" half of the project, separate from (and blended on
top of, never replacing) the rule-based decision-making/lane-keeping stack
in `erlc_autopilot/control/driving_policy.py` and `erlc_autopilot/perception/`.
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Optional

from ..pipeline import AutopilotPipeline
from ..training.features import extract_features
from ..training.recorder import DrivingRecorder
from .human_input import HumanInputReader

TICK_HZ = 20
DT = 1.0 / TICK_HZ


class RecordRunner:
    def __init__(self, frame_source, out_path: str,
                 speed_fn: Optional[Callable[[], float]] = None,
                 model_dir: str = "models"):
        """`speed_fn`, if given, is a zero-arg callable returning the
        current speed in m/s (e.g. the simulator's ground truth, for
        testing this whole pipeline without a real human/game). If not
        given, speed is dead-reckoned from the human's own throttle/brake
        each tick -- there's no OCR speed reader wired up in recording mode
        since recording doesn't need to actuate anything, just observe."""
        self.frame_source = frame_source
        self.out_path = out_path
        self.pipeline = AutopilotPipeline(model_dir=model_dir)
        self.human_input = HumanInputReader()
        self.recorder: Optional[DrivingRecorder] = None
        self._speed_fn = speed_fn
        self._dead_reckon_mps = 0.0
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.frames_logged = 0
        self.error: Optional[str] = None

    def start(self) -> None:
        if not self.human_input.available():
            raise RuntimeError(
                "Can't read your real keyboard state (the 'keyboard' package isn't "
                "available, or couldn't get the permission it needs on this machine). "
                "Recording needs this to know what you actually pressed each frame -- "
                "without it there's nothing honest to learn from.")
        self.frame_source.start()
        self.recorder = DrivingRecorder(self.out_path)
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="autopilot-recorder")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self.recorder:
            self.recorder.close()

    def is_alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def _measure_speed(self, dt: float, throttle: float, brake: float) -> float:
        if self._speed_fn is not None:
            return self._speed_fn()
        accel = throttle * 3.0 - brake * 6.0 - 0.15
        self._dead_reckon_mps = max(0.0, min(45.0, self._dead_reckon_mps + accel * dt))
        return self._dead_reckon_mps

    def _run(self) -> None:
        last = time.time()
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
                    time.sleep(0.01)
                    continue
                steer, throttle, brake = self.human_input.read()
                speed_mps = self._measure_speed(dt, throttle, brake)
                lane, detections, light, ev_signals = self.pipeline.perceive_only(frame)
                features = extract_features(lane, detections, light, ev_signals, speed_mps)
                self.recorder.add(features, steer, throttle, brake)
                self.frames_logged += 1
            except Exception as exc:
                self.error = f"{exc.__class__.__name__}: {exc}"
                print(f"\n[recorder] tick error ({self.error}) -- continuing.")
                time.sleep(0.2)

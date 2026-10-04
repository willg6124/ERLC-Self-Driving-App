"""Frame sources for the autopilot pipeline.

Two implementations share one tiny interface (`start`/`stop`/`get_frame`)
so `erlc_autopilot.pipeline.AutopilotPipeline` and everything downstream
never has to know or care which one is feeding it:

  * `ScreenCaptureSource` -- grabs the calibrated region of the real
    screen during actual ER:LC gameplay (Windows, production use).
  * `SimFrameSource` -- wraps the bundled procedural simulator, so the
    exact same wizard -> overlay -> runner app can also be driven and
    tested entirely inside a dev machine / CI sandbox with no Roblox
    installed at all.

This is the literal implementation of the "decoupled from where the frame
came from" promise in pipeline.py.
"""
from __future__ import annotations

import threading
import time
from typing import Optional

import cv2
import numpy as np

from .window_capture import WindowRegion

# The perception stack (lane detector, object detector, EV detector) was
# tuned against this resolution -- real captures are resized down/up to
# match so the same gains/thresholds work regardless of the user's actual
# game resolution.
PIPELINE_FRAME_SIZE = (960, 540)


class ScreenCaptureSource:
    """Grabs the calibrated screen region at a fixed rate on its own
    background thread -- so a slow perception tick never throttles
    capture, the pipeline always consumes the freshest frame available --
    and resizes it to the resolution the CV stack expects, same as a real
    camera driver would. Uses `mss` (cross-platform) but is intended to
    run against a real ER:LC window on Windows.
    """

    def __init__(self, region: WindowRegion, target_fps: int = 30):
        self.region = region
        self.target_fps = target_fps
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._mss = None
        self.frames_captured = 0
        self.last_capture_error: Optional[str] = None

    def start(self) -> None:
        import mss  # imported lazily -- never needed in sim mode, and not
        # installed at all in this dev sandbox

        self._mss = mss.mss()
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="screen-capture")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._mss is not None:
            try:
                self._mss.close()
            except Exception:
                pass

    def set_region(self, region: WindowRegion) -> None:
        self.region = region

    def _run(self) -> None:
        dt = 1.0 / self.target_fps
        while self._running:
            t0 = time.time()
            try:
                shot = self._mss.grab(self.region.to_mss_region())
                frame = np.array(shot)[:, :, :3]  # BGRA -> BGR
                frame = cv2.resize(frame, PIPELINE_FRAME_SIZE, interpolation=cv2.INTER_AREA)
                with self._lock:
                    self._frame = frame
                self.frames_captured += 1
                self.last_capture_error = None
            except Exception as exc:  # keep capturing even if one grab hiccups
                self.last_capture_error = str(exc)
            elapsed = time.time() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)

    def get_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._frame is None else self._frame.copy()


class SimFrameSource:
    """Wraps the bundled procedural simulator (the same one behind the
    `/dashboard` test harness) behind the exact same interface as the real
    screen capture, so the real wizard -> overlay -> runner app can be
    exercised end-to-end without Roblox installed -- this is how the live
    app's orchestration logic gets tested in this repo/sandbox.
    """

    def __init__(self, world=None):
        from ..sim.world import World
        from ..sim.renderer import render

        self._render = render
        self.world = world or World()
        self.last_meta: dict = {}

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def get_frame(self) -> Optional[np.ndarray]:
        frame, meta = self._render(self.world)
        self.last_meta = meta
        return frame

    @property
    def ego_speed_mps(self) -> float:
        return self.world.ego.speed

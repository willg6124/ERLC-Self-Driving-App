"""Reads which of WASD you're actually holding down right now, so recording
mode can pair each frame's perception features with what you really did --
the "labels" for imitation learning. Uses the same `keyboard` package
already relied on for the F9 kill switch (global low-level key state,
works even without window focus, Windows-only).

Same lesson as the Tesseract bug: never let a missing/misbehaving OS
dependency here throw out of the recording loop uncaught -- `available()`
is checked once up front (and the caller refuses to start recording
without it, since silently logging all-zero actions would quietly produce
a garbage dataset), and `read()` itself never raises.
"""
from __future__ import annotations

from typing import Tuple


class HumanInputReader:
    def __init__(self):
        self._keyboard = None
        try:
            import keyboard

            # cheapest possible real call to confirm the backend actually
            # works on this machine/permission level, not just that the
            # package imported
            keyboard.is_pressed("w")
            self._keyboard = keyboard
        except Exception:
            self._keyboard = None

    def available(self) -> bool:
        return self._keyboard is not None

    def read(self) -> Tuple[float, float, float]:
        """Returns (steer, throttle, brake) approximating what the
        WASD-only control scheme can express: W/S give throttle/brake as
        plain 0/1 (that's all a held digital key can mean), A/D give
        steer -1/+1 (both held cancels out, like fighting the wheel)."""
        if self._keyboard is None:
            return 0.0, 0.0, 0.0
        try:
            k = self._keyboard
            steer = (1.0 if k.is_pressed("d") else 0.0) - (1.0 if k.is_pressed("a") else 0.0)
            throttle = 1.0 if k.is_pressed("w") else 0.0
            brake = 1.0 if k.is_pressed("s") else 0.0
            return steer, throttle, brake
        except Exception:
            return 0.0, 0.0, 0.0

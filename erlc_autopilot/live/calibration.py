"""Persists the one-time setup wizard's results (capture region, speed
readout region, chosen model tier, input backend) to disk so the app
remembers them on every subsequent launch instead of re-running the
wizard every time.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import List, Optional

DEFAULT_PATH = os.path.join(os.path.expanduser("~"), ".erlc_autopilot", "calibration.json")

_FIELDS = (
    "capture_left", "capture_top", "capture_width", "capture_height",
    "speed_roi", "model_tier", "input_backend", "calibrated",
)


@dataclass
class Calibration:
    capture_left: int = 0
    capture_top: int = 0
    capture_width: int = 1920
    capture_height: int = 1080
    # normalized (0..1) crop of the capture region where the in-game speed
    # readout lives -- (x1, y1, x2, y2); ER:LC's default HUD puts it
    # bottom-right, but UI scale/position varies with resolution/aspect,
    # hence calibrated per-user rather than hardcoded
    speed_roi: List[float] = field(default_factory=lambda: [0.86, 0.86, 0.99, 0.97])
    model_tier: str = "standard"  # "lite" | "standard" | "pro"
    input_backend: str = "auto"   # "auto" | "pydirectinput" | "pynput" | "dry_run"
    calibrated: bool = False

    def save(self, path: str = DEFAULT_PATH) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str = DEFAULT_PATH) -> "Calibration":
        if not os.path.exists(path):
            return cls()
        try:
            with open(path) as f:
                data = json.load(f)
            return cls(**{k: v for k, v in data.items() if k in _FIELDS})
        except Exception:
            return cls()

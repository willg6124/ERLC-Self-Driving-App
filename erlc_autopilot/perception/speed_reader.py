"""Reads the car's current speed from ER:LC's on-screen speedometer via
OCR.

Unlike the bundled simulator (which can just hand the pipeline its own
ground-truth speed), real gameplay offers no such signal -- the dashcam
frame is the *only* source of truth, same as every real sensor-based
system has to work from what it can actually observe. `SpeedReader` crops
the calibrated HUD speed readout, cleans it up for OCR, and parses out a
number.

Gracefully degrades: if `pytesseract`/the `tesseract` binary aren't
installed (never are in this dev sandbox, and not every user will bother
installing a system binary), `read_mph()` returns None and the caller
falls back to a rough dead-reckoning estimate integrated from recent
throttle/brake commands -- approximate, but keeps the control loop from
driving blind on a frozen/zero speed rather than failing outright.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    import pytesseract

    _HAS_TESSERACT = True
except Exception:
    pytesseract = None
    _HAS_TESSERACT = False

MPH_TO_MPS = 0.44704


@dataclass
class SpeedReaderConfig:
    # normalized (0..1) crop of the *full capture frame* -- not the
    # resized pipeline frame -- where the speed readout lives. Matches
    # Calibration.speed_roi from the wizard.
    roi: List[float] = field(default_factory=lambda: [0.86, 0.86, 0.99, 0.97])


class SpeedReader:
    def __init__(self, config: Optional[SpeedReaderConfig] = None):
        self.cfg = config or SpeedReaderConfig()
        self._dead_reckon_mps: float = 0.0

    def available(self) -> bool:
        return _HAS_TESSERACT

    def _crop(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = self.cfg.roi
        x1, x2 = int(x1 * w), int(x2 * w)
        y1, y2 = int(y1 * h), int(y2 * h)
        y1, y2 = max(0, y1), max(1, y2)
        x1, x2 = max(0, x1), max(1, x2)
        return frame[y1:y2, x1:x2]

    def read_mph(self, frame: np.ndarray) -> Optional[float]:
        """OCR'd speed in mph, or None if OCR isn't available or didn't
        find a confident digit string this frame (caller should fall back
        to dead reckoning / the previous good reading)."""
        if not _HAS_TESSERACT:
            return None
        roi = self._crop(frame)
        if roi.size == 0:
            return None
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # the HUD speed readout is bright text on a dark translucent panel
        # -- threshold to pure black/white so tesseract sees clean digit
        # glyphs instead of anti-aliased grey, then upscale (tesseract is
        # tuned for document-sized text, not a 20px-tall HUD element)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        thresh = cv2.resize(thresh, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        text = pytesseract.image_to_string(
            thresh, config="--psm 7 -c tessedit_char_whitelist=0123456789")
        digits = re.sub(r"[^0-9]", "", text)
        if not digits:
            return None
        try:
            mph = float(digits[:3])  # no ER:LC vehicle realistically does 1000+
        except ValueError:
            return None
        if mph > 200:  # OCR misread guard
            return None
        return mph

    def estimate_mps(self, frame: np.ndarray, dt: float, throttle: float, brake: float) -> Tuple[float, bool]:
        """Best available speed estimate in m/s, plus whether it came from
        real OCR this tick (True) or degraded dead-reckoning (False) --
        surfaced in the HUD so you know when the bot is driving with a
        real speed signal vs. guessing."""
        mph = self.read_mph(frame)
        if mph is not None:
            self._dead_reckon_mps = mph * MPH_TO_MPS
            return self._dead_reckon_mps, True
        # degraded mode: tesseract not installed, or a transient misread
        # -- integrate a rough accel model from the last commanded pedals
        # so the control loop still has *something* to reason about
        accel = throttle * 3.0 - brake * 6.0 - 0.15
        self._dead_reckon_mps = max(0.0, min(45.0, self._dead_reckon_mps + accel * dt))
        return self._dead_reckon_mps, False

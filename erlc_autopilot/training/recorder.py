"""Writes one row per tick -- perception features plus the steer/throttle/
brake that actually happened that tick -- to a plain CSV file. Deliberately
not pandas/numpy-file based: a CSV is trivially appendable, human-readable,
and good enough for the dataset sizes a single recorded driving session
produces (hundreds to a few thousand rows), and keeps `requirements.txt`
unchanged.
"""
from __future__ import annotations

import csv
import os
from typing import List, Optional, TextIO

from .features import FEATURE_NAMES

HEADER = FEATURE_NAMES + ["steer", "throttle", "brake"]


class DrivingRecorder:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        is_new = not os.path.exists(path) or os.path.getsize(path) == 0
        self._fh: Optional[TextIO] = open(path, "a", newline="")
        self._writer = csv.writer(self._fh)
        if is_new:
            self._writer.writerow(HEADER)
            self._fh.flush()
        self.rows_written = 0

    def add(self, features: List[float], steer: float, throttle: float, brake: float) -> None:
        self._writer.writerow([*features, steer, throttle, brake])
        self.rows_written += 1
        if self.rows_written % 20 == 0:
            self._fh.flush()  # don't lose a whole session if the process is killed mid-recording

    def close(self) -> None:
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
            self._fh = None

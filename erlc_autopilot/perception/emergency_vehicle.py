"""
Emergency-vehicle (police / fire / EMS) light-bar detection.

A "Move Over Law" compliant driver -- and this autopilot -- needs to
notice an active light bar (a tight cluster of very saturated red AND
blue, or red and white, pixels near the roofline of a nearby vehicle)
and react by slowing down and biasing away from it, whether it's
overtaking from behind or already stopped on the shoulder ahead.

This is intentionally independent of vehicle *classification*: the light
bar itself is the distinctive, reliable signal. A car can be red, a car
can be blue, but a tight cluster of saturated red+blue pixels that
persists/flickers across several frames essentially never happens by
coincidence on an ordinary vehicle.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .object_detection import Detection


@dataclass
class EmergencyVehicleSignal:
    bbox: Tuple[int, int, int, int]
    distance_m: float
    confidence: float
    flashing: bool  # confirmed via multi-frame history, not just one lucky frame


class EmergencyVehicleDetector:
    """Scans the roof region of every already-detected 'vehicle' box for an
    active light bar. A short rolling history per rough screen location is
    used to tell a genuinely flashing bar (color present, and either both
    colors at once or alternating across frames) apart from a one-frame
    fluke."""

    def __init__(self, history_len: int = 6, max_tracked: int = 64):
        self.history_len = history_len
        self.max_tracked = max_tracked
        self._history: Dict[Tuple[int, int], Deque[Optional[str]]] = {}

    @staticmethod
    def _bucket(cx: float, cy: float) -> Tuple[int, int]:
        return (int(cx // 40), int(cy // 40))

    @staticmethod
    def _color_fractions(region: np.ndarray) -> Optional[Tuple[float, float]]:
        if region.size == 0:
            return None
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        red1 = cv2.inRange(hsv, (0, 140, 140), (8, 255, 255))
        red2 = cv2.inRange(hsv, (170, 140, 140), (180, 255, 255))
        red_mask = cv2.bitwise_or(red1, red2)
        blue_mask = cv2.inRange(hsv, (100, 120, 120), (130, 255, 255))
        total = region.shape[0] * region.shape[1]
        if total == 0:
            return None
        return float(np.count_nonzero(red_mask)) / total, float(np.count_nonzero(blue_mask)) / total

    def _prune(self):
        if len(self._history) > self.max_tracked:
            # drop the oldest-looking (shortest/emptiest) entries first
            stale = sorted(self._history.items(), key=lambda kv: len(kv[1]))[: len(self._history) - self.max_tracked]
            for k, _ in stale:
                self._history.pop(k, None)

    def detect(self, frame: np.ndarray, vehicle_detections: List[Detection]) -> List[EmergencyVehicleSignal]:
        h, w = frame.shape[:2]
        signals: List[EmergencyVehicleSignal] = []
        for d in vehicle_detections:
            # scan anything roughly vehicle-shaped OR pedestrian-shaped --
            # a light bar sitting on a distant/angled vehicle can get
            # mis-bucketed by the coarse aspect-ratio heuristic upstream,
            # but the color signature on its roof is what actually matters
            x1, y1, x2, y2 = d.bbox
            roof_h = max(3, int((y2 - y1) * 0.5))
            rx1, ry1 = max(0, x1), max(0, y1 - roof_h)
            rx2, ry2 = min(w, x2), min(h, y1 + 2)
            if rx2 <= rx1 or ry2 <= ry1:
                continue
            fracs = self._color_fractions(frame[ry1:ry2, rx1:rx2])
            if fracs is None:
                continue
            red_frac, blue_frac = fracs

            dominant: Optional[str] = None
            if red_frac > 0.08 and blue_frac > 0.08:
                dominant = "both"
            elif red_frac > 0.15:
                dominant = "red"
            elif blue_frac > 0.15:
                dominant = "blue"

            key = self._bucket((x1 + x2) / 2, y1)
            hist = self._history.setdefault(key, deque(maxlen=self.history_len))
            hist.append(dominant)

            present = [c for c in hist if c is not None]
            flashing = dominant == "both" or (len(present) >= 2 and len(set(present)) > 1)
            if dominant is not None:
                confidence = min(1.0, 0.5 + 0.1 * len(present) + (0.2 if dominant == "both" else 0.0))
                signals.append(EmergencyVehicleSignal(
                    bbox=(x1, ry1, x2, y2), distance_m=d.distance_m,
                    confidence=confidence, flashing=flashing))

        self._prune()
        signals.sort(key=lambda s: s.distance_m)
        return signals

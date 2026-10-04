"""
Traffic-light state recognition: finds dark signal-head blobs in the upper
portion of the frame and classifies which lamp is lit via HSV color
masking, same technique real-world traffic-light classifiers in ADAS
research prototypes use for a cheap, explainable first pass.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

HSV_RANGES = {
    "red": [((0, 120, 150), (10, 255, 255)), ((170, 120, 150), (180, 255, 255))],
    "yellow": [((18, 120, 150), (33, 255, 255))],
    "green": [((45, 80, 120), (90, 255, 255))],
}


@dataclass
class TrafficLightDetection:
    state: str  # "red" | "yellow" | "green" | "unknown"
    bbox: Tuple[int, int, int, int]
    distance_m: float
    confidence: float


class TrafficLightDetector:
    def __init__(self, focal_px: float = 620.0, lamp_height_m: float = 0.35):
        self.focal_px = focal_px
        self.lamp_height_m = lamp_height_m

    def detect(self, frame: np.ndarray) -> Optional[TrafficLightDetection]:
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi_y2 = int(h * 0.55)
        best = None
        for state, ranges in HSV_RANGES.items():
            mask = np.zeros((h, w), dtype=np.uint8)
            for lo, hi in ranges:
                mask |= cv2.inRange(hsv, lo, hi)
            mask[roi_y2:, :] = 0
            mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area = cv2.contourArea(c)
                if area < 10:
                    continue
                x, y, bw, bh = cv2.boundingRect(c)
                aspect = bw / max(1, bh)
                if not (0.5 < aspect < 2.0):
                    continue
                # a traffic-light lamp renders as a roughly circular blob;
                # lane paint (dashes/edges) renders as thin elongated
                # slivers -- circularity rejects most of those.
                perimeter = cv2.arcLength(c, True)
                circularity = 4 * np.pi * area / max(1e-6, perimeter ** 2)
                if circularity < 0.55:
                    continue
                # a real signal head sits in a dark housing; lane paint
                # sits on the (much brighter) road surface -- check the
                # ring immediately around the blob is dark to reject paint.
                pad = max(3, bw // 2, bh // 2)
                ry1, ry2 = max(0, y - pad), min(h, y + bh + pad)
                rx1, rx2 = max(0, x - pad), min(w, x + bw + pad)
                ring_region = gray[ry1:ry2, rx1:rx2].copy()
                inner = gray[y:y + bh, x:x + bw]
                if ring_region.size == 0:
                    continue
                ring_region[y - ry1:y - ry1 + bh, x - rx1:x - rx1 + bw] = 255
                dark_pixels = ring_region[ring_region < 40]
                if dark_pixels.size < 0.25 * ring_region.size:
                    continue  # not surrounded by a dark housing -> not a light
                score = area
                if best is None or score > best[0]:
                    best = (score, state, (x, y, x + bw, y + bh))
        if best is None:
            return None
        _, state, bbox = best
        bh = max(1, bbox[3] - bbox[1])
        distance = round(self.focal_px * self.lamp_height_m / bh, 1)
        return TrafficLightDetection(state=state, bbox=bbox, distance_m=distance, confidence=0.85)

"""
Obstacle / traffic-light perception.

Two detector backends are provided behind the same `Detector` interface:

  * `HeuristicDetector` -- a dependency-light classical CV detector (HSV
    saturation/value segmentation + contour analysis + pinhole-camera
    distance estimation from apparent bounding-box height). Works out of
    the box against the bundled simulator and is a reasonable starting
    point against the real game too.

  * `MobileSSDDetector` -- a real pretrained MobileNet-SSD (Caffe) object
    detector via `cv2.dnn`, for production use against real ER:LC screen
    captures where vehicles/pedestrians won't be flat simulator colors.
    Model weights are not vendored in the repo (see models/README.md) --
    this class degrades gracefully to the heuristic detector if the
    weight files are not present.

Both return a common `Detection` type so the rest of the stack (control /
dashboard) never has to care which backend is active.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

CAMERA_FOCAL_PX = 620.0  # matches erlc_autopilot.sim.renderer.FOV_SCALE
ASSUMED_HEIGHTS = {"vehicle": 1.5, "pedestrian": 1.6}


@dataclass
class Detection:
    cls: str                      # "vehicle" | "pedestrian"
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    distance_m: float
    confidence: float


class HeuristicDetector:
    """Classical contour-based obstacle detector + distance estimation."""

    def __init__(self, focal_px: float = CAMERA_FOCAL_PX, min_area: int = 90):
        self.focal_px = focal_px
        self.min_area = min_area

    def _estimate_distance(self, cls: str, box_h: int) -> float:
        if box_h <= 1:
            return 999.0
        return round(self.focal_px * ASSUMED_HEIGHTS[cls] / box_h, 1)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # road is a flat low-saturation grey; most non-road, non-sky, non-lane
        # paint blobs below the horizon are candidate obstacles.
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        # NOTE: a narrow-vertical-FOV dashcam compresses a lot of forward
        # distance into a small band of rows near the true vanishing point
        # -- a vehicle/pedestrian 20-50m out can sit well above the naive
        # "just below the horizon" guess this used to be (0.46*h), which
        # was excluding anything past ~8-10m entirely. Cast the net wider;
        # `not_sky` below still screens out genuine sky pixels.
        horizon = int(h * 0.18)
        road_region = np.zeros((h, w), dtype=np.uint8)
        road_region[horizon:h - 25, :] = 255

        # anything reasonably saturated/bright that isn't the yellow lane
        # dashes, white lane paint, the grass shoulder, or the sky
        colorful = cv2.inRange(sat, 60, 255)
        not_lane_yellow = cv2.bitwise_not(cv2.inRange(hsv, (10, 60, 60), (42, 255, 255)))
        not_grass = cv2.bitwise_not(cv2.inRange(hsv, (45, 60, 70), (75, 160, 170)))
        not_sky = cv2.bitwise_not(cv2.inRange(hsv, (85, 0, 150), (135, 255, 255)))
        not_too_bright = cv2.inRange(val, 0, 240)
        candidate = cv2.bitwise_and(colorful, not_lane_yellow)
        candidate = cv2.bitwise_and(candidate, not_grass)
        candidate = cv2.bitwise_and(candidate, not_sky)
        candidate = cv2.bitwise_and(candidate, not_too_bright)
        candidate = cv2.bitwise_and(candidate, road_region)

        # erode first to kill thin 1-2px anti-aliased fringes along lane
        # paint edges, which otherwise masquerade as tiny "vehicles"
        candidate = cv2.erode(candidate, np.ones((2, 2), np.uint8), iterations=1)
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        candidate = cv2.dilate(candidate, np.ones((3, 3), np.uint8), iterations=1)

        contours, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: List[Detection] = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            aspect = bw / max(1, bh)
            cls: Optional[str] = None
            if 0.1 < aspect < 0.55 and bh > 10:
                cls = "pedestrian"
            elif 0.5 <= aspect < 3.0 and bw > 8:
                cls = "vehicle"
            if cls is None:
                continue
            dist = self._estimate_distance(cls, bh)
            confidence = float(np.clip(area / 1200.0, 0.2, 0.98))
            detections.append(Detection(cls=cls, bbox=(x, y, x + bw, y + bh),
                                         distance_m=dist, confidence=confidence))
        detections.sort(key=lambda d: d.distance_m)
        return detections


class MobileSSDDetector:
    """Real pretrained detector (MobileNet-SSD / Caffe) for production use
    against actual ER:LC screen captures. Falls back to HeuristicDetector
    automatically if weight files aren't present on disk."""

    VOC_CLASSES = [
        "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
        "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
        "pottedplant", "sheep", "sofa", "train", "tvmonitor",
    ]
    VEHICLE_CLASSES = {"car", "bus", "motorbike", "bicycle"}

    def __init__(self, model_dir: str = "models", focal_px: float = CAMERA_FOCAL_PX,
                 conf_threshold: float = 0.45):
        prototxt = os.path.join(model_dir, "MobileNetSSD_deploy.prototxt")
        caffemodel = os.path.join(model_dir, "MobileNetSSD_deploy.caffemodel")
        self.focal_px = focal_px
        self.conf_threshold = conf_threshold
        self.net = None
        self.fallback = HeuristicDetector(focal_px=focal_px)
        if os.path.exists(prototxt) and os.path.exists(caffemodel):
            self.net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if self.net is None:
            return self.fallback.detect(frame)
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
        self.net.setInput(blob)
        out = self.net.forward()
        detections: List[Detection] = []
        for i in range(out.shape[2]):
            conf = float(out[0, 0, i, 2])
            if conf < self.conf_threshold:
                continue
            class_id = int(out[0, 0, i, 1])
            if class_id >= len(self.VOC_CLASSES):
                continue
            name = self.VOC_CLASSES[class_id]
            if name == "person":
                cls = "pedestrian"
            elif name in self.VEHICLE_CLASSES:
                cls = "vehicle"
            else:
                continue
            box = out[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            bh = max(1, y2 - y1)
            dist = round(self.focal_px * ASSUMED_HEIGHTS[cls] / bh, 1)
            detections.append(Detection(cls=cls, bbox=(x1, y1, x2, y2), distance_m=dist, confidence=conf))
        detections.sort(key=lambda d: d.distance_m)
        return detections


def build_default_detector(model_dir: str = "models"):
    """Use the real MobileNet-SSD model if weights are available, otherwise
    fall back to the dependency-light heuristic detector automatically."""
    detector = MobileSSDDetector(model_dir=model_dir)
    return detector if detector.net is not None else HeuristicDetector()

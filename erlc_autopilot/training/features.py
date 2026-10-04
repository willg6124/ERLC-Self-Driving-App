"""Turns one tick's perception outputs into a fixed-order numeric feature
vector. The *exact same* function is used both when recording a driving
session (paired with whatever steer/throttle/brake actually happened that
tick) and at inference time (paired with the trained model's prediction),
so there's no risk of train/serve skew between the two.

Deliberately feature-based rather than raw-pixel-based: a few numbers
(lane offset/curvature/confidence, nearest obstacle distances, traffic
light state, speed) is enough for a small, fast, CPU-only scikit-learn
model to learn a *driving style* on top of the rule-based policy, without
needing a GPU, thousands of recorded frames, or shipping a vision model of
its own -- the existing lane/object/light/EV detectors already did the
hard perception work; this just hands their output to the learner.
"""
from __future__ import annotations

from typing import List, Optional

FRAME_CENTER_X = 480.0
VEHICLE_LANE_MARGIN_PX = 200.0
PEDESTRIAN_LANE_MARGIN_PX = 140.0
NO_OBJECT_DISTANCE_M = 999.0

LIGHT_STATE_CODE = {"green": 1.0, "yellow": 2.0, "red": 3.0}

FEATURE_NAMES = [
    "lane_offset_norm",
    "lane_curvature",
    "lane_confidence",
    "lookahead_offset_norm",
    "speed_mps",
    "lead_vehicle_present",
    "lead_vehicle_distance_m",
    "pedestrian_present",
    "pedestrian_distance_m",
    "light_present",
    "light_state_code",
    "light_distance_m",
    "ev_active",
    "ev_distance_m",
]


def _lookahead_offset_norm(lane) -> Optional[float]:
    if lane.left_line is None or lane.right_line is None:
        return None
    far_left_x, far_right_x = lane.left_line[2], lane.right_line[2]
    far_width = far_right_x - far_left_x
    if far_width < 20:
        return None
    far_center = (far_left_x + far_right_x) / 2.0
    return max(-2.5, min(2.5, (FRAME_CENTER_X - far_center) / (far_width / 2.0)))


def extract_features(lane, detections, light, ev_signals, speed_mps: float) -> List[float]:
    """Returns a list of floats in `FEATURE_NAMES` order."""
    lookahead = _lookahead_offset_norm(lane)
    if lookahead is None:
        lookahead = lane.offset_norm

    lead = next((d for d in detections if d.cls == "vehicle"
                 and abs((d.bbox[0] + d.bbox[2]) / 2 - FRAME_CENTER_X) < VEHICLE_LANE_MARGIN_PX), None)
    ped = next((d for d in detections if d.cls == "pedestrian"
                and abs((d.bbox[0] + d.bbox[2]) / 2 - FRAME_CENTER_X) < PEDESTRIAN_LANE_MARGIN_PX), None)
    active_ev = [e for e in (ev_signals or []) if e.flashing]
    nearest_ev = min(active_ev, key=lambda e: e.distance_m) if active_ev else None

    return [
        float(lane.offset_norm),
        float(lane.curvature),
        float(lane.confidence),
        float(lookahead),
        float(speed_mps),
        1.0 if lead is not None else 0.0,
        float(lead.distance_m) if lead is not None else NO_OBJECT_DISTANCE_M,
        1.0 if ped is not None else 0.0,
        float(ped.distance_m) if ped is not None else NO_OBJECT_DISTANCE_M,
        1.0 if light is not None else 0.0,
        LIGHT_STATE_CODE.get(light.state, 0.0) if light is not None else 0.0,
        float(light.distance_m) if light is not None else NO_OBJECT_DISTANCE_M,
        1.0 if nearest_ev is not None else 0.0,
        float(nearest_ev.distance_m) if nearest_ev is not None else NO_OBJECT_DISTANCE_M,
    ]

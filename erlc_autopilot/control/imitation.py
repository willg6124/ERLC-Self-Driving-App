"""Loads a trained imitation model (see `train_model.py`) and turns one
tick's perception features into a predicted (steer, throttle, brake) --
entirely optional, and only ever used as a *blend* on top of the
rule-based `DrivingPolicy`, never a replacement for it. If no trained
model exists on disk yet (the common case until you've recorded and
trained one), everything using this degrades to pure rule-based driving
with zero behavior change -- same pattern as the MobileNet-SSD detector
falling back to the heuristic one when its weight files aren't present.
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

from ..training.features import FEATURE_NAMES


class ImitationModel:
    def __init__(self, sk_model):
        self._model = sk_model

    @classmethod
    def load(cls, path: str) -> Optional["ImitationModel"]:
        if not os.path.exists(path):
            return None
        try:
            import joblib

            payload = joblib.load(path)
            model = payload["model"] if isinstance(payload, dict) else payload
            return cls(model)
        except Exception as exc:
            print(f"(Couldn't load imitation model at {path}: {exc} -- driving purely rule-based.)")
            return None

    def predict(self, features: List[float]) -> Optional[Tuple[float, float, float]]:
        try:
            pred = self._model.predict([features])[0]
            steer, throttle, brake = float(pred[0]), float(pred[1]), float(pred[2])
            return (max(-1.0, min(1.0, steer)),
                    max(0.0, min(1.0, throttle)),
                    max(0.0, min(1.0, brake)))
        except Exception:
            return None


def build_imitation_model_for_tier(tier: str, model_dir: str = "models") -> Optional[ImitationModel]:
    """Only the Pro tier ever blends a learned model in (see
    AutopilotConfig.for_tier and DrivingPolicy.step) -- Lite/Standard stay
    100% rule-based regardless of whether a trained model file exists."""
    if tier != "pro":
        return None
    return ImitationModel.load(os.path.join(model_dir, "imitation_model.joblib"))


__all__ = ["ImitationModel", "build_imitation_model_for_tier", "FEATURE_NAMES"]

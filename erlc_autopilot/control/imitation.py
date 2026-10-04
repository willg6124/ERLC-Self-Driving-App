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
    """Picks whichever trained steering model should be blended in, if
    any -- same safety-gated blend hook either way (see
    DrivingPolicy.step): only `steer` is ever used, throttle/brake and
    every hard safety rule stay 100% rule-based regardless.

    Preference order:
      1. models/rl_policy.joblib -- the self-training (reinforcement
         learning / Evolution Strategies, see train_rl.py) model. This
         needs no recorded driving data at all, so it's offered on every
         tier, not just Pro.
      2. models/imitation_model.joblib -- the older imitation-learning
         path (train_model.py), which does need you to have recorded
         your own driving first. Kept Pro-only, same as before, so
         Lite/Standard behavior is unchanged unless you've actually
         trained an RL model.
    """
    rl_model = ImitationModel.load(os.path.join(model_dir, "rl_policy.joblib"))
    if rl_model is not None:
        return rl_model
    if tier != "pro":
        return None
    return ImitationModel.load(os.path.join(model_dir, "imitation_model.joblib"))


__all__ = ["ImitationModel", "build_imitation_model_for_tier", "FEATURE_NAMES"]

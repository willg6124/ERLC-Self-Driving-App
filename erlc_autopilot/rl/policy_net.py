"""A tiny pure-numpy MLP: 14 perception features -> (steer, throttle,
brake). No PyTorch/TensorFlow/GPU needed -- this is small and cheap
enough (a few hundred parameters) that plain numpy matrix multiplies are
all it takes, both to run it live and to train it with Evolution
Strategies (see train_es.py).

Exposes the same `.predict(X)` sklearn-style interface the imitation
model's RandomForestRegressor does, so a trained RL policy is a drop-in
replacement in `models/*.joblib` for `ImitationModel.load()` -- no
changes needed anywhere else in the app.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

HIDDEN_UNITS = 24


class MLPPolicy:
    def __init__(self, n_in: int, n_hidden: int = HIDDEN_UNITS, n_out: int = 3,
                 obs_mean: Optional[np.ndarray] = None, obs_std: Optional[np.ndarray] = None,
                 seed: int = 0):
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.obs_mean = np.zeros(n_in) if obs_mean is None else np.asarray(obs_mean, dtype=np.float64)
        self.obs_std = np.ones(n_in) if obs_std is None else np.asarray(obs_std, dtype=np.float64)
        self.obs_std[self.obs_std < 1e-6] = 1.0

        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 0.15, (n_in, n_hidden))
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.normal(0, 0.15, (n_hidden, n_out))
        self.b2 = np.zeros(n_out)
        self._warm_start_steering()

    def _warm_start_steering(self):
        """Hand-initialize the steer output to roughly mimic a simple
        proportional controller on `lane_offset_norm` (feature 0) before
        any training happens -- a cheap, legitimate "warm start" (like
        picking sane initial PID gains) so Evolution Strategies spends its
        limited generations refining a reasonable policy instead of
        random-walking up from a useless one. Throttle/brake start at a
        neutral/unbiased point since only `steer` is ever actually used
        by the live driving policy (see control/driving_policy.py)."""
        self.W1[:, 0] = 0.0
        self.W1[0, 0] = 3.0   # strong initial response to lane_offset_norm through tanh hidden unit 0
        self.b1[0] = 0.0
        self.W2[:, 0] = 0.0
        self.W2[0, 0] = -1.5  # negative: steer away from a positive (rightward) offset

    def num_params(self) -> int:
        return self.W1.size + self.b1.size + self.W2.size + self.b2.size

    def get_flat_weights(self) -> np.ndarray:
        return np.concatenate([self.W1.ravel(), self.b1.ravel(), self.W2.ravel(), self.b2.ravel()])

    def set_flat_weights(self, flat: np.ndarray) -> None:
        flat = np.asarray(flat, dtype=np.float64)
        i = 0
        n = self.n_in * self.n_hidden
        self.W1 = flat[i:i + n].reshape(self.n_in, self.n_hidden); i += n
        n = self.n_hidden
        self.b1 = flat[i:i + n].copy(); i += n
        n = self.n_hidden * self.n_out
        self.W2 = flat[i:i + n].reshape(self.n_hidden, self.n_out); i += n
        n = self.n_out
        self.b2 = flat[i:i + n].copy(); i += n

    def clone_with_weights(self, flat: np.ndarray) -> "MLPPolicy":
        clone = MLPPolicy(self.n_in, self.n_hidden, self.n_out, self.obs_mean, self.obs_std)
        clone.set_flat_weights(flat)
        return clone

    def _forward(self, X: np.ndarray) -> np.ndarray:
        x = (X - self.obs_mean) / self.obs_std
        x = np.clip(x, -8.0, 8.0)
        h = np.tanh(x @ self.W1 + self.b1)
        o = h @ self.W2 + self.b2
        steer = np.tanh(o[:, 0])
        throttle = 1.0 / (1.0 + np.exp(-o[:, 1]))
        brake = 1.0 / (1.0 + np.exp(-o[:, 2]))
        return np.stack([steer, throttle, brake], axis=1)

    def predict(self, X) -> np.ndarray:
        """sklearn-style: X is array-like of shape (n_samples, n_features),
        returns shape (n_samples, 3) = (steer, throttle, brake)."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[None, :]
        return self._forward(X)

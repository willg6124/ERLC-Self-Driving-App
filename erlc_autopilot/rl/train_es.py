"""Evolution Strategies trainer: a genuinely self-training ("trains
itself through trial and error", zero human demonstrations needed)
reinforcement-learning algorithm, plugged straight into the bundled
simulator.

How it works (OpenAI-ES / "natural evolution strategies" style, mirrored
sampling):
  1. Start from one weight vector `theta` (a small MLP, see policy_net.py).
  2. Each generation, sample a batch of random perturbations and try
     `theta + sigma*eps` AND `theta - sigma*eps` for each one (mirrored
     sampling -- halves the variance of the gradient estimate for free).
  3. Drive the actual bundled simulator with each perturbed network
     plugged in as the live steering model (same blend hook used by
     imitation learning) for a short episode, and score it: reward for
     staying lane-centered and steering smoothly, heavy penalty for
     collisions/off-road.
  4. Nudge `theta` towards whichever perturbations scored better
     (reward-weighted average direction) and repeat.

No gradients, no backprop, no PyTorch/TensorFlow -- just running the
simulator forward many times and keeping what works, which is the
literal definition of "trains itself" the user asked for.
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

from ..config import AutopilotConfig
from ..control.imitation import ImitationModel
from ..pipeline import AutopilotPipeline
from ..sim.renderer import render
from ..sim.world import World
from ..training.features import FEATURE_NAMES
from .policy_net import MLPPolicy


@dataclass
class EpisodeResult:
    reward: float
    collisions: int
    off_road_time: float
    mean_abs_offset: float
    final_speed_mph: float
    disengaged: bool


def rollout(flat_weights: np.ndarray, seed: int, tier: str = "standard",
            episode_ticks: int = 150, obs_mean: Optional[np.ndarray] = None,
            obs_std: Optional[np.ndarray] = None, blend_weight: float = 1.0,
            dt: float = 1.0 / 20.0) -> EpisodeResult:
    """Runs the REAL production pipeline (same DrivingPolicy.step(), same
    World physics, same perception stack) for one short episode with the
    candidate network plugged in as the steering model. This is
    deliberately not a separate simplified training environment: using
    the exact code path that ends up driving for real means there is
    zero train/inference mismatch, at the cost of training being slower
    than a bespoke fast simulator would be (each tick pays for real
    OpenCV lane/object detection, same as live driving does)."""
    n_in = len(FEATURE_NAMES)
    net = MLPPolicy(n_in, obs_mean=obs_mean, obs_std=obs_std)
    net.set_flat_weights(flat_weights)

    cfg = AutopilotConfig.for_tier(tier)
    cfg.imitation_blend_weight = blend_weight
    pipe = AutopilotPipeline(cfg)
    pipe.set_engaged(True)
    pipe.policy.imitation_model = ImitationModel(net)

    world = World(seed=seed)

    total_reward = 0.0
    prev_steer = 0.0
    abs_offsets: List[float] = []
    prev_collisions = 0
    prev_offroad = 0.0
    disengaged_ticks = 0

    for _ in range(episode_ticks):
        frame, _ = render(world)
        result = pipe.tick(frame=frame, speed_mps=world.ego.speed, dt=dt)
        cmd = result.command
        world.update(dt, cmd.steer, cmd.throttle, cmd.brake)

        offset = float(result.lane.offset_norm)
        abs_offsets.append(abs(offset))
        r = -(offset ** 2)
        r -= 0.1 * (cmd.steer - prev_steer) ** 2
        prev_steer = cmd.steer

        if world.collisions > prev_collisions:
            r -= 3.0 * (world.collisions - prev_collisions)
            prev_collisions = world.collisions
        if world.off_road_time > prev_offroad + 1e-9:
            r -= 0.2
            prev_offroad = world.off_road_time
        if not cmd.engaged:
            disengaged_ticks += 1
            r -= 0.05

        total_reward += r

    avg_reward = total_reward / max(1, episode_ticks)
    return EpisodeResult(
        reward=avg_reward,
        collisions=world.collisions,
        off_road_time=world.off_road_time,
        mean_abs_offset=float(np.mean(abs_offsets)) if abs_offsets else 0.0,
        final_speed_mph=world.ego.speed * 2.237,
        disengaged=disengaged_ticks > episode_ticks * 0.3,
    )


def _worker_init() -> None:
    """Runs once per worker process at pool startup. OpenCV spawns its own
    internal thread pool per process by default (for Canny/blur/etc); with
    N worker *processes* each also fanning out to multiple cv2 *threads*,
    a 2-core machine running --workers 2 ends up with 4+ threads fighting
    over 2 cores, which was measured to cause a ~20x slowdown (every
    tick's cv2 calls paying constant-sized work but getting a fraction of
    a core's attention amid constant context-switching) versus a single
    process running the exact same workload uncontended. Each worker
    process already IS the unit of parallelism here, so cv2 itself should
    stick to one thread.
    """
    try:
        import cv2

        cv2.setNumThreads(1)
    except Exception:
        pass


def _eval_worker(args) -> float:
    flat_weights, seed, tier, episode_ticks, obs_mean, obs_std, blend_weight = args
    return rollout(flat_weights, seed, tier, episode_ticks, obs_mean, obs_std, blend_weight).reward


def _collect_obs_stats(tier: str, n_episodes: int = 3, episode_ticks: int = 100,
                        seed0: int = 10_000) -> Tuple[np.ndarray, np.ndarray]:
    """Runs a few purely rule-based (no learned blend) episodes just to
    get realistic mean/std for each of the 14 features, so the network's
    input normalization matches what it'll actually see."""
    from ..training.features import extract_features

    cfg = AutopilotConfig.for_tier(tier)
    rows: List[List[float]] = []
    for i in range(n_episodes):
        pipe = AutopilotPipeline(cfg)
        pipe.set_engaged(True)
        world = World(seed=seed0 + i)
        dt = 1.0 / 20.0
        for _ in range(episode_ticks):
            frame, _ = render(world)
            lane, detections, light, ev = pipe.perceive_only(frame)
            rows.append(extract_features(lane, detections, light, ev, world.ego.speed))
            result = pipe.tick(frame=frame, speed_mps=world.ego.speed, dt=dt)
            cmd = result.command
            world.update(dt, cmd.steer, cmd.throttle, cmd.brake)
    arr = np.array(rows, dtype=np.float64)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean, std


def train(tier: str = "standard", generations: int = 40, population: int = 12,
          episode_ticks: int = 150, sigma: float = 0.1, lr: float = 0.05,
          workers: int = 2, seed: int = 0, out_path: str = "models/rl_policy.joblib",
          progress_cb: Optional[Callable[[int, int, float, float, dict], None]] = None,
          checkpoint_every: int = 5) -> Tuple[np.ndarray, List[dict]]:
    try:
        import cv2

        cv2.setNumThreads(1)
    except Exception:
        pass

    n_in = len(FEATURE_NAMES)
    rng = np.random.default_rng(seed)

    if progress_cb is None:
        def progress_cb(gen, total_gens, mean_r, best_r, extra):
            # flush explicitly: stdout is fully (not line-) buffered
            # whenever it's not a terminal (e.g. piped through `tee`, or
            # redirected to a log file), so without this a long run can
            # look completely silent/stuck for many generations even
            # though it's progressing normally.
            print(f"[gen {gen + 1}/{total_gens}] mean_reward={mean_r:+.4f} "
                  f"best_reward={best_r:+.4f} {extra}", flush=True)

    print("Collecting baseline feature statistics (a few rule-based warm-up episodes)...")
    obs_mean, obs_std = _collect_obs_stats(tier)

    base_net = MLPPolicy(n_in, obs_mean=obs_mean, obs_std=obs_std, seed=seed)
    theta = base_net.get_flat_weights()
    d = theta.size
    pop_half = max(1, population // 2)

    executor = ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) if workers > 1 else None
    history: List[dict] = []
    t0 = time.time()
    try:
        for gen in range(generations):
            env_seed = int(rng.integers(0, 1_000_000))
            eps = rng.normal(0.0, 1.0, (pop_half, d))
            candidates = []
            for e in eps:
                candidates.append(theta + sigma * e)
                candidates.append(theta - sigma * e)

            args = [(c, env_seed, tier, episode_ticks, obs_mean, obs_std, 1.0) for c in candidates]
            if executor is not None:
                rewards = list(executor.map(_eval_worker, args))
            else:
                rewards = [_eval_worker(a) for a in args]
            rewards = np.asarray(rewards, dtype=np.float64)

            # Centered-rank fitness shaping: robust to reward scale/outliers
            # (a single catastrophic collision shouldn't dominate the
            # gradient estimate the way a raw reward would).
            order = np.argsort(rewards)
            ranks = np.empty_like(order)
            ranks[order] = np.arange(len(rewards))
            shaped = (ranks / max(1, len(rewards) - 1)) - 0.5

            grad = np.zeros(d)
            for i, e in enumerate(eps):
                grad += (shaped[2 * i] - shaped[2 * i + 1]) * e
            grad /= pop_half
            theta = theta + (lr / sigma) * grad

            mean_reward = float(np.mean(rewards))
            best_reward = float(np.max(rewards))
            elapsed = time.time() - t0
            history.append({"generation": gen, "mean_reward": mean_reward,
                             "best_reward": best_reward, "elapsed_s": round(elapsed, 1)})
            progress_cb(gen, generations, mean_reward, best_reward,
                        {"elapsed_s": round(elapsed, 1)})

            if (gen + 1) % checkpoint_every == 0 or gen == generations - 1:
                _save(theta, obs_mean, obs_std, out_path, n_in)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    _save(theta, obs_mean, obs_std, out_path, n_in)
    return theta, history


def _save(theta: np.ndarray, obs_mean: np.ndarray, obs_std: np.ndarray, out_path: str, n_in: int) -> None:
    import joblib

    net = MLPPolicy(n_in, obs_mean=obs_mean, obs_std=obs_std)
    net.set_flat_weights(theta)
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    joblib.dump({"model": net, "feature_names": FEATURE_NAMES,
                 "label_names": ["steer", "throttle", "brake"]}, out_path)

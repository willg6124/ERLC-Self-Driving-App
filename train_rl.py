"""Train a self-driving steering model purely through trial and error
(reinforcement learning / Evolution Strategies) against the bundled
simulator -- no recorded driving data needed at all, unlike
train_model.py's imitation-learning path.

Usage:
    python train_rl.py
    python train_rl.py --tier pro --generations 150 --population 16 --episode-ticks 200

This is CPU-only and uses no PyTorch/TensorFlow/GPU -- just numpy -- so it
runs on any Windows gaming PC with zero extra multi-gigabyte downloads.
It will take a while (each generation has to actually drive the
simulator many times); let it run in the background, check the printed
reward curve, and stop it (Ctrl+C) whenever you're happy -- a checkpoint
is saved to --out every few generations, so progress is never lost.

The result is saved to models/rl_policy.joblib by default. Once that
file exists, run_live.py automatically blends its steering suggestions
on top of the normal rule-based autopilot (same safety-gated blend
mechanism as an imitation-learned model -- every hard safety rule
stays fully rule-based regardless: this only ever changes *how* the
car steers, never *whether* it brakes/stops).
"""
from __future__ import annotations

import argparse
import time

from erlc_autopilot.rl.train_es import train


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", default="standard", choices=["lite", "standard", "pro"],
                    help="Which config preset to train/drive with (default: standard)")
    ap.add_argument("--generations", type=int, default=40,
                    help="How many ES generations to run (more = better, but slower). Default 40.")
    ap.add_argument("--population", type=int, default=12,
                    help="Candidates per generation (mirrored pairs, so this should be even). Default 12.")
    ap.add_argument("--episode-ticks", type=int, default=150,
                    help="Simulated ticks per candidate per generation (at 20 ticks/s). Default 150 (~7.5s).")
    ap.add_argument("--sigma", type=float, default=0.1, help="Perturbation noise std-dev. Default 0.1.")
    ap.add_argument("--lr", type=float, default=0.05, help="Learning rate. Default 0.05.")
    ap.add_argument("--workers", type=int, default=2,
                    help="Parallel worker processes (match your CPU core count). Default 2.")
    ap.add_argument("--seed", type=int, default=0, help="Random seed. Default 0.")
    ap.add_argument("--out", default="models/rl_policy.joblib", help="Output path for the trained model.")
    ap.add_argument("--checkpoint-every", type=int, default=5,
                    help="Save a checkpoint every N generations. Default 5.")
    args = ap.parse_args()

    if args.population % 2 != 0:
        args.population += 1
        print(f"(--population must be even for mirrored sampling -- bumped to {args.population})")

    print("=" * 60)
    print(" ERLC AUTOPILOT -- self-training (reinforcement learning)")
    print("=" * 60)
    print(f"tier={args.tier} generations={args.generations} population={args.population} "
          f"episode_ticks={args.episode_ticks} workers={args.workers}")
    print("No demonstrations needed -- this learns purely by driving the bundled")
    print("simulator over and over and keeping whatever steering tweaks scored better.")
    print(f"Saving checkpoints to {args.out} every {args.checkpoint_every} generation(s).")
    print("Press Ctrl+C any time -- the latest checkpoint is already on disk.\n")

    t0 = time.time()
    try:
        _, history = train(
            tier=args.tier, generations=args.generations, population=args.population,
            episode_ticks=args.episode_ticks, sigma=args.sigma, lr=args.lr,
            workers=args.workers, seed=args.seed, out_path=args.out,
            checkpoint_every=args.checkpoint_every,
        )
    except KeyboardInterrupt:
        print("\nStopped early -- the latest checkpoint is already saved.")
        return

    dt = time.time() - t0
    print(f"\nDone in {dt / 60:.1f} min. Final model saved to {args.out}")
    if history:
        best = max(h["best_reward"] for h in history)
        print(f"Best single-episode reward seen during training: {best:+.4f}")
    print("\nNow just run run_live.py (or run_live.py --sim) as usual -- it will")
    print("automatically pick up and blend in the trained model.")


if __name__ == "__main__":
    main()

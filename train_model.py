#!/usr/bin/env python3
"""Trains a small imitation-learning model from your own recorded ER:LC
driving (see `python run_live.py --record ...`) and saves it where
`run_live.py --tier pro` automatically picks it up and blends it into the
rule-based steering, without replacing any of the rule-based safety
behavior (e-braking, red lights, emergency-vehicle yield/stop, disengage
all stay 100% rule-based regardless -- see erlc_autopilot/control/
driving_policy.py and imitation.py for exactly how it's blended in).

    python run_live.py --record recordings/session1.csv   # drive manually
    python train_model.py                                  # train on it
    python run_live.py --tier pro                           # now blended in

Works on however many recorded sessions you have (pass --data with a glob,
or just drop more CSVs into recordings/ and rerun -- it picks up every
*.csv there by default).
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

import numpy as np

from erlc_autopilot.training.features import FEATURE_NAMES

LABEL_NAMES = ["steer", "throttle", "brake"]


def load_dataset(pattern: str):
    paths = sorted(glob.glob(pattern))
    if not paths:
        return None, None, []
    X, y = [], []
    for path in paths:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != FEATURE_NAMES + LABEL_NAMES:
                print(f"Skipping {path}: columns don't match the current feature set "
                      f"(it may have been recorded with an older version of the app).")
                continue
            for row in reader:
                try:
                    X.append([float(row[name]) for name in FEATURE_NAMES])
                    y.append([float(row[name]) for name in LABEL_NAMES])
                except (TypeError, ValueError):
                    continue
    if not X:
        return None, None, paths
    return np.array(X, dtype=np.float64), np.array(y, dtype=np.float64), paths


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="recordings/*.csv",
                         help="Glob pattern for recorded session CSVs (default: recordings/*.csv)")
    parser.add_argument("--out", default="models/imitation_model.joblib",
                         help="Where to save the trained model (default: models/imitation_model.joblib)")
    parser.add_argument("--test-size", type=float, default=0.2,
                         help="Fraction of rows held out to report accuracy on (default: 0.2)")
    args = parser.parse_args()

    X, y, paths = load_dataset(args.data)
    if X is None:
        print(f"No usable recorded data found matching '{args.data}'.")
        if paths:
            print(f"(Found {len(paths)} file(s) there, but none had rows matching the "
                  f"current feature set.)")
        else:
            print("Record a driving session first:\n  python run_live.py --record recordings/session1.csv")
        sys.exit(1)

    print(f"Loaded {len(X)} rows from {len(paths)} file(s): {', '.join(os.path.basename(p) for p in paths)}")
    if len(X) < 200:
        print(f"Warning: only {len(X)} rows -- that's a very short recording. The model will "
              f"likely be weak/overfit; more recorded driving (ideally several minutes, varied "
              f"roads/traffic) will make it meaningfully better. Still training on what's here.")

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split

    test_size = args.test_size if len(X) * args.test_size >= 10 else 0.0
    if test_size > 0:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
    else:
        X_train, y_train = X, y
        X_test, y_test = None, None

    model = RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_leaf=3,
                                   random_state=0, n_jobs=-1)
    model.fit(X_train, y_train)

    if X_test is not None:
        score = model.score(X_test, y_test)
        preds = model.predict(X_test)
        mae = np.mean(np.abs(preds - y_test), axis=0)
        print(f"Held-out R^2: {score:.3f}  (1.0 = perfect, 0 = no better than predicting the mean)")
        print(f"Held-out mean abs error -- steer: {mae[0]:.3f}  throttle: {mae[1]:.3f}  brake: {mae[2]:.3f}")
    else:
        print("Dataset too small for a held-out test split -- trained on all of it with no accuracy check.")

    import joblib

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    joblib.dump({"model": model, "feature_names": FEATURE_NAMES, "label_names": LABEL_NAMES}, args.out)
    print(f"Saved to {args.out} -- run with --tier pro to use it.")


if __name__ == "__main__":
    main()

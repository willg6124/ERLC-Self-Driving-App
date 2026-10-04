"""Imitation-learning support: turning a recorded driving session (your own
real ER:LC driving, or a synthetic demonstration) into a small model that
blends in on top of the rule-based policy at the Pro tier.

See `erlc_autopilot/training/features.py` (what gets recorded/predicted
from), `recorder.py` (writes a session to disk), and `../control/imitation.py`
(loads a trained model back for the driving policy to blend from).
Training itself lives in the top-level `train_model.py` script.
"""

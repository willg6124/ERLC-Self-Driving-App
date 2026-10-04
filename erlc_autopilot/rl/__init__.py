"""Self-training (reinforcement learning) driving model.

Unlike `erlc_autopilot/training` (imitation learning -- needs you to
record yourself driving first), this trains a small neural-network
steering policy purely through trial and error against the bundled
simulator: no human demonstrations required. It uses Evolution Strategies
(a genuinely "trains itself" / natural-selection style algorithm: try a
population of randomly perturbed policies, keep nudging the weights
towards whichever perturbations scored a higher reward) rather than
gradient-based deep RL, specifically so it:

  * needs zero GPU / zero heavy ML framework (pure numpy + the
    scikit-learn-style `.predict()` interface already used by the
    imitation-learning path) -- a few hundred KB, installs instantly on
    any Windows gaming PC;
  * is simple enough to audit/understand end to end;
  * plugs into the exact same safety-gated blend mechanism as the
    imitation model (`DrivingPolicy.imitation_model`), so every hard
    safety rule (pedestrian e-braking, red-light stop, emergency-vehicle
    yield/pull-over, disengage-on-lost-lane) stays 100% deterministic and
    rule-based -- the trained network only ever nudges *how* the car
    steers, never *whether* it brakes or stops.
"""
from .policy_net import MLPPolicy
from .train_es import train, EpisodeResult

__all__ = ["MLPPolicy", "train", "EpisodeResult"]

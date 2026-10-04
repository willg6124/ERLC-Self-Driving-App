"""Central, runtime-tweakable configuration for the autopilot. The live
dashboard can hot-patch any of these fields (see dashboard/server.py) so you
can tune gains while watching the car drive instead of editing files."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class AutopilotConfig:
    # --- steering ---
    steer_kp: float = 1.35
    steer_ki: float = 0.015
    steer_kd: float = 0.55
    curvature_feedforward: float = 0.0   # experimental; pixel-space curvature isn't metrically
                                          # calibrated so a nonzero gain here can destabilize steering
    # Pure-pursuit-style lookahead blend: steer off a weighted mix of the
    # near-field lane offset (stable, low noise) and a far-field offset
    # measured near the top of the detected lane lines (anticipates the
    # road ahead instead of only reacting to where it already is) --
    # smooths out curve-cutting/oscillation versus near-offset-only PID.
    lookahead_weight: float = 0.35       # 0 = old behavior (near-offset only)

    # --- speed ---
    cruise_speed_mps: float = 13.5       # ~30 mph, typical ER:LC residential cap
    max_speed_mps: float = 24.0          # ~54 mph, highway cap
    curve_speed_gain: float = 9.0        # higher = slow down more in curves
    min_confidence_speed_mps: float = 5.0
    speed_kp: float = 0.55
    speed_ki: float = 0.08
    speed_kd: float = 0.02

    # --- adaptive cruise / safety following ---
    follow_time_headway_s: float = 1.8
    min_follow_gap_m: float = 6.0
    hard_brake_gap_m: float = 3.0
    # Time-to-collision anticipation: distance alone only reacts once a
    # lead car is already close. Estimating closing speed frame-to-frame
    # and reacting to *how fast the gap is shrinking* catches a sudden
    # cut-in or a lead car braking hard well before distance alone would,
    # same idea real adaptive cruise control radar systems use.
    ttc_hard_brake_s: float = 2.0        # closing this fast -> treat like hard_brake_gap_m
    ttc_caution_s: float = 4.5           # closing this fast -> start easing off early

    # --- traffic lights ---

    stop_line_margin_m: float = 4.0
    light_reaction_distance_m: float = 32.0
    # Once fully stopped at a red/yellow light the signal head itself is
    # commonly out of the dashcam's vertical field of view (it's mounted
    # high and we're now right underneath it) -- there's no fresh visual
    # cue telling us when it turns green. After waiting this long with no
    # re-detection we cautiously proceed rather than deadlocking forever,
    # the same fallback a real AV stack uses when it loses sight of a
    # signal it's already stopped at.
    max_blind_light_stop_s: float = 8.0

    # --- pedestrians ---
    pedestrian_emergency_distance_m: float = 14.0
    pedestrian_lane_margin_px: float = 140.0

    # how far off-center (px) a "vehicle" detection can be and still count
    # as a lead car to follow/brake for, rather than a car in another lane
    # or parked on the shoulder (e.g. a pulled-over emergency vehicle) that
    # happens to be the nearest vehicle-class detection this frame
    vehicle_lane_margin_px: float = 200.0

    # --- emergency vehicles (police / fire / ems "Move Over Law") ---
    emergency_vehicle_reaction_distance_m: float = 35.0
    emergency_vehicle_yield_speed_mps: float = 6.0   # ~13 mph crawl while first noticing it
    emergency_vehicle_steer_bias: float = 0.30       # how hard to lean away at max proximity, in steer-error units
    # Literal "pull over": once it's this close (actually overtaking /
    # about to pass, or we're right on top of one parked ahead), fully
    # stop at the shoulder lean rather than just crawling past it --
    # released again once it's pulled away past `..._resume_distance_m`
    # (a wider gap than the stop trigger, so it doesn't flicker on/off
    # right at the boundary as the distance estimate jitters frame to frame).
    emergency_vehicle_stop_distance_m: float = 12.0
    emergency_vehicle_resume_distance_m: float = 22.0

    # --- imitation learning (optional, blended on top of the rule-based
    # policy at the Pro tier only if a trained model is present under
    # models/imitation_model.joblib -- see train_model.py) ---
    imitation_blend_weight: float = 0.35  # 0 = ignore the learned model entirely

    # --- engagement / safety ---
    min_lane_confidence_to_drive: float = 0.15
    disengage_after_lost_frames: int = 20  # ~1s @20fps with zero lane confidence -- fail safe, fast

    # which "model tier" produced this config -- purely informational
    # (the wizard sets it), but surfaced in telemetry/HUD so you can see
    # which preset + detector backend is actually driving
    model_tier: str = "standard"

    def to_dict(self):
        return asdict(self)

    def update(self, patch: dict):
        for k, v in patch.items():
            if hasattr(self, k) and k != "model_tier":
                setattr(self, k, type(getattr(self, k))(v))

    @classmethod
    def for_tier(cls, tier: str) -> "AutopilotConfig":
        """Setup-wizard model-tier presets. Real differences, not just a
        label: Lite trades accuracy for raw speed on low-end PCs (bigger
        safety margins to compensate for a coarser/faster perception
        backend), Standard is the well-tested default, Pro tightens
        everything up for a snappier, more assertive drive once you trust
        it (and is where a future imitation-learned steering model would
        get blended in on top of the rule-based stack)."""
        cfg = cls()
        cfg.model_tier = tier
        if tier == "lite":
            cfg.cruise_speed_mps = 11.0
            cfg.follow_time_headway_s = 2.3
            cfg.min_follow_gap_m = 8.0
            cfg.pedestrian_emergency_distance_m = 18.0
            cfg.emergency_vehicle_reaction_distance_m = 40.0
            cfg.steer_kp = 1.15
            cfg.lookahead_weight = 0.2     # cheaper/noisier detector -> trust the far point less
            cfg.ttc_caution_s = 5.5        # bigger margins all around on weaker hardware
        elif tier == "pro":
            cfg.cruise_speed_mps = 16.0
            cfg.follow_time_headway_s = 1.4
            cfg.min_follow_gap_m = 5.0
            cfg.steer_kp = 1.5
            cfg.steer_kd = 0.65
            cfg.lookahead_weight = 0.45
            cfg.ttc_caution_s = 4.0
            # the only tier that blends a trained imitation model in, if
            # models/imitation_model.joblib exists (see train_model.py) --
            # purely rule-based otherwise, same as the other tiers
            cfg.imitation_blend_weight = 0.35
        return cfg


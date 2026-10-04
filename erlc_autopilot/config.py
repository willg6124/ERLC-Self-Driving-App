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
    emergency_vehicle_yield_speed_mps: float = 6.0   # ~13 mph crawl while passing/being passed
    emergency_vehicle_steer_bias: float = 0.30       # how hard to lean away, in steer-error units

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
        elif tier == "pro":
            cfg.cruise_speed_mps = 16.0
            cfg.follow_time_headway_s = 1.4
            cfg.min_follow_gap_m = 5.0
            cfg.steer_kp = 1.5
            cfg.steer_kd = 0.65
        return cfg


"""
Fuses lane-keeping, adaptive cruise control, traffic-light compliance,
pedestrian avoidance and a safety watchdog into one steer/throttle/brake
command per tick. This is the "brain" of the autopilot -- everything
upstream (perception) only produces measurements, everything downstream
(input_driver) only turns a Command into keypresses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from ..config import AutopilotConfig
from ..perception.emergency_vehicle import EmergencyVehicleSignal
from ..perception.lane_detection import LaneResult
from ..perception.object_detection import Detection
from ..perception.traffic_light import TrafficLightDetection
from ..training.features import extract_features
from .pid import PID

FRAME_CENTER_X = 480.0  # half of the standard 960px-wide capture/sim frame


@dataclass
class Command:
    steer: float = 0.0       # -1 (full left) .. 1 (full right)
    throttle: float = 0.0    # 0 .. 1
    brake: float = 0.0       # 0 .. 1
    target_speed: float = 0.0
    engaged: bool = True
    status: str = "cruising"
    turn_signal: Optional[str] = None  # "left" | "right" | None -- the real
    # input driver taps Q/E for this so the car actually signals before it
    # leans away from an emergency vehicle, same as a human would
    events: List[str] = field(default_factory=list)


class DrivingPolicy:
    def __init__(self, config: Optional[AutopilotConfig] = None):
        self.cfg = config or AutopilotConfig()
        self._build_pids()
        self._lost_lane_frames = 0
        self._engaged = True
        self._remembered_light = None  # (state, distance_m) -- survives brief frames
        # where a near traffic-light head goes out of the camera's narrow
        # vertical FOV right as the car gets close to the stop line
        self._blind_light_timer = 0.0  # seconds spent stopped with no fresh
        # re-detection of a remembered red/yellow light -- caps how long we
        # wait before cautiously proceeding (see max_blind_light_stop_s)
        self._last_lead_distance: Optional[float] = None  # for closing-speed/TTC estimation
        self._ev_committed_stop = False  # true once we've actually pulled
        # over and stopped for a close emergency vehicle, held with
        # hysteresis until it's clearly moved away (see
        # emergency_vehicle_stop_distance_m / ..._resume_distance_m)
        # Optional learned steering model (see erlc_autopilot/control/imitation.py
        # and train_model.py) -- None unless the Pro tier found a trained
        # models/imitation_model.joblib on disk. Only ever nudges the
        # rule-based steer output by `imitation_blend_weight`; every safety
        # behavior below (e-braking, red lights, EV yield, disengage) stays
        # fully rule-based and is applied after this blend, same as today.
        self.imitation_model = None

    def _build_pids(self):
        c = self.cfg
        self.steer_pid = PID(c.steer_kp, c.steer_ki, c.steer_kd, output_limits=(-1, 1),
                              integral_limits=(-0.6, 0.6))
        self.speed_pid = PID(c.speed_kp, c.speed_ki, c.speed_kd, output_limits=(-1, 1),
                              integral_limits=(-2.0, 2.0))

    def apply_config(self, config: AutopilotConfig):
        self.cfg = config
        self._build_pids()

    def set_engaged(self, engaged: bool):
        self._engaged = engaged
        if engaged:
            self.steer_pid.reset()
            self.speed_pid.reset()
            self._remembered_light = None
            self._blind_light_timer = 0.0
            self._last_lead_distance = None
            self._ev_committed_stop = False

    def _resolve_light(self, light: Optional[TrafficLightDetection], speed_mps: float,
                        dt: float, events: List[str]) -> Optional[TrafficLightDetection]:
        """Combine the fresh detection with short-term memory so a real red
        light commitment doesn't evaporate just because the signal head
        scrolls out of the camera's narrow vertical FOV in the final
        approach (very common once the car is within ~15m, since a tall
        pole-mounted light goes above frame at close range). Memory decays
        by distance travelled and is cleared once we'd have passed it or a
        fresh green is actually seen.

        If we end up fully stopped right under a light we can no longer see
        at all (common once stopped -- the head is now above frame), there
        is no visual cue left telling us when it changes to green. Rather
        than wait forever, `max_blind_light_stop_s` of silence makes us
        cautiously proceed, like a real driver creeping through a signal
        they've lost sight of after a safe wait.
        """
        if light is not None:
            if light.state == "green":
                self._remembered_light = None
                self._blind_light_timer = 0.0
                return light
            # Guard against a spurious far-away false-positive detection
            # (e.g. a stray red pixel blob) snapping in right as the real,
            # close light scrolls out of the camera's frame and getting
            # mistaken for "actually there's lots of room now" -- while we
            # are actively committed to a close light, only accept a fresh
            # detection if it is at least roughly consistent with the
            # car continuing to approach the same light (distance should
            # only shrink, modulo a little detector noise).
            mem = self._remembered_light
            if (mem is not None and mem.distance_m < 25.0
                    and light.distance_m > mem.distance_m + 8.0):
                light = None  # treat as if nothing fresh was seen this tick
            else:
                self._remembered_light = TrafficLightDetection(
                    state=light.state, bbox=light.bbox,
                    distance_m=light.distance_m, confidence=light.confidence)
                self._blind_light_timer = 0.0
                return light

        if self._remembered_light is not None:
            if speed_mps < 0.5:
                self._blind_light_timer += dt
                if self._blind_light_timer > self.cfg.max_blind_light_stop_s:
                    events.append(
                        f"traffic_light: lost sight of signal after stopping, "
                        f"proceeding cautiously after {self._blind_light_timer:.0f}s")
                    self._remembered_light = None
                    self._blind_light_timer = 0.0
                    return None
            traveled = max(0.0, speed_mps * dt)
            remaining = self._remembered_light.distance_m - traveled
            if remaining > -2.0:
                self._remembered_light = TrafficLightDetection(
                    state=self._remembered_light.state,
                    bbox=self._remembered_light.bbox,
                    distance_m=max(0.0, remaining),
                    confidence=self._remembered_light.confidence)
                return self._remembered_light
            self._remembered_light = None
        return None

    def _handle_emergency_vehicles(self, ev_signals: List[EmergencyVehicleSignal],
                                    events: List[str]):
        """"Move Over Law" response, proportional to proximity rather than a
        single fixed crawl speed: lean progressively harder away from
        whichever side the active light bar is on as it gets closer, and
        once it's genuinely close (actually overtaking, or we've reached a
        parked one ahead) pull fully onto the shoulder lean and come to a
        complete stop -- not just a slow crawl past it -- until it's
        clearly moved away again. Returns (steer_bias, turn_signal, speed_cap).
        """
        c = self.cfg
        active = [e for e in ev_signals if e.flashing and e.distance_m < c.emergency_vehicle_reaction_distance_m]
        if not active:
            if self._ev_committed_stop:
                events.append("emergency_vehicle: clear, resuming normal driving")
            self._ev_committed_stop = False
            return 0.0, None, None

        nearest = min(active, key=lambda e: e.distance_m)
        cx = (nearest.bbox[0] + nearest.bbox[2]) / 2.0
        lean_right = cx < FRAME_CENTER_X  # it's on our left -> lean right, away from it
        # 0 = just entered reaction distance, 1 = right on top of it -- used
        # to scale both the steer lean and the speed cap smoothly instead of
        # snapping straight to a fixed crawl the instant it's noticed
        proximity = 1.0 - max(0.0, min(1.0, nearest.distance_m / c.emergency_vehicle_reaction_distance_m))
        bias_mag = c.emergency_vehicle_steer_bias * (0.45 + 0.55 * proximity)
        bias = bias_mag if lean_right else -bias_mag
        signal = "right" if lean_right else "left"

        if self._ev_committed_stop:
            if nearest.distance_m > c.emergency_vehicle_resume_distance_m:
                self._ev_committed_stop = False
            else:
                events.append(f"emergency_vehicle: pulled over, holding stop while it passes "
                               f"({nearest.distance_m:.1f}m)")
                return bias, signal, 0.0

        if nearest.distance_m < c.emergency_vehicle_stop_distance_m:
            self._ev_committed_stop = True
            events.append(f"emergency_vehicle: pulling over and stopping, {nearest.distance_m:.1f}m away")
            return bias, signal, 0.0

        speed_cap = c.emergency_vehicle_yield_speed_mps * (1.0 - 0.5 * proximity)
        events.append(f"emergency_vehicle: yielding, active light bar {nearest.distance_m:.1f}m away")
        return bias, signal, max(0.0, speed_cap)

    @staticmethod
    def _lookahead_offset_norm(lane: LaneResult) -> Optional[float]:
        """A second, farther-ahead lane-offset estimate taken near the top
        of the detected lane lines (as opposed to `lane.offset_norm`, which
        is evaluated close to the car) -- blending this in gives the
        steering controller a pure-pursuit-style preview of where the road
        is headed instead of only correcting for where it already drifted,
        which is what was causing oscillation/curve-cutting on bends.
        Returns None if either lane line wasn't actually detected this
        frame (caller should just fall back to the near-field offset)."""
        if lane.left_line is None or lane.right_line is None:
            return None
        # line tuples are (x_bottom, y_bottom, x_top, y_top, max_y_seen)
        far_left_x, far_right_x = lane.left_line[2], lane.right_line[2]
        far_width = far_right_x - far_left_x
        if far_width < 20:  # degenerate/crossed lines -- not trustworthy
            return None
        far_center = (far_left_x + far_right_x) / 2.0
        far_offset_px = FRAME_CENTER_X - far_center
        return max(-2.5, min(2.5, far_offset_px / (far_width / 2.0)))

    @staticmethod
    def _bbox_overlaps(a, b) -> bool:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return False
        inter = (ix2 - ix1) * (iy2 - iy1)
        a_area = max(1, (ax2 - ax1) * (ay2 - ay1))
        return inter / a_area > 0.3

    def _target_speed(self, lane: LaneResult, detections: List[Detection],
                       light: Optional[TrafficLightDetection], events: List[str],
                       confirmed_ev_boxes: Optional[list] = None, dt: float = 0.05) -> float:
        c = self.cfg
        confirmed_ev_boxes = confirmed_ev_boxes or []
        target = c.cruise_speed_mps

        # slow down for curves (feedforward from measured curvature), but
        # never below a reasonable "taking a bend" floor
        curve_penalty = min(target * 0.7, abs(lane.curvature) * c.curve_speed_gain)
        target -= curve_penalty

        # scale right down if perception confidence is low -- a real
        # lane-keep system creeps cautiously (or stops) rather than
        # barrelling ahead blind on stale/guessed geometry
        if lane.confidence < 0.5:
            target = min(target, c.min_confidence_speed_mps * lane.confidence * 2.0)

        # adaptive cruise control against the nearest vehicle ahead in-lane
        # -- gated to roughly our own travel lane so a car in the opposite
        # lane, or a vehicle (including an emergency vehicle) parked off to
        # the side on the shoulder, doesn't get treated as a lead car to
        # brake behind
        lead = next((d for d in detections if d.cls == "vehicle"
                     and abs((d.bbox[0] + d.bbox[2]) / 2 - 480) < c.vehicle_lane_margin_px), None)
        if lead is not None:
            # Closing-speed / time-to-collision anticipation: distance alone
            # only reacts once a lead car is already close, which is too
            # late for a sudden cut-in or a lead car braking hard right in
            # front of us. Estimating how fast the gap is actually shrinking
            # frame-to-frame (closing_speed) and reacting to the *time* that
            # implies until contact catches these cases well before distance
            # alone would -- the same anticipation a real radar-based ACC
            # system gets for free that a single-frame vision distance
            # estimate doesn't.
            closing_speed = 0.0
            if self._last_lead_distance is not None and dt > 1e-3:
                closing_speed = max(0.0, (self._last_lead_distance - lead.distance_m) / dt)
            self._last_lead_distance = lead.distance_m
            ttc = (lead.distance_m / closing_speed) if closing_speed > 0.5 else float("inf")

            safe_gap = max(c.min_follow_gap_m, target * c.follow_time_headway_s)
            if ttc < c.ttc_hard_brake_s:
                target = 0.0
                events.append(f"hard_brake: closing fast on lead car (TTC {ttc:.1f}s)")
            elif lead.distance_m < safe_gap or ttc < c.ttc_caution_s:
                frac = max(0.0, (lead.distance_m - c.hard_brake_gap_m) /
                           max(1.0, safe_gap - c.hard_brake_gap_m))
                if ttc < c.ttc_caution_s:
                    # scale down further the faster we're closing, even if
                    # the raw gap itself still looks comfortable
                    ttc_frac = max(0.0, min(1.0, ttc / c.ttc_caution_s))
                    frac = min(frac, ttc_frac)
                target = min(target, target * frac)
                tag = "closing" if ttc < c.ttc_caution_s else f"{lead.distance_m:.1f}m ahead"
                events.append(f"adaptive_cruise: lead car {tag}")
            if lead.distance_m < c.hard_brake_gap_m:
                target = 0.0
                events.append("hard_brake: lead car too close")
        else:
            self._last_lead_distance = None

        # traffic-light compliance
        if light is not None and light.distance_m < c.light_reaction_distance_m:
            if light.state in ("red", "yellow"):
                stop_dist = max(0.0, light.distance_m - c.stop_line_margin_m)
                # physics-based approach speed: v = sqrt(2 * a * d) for a
                # comfortable deceleration, so the car brakes smoothly from
                # well before the stop line rather than barely slowing down
                # until it's almost on top of it
                comfortable_decel = 3.0
                approach_speed = (2 * comfortable_decel * stop_dist) ** 0.5
                target = min(target, approach_speed)
                if stop_dist < 1.5:
                    target = 0.0
                events.append(f"traffic_light: stopping for {light.state} ({light.distance_m:.1f}m)")

        # pedestrian emergency braking -- skip boxes that are actually a
        # confirmed (flashing light bar) emergency vehicle the coarse shape
        # heuristic happened to misclassify as a pedestrian; that case is
        # handled by the dedicated "Move Over Law" yield response instead,
        # which slows and leans away rather than slamming to a full stop
        # blocking the road right next to a police car / fire truck.
        for d in detections:
            if d.cls != "pedestrian":
                continue
            if any(self._bbox_overlaps(d.bbox, evb) for evb in confirmed_ev_boxes):
                continue
            in_path = abs((d.bbox[0] + d.bbox[2]) / 2 - 480) < c.pedestrian_lane_margin_px
            if in_path and d.distance_m < c.pedestrian_emergency_distance_m:
                target = 0.0
                events.append(f"EMERGENCY_BRAKE: pedestrian {d.distance_m:.1f}m ahead")

        return max(0.0, target)

    def step(self, lane: LaneResult, detections: List[Detection],
             light: Optional[TrafficLightDetection], speed_mps: float, dt: float,
             ev_signals: Optional[List[EmergencyVehicleSignal]] = None) -> Command:
        c = self.cfg
        events: List[str] = []
        status = "cruising"

        if lane.confidence < c.min_lane_confidence_to_drive:
            self._lost_lane_frames += 1
        else:
            self._lost_lane_frames = 0

        if self._lost_lane_frames > c.disengage_after_lost_frames:
            self._engaged = False
            status = "disengaged: lane lost"
            events.append("AUTOPILOT_DISENGAGED: could not find lane markings")

        if not self._engaged:
            if status == "cruising":
                status = "disengaged"
            return Command(steer=0.0, throttle=0.0, brake=0.6, target_speed=0.0,
                            engaged=False, status=status, events=events)

        ev_bias, turn_signal, ev_speed_cap = self._handle_emergency_vehicles(ev_signals or [], events)

        # steering: PID on a near+far blended lane offset (pure-pursuit-style
        # lookahead -- see _lookahead_offset_norm) + curvature feedforward +
        # a lean away from an active emergency vehicle, if one is present.
        # The PID's derivative term still measures off the near-field offset
        # alone (lower-noise, and avoids derivative kick from the far point
        # jumping around when a lane line briefly drops out).
        lookahead_offset = self._lookahead_offset_norm(lane)
        if lookahead_offset is None:
            blended_offset = lane.offset_norm
        else:
            w = max(0.0, min(1.0, c.lookahead_weight))
            blended_offset = (1 - w) * lane.offset_norm + w * lookahead_offset
        error = -blended_offset + ev_bias
        steer = self.steer_pid.step(error, lane.offset_norm, dt)
        steer += c.curvature_feedforward * lane.curvature

        # Optional: blend in a trained imitation model's steering suggestion
        # (Pro tier only, and only if models/imitation_model.joblib exists --
        # see train_model.py). Throttle/brake and every safety override below
        # (e-braking, red lights, EV yield/stop, disengage) stay entirely
        # rule-based regardless, so a learned model can only ever nudge how
        # the car steers, never override why it slows or stops.
        if self.imitation_model is not None and c.imitation_blend_weight > 0:
            features = extract_features(lane, detections, light, ev_signals, speed_mps)
            learned = self.imitation_model.predict(features)
            if learned is not None:
                w = max(0.0, min(1.0, c.imitation_blend_weight))
                steer = (1 - w) * steer + w * learned[0]
                events.append(f"imitation: blended learned steer (weight={w:.2f})")

        steer = max(-1.0, min(1.0, steer))

        confirmed_ev_boxes = [e.bbox for e in (ev_signals or []) if e.flashing]
        effective_light = self._resolve_light(light, speed_mps, dt, events)
        target_speed = self._target_speed(lane, detections, effective_light, events, confirmed_ev_boxes, dt)
        if ev_speed_cap is not None:
            target_speed = min(target_speed, ev_speed_cap)
        speed_error = target_speed - speed_mps
        pedal = self.speed_pid.step(speed_error, speed_mps, dt)
        if target_speed <= 0.05 and speed_mps < 0.5:
            throttle, brake = 0.0, 0.3
        elif pedal >= 0:
            throttle, brake = pedal, 0.0
        else:
            throttle, brake = 0.0, -pedal

        if any(ev.startswith("EMERGENCY_BRAKE") for ev in events):
            status = "emergency_braking"
            throttle, brake = 0.0, 1.0
        elif any(ev.startswith("hard_brake") for ev in events):
            status = "braking"
        elif any(ev.startswith("traffic_light") for ev in events) and target_speed < 1.0:
            status = "stopped_at_light"
        elif any(ev.startswith("emergency_vehicle") for ev in events):
            status = "yielding"
        elif any(ev.startswith("adaptive_cruise") for ev in events):
            status = "following"

        return Command(steer=steer, throttle=throttle, brake=brake, target_speed=target_speed,
                        engaged=True, status=status, turn_signal=turn_signal, events=events)

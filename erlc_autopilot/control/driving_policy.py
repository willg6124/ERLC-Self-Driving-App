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
        """"Move Over Law" response: for the nearest confirmed active light
        bar within reaction distance, lean the lane-keeping target away from
        whichever side of the frame it's on and cap speed to a cautious
        crawl, whether it's overtaking from behind or already stopped on
        the shoulder ahead. Returns (steer_bias, turn_signal, speed_cap).
        """
        c = self.cfg
        active = [e for e in ev_signals if e.flashing and e.distance_m < c.emergency_vehicle_reaction_distance_m]
        if not active:
            return 0.0, None, None
        nearest = min(active, key=lambda e: e.distance_m)
        cx = (nearest.bbox[0] + nearest.bbox[2]) / 2.0
        if cx < FRAME_CENTER_X:
            # it's on our left -> lean right, away from it
            bias, signal = c.emergency_vehicle_steer_bias, "right"
        else:
            bias, signal = -c.emergency_vehicle_steer_bias, "left"
        events.append(f"emergency_vehicle: yielding, active light bar {nearest.distance_m:.1f}m away")
        return bias, signal, c.emergency_vehicle_yield_speed_mps

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
                       confirmed_ev_boxes: Optional[list] = None) -> float:
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
            safe_gap = max(c.min_follow_gap_m, target * c.follow_time_headway_s)
            if lead.distance_m < safe_gap:
                frac = max(0.0, (lead.distance_m - c.hard_brake_gap_m) /
                           max(1.0, safe_gap - c.hard_brake_gap_m))
                target = min(target, target * frac)
                events.append(f"adaptive_cruise: lead car {lead.distance_m:.1f}m ahead")
            if lead.distance_m < c.hard_brake_gap_m:
                target = 0.0
                events.append("hard_brake: lead car too close")

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

        # steering: PID on lane offset + curvature feedforward + a lean away
        # from an active emergency vehicle, if one is present
        error = -lane.offset_norm + ev_bias
        steer = self.steer_pid.step(error, lane.offset_norm, dt)
        steer += c.curvature_feedforward * lane.curvature
        steer = max(-1.0, min(1.0, steer))

        confirmed_ev_boxes = [e.bbox for e in (ev_signals or []) if e.flashing]
        effective_light = self._resolve_light(light, speed_mps, dt, events)
        target_speed = self._target_speed(lane, detections, effective_light, events, confirmed_ev_boxes)
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

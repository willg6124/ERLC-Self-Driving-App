"""
A lightweight, self-contained driving world used to test and demo the
autopilot entirely inside this sandbox (no Roblox / ER:LC required to see
it work). It models:

  * a winding multi-lane road (procedurally generated, loops forever)
  * intersections with cycling traffic lights
  * crosswalks with pedestrians that occasionally cross
  * other AI traffic that drives along the road at various speeds
  * simple bicycle-model physics for the ego vehicle

The autopilot never reads any of this ground-truth state directly for its
driving decisions -- it only ever sees the rendered camera frame produced
by `renderer.py`, exactly like a real screen-capture bot would only see
pixels from the ER:LC game window. Ground truth is used here only to
render the world and to score the drive (off-road %, collisions, etc).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple

LANE_WIDTH = 3.6
NUM_LANES_EACH_WAY = 1
ROAD_HALF_WIDTH = LANE_WIDTH * NUM_LANES_EACH_WAY
SHOULDER = 1.5
TRACK_LENGTH = 2200.0  # meters, then loops
SAMPLE_STEP = 0.5
INTERSECTION_SPACING = 275.0
CAR_LENGTH = 4.6
CAR_WIDTH = 1.9


def _curvature_profile(s: float) -> float:
    """Smooth, ER:LC-ish winding road curvature (1/radius) at arclength s."""
    return (
        0.012 * math.sin(s / 140.0)
        + 0.006 * math.sin(s / 61.0 + 1.3)
        + 0.003 * math.sin(s / 23.0 + 0.4)
    )


class Centerline:
    """Precomputed dense polyline for the road centerline + heading."""

    def __init__(self, length: float = TRACK_LENGTH, step: float = SAMPLE_STEP):
        self.step = step
        n = int(length / step)
        xs = [0.0] * n
        ys = [0.0] * n
        headings = [0.0] * n
        x = y = heading = 0.0
        for i in range(n):
            s = i * step
            curvature = _curvature_profile(s)
            heading += curvature * step
            x += math.cos(heading) * step
            y += math.sin(heading) * step
            xs[i], ys[i], headings[i] = x, y, heading
        self.xs, self.ys, self.headings = xs, ys, headings
        self.n = n
        self.length = length
        self._full_search_cooldown = 0

    def index_at(self, s: float) -> int:
        s = s % self.length
        return int((s / self.length) * self.n) % self.n

    def pose_at(self, s: float) -> Tuple[float, float, float]:
        i = self.index_at(s)
        return self.xs[i], self.ys[i], self.headings[i]

    def nearest(self, x: float, y: float, hint_s: float, search: float = 25.0) -> float:
        """Local search for the arclength `s` whose centerline point is
        closest to (x, y), seeded around `hint_s` so it stays cheap. Falls
        back to a full-track search if the car has drifted far enough that
        the local window no longer contains the true nearest point (e.g.
        after running wide through a sharp bend), so tracking can never
        permanently "lose lock" and teleport to a bogus arclength."""
        best_s, best_d = self._local_search(x, y, hint_s, search)
        if best_d > 15.0 ** 2 and self._full_search_cooldown <= 0:
            best_s, best_d = self._local_search(x, y, hint_s, search=None)
            self._full_search_cooldown = 20  # ~1s @20Hz before trying another expensive full scan
        else:
            self._full_search_cooldown -= 1
        return best_s

    def _local_search(self, x: float, y: float, hint_s: float, search):
        best_s, best_d = hint_s, float("inf")
        if search is None:
            rng = range(self.n)
        else:
            lo = int((hint_s - search) / self.step)
            hi = int((hint_s + search) / self.step)
            rng = range(lo, hi + 1)
        for k in rng:
            i = k % self.n
            dx = self.xs[i] - x
            dy = self.ys[i] - y
            d = dx * dx + dy * dy
            if d < best_d:
                best_d = d
                best_s = (k % self.n) * self.step
        return best_s, best_d


@dataclass
class TrafficLight:
    s: float
    state: str = "green"
    timer: float = 0.0
    cycle: Tuple[float, float, float] = (9.0, 2.5, 7.0)  # green, yellow, red

    def update(self, dt: float):
        self.timer += dt
        g, y, r = self.cycle
        total = g + y + r
        t = self.timer % total
        if t < g:
            self.state = "green"
        elif t < g + y:
            self.state = "yellow"
        else:
            self.state = "red"


@dataclass
class Pedestrian:
    s: float
    lateral: float
    direction: int
    speed: float = 0.9
    active: bool = True

    def update(self, dt: float):
        self.lateral += self.direction * self.speed * dt
        if abs(self.lateral) > ROAD_HALF_WIDTH + SHOULDER + 1.0:
            self.active = False


@dataclass
class TrafficCar(object):
    s: float
    lane_offset: float
    speed: float
    color: Tuple[int, int, int]

    def update(self, dt: float, lead_gap: float):
        # very small car-following behaviour so traffic looks alive
        if lead_gap < 12.0:
            self.speed = max(0.0, self.speed - 4.0 * dt)
        else:
            self.speed = min(14.0, self.speed + 2.0 * dt)
        self.s += self.speed * dt


@dataclass
class EmergencyVehicle:
    """A police cruiser / fire truck / ambulance with an active light bar --
    either overtaking from behind (responding, passing through) or parked
    on the shoulder ahead (e.g. already handling a stop). Either way, a
    real driver (and this autopilot) is expected to slow down and move
    away from it -- the "Move Over Law" most US states have."""
    s: float
    lane_offset: float
    speed: float
    kind: str  # "police" | "fire" | "ems"
    lights_on: bool = True
    parked: bool = False

    def update(self, dt: float):
        if not self.parked:
            self.s += self.speed * dt


@dataclass
class EgoCar:
    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0
    speed: float = 0.0
    steer_angle: float = 0.0  # radians, current wheel angle
    wheelbase: float = 2.8
    max_steer: float = math.radians(32)
    max_speed: float = 24.0  # ~54 mph, ER:LC-ish residential/highway mix
    s_hint: float = 0.0

    def step(self, dt: float, steer_cmd: float, throttle: float, brake: float):
        # steer_cmd, throttle, brake all in [-1, 1] / [0, 1]
        target_steer = max(-1.0, min(1.0, steer_cmd)) * self.max_steer
        # steering actuator has a slew rate, like a real wheel/controller
        max_rate = math.radians(220) * dt
        delta = target_steer - self.steer_angle
        delta = max(-max_rate, min(max_rate, delta))
        self.steer_angle += delta

        accel = throttle * 6.5 - brake * 9.0 - 0.25  # rolling resistance
        self.speed = max(0.0, min(self.max_speed, self.speed + accel * dt))

        # bicycle model
        if abs(self.steer_angle) > 1e-5:
            turn_radius = self.wheelbase / math.tan(self.steer_angle)
            angular_velocity = self.speed / turn_radius
        else:
            angular_velocity = 0.0
        self.heading += angular_velocity * dt
        self.x += math.cos(self.heading) * self.speed * dt
        self.y += math.sin(self.heading) * self.speed * dt


class World:
    def __init__(self, seed: int = 7):
        self.rng = random.Random(seed)
        self.centerline = Centerline()
        self.ego = EgoCar()
        cx, cy, ch = self.centerline.pose_at(0.0)
        # spawn centered in the right-hand lane (not straddling the yellow
        # centerline) -- same as where a real driver/bot would start
        nx, ny = -math.sin(ch), math.cos(ch)
        lane_center = LANE_WIDTH / 2
        self.ego.x = cx + nx * lane_center
        self.ego.y = cy + ny * lane_center
        self.ego.heading = ch

        self.lights: List[TrafficLight] = [
            TrafficLight(s=s, timer=self.rng.uniform(0, 20))
            for s in self._frange(INTERSECTION_SPACING, TRACK_LENGTH, INTERSECTION_SPACING)
        ]
        self.traffic: List[TrafficCar] = []
        for i in range(10):
            s0 = self.rng.uniform(20, TRACK_LENGTH)
            lane = self.rng.choice([-1, 1]) * (LANE_WIDTH / 2)
            color = self.rng.choice(
                [(60, 60, 220), (40, 170, 60), (210, 210, 40), (200, 80, 20), (230, 230, 230)]
            )
            self.traffic.append(TrafficCar(s=s0, lane_offset=lane, speed=self.rng.uniform(6, 11), color=color))
        self.pedestrians: List[Pedestrian] = []
        self.emergency_vehicles: List[EmergencyVehicle] = []
        self.time = 0.0
        self.collisions = 0
        self.off_road_time = 0.0
        self.ran_red_lights = 0
        self._last_light_state_at_stop = {}

    @staticmethod
    def _frange(a, b, step):
        v = a
        while v < b:
            yield v
            v += step

    def nearest_light_ahead(self, s: float):
        best = None
        best_d = 1e9
        for lt in self.lights:
            d = (lt.s - s) % TRACK_LENGTH
            if d < best_d:
                best_d = d
                best = lt
        return best, best_d

    def maybe_spawn_pedestrian(self):
        if self.rng.random() < 0.004 and len(self.pedestrians) < 3:
            light, dist = self.nearest_light_ahead(self.ego.s_hint)
            if light and dist < 40:
                direction = self.rng.choice([-1, 1])
                self.pedestrians.append(
                    Pedestrian(s=light.s + self.rng.uniform(-1, 1), lateral=-direction * 5.0, direction=direction)
                )

    def spawn_emergency_vehicle(self, kind: str = "police", mode: str = "overtaking"):
        """mode='overtaking': spawns behind, catching up and passing in our
        lane (classic "pull over and let them by" scenario). mode='parked':
        spawns ahead already stopped on the shoulder with lights on (classic
        "move over a lane / slow down passing a traffic stop" scenario)."""
        if mode == "overtaking":
            self.emergency_vehicles.append(EmergencyVehicle(
                s=(self.ego.s_hint - 35.0) % TRACK_LENGTH, lane_offset=0.0,
                speed=self.ego.speed + 9.0, kind=kind, lights_on=True, parked=False))
        else:
            self.emergency_vehicles.append(EmergencyVehicle(
                s=self.ego.s_hint + 18.0, lane_offset=LANE_WIDTH * 0.85,
                speed=0.0, kind=kind, lights_on=True, parked=True))

    def update(self, dt: float, steer_cmd: float, throttle: float, brake: float):
        self.time += dt
        self.ego.step(dt, steer_cmd, throttle, brake)
        self.ego.s_hint = self.centerline.nearest(self.ego.x, self.ego.y, self.ego.s_hint)

        for lt in self.lights:
            lt.update(dt)

        self.maybe_spawn_pedestrian()
        for p in self.pedestrians:
            p.update(dt)
        self.pedestrians = [p for p in self.pedestrians if p.active]

        for e in self.emergency_vehicles:
            if not e.parked:
                # always closing in relative to our current speed, like a
                # real responding unit trying to get past
                e.speed = self.ego.speed + 9.0
            e.update(dt)
            if not e.parked:
                # swing into the oncoming lane while actually passing us --
                # this is a forward-facing-dashcam sim, so an overtake from
                # directly behind only becomes visible right as it draws
                # alongside, same as in a real third-person chase camera.
                rel = (e.s - self.ego.s_hint + TRACK_LENGTH / 2) % TRACK_LENGTH - TRACK_LENGTH / 2
                if -8.0 < rel < 18.0:
                    e.lane_offset = -ROAD_HALF_WIDTH * 0.62
                else:
                    e.lane_offset = 0.0

        def _keep_ev(e):
            rel = (e.s - self.ego.s_hint + TRACK_LENGTH / 2) % TRACK_LENGTH - TRACK_LENGTH / 2
            return -40.0 < rel < 250.0
        self.emergency_vehicles = [e for e in self.emergency_vehicles if _keep_ev(e)]

        # scoring: off-road / collision detection (ground truth, for telemetry only)
        cx, cy, cheading = self.centerline.pose_at(self.ego.s_hint)
        ego_lateral_now = -(self.ego.x - cx) * math.sin(cheading) + (self.ego.y - cy) * math.cos(cheading)

        self.traffic.sort(key=lambda c: c.s)
        n = len(self.traffic)
        for i, car in enumerate(self.traffic):
            nxt = self.traffic[(i + 1) % n]
            gap = (nxt.s - car.s) % TRACK_LENGTH
            # Car-following only ever looked at the NEXT traffic car, never
            # at the ego -- so a stopped/slow ego sitting in a traffic
            # car's own lane was completely invisible to it and it would
            # just drive straight through (verified: this is exactly what
            # caused dozens of "collisions" per test run, a trailing car
            # plowing through a standing-start ego with zero braking,
            # which is not something any perception/control fix on the
            # autopilot side could ever prevent). Treat the ego as a
            # same-lane obstacle for gap-keeping too.
            if abs(car.lane_offset - ego_lateral_now) < CAR_WIDTH:
                ego_gap = (self.ego.s_hint - car.s) % TRACK_LENGTH
                if 0.0 < ego_gap < gap:
                    gap = ego_gap
            car.update(dt, gap if gap > 0.1 else TRACK_LENGTH)
            car.s %= TRACK_LENGTH
        lateral = ego_lateral_now
        if abs(lateral) > ROAD_HALF_WIDTH + SHOULDER:
            self.off_road_time += dt
        self._last_lateral = lateral

        for car in self.traffic:
            d_s = min((car.s - self.ego.s_hint) % TRACK_LENGTH, (self.ego.s_hint - car.s) % TRACK_LENGTH)
            if d_s < CAR_LENGTH and abs(lateral - car.lane_offset) < CAR_WIDTH:
                self.collisions += 1
        for p in self.pedestrians:
            d_s = min((p.s - self.ego.s_hint) % TRACK_LENGTH, (self.ego.s_hint - p.s) % TRACK_LENGTH)
            if d_s < CAR_LENGTH * 0.6 and abs(lateral - p.lateral) < 1.0:
                self.collisions += 1

    @property
    def lateral_offset(self) -> float:
        return getattr(self, "_last_lateral", 0.0)

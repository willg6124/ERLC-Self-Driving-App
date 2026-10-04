"""
Renders the ground-truth World into a pseudo-3D dashcam-style frame, the
same way a screen-capture region of the ER:LC window would look to the
autopilot. This is the ONLY thing the perception stack is allowed to see.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import cv2
import numpy as np

from .world import CAR_LENGTH, CAR_WIDTH, ROAD_HALF_WIDTH, SHOULDER, World

EV_COLORS = {"police": (40, 40, 30), "fire": (20, 20, 170), "ems": (120, 40, 220)}
# NOTE: "ems" intentionally avoids a near-white body -- a solid near-white
# box has ~zero HSV saturation and the heuristic object detector's
# "colorful" obstacle mask requires saturation > 60 to tell a vehicle apart
# from the grey road surface, so an all-white ambulance was previously
# invisible to perception at every distance. This rose/magenta tone is
# chosen to clear every exclusion band (lane-yellow, grass, sky, overbright)
# in HeuristicDetector while still reading as a distinct "EMS" vehicle.

WIDTH, HEIGHT = 960, 540
CAM_HEIGHT = 1.25
CAM_PITCH = math.radians(10.5)
FOV_SCALE = 620.0
RENDER_DISTANCE = 70.0
HORIZON_V = int(HEIGHT * 0.42)


def _project(dx: float, dy: float, heading: float, height_above_ground: float = 0.0) -> Optional[Tuple[int, int]]:
    """World-relative ground offset (+ optional height above ground, e.g. for
    drawing roofs/traffic-light heads) -> screen pixel, or None if behind
    the camera. World frame is right-handed with Y (here just "up") positive
    upward; image v grows downward."""
    forward = dx * math.cos(heading) + dy * math.sin(heading)
    right = -dx * math.sin(heading) + dy * math.cos(heading)
    if forward < 0.4:
        return None
    y_world = height_above_ground - CAM_HEIGHT  # relative to camera, up positive
    # rotate into the pitched-down camera frame (camera physically tilts
    # down by CAM_PITCH, so world points rotate by -CAM_PITCH in its frame)
    y_cam = y_world * math.cos(CAM_PITCH) + forward * math.sin(CAM_PITCH)
    z_cam = -y_world * math.sin(CAM_PITCH) + forward * math.cos(CAM_PITCH)
    if z_cam < 0.3:
        return None
    u = FOV_SCALE * right / z_cam + WIDTH / 2
    v = HEIGHT / 2 - FOV_SCALE * y_cam / z_cam
    return int(u), int(v)


def _sky_and_ground(frame: np.ndarray):
    sky_top = np.array([235, 180, 110], dtype=np.float32)
    sky_bot = np.array([250, 220, 180], dtype=np.float32)
    for v in range(0, HORIZON_V):
        t = v / max(1, HORIZON_V)
        color = sky_top * (1 - t) + sky_bot * t
        frame[v, :] = color
    frame[HORIZON_V:, :] = (70, 110, 70)


def _draw_ground_polyline(frame, world: World, lateral_offset_m: float, color, thickness=3, dashed=False):
    ego = world.ego
    cl = world.centerline
    pts = []
    # geometric spacing: dense close to the camera (where perspective warps
    # fastest) and coarse far away -- uniform spacing here would "skip"
    # across the screen for the first couple of samples and leave gaps.
    for d in np.geomspace(1.0, RENDER_DISTANCE, 160):
        s = ego.s_hint + d
        cx, cy, ch = cl.pose_at(s)
        nx, ny = -math.sin(ch), math.cos(ch)
        wx = cx + nx * lateral_offset_m
        wy = cy + ny * lateral_offset_m
        p = _project(wx - ego.x, wy - ego.y, ego.heading)
        if p is not None:
            pts.append((p, d))
    if len(pts) < 2:
        return
    max_jump = WIDTH * 0.6
    for i in range(len(pts) - 1):
        (p0, d0), (p1, d1) = pts[i], pts[i + 1]
        if dashed and int(d0 * 1.4) % 2 == 0:
            continue
        if abs(p1[0] - p0[0]) > max_jump or abs(p1[1] - p0[1]) > max_jump:
            continue
        thick = max(1, int(thickness * (1.0 - min(d0, 45) / 55)))
        cv2.line(frame, p0, p1, color, thick, cv2.LINE_AA)


def _draw_box_at(frame, world: World, s: float, lateral_m: float, heading_rel: float,
                  length: float, width: float, height: float, color, label: str = ""):
    ego = world.ego
    cl = world.centerline
    cx, cy, ch = cl.pose_at(s)
    nx, ny = -math.sin(ch), math.cos(ch)
    cx += nx * lateral_m
    cy += ny * lateral_m
    fx, fy = math.cos(ch), math.sin(ch)
    # build an oriented rectangle footprint then project base + top
    half_l, half_w = length / 2, width / 2
    local = [(-half_l, -half_w), (half_l, -half_w), (half_l, half_w), (-half_l, half_w)]
    base_pts = []
    top_pts = []
    behind = False
    for lx, lw in local:
        wx = cx + fx * lx - ny * lw
        wy = cy + fy * lx + nx * lw
        p0 = _project(wx - ego.x, wy - ego.y, ego.heading, 0.0)
        p1 = _project(wx - ego.x, wy - ego.y, ego.heading, height)
        if p0 is None or p1 is None:
            behind = True
            break
        base_pts.append(p0)
        top_pts.append(p1)
    if behind or not base_pts:
        return None
    base_pts = np.array(base_pts, dtype=np.int32)
    top_pts = np.array(top_pts, dtype=np.int32)
    body = np.array([top_pts[0], top_pts[1], base_pts[1], base_pts[0]], dtype=np.int32)
    cv2.fillConvexPoly(frame, np.array([base_pts[0], base_pts[1], top_pts[1], top_pts[0]]), color, cv2.LINE_AA)
    cv2.fillConvexPoly(frame, np.array([base_pts[1], base_pts[2], top_pts[2], top_pts[1]]),
                        tuple(int(c * 0.8) for c in color), cv2.LINE_AA)
    cv2.fillConvexPoly(frame, top_pts, tuple(min(255, int(c * 1.1)) for c in color), cv2.LINE_AA)
    all_pts = np.vstack([base_pts, top_pts])
    bbox = (int(all_pts[:, 0].min()), int(top_pts[:, 1].min()),
            int(all_pts[:, 0].max()), int(base_pts[:, 1].max()))
    if label:
        cv2.putText(frame, label, (bbox[0], max(12, bbox[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return bbox


def _draw_traffic_light(frame, world: World, light, dist):
    if dist > RENDER_DISTANCE:
        return None
    ego = world.ego
    cl = world.centerline
    cx, cy, ch = cl.pose_at(light.s)
    nx, ny = -math.sin(ch), math.cos(ch)
    pole_x = cx + nx * (ROAD_HALF_WIDTH + 1.0)
    pole_y = cy + ny * (ROAD_HALF_WIDTH + 1.0)
    base = _project(pole_x - ego.x, pole_y - ego.y, ego.heading, 0.0)
    top = _project(pole_x - ego.x, pole_y - ego.y, ego.heading, 4.6)
    if base is None or top is None:
        return None
    forward = max(dist, 1.0)
    head_h_px = int((1.1 / forward) * FOV_SCALE)
    head_w_px = max(4, int((0.5 / forward) * FOV_SCALE))
    cv2.line(frame, base, top, (40, 40, 40), max(1, head_w_px // 3), cv2.LINE_AA)
    box_tl = (top[0] - head_w_px, top[1] - head_h_px)
    box_br = (top[0] + head_w_px, top[1])
    cv2.rectangle(frame, box_tl, box_br, (25, 25, 25), -1)
    colors = {"red": (0, 0, 255), "yellow": (0, 220, 255), "green": (0, 200, 0)}
    lit = colors[light.state]
    cy_lamp = {
        "red": box_tl[1] + head_h_px // 4,
        "yellow": box_tl[1] + head_h_px // 2,
        "green": box_br[1] - head_h_px // 4,
    }
    for name, col in colors.items():
        c = col if name == light.state else tuple(int(v * 0.25) for v in col)
        radius = max(2, head_w_px // 2)
        cv2.circle(frame, (top[0], cy_lamp[name]), radius, c, -1, cv2.LINE_AA)
    bbox = (box_tl[0], box_tl[1], box_br[0], box_br[1])
    return bbox, light.state


def render(world: World) -> Tuple[np.ndarray, dict]:
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    _sky_and_ground(frame)

    meta = {"traffic_lights": [], "vehicles": [], "pedestrians": [], "emergency_vehicles": []}

    # road surface (filled polygon strip) drawn first via dense edge sampling
    left_pts, right_pts = [], []
    ego = world.ego
    cl = world.centerline
    for d in np.geomspace(1.0, RENDER_DISTANCE, 160):
        s = ego.s_hint + d
        cx, cy, ch = cl.pose_at(s)
        nx, ny = -math.sin(ch), math.cos(ch)
        lx, ly = cx + nx * -(ROAD_HALF_WIDTH + SHOULDER), cy + ny * -(ROAD_HALF_WIDTH + SHOULDER)
        rx, ry = cx + nx * (ROAD_HALF_WIDTH + SHOULDER), cy + ny * (ROAD_HALF_WIDTH + SHOULDER)
        pl = _project(lx - ego.x, ly - ego.y, ego.heading)
        pr = _project(rx - ego.x, ry - ego.y, ego.heading)
        if pl is not None:
            left_pts.append(pl)
        if pr is not None:
            right_pts.append(pr)
    if len(left_pts) > 1 and len(right_pts) > 1:
        poly = np.array(left_pts + right_pts[::-1], dtype=np.int32)
        cv2.fillPoly(frame, [poly], (58, 58, 60), cv2.LINE_AA)

    # lane markings
    _draw_ground_polyline(frame, world, -(ROAD_HALF_WIDTH + SHOULDER * 0.3), (235, 235, 235), 3)
    _draw_ground_polyline(frame, world, (ROAD_HALF_WIDTH + SHOULDER * 0.3), (235, 235, 235), 3)
    _draw_ground_polyline(frame, world, 0.0, (0, 215, 255), 3, dashed=True)

    # crosswalks near lights
    for lt in world.lights:
        d = (lt.s - ego.s_hint) % 2200.0
        if d < RENDER_DISTANCE:
            for off in np.arange(-ROAD_HALF_WIDTH - SHOULDER, ROAD_HALF_WIDTH + SHOULDER, 0.6):
                cx, cy, ch = cl.pose_at(lt.s - 2.0)
                nx, ny = -math.sin(ch), math.cos(ch)
                wx, wy = cx + nx * off, cy + ny * off
                p = _project(wx - ego.x, wy - ego.y, ego.heading)
                if p:
                    size = max(1, int(14 * (1 - min(d, 50) / 60)))
                    cv2.rectangle(frame, (p[0] - size, p[1] - 2), (p[0] + size, p[1] + 2), (230, 230, 230), -1)

    # other traffic
    for car in sorted(world.traffic, key=lambda c: -((c.s - ego.s_hint) % 2200.0)):
        d = (car.s - ego.s_hint) % 2200.0
        if 0.5 < d < RENDER_DISTANCE:
            bbox = _draw_box_at(frame, world, ego.s_hint + d, car.lane_offset, 0.0,
                                 CAR_LENGTH, CAR_WIDTH, 1.5, car.color)
            if bbox:
                meta["vehicles"].append({"bbox": bbox, "distance_m": round(d, 1)})

    # emergency vehicles (police / fire / ems) -- flashing light bar on the roof
    for ev in sorted(world.emergency_vehicles, key=lambda c: -((c.s - ego.s_hint) % 2200.0)):
        d = (ev.s - ego.s_hint + 1100.0) % 2200.0 - 1100.0
        if -20.0 < d < RENDER_DISTANCE:
            d_draw = max(d, 0.5)
            body_color = EV_COLORS.get(ev.kind, (50, 50, 50))
            bbox = _draw_box_at(frame, world, ego.s_hint + d_draw, ev.lane_offset, 0.0,
                                 CAR_LENGTH, CAR_WIDTH, 1.6, body_color, label=ev.kind.upper())
            if bbox and ev.lights_on:
                # flashing red/blue bar -- alternates fast enough to read as
                # "active lights" rather than a solid-colored vehicle
                phase = int(world.time * 6.0) % 2
                bar_colors = [(0, 0, 255), (255, 60, 0)] if phase == 0 else [(255, 60, 0), (0, 0, 255)]
                x1, y1, x2, y2 = bbox
                mid = (x1 + x2) // 2
                bar_h = max(2, (y2 - y1) // 8)
                cv2.rectangle(frame, (x1, y1 - bar_h), (mid, y1), bar_colors[0], -1)
                cv2.rectangle(frame, (mid, y1 - bar_h), (x2, y1), bar_colors[1], -1)
            if bbox:
                meta.setdefault("emergency_vehicles", []).append(
                    {"bbox": bbox, "distance_m": round(d, 1), "kind": ev.kind,
                     "lights_on": ev.lights_on, "parked": ev.parked})

    # pedestrians
    for p in world.pedestrians:
        d = (p.s - ego.s_hint) % 2200.0
        if 0.5 < d < RENDER_DISTANCE:
            bbox = _draw_box_at(frame, world, ego.s_hint + d, p.lateral, 0.0, 0.5, 0.5, 1.7, (40, 30, 200),
                                 label="PED")
            if bbox:
                meta["pedestrians"].append({"bbox": bbox, "distance_m": round(d, 1)})

    # traffic lights
    light, dist = world.nearest_light_ahead(ego.s_hint)
    if light:
        res = _draw_traffic_light(frame, world, light, dist)
        if res:
            bbox, state = res
            meta["traffic_lights"].append({"bbox": bbox, "state": state, "distance_m": round(dist, 1)})

    # simple hood/dash silhouette for realism
    cv2.rectangle(frame, (0, HEIGHT - 28), (WIDTH, HEIGHT), (20, 20, 20), -1)

    return frame, meta

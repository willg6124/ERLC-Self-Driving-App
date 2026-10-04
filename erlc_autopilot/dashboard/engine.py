"""Background simulation engine: owns the World + AutopilotPipeline, steps
them on a fixed-rate thread, and exposes thread-safe snapshots for the
Flask dashboard to render. This is what lets you *watch* the autopilot
drive in real time, right here in the browser preview, with zero Roblox
dependency."""
from __future__ import annotations

import math
import threading
import time
from typing import Optional

import cv2

from ..config import AutopilotConfig
from ..pipeline import AutopilotPipeline
from ..sim.renderer import render
from ..sim.world import Pedestrian, TrafficCar, World
from ..telemetry.logger import TelemetryLogger

TICK_HZ = 20
DT = 1.0 / TICK_HZ


class SimulationEngine:
    def __init__(self):
        self.lock = threading.Lock()
        self.config = AutopilotConfig()
        self.pipeline = AutopilotPipeline(self.config)
        self.world = World()
        self.logger = TelemetryLogger(csv_path="logs/session.csv")
        self._frame_jpeg = b""
        self._telemetry = {}
        self._running = True
        self._engaged = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._speed_multiplier = 1.0
        self._disengaged_time = 0.0
        self._auto_reset_after_s = 6.0

    def start(self):
        self._thread.start()

    def stop(self):
        self._running = False

    # --- public controls, called from Flask request handlers ---
    def toggle_engaged(self):
        with self.lock:
            self._engaged = not self._engaged
            self.pipeline.set_engaged(self._engaged)

    def reset(self):
        with self.lock:
            self.world = World(seed=int(time.time()) % 10000)
            self.pipeline = AutopilotPipeline(self.config)
            self._engaged = True
            self.logger = TelemetryLogger(csv_path="logs/session.csv")

    def update_config(self, patch: dict):
        with self.lock:
            self.config.update(patch)
            self.pipeline.apply_config(self.config)

    def trigger_pedestrian(self):
        with self.lock:
            w = self.world
            # Spawned close to the shoulder and already stepping toward the
            # lane at a brisk walking pace, timed so it's genuinely in the
            # ego's path by the time the car arrives -- reliably exercises
            # the emergency-braking path instead of a crossing pedestrian
            # that the car simply passes before they reach the lane.
            w.pedestrians.append(Pedestrian(s=w.ego.s_hint + 20, lateral=-1.8, direction=1, speed=1.2))

    def trigger_cutin(self):
        with self.lock:
            w = self.world
            w.traffic.append(TrafficCar(s=(w.ego.s_hint + 16) % w.centerline.length, lane_offset=0.0,
                                         speed=w.ego.speed * 0.5, color=(30, 30, 220)))

    def trigger_emergency_vehicle(self, kind: str = "police", mode: str = "overtaking"):
        with self.lock:
            self.world.spawn_emergency_vehicle(kind=kind, mode=mode)

    def force_red_light(self):
        with self.lock:
            w = self.world
            light, _ = w.nearest_light_ahead(w.ego.s_hint)
            if light:
                light.s = w.ego.s_hint + 40
                light.state = "red"
                g, y, _r = light.cycle
                light.timer = g + y  # land right at the start of the red phase

    # --- main loop ---
    def _run(self):
        last = time.time()
        while self._running:
            now = time.time()
            elapsed = now - last
            if elapsed < DT:
                time.sleep(DT - elapsed)
                now = time.time()
            last = now

            with self.lock:
                frame, meta = render(self.world)
                result = self.pipeline.tick(frame, self.world.ego.speed, DT)
                cmd = result.command
                self.world.update(DT, cmd.steer, cmd.throttle, cmd.brake)
                self.logger.log_tick(self.world.time, self.world.ego.speed, cmd, result.lane, DT)

                # a real lane-keep system hands control back to a human when
                # it loses confidence for too long; here there's no human,
                # so after a short grace period we just start a fresh drive
                # rather than leaving the dashboard stuck showing a car
                # stranded in a field.
                if not cmd.engaged or abs(self.world.lateral_offset) > 12.0:
                    self._disengaged_time += DT
                else:
                    self._disengaged_time = 0.0
                if self._disengaged_time > self._auto_reset_after_s:
                    self.logger.push_event(self.world.time, "AUTO_RESET: starting a fresh drive")
                    self.world = World(seed=int(time.time()) % 10000)
                    self.pipeline.set_engaged(True)
                    self._disengaged_time = 0.0
                    continue

                ok, buf = cv2.imencode(".jpg", result.debug_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ok:
                    self._frame_jpeg = buf.tobytes()

                light, dist = self.world.nearest_light_ahead(self.world.ego.s_hint)
                self._telemetry = {
                    "speed_mps": round(self.world.ego.speed, 2),
                    "speed_mph": round(self.world.ego.speed * 2.23694, 1),
                    "target_speed_mph": round(cmd.target_speed * 2.23694, 1),
                    "steer": round(cmd.steer, 3),
                    "throttle": round(cmd.throttle, 2),
                    "brake": round(cmd.brake, 2),
                    "status": cmd.status,
                    "engaged": cmd.engaged,
                    "turn_signal": cmd.turn_signal,
                    "lane_offset": round(result.lane.offset_norm, 3),
                    "lane_confidence": round(result.lane.confidence, 2),
                    "curvature": round(result.lane.curvature, 4),
                    "fps": round(result.fps, 1),
                    "nearest_light": {"state": light.state, "distance_m": round(dist, 1)} if light and dist < 70 else None,
                    "vehicles_ahead": len(meta["vehicles"]),
                    "pedestrians_ahead": len(meta["pedestrians"]),
                    "emergency_vehicles": [
                        {"distance_m": round(ev.distance_m, 1), "flashing": ev.flashing,
                         "confidence": round(ev.confidence, 2)}
                        for ev in result.emergency_vehicles
                    ],
                    "stats": {
                        "distance_m": round(self.logger.stats.distance_m, 1),
                        "time_engaged_s": round(self.logger.stats.time_engaged_s, 1),
                        "time_total_s": round(self.logger.stats.time_total_s, 1),
                        "collisions": self.world.collisions,
                        "emergency_brakes": self.logger.stats.emergency_brakes,
                        "lights_stopped_for": self.logger.stats.lights_stopped_for,
                        "max_speed_mph": round(self.logger.stats.max_speed_mps * 2.23694, 1),
                        "avg_abs_offset": round(self.logger.stats.avg_abs_offset, 3),
                        "off_road_time_s": round(self.world.off_road_time, 1),
                    },
                    "events": list(self.logger.events),
                    "config": self.config.to_dict(),
                }

    def get_frame(self) -> bytes:
        with self.lock:
            return self._frame_jpeg

    def get_telemetry(self) -> dict:
        with self.lock:
            return dict(self._telemetry)

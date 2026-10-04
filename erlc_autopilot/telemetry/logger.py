"""Session telemetry: a rolling event feed for the dashboard plus an
optional CSV trace (speed/steer/offset/etc per tick) for later analysis --
handy for tuning PID gains offline in a spreadsheet/notebook."""
from __future__ import annotations

import csv
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional


@dataclass
class SessionStats:
    distance_m: float = 0.0
    time_engaged_s: float = 0.0
    time_total_s: float = 0.0
    collisions: int = 0
    emergency_brakes: int = 0
    lights_stopped_for: int = 0
    max_speed_mps: float = 0.0
    avg_abs_offset: float = 0.0
    _offset_samples: int = 0


class TelemetryLogger:
    def __init__(self, csv_path: Optional[str] = None, event_feed_size: int = 60):
        self.csv_path = csv_path
        self.stats = SessionStats()
        self.events: Deque[dict] = deque(maxlen=event_feed_size)
        self._csv_file = None
        self._csv_writer = None
        if csv_path:
            os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
            self._csv_file = open(csv_path, "w", newline="")
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow(
                ["t", "speed_mps", "target_speed", "steer", "throttle", "brake",
                 "offset_norm", "curvature", "confidence", "status"]
            )

    def log_tick(self, t: float, speed: float, command, lane, dt: float):
        self.stats.time_total_s += dt
        if command.engaged:
            self.stats.time_engaged_s += dt
        self.stats.distance_m += speed * dt
        self.stats.max_speed_mps = max(self.stats.max_speed_mps, speed)
        self.stats._offset_samples += 1
        n = self.stats._offset_samples
        self.stats.avg_abs_offset += (abs(lane.offset_norm) - self.stats.avg_abs_offset) / n

        for ev in command.events:
            self.push_event(t, ev)

        if self._csv_writer:
            self._csv_writer.writerow([f"{t:.2f}", f"{speed:.2f}", f"{command.target_speed:.2f}",
                                        f"{command.steer:.3f}", f"{command.throttle:.2f}",
                                        f"{command.brake:.2f}", f"{lane.offset_norm:.3f}",
                                        f"{lane.curvature:.4f}", f"{lane.confidence:.2f}", command.status])

    def push_event(self, t: float, message: str):
        severity = "critical" if message.startswith("EMERGENCY") or "AUTOPILOT_DISENGAGED" in message else (
            "warning" if message.startswith(("hard_brake", "traffic_light")) else "info")
        self.events.appendleft({"t": round(t, 1), "message": message, "severity": severity,
                                 "wall_time": time.strftime("%H:%M:%S")})
        if message.startswith("EMERGENCY"):
            self.stats.emergency_brakes += 1
        if message.startswith("traffic_light"):
            self.stats.lights_stopped_for += 1

    def close(self):
        if self._csv_file:
            self._csv_file.close()

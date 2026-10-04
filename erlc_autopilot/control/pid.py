"""A small, dependency-free PID controller with anti-windup clamping and
derivative-on-measurement filtering (avoids derivative kick on setpoint
changes, which matters since our "setpoint" -- lane center -- is always 0
but the measurement is noisy CV output)."""
from __future__ import annotations


class PID:
    def __init__(self, kp: float, ki: float, kd: float, output_limits=(-1.0, 1.0),
                 integral_limits=(-1.0, 1.0)):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.output_limits = output_limits
        self.integral_limits = integral_limits
        self._integral = 0.0
        self._prev_measurement = None
        self._prev_error = 0.0

    def reset(self):
        self._integral = 0.0
        self._prev_measurement = None

    def step(self, error: float, measurement: float, dt: float) -> float:
        if dt <= 0:
            return 0.0
        self._integral += error * dt
        lo, hi = self.integral_limits
        self._integral = max(lo, min(hi, self._integral))

        if self._prev_measurement is None:
            derivative = 0.0
        else:
            derivative = -(measurement - self._prev_measurement) / dt
        self._prev_measurement = measurement
        self._prev_error = error

        output = self.kp * error + self.ki * self._integral + self.kd * derivative
        lo, hi = self.output_limits
        return max(lo, min(hi, output))

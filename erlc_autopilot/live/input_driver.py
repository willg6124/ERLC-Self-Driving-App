"""Turns a `Command` (steer/throttle/brake/turn_signal/...) into real
keyboard input during live ER:LC gameplay -- the actuator half of
"decoupled from where the command goes" (see pipeline.py).

Exact control scheme (fixed, not configurable -- matches the real ER:LC
vehicle bindings this app is built around):
    W / A / S / D   drive (throttle / steer-left / brake-reverse / steer-right)
    P               parking brake
    Q / E           left / right turn signal
    G               hazards
    H               horn

WASD are digital keys in Roblox (no analog pedal), but the control policy
produces continuous [0,1]/[-1,1] values. `KeyboardActuator` converts those
to effectively-analog behaviour with a small software-PWM loop running
independently of the (slower) perception tick rate, so steering stays
smooth and control latency stays low regardless of CV frame rate -- this
is also where the "make it fast" requirement is actually earned: the
actuation loop runs at `PWM_HZ`, decoupled from whatever the vision
pipeline is currently doing.
"""
from __future__ import annotations

import threading
import time
from typing import Optional, Protocol


class _KeyBackend(Protocol):
    def key_down(self, key: str) -> None: ...
    def key_up(self, key: str) -> None: ...
    def tap(self, key: str) -> None: ...


class DryRunBackend:
    """No-op backend that just records what it *would* send. Used for
    sim-mode testing (there's no real OS/game to send keys to in this
    sandbox) and doubles as a safe "rehearsal" mode on a real machine
    before trusting the bot to actually touch your keyboard.
    """

    def __init__(self):
        self.log: list = []
        self._down: set = set()

    def key_down(self, key: str) -> None:
        if key not in self._down:
            self.log.append(("down", key, time.time()))
            self._down.add(key)

    def key_up(self, key: str) -> None:
        if key in self._down:
            self.log.append(("up", key, time.time()))
            self._down.discard(key)

    def tap(self, key: str) -> None:
        self.log.append(("tap", key, time.time()))


class PyDirectInputBackend:
    """Real key injection via `pydirectinput`, which sends DirectInput
    style scan codes rather than the higher-level SendInput virtual-key
    events `pynput`/`pyautogui` use by default. Roblox, like most games
    built against DirectInput/XInput polling, frequently ignores the
    latter -- this is the backend that actually registers in the live
    game on Windows. Windows-only.
    """

    def __init__(self):
        import pydirectinput

        pydirectinput.PAUSE = 0.0
        pydirectinput.FAILSAFE = False
        self._lib = pydirectinput

    def key_down(self, key: str) -> None:
        self._lib.keyDown(key)

    def key_up(self, key: str) -> None:
        self._lib.keyUp(key)

    def tap(self, key: str) -> None:
        self._lib.keyDown(key)
        self._lib.keyUp(key)


class PynputBackend:
    """Fallback backend for dev machines / platforms where
    `pydirectinput` isn't available. Uses standard OS-level key events,
    which work fine for most desktop apps but may not register in Roblox
    on Windows -- prefer `PyDirectInputBackend` there.
    """

    def __init__(self):
        from pynput.keyboard import Controller

        self._ctrl = Controller()

    def key_down(self, key: str) -> None:
        self._ctrl.press(key)

    def key_up(self, key: str) -> None:
        self._ctrl.release(key)

    def tap(self, key: str) -> None:
        self._ctrl.press(key)
        self._ctrl.release(key)


def select_backend(name: str = "auto") -> _KeyBackend:
    if name == "dry_run":
        return DryRunBackend()
    if name == "pydirectinput":
        return PyDirectInputBackend()
    if name == "pynput":
        return PynputBackend()
    if name == "auto":
        try:
            return PyDirectInputBackend()
        except Exception:
            pass
        try:
            return PynputBackend()
        except Exception:
            pass
        return DryRunBackend()
    raise ValueError(f"unknown input backend: {name}")


class KeyboardActuator:
    """Converts a stream of `Command`s into held/tapped WASD + P/Q/E/G/H
    key events on a dedicated high-rate thread.
    """

    PWM_HZ = 50  # actuation rate, independent of (and faster than) the
    # perception tick rate -- this is what keeps control latency low

    def __init__(self, backend: str = "auto"):
        self.backend_name = backend
        self._backend: Optional[_KeyBackend] = None
        self._lock = threading.Lock()
        self._steer = 0.0
        self._throttle = 0.0
        self._brake = 0.0
        self._engaged = False
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._turn_signal: Optional[str] = None
        self._hazards_on = False
        self._phase = {"w": 0.0, "s": 0.0, "a": 0.0, "d": 0.0}

    def start(self) -> None:
        self._backend = select_backend(self.backend_name)
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="keyboard-actuator")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        self._release_all()

    def release_all(self) -> None:
        """Public wrapper so callers outside this module (e.g. the
        runner's error handler) can snap back to a safe all-keys-up state
        without reaching into a private method."""
        with self._lock:
            self._steer = 0.0
            self._throttle = 0.0
            self._brake = 0.0
        self._release_all()

    def apply(self, command) -> None:
        """Called once per perception tick (~20 Hz); just updates the
        target state that the PWM thread (running at `PWM_HZ`) actuates
        continuously in between calls."""
        with self._lock:
            self._steer = command.steer
            self._throttle = command.throttle
            self._brake = command.brake
            self._engaged = command.engaged
            if command.turn_signal != self._turn_signal:
                self._tap_turn_signal(command.turn_signal)
                self._turn_signal = command.turn_signal

    def _tap_turn_signal(self, signal: Optional[str]) -> None:
        if self._backend is None:
            return
        if signal == "left":
            self._backend.tap("q")
        elif signal == "right":
            self._backend.tap("e")
        elif signal is None and self._turn_signal is not None:
            # cancel whichever one was previously active
            self._backend.tap("q" if self._turn_signal == "left" else "e")

    def set_hazards(self, on: bool) -> None:
        if self._backend is None or on == self._hazards_on:
            return
        self._backend.tap("g")
        self._hazards_on = on

    def honk(self) -> None:
        if self._backend is not None:
            self._backend.tap("h")

    def set_parking_brake(self, on: bool) -> None:
        if self._backend is not None and on:
            self._backend.tap("p")

    def emergency_stop(self) -> None:
        """Kill switch: instantly release every key and disengage. Safe
        to call from any thread (e.g. the global hotkey listener)."""
        with self._lock:
            self._engaged = False
        self._release_all()

    def _run(self) -> None:
        dt = 1.0 / self.PWM_HZ
        while self._running:
            t0 = time.time()
            with self._lock:
                steer, throttle, brake, engaged = self._steer, self._throttle, self._brake, self._engaged
            if not engaged:
                self._release_all()
            else:
                self._actuate(steer, throttle, brake)
            elapsed = time.time() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)

    def _duty(self, key: str, value: float) -> None:
        """Software PWM: holds `key` down for `value` (0..1) fraction of
        each actuation window, released the rest of the time -- turns a
        digital key into an effectively-analog throttle/brake/steer
        input, the standard trick for driving games from a continuous
        control policy."""
        value = max(0.0, min(1.0, value))
        self._phase[key] += value
        if self._phase[key] >= 1.0:
            self._phase[key] -= 1.0
            self._backend.key_down(key)
        else:
            self._backend.key_up(key)

    def _actuate(self, steer: float, throttle: float, brake: float) -> None:
        self._duty("w", throttle)
        self._duty("s", brake)
        if steer >= 0:
            self._duty("d", steer)
            self._backend.key_up("a")
            self._phase["a"] = 0.0
        else:
            self._duty("a", -steer)
            self._backend.key_up("d")
            self._phase["d"] = 0.0

    def _release_all(self) -> None:
        if self._backend is None:
            return
        for k in ("w", "a", "s", "d"):
            self._backend.key_up(k)
            self._phase[k] = 0.0

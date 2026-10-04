"""A global "stop the car right now" hotkey.

A keyboard emergency brake is only as good as how fast you can hit it --
if it only worked while the overlay window had focus, you'd have to
alt-tab away from the game to use it, which defeats the point. This uses
the `keyboard` library's global low-level hook so the hotkey fires no
matter which window is focused (including ER:LC itself). Bound to F9 by
default.

If the `keyboard` package can't install its hook (missing dependency, or
on some Windows setups a non-admin process can't install a global hook),
`arm()` returns False and the caller should fall back to "use the
Disengage button in the overlay" -- never crash the app over this.
"""
from __future__ import annotations

from typing import Callable, Optional


class KillSwitch:
    def __init__(self, on_trigger: Callable[[], None], hotkey: str = "f9"):
        self.on_trigger = on_trigger
        self.hotkey = hotkey
        self._armed = False

    def arm(self) -> bool:
        try:
            import keyboard

            keyboard.add_hotkey(self.hotkey, self.on_trigger)
            self._armed = True
            return True
        except Exception:
            return False

    def disarm(self) -> None:
        if not self._armed:
            return
        try:
            import keyboard

            keyboard.remove_hotkey(self.hotkey)
        except Exception:
            pass
        self._armed = False

    @property
    def armed(self) -> bool:
        return self._armed

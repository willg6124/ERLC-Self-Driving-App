"""Locates the Roblox/ER:LC game window on Windows so the capture region
can be calibrated automatically instead of the user having to eyeball
pixel coordinates by hand. Falls back gracefully (returns None) on
non-Windows platforms or if `pywin32` isn't installed -- the setup
wizard's manual click-and-drag calibration step always works either way,
this is just the "do it for me" shortcut.
"""
from __future__ import annotations

import platform
from dataclasses import dataclass
from typing import Optional


@dataclass
class WindowRegion:
    left: int
    top: int
    width: int
    height: int

    def to_mss_region(self) -> dict:
        return {"left": self.left, "top": self.top, "width": self.width, "height": self.height}

    def to_dict(self) -> dict:
        return {"left": self.left, "top": self.top, "width": self.width, "height": self.height}


def find_roblox_window() -> Optional[WindowRegion]:
    """Best-effort: ER:LC runs inside the Roblox client window, whose
    title is typically just "Roblox" (sometimes the place name, depending
    on client version/fullscreen mode). Returns None if not found, not on
    Windows, or pywin32 isn't installed -- callers should fall back to
    manual calibration in that case, never crash the wizard over it.
    """
    if platform.system() != "Windows":
        return None
    try:
        import win32gui  # pywin32, Windows-only

        found = []

        def _enum(hwnd, _):
            if not win32gui.IsWindowVisible(hwnd):
                return
            title = win32gui.GetWindowText(hwnd)
            if not title:
                return
            if "roblox" in title.lower():
                l, t, r, b = win32gui.GetWindowRect(hwnd)
                if r - l > 200 and b - t > 200:
                    found.append(WindowRegion(l, t, r - l, b - t))

        win32gui.EnumWindows(_enum, None)
        if found:
            # several hidden/offscreen helper windows can share "Roblox" in
            # their title -- the real game window is overwhelmingly likely
            # to be the largest one
            found.sort(key=lambda r: r.width * r.height, reverse=True)
            return found[0]
    except Exception:
        pass
    return None


def primary_screen_size() -> Optional[WindowRegion]:
    """Fallback capture region covering the whole primary monitor, for
    users running ER:LC borderless-fullscreen (no discoverable window
    border) or when window auto-detection fails."""
    try:
        import mss

        with mss.mss() as sct:
            mon = sct.monitors[1]  # [0] is "all monitors combined"
            return WindowRegion(mon["left"], mon["top"], mon["width"], mon["height"])
    except Exception:
        return None

"""A small always-on-top HUD overlay, built with plain Tkinter instead of
a browser.

The previous wizard/overlay was a local Flask site opened in a pywebview
native window -- and pywebview can silently fall back to a legacy
Internet-Explorer-based renderer on Windows machines without the Edge
WebView2 Runtime registered, which has no fetch()/modern JS at all, so
every button quietly did nothing. Tkinter ships with the standard
python.org Windows installer, runs as plain native widgets with zero
browser engine involved, and talks to the `LiveRunner` directly in the
same process (no HTTP, no JSON, nothing that can go silently wrong
between a button click and the actual Python callback running).

This is deliberately optional: `run_live.py` tries it and falls back to
plain console output if `tkinter`/`Pillow` aren't available (e.g. this
dev sandbox, or a stripped-down Python install) -- the autopilot itself
runs identically either way, this is purely the on-screen status window.
"""
from __future__ import annotations

import cv2

from .runner import LiveRunner


def run_overlay(runner: LiveRunner) -> None:
    """Blocks until the overlay window is closed. Raises ImportError if
    tkinter/Pillow aren't installed -- callers should catch that and fall
    back to `console_loop` instead."""
    import tkinter as tk
    from PIL import Image, ImageTk

    root = tk.Tk()
    root.title("ERLC Autopilot")
    root.attributes("-topmost", True)
    root.configure(bg="#0c0f14")
    root.geometry("300x430+40+40")
    root.resizable(False, False)

    FG = "#e8edf5"
    MUTED = "#8a93a3"
    ACCENT = "#5fd4ff"
    GOOD = "#35d07f"
    BAD = "#ff4d5e"
    PANEL = "#151a22"

    header = tk.Frame(root, bg="#0c0f14")
    header.pack(fill="x", padx=10, pady=(10, 4))
    dot = tk.Label(header, text="\u25cf", fg=GOOD, bg="#0c0f14", font=("Segoe UI", 12))
    dot.pack(side="left")
    tk.Label(header, text=" ERLC AUTOPILOT", fg=FG, bg="#0c0f14",
              font=("Segoe UI", 10, "bold")).pack(side="left")
    tier_label = tk.Label(header, text="--", fg=MUTED, bg="#0c0f14", font=("Segoe UI", 8))
    tier_label.pack(side="right")

    cam_label = tk.Label(root, bg="black")
    cam_label.pack(padx=10, pady=4)

    ev_banner = tk.Label(root, text="\u26a0 EMERGENCY VEHICLE \u2014 YIELDING", fg="#ffd6f6",
                          bg="#2a1230", font=("Segoe UI", 8, "bold"))

    stats = tk.Frame(root, bg="#0c0f14")
    stats.pack(fill="x", padx=10, pady=4)
    speed_box = tk.Frame(stats, bg=PANEL)
    speed_box.pack(side="left", expand=True, fill="x", padx=(0, 4))
    tk.Label(speed_box, text="SPEED", fg=MUTED, bg=PANEL, font=("Segoe UI", 7)).pack()
    speed_val = tk.Label(speed_box, text="0", fg=ACCENT, bg=PANEL, font=("Segoe UI", 18, "bold"))
    speed_val.pack()
    tk.Label(speed_box, text="mph", fg=MUTED, bg=PANEL, font=("Segoe UI", 7)).pack()

    target_box = tk.Frame(stats, bg=PANEL)
    target_box.pack(side="left", expand=True, fill="x", padx=(4, 0))
    tk.Label(target_box, text="TARGET", fg=MUTED, bg=PANEL, font=("Segoe UI", 7)).pack()
    target_val = tk.Label(target_box, text="0", fg=ACCENT, bg=PANEL, font=("Segoe UI", 18, "bold"))
    target_val.pack()
    tk.Label(target_box, text="mph", fg=MUTED, bg=PANEL, font=("Segoe UI", 7)).pack()

    status_label = tk.Label(root, text="starting...", fg=GOOD, bg="#0c0f14",
                             font=("Segoe UI", 9, "bold"))
    status_label.pack(pady=(4, 0))
    signal_label = tk.Label(root, text="", fg=MUTED, bg="#0c0f14", font=("Segoe UI", 8))
    signal_label.pack()

    btn_row = tk.Frame(root, bg="#0c0f14")
    btn_row.pack(fill="x", padx=10, pady=8)

    def do_toggle():
        engaged = runner.toggle_engaged()
        _refresh_toggle_text(engaged)

    def _refresh_toggle_text(engaged: bool):
        toggle_btn.config(text="\u23f8 Disengage" if engaged else "\u25b6 Engage")

    toggle_btn = tk.Button(btn_row, text="\u23f8 Disengage", command=do_toggle,
                            bg=ACCENT, fg="#06131c", font=("Segoe UI", 9, "bold"), relief="flat")
    toggle_btn.pack(side="left", expand=True, fill="x", padx=2)
    tk.Button(btn_row, text="\U0001f4ef", command=runner.honk, bg="#232a36", fg=FG,
              relief="flat").pack(side="left", padx=2)

    hazards_state = {"on": False}

    def do_hazards():
        hazards_state["on"] = not hazards_state["on"]
        runner.set_hazards(hazards_state["on"])

    tk.Button(btn_row, text="\u26a0", command=do_hazards, bg="#232a36", fg=FG,
              relief="flat").pack(side="left", padx=2)

    def do_stop():
        runner.emergency_stop()
        _refresh_toggle_text(False)

    tk.Button(root, text="\u23f9 KILL SWITCH (or press F9)", command=do_stop,
              bg="#3a1414", fg="#ffc9ce", font=("Segoe UI", 8, "bold"),
              relief="flat").pack(fill="x", padx=10, pady=(0, 10))

    def refresh():
        t = runner.get_telemetry()
        if t:
            speed_val.config(text=str(round(t.get("speed_mph", 0))))
            target_val.config(text=str(round(t.get("target_speed_mph", 0))))
            status_label.config(text=str(t.get("status", "--")).replace("_", " "))
            tier_label.config(text=str(t.get("model_tier", "--")).upper())
            engaged = bool(t.get("engaged"))
            dot.config(fg=GOOD if engaged else BAD)
            status_label.config(fg=GOOD if engaged else BAD)
            _refresh_toggle_text(engaged)

            evs = t.get("emergency_vehicles") or []
            if any(e.get("flashing") for e in evs):
                ev_banner.pack(fill="x", padx=10, pady=(0, 2), before=stats)
            else:
                ev_banner.pack_forget()

            signal = t.get("turn_signal")
            if signal:
                signal_label.config(text="signal: " + str(signal).upper())
            elif t.get("speed_is_real") is False:
                signal_label.config(text="speed: estimated (no OCR)")
            else:
                signal_label.config(text="")

        frame_bytes = runner.get_frame()
        if frame_bytes:
            try:
                import numpy as np
                arr = np.frombuffer(frame_bytes, dtype="uint8")
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is not None:
                    frame = cv2.resize(frame, (280, 158))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame)
                    photo = ImageTk.PhotoImage(img)
                    cam_label.config(image=photo)
                    cam_label.image = photo  # keep a reference, tkinter needs it
            except Exception:
                pass

        root.after(150, refresh)

    def on_close():
        runner.stop()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.after(150, refresh)
    root.mainloop()

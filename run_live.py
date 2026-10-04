#!/usr/bin/env python3
"""Drives ER:LC for you: captures your screen, drives with real keyboard
input (W/A/S/D, P, Q/E, G, H). No web browser, no setup wizard -- just:

    python run_live.py

1. Launch Roblox, join ER:LC, get into a car.
2. Run this script. On Windows it auto-detects the Roblox window (falls
   back to your whole primary monitor if it can't find it -- still works
   fine if Roblox is fullscreen/maximized). The region is remembered in
   ~/.erlc_autopilot/calibration.json so this only has to happen once;
   pass --recalibrate to redo it.
3. It counts down a few seconds so you can switch back to the game, then
   starts driving. Status prints live to this console, plus a small
   always-on-top overlay window if `tkinter`/`Pillow` are available
   (they ship with the standard python.org Windows installer).
4. Press F9 at any time to instantly stop (kill switch), or Ctrl+C here.

On any OS other than Windows (e.g. a dev sandbox with no Roblox), this
automatically drives the bundled simulator instead (Sim Mode) so the
whole app can still be exercised without Windows/Roblox installed -- pass
--sim to force this on Windows too.
"""
from __future__ import annotations

import argparse
import platform
import sys
import time

from erlc_autopilot.config import AutopilotConfig
from erlc_autopilot.live.calibration import Calibration
from erlc_autopilot.live.capture import ScreenCaptureSource, SimFrameSource
from erlc_autopilot.live.input_driver import KeyboardActuator
from erlc_autopilot.live.runner import LiveRunner
from erlc_autopilot.live.window_capture import WindowRegion, find_roblox_window, primary_screen_size
from erlc_autopilot.perception.speed_reader import SpeedReader, SpeedReaderConfig

IS_WINDOWS = platform.system() == "Windows"


def _auto_calibrate(cal: Calibration) -> Calibration:
    print("Looking for the Roblox window...")
    region = find_roblox_window()
    if region is not None:
        print(f"  found it: {region.width}x{region.height} at ({region.left},{region.top})")
    else:
        region = primary_screen_size()
        if region is not None:
            print(f"  couldn't find a 'Roblox' window -- using your primary monitor instead "
                  f"({region.width}x{region.height}). Make sure ER:LC is running and visible.")
        else:
            region = WindowRegion(0, 0, 1920, 1080)
            print("  couldn't detect anything -- defaulting to 1920x1080 at (0,0). "
                  "Edit ~/.erlc_autopilot/calibration.json if that's wrong for your setup.")
    cal.capture_left, cal.capture_top = region.left, region.top
    cal.capture_width, cal.capture_height = region.width, region.height
    cal.calibrated = True
    cal.save()
    return cal


def _console_loop(runner: LiveRunner) -> None:
    print("\nNo tkinter/Pillow overlay available -- printing status to this console instead.")
    print("Press Ctrl+C to stop (or F9 from anywhere, including inside the game).\n")
    try:
        while True:
            t = runner.get_telemetry()
            if t:
                evs = t.get("emergency_vehicles") or []
                ev_flag = " [EV!]" if any(e.get("flashing") for e in evs) else ""
                sig = f" signal={t.get('turn_signal')}" if t.get("turn_signal") else ""
                line = (f"\rspeed={t.get('speed_mph', 0):5.1f}mph  "
                        f"target={t.get('target_speed_mph', 0):5.1f}mph  "
                        f"status={str(t.get('status', '--')):18s}  "
                        f"engaged={t.get('engaged')!s:5s}{sig}{ev_flag}   ")
                sys.stdout.write(line)
                sys.stdout.flush()
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nStopping...")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sim", action="store_true",
                         help="Force Sim Mode even on Windows (drive the bundled simulator).")
    parser.add_argument("--tier", choices=["lite", "standard", "pro"], default=None,
                         help="Model tier (default: remembered from last run, or 'standard').")
    parser.add_argument("--recalibrate", action="store_true",
                         help="Re-detect the capture region instead of reusing the saved one.")
    parser.add_argument("--region", type=str, default=None,
                         help="Manually set the capture region as left,top,width,height "
                              "(skips auto-detection).")
    parser.add_argument("--no-overlay", action="store_true",
                         help="Skip the tkinter overlay window even if available; console only.")
    parser.add_argument("--input-backend", choices=["auto", "pydirectinput", "pynput", "dry_run"],
                         default=None, help="Override how keys are sent (default: remembered/auto).")
    args = parser.parse_args()

    force_sim = args.sim or not IS_WINDOWS
    cal = Calibration.load()
    if args.tier:
        cal.model_tier = args.tier
    if args.input_backend:
        cal.input_backend = args.input_backend

    print("=" * 60)
    print(" ERLC AUTOPILOT")
    print("=" * 60)
    print("Controls it drives with: W A S D | P parking brake | Q/E turn")
    print("signals | G hazards | H horn | F9 KILL SWITCH (works anywhere)")
    print("This drives your keyboard for you -- keep a hand near it and")
    print("watch the road. It will make mistakes.")
    print("=" * 60)

    if force_sim:
        print("Sim Mode: no Windows/ER:LC capture on this machine -- driving the")
        print("bundled simulator instead so the app can still be exercised.")
        frame_source = SimFrameSource()
        actuator = KeyboardActuator(backend="dry_run")
        speed_reader = None
        ground_truth = lambda: frame_source.world.ego.speed
        advance_sim = lambda cmd, dt: frame_source.world.update(dt, cmd.steer, cmd.throttle, cmd.brake)
    else:
        if args.region:
            try:
                l, t, w, h = (int(x) for x in args.region.split(","))
                cal.capture_left, cal.capture_top, cal.capture_width, cal.capture_height = l, t, w, h
                cal.calibrated = True
                cal.save()
            except ValueError:
                print(f"Couldn't parse --region '{args.region}' as left,top,width,height; ignoring.")
        if not cal.calibrated or args.recalibrate:
            cal = _auto_calibrate(cal)
        print(f"Capture region: {cal.capture_width}x{cal.capture_height} at "
              f"({cal.capture_left},{cal.capture_top})  [--recalibrate to redo]")
        region = WindowRegion(cal.capture_left, cal.capture_top, cal.capture_width, cal.capture_height)
        frame_source = ScreenCaptureSource(region)
        actuator = KeyboardActuator(backend=cal.input_backend)
        speed_reader = SpeedReader(SpeedReaderConfig(roi=list(cal.speed_roi)))
        if not speed_reader.available():
            print("Note: Tesseract OCR not found -- speed will be an estimate, not a real "
                  "reading. See requirements.txt for how to install it.")
        ground_truth = None
        advance_sim = None

    config = AutopilotConfig.for_tier(cal.model_tier)
    runner = LiveRunner(frame_source, actuator, config=config,
                         speed_reader=speed_reader, ground_truth_speed=ground_truth,
                         advance_sim=advance_sim)
    if not force_sim:
        print(f"Model tier: {cal.model_tier}  |  input backend: {cal.input_backend}")
        for i in range(3, 0, -1):
            print(f"Starting in {i}... (switch to the ER:LC window now)")
            time.sleep(1)

    runner.start()
    if not runner.kill_switch or not runner.kill_switch.armed:
        print("Note: global F9 kill switch isn't armed (the 'keyboard' package may be "
              "missing, or needs admin on this machine) -- use Ctrl+C or the overlay's "
              "kill switch button instead.")

    overlay_shown = False
    if not args.no_overlay:
        try:
            from erlc_autopilot.live.tk_overlay import run_overlay
            overlay_shown = True
            run_overlay(runner)  # blocks until the window is closed
        except ImportError as exc:
            print(f"(No overlay window: {exc})")
        except Exception as exc:
            print(f"(Overlay window failed to start: {exc})")

    if not overlay_shown:
        _console_loop(runner)

    runner.stop()
    print("Stopped.")


if __name__ == "__main__":
    main()

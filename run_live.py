#!/usr/bin/env python3
"""Launches the real ERLC Autopilot app: a first-run setup wizard, then a
small always-on-top HUD overlay that drives actual ER:LC gameplay by
reading your screen and sending real keyboard input.

    python run_live.py

On Windows with ER:LC running, this opens the setup wizard the first time
(model tier, screen/speedometer calibration, key bindings + safety
acknowledgement), then a small floating overlay while the autopilot
drives. On any other platform (including this dev sandbox), it
automatically falls back to Sim Mode -- the exact same wizard/overlay UI
and control loop, but driving the bundled simulator instead of a real
game, so you can see and test the whole app without Windows/Roblox.

Opens in a native-feeling window via `pywebview` if installed; otherwise
falls back to just starting the local web server and printing the URL to
open in a normal browser (this is what happens in this sandbox, and is a
perfectly fine way to run it on Windows too if you'd rather not install
pywebview).
"""
from __future__ import annotations

import argparse
import os
import platform
import threading
import time

from erlc_autopilot.live.ui_server import create_app


def _run_flask(app, port: int):
    app.run(host="0.0.0.0", port=port, threaded=True, debug=False, use_reloader=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8800)))
    parser.add_argument("--sim", action="store_true",
                         help="Force Sim Mode even on Windows (drive the bundled "
                              "simulator instead of a real screen capture).")
    parser.add_argument("--no-window", action="store_true",
                         help="Skip the pywebview native window and just run the "
                              "local server (open the printed URL in your browser).")
    args = parser.parse_args()

    force_sim = args.sim or platform.system() != "Windows"
    app = create_app(force_sim=force_sim)

    server_thread = threading.Thread(target=_run_flask, args=(app, args.port), daemon=True)
    server_thread.start()
    time.sleep(0.6)  # let Flask bind before we try to load it in a window

    url = f"http://127.0.0.1:{args.port}/"
    if force_sim:
        print(f"[ERLC Autopilot] Sim Mode (no Windows/ER:LC detected) -- {url}")
    else:
        print(f"[ERLC Autopilot] Live Mode -- {url}")

    if not args.no_window:
        try:
            import webview  # pywebview

            window = webview.create_window("ERLC Autopilot", url, width=760, height=720)
            webview.start()
            return
        except Exception as exc:
            print(f"[ERLC Autopilot] Native window unavailable ({exc}); "
                  f"open {url} in your browser instead.")

    print(f"[ERLC Autopilot] Open {url} in your browser. Press Ctrl+C to quit.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

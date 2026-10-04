#!/usr/bin/env python3
"""Launches the live autopilot dashboard (simulator + web UI).

    python run_dashboard.py

Then open http://localhost:8000 (or whatever PORT you set). This runs the
whole perception -> control -> actuation loop against the bundled
simulator so you can see the autopilot drive, tune it, and stress-test its
safety behaviors (pedestrians, cut-ins, red lights) without ER:LC/Roblox
installed at all.
"""
from erlc_autopilot.dashboard.server import main

if __name__ == "__main__":
    main()

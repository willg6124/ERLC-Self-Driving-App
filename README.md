# ERLC Autopilot

A self-driving bot for **Emergency Response: Liberty County (ER:LC)** on
Roblox. It watches your screen while you play, figures out the lane,
traffic, lights, and emergency vehicles around you with computer vision,
and drives for you by sending real keyboard input — the same way Tesla's
Autopilot (camera in, control policy, actuator out) works, just aimed at
a game instead of a real car.

It ships with two things:

1. **The real app** (`run_live.py`) — a zero-config console app (plus an
   optional small always-on-top status window, *not* a browser) that
   captures your actual screen and drives actual ER:LC gameplay on your
   Windows PC. No web pages, no embedded browser — just run it.
2. **A bundled simulator + dashboard** (`run_dashboard.py`) — a
   procedural fake ER:LC used as a test harness, so the whole perception
   and driving-policy stack can be built, tuned, and safety-tested without
   Roblox installed at all. Every scenario button on it (pedestrian
   crossing, car cut-in, forced red light, emergency vehicles) exercises
   the exact same code that drives the real game.

## Quick start

```
pip install -r requirements.txt
```

**Try it without ER:LC first** (works on any OS):

```
python run_dashboard.py          # in-browser simulator + live tuning dashboard (dev tool only)
python run_live.py --sim         # the *real* app, driving the bundled simulator instead of ER:LC
```

**Drive real ER:LC** (Windows only):

```
python run_live.py
```

That's it — no setup wizard, no web page to click through. On first run:

1. Launch Roblox, join ER:LC, get into a car.
2. Run `python run_live.py`. It auto-detects the Roblox window (falls back
   to your whole primary monitor if it can't find one — still fine if
   Roblox is fullscreen/maximized), picks the **Standard** model tier by
   default, counts down 3 seconds so you can switch back to the game, and
   starts driving. Status prints live in the console: speed, target
   speed, status, engagement, turn signal, any emergency-vehicle alert.
3. If `tkinter`/`Pillow` are available (they ship with the standard
   python.org Windows installer, so usually nothing extra to install),
   a small always-on-top native window also appears with a mini camera
   preview and Engage/Disengage/Honk/Hazards buttons. This is a plain
   Tkinter window, not a browser — if it can't load for any reason the
   app automatically keeps running with console-only output instead.
4. Press **F9** any time to instantly stop (kill switch) — or Ctrl+C in
   the console, or the overlay's kill switch button.

Useful flags: `--tier lite|standard|pro`, `--recalibrate` (re-detect the
window instead of reusing the saved region), `--region L,T,W,H` (set the
capture region manually), `--no-overlay` (console only). The detected
capture region is remembered in `~/.erlc_autopilot/calibration.json` so
auto-detection only has to run once.

## Controls

The bot drives with the exact same keys you would — it's not a mod, it's
just pressing keys:

| Key(s)  | Action |
|---------|--------|
| `W A S D` | Drive (throttle / steer left / brake / steer right) |
| `P`     | Parking brake |
| `Q`     | Left turn signal |
| `E`     | Right turn signal |
| `G`     | Hazards |
| `H`     | Horn |
| `F9`    | **Kill switch** — instantly disengages and releases every key, from anywhere, even if ER:LC has focus |

## Safety — read this

This drives your keyboard for you. It will make mistakes. Keep a hand
near the keyboard and your eyes on the screen; it is driver assistance,
not a replacement for supervision. Hit **F9**, Ctrl+C, or the overlay's
**Disengage** button any time something looks wrong. It also
auto-disengages on its own if it loses the lane for about a second
(`disengage_after_lost_frames` in config) rather than continuing to
guess blind. Using automation/macros in an online game may be against
that game's or platform's rules — that's between you and whoever's rules
those are; this project doesn't check for you.

## Model tiers

| Tier | What's different |
|------|-------------------|
| **Lite** | Forces the lightweight heuristic detector (no neural net) and widens every safety margin to compensate — lowest CPU use, best for weaker PCs. |
| **Standard** | The default. Real MobileNet-SSD object detection if the weight files are present (see `models/README.md`), heuristic fallback otherwise. |
| **Pro** | Tighter follow distance, snappier steering gains, more lookahead weight for a more assertive drive once you trust it. Also the only tier that blends in a model trained on *your own* recorded driving, if you've trained one (see "Training your own driving" below) — purely rule-based otherwise. |

## How it actually works

```
screen capture  ─┐                                   ┌─ keyboard (W A S D P Q E G H)
  (mss)          │                                    │  pydirectinput, Windows
                 ▼                                    │
          AutopilotPipeline                            │
   lane detection ─┬─ object detection ─┬─ traffic     │
   (CV, HSV/edges) │  (heuristic or     │  light + EV   │
                    │   MobileNet-SSD)   │  detection    │
                    └─────────┬──────────┴──────┬────────┘
                               ▼                 ▼
                          DrivingPolicy (PID steering + speed,
                          adaptive cruise, red-light compliance,
                          pedestrian e-brake, "Move Over Law" EV yield)
                               │
                               ▼
                       Command(steer, throttle, brake, turn_signal, ...)
```

- **Lane keeping**: classical CV (HSV threshold + edge/contour fit for lane
  lines), PID steering control blended with a second, farther-ahead
  "lookahead" offset taken near the top of the detected lane lines
  (pure-pursuit-style) — anticipates where the road is actually headed
  instead of only correcting for where it already drifted, which cuts
  down on oscillation/curve-cutting versus near-offset-only PID.
- **Object detection**: a dependency-light HSV/contour heuristic by
  default; drops in a real MobileNet-SSD (`cv2.dnn`) automatically if you
  add weight files under `models/` (see `models/README.md`).
- **Adaptive cruise**: reacts to *time-to-collision*, not just raw gap —
  it estimates how fast the gap to the car ahead is actually shrinking
  frame-to-frame and starts easing off early if that implies contact soon,
  the same anticipation a real radar-based ACC gets for free, catching a
  sudden cut-in or hard-braking lead car earlier than distance alone would.
- **Traffic lights**: color-region detection + distance estimation from
  apparent size, with short-term memory so a real stop commitment doesn't
  evaporate just because the signal scrolls out of frame on final approach.
- **Emergency vehicles**: scans the roof region of every detected object
  for a flashing red+blue light-bar signature (confirmed over a few
  frames, not a single lucky one) to catch police/fire/EMS regardless of
  how the coarse shape heuristic classified the box. On a confirmed
  sighting the car leans away and slows, proportional to how close it is —
  and once it's genuinely close (actually overtaking, or already reached
  one parked ahead) it pulls fully onto the shoulder lean and comes to a
  complete stop, not just a slow crawl past it, until the vehicle's clearly
  moved away again — literally "pull over for police, move over for
  fire/EMS," not just slow down near them.
- **Imitation learning (optional, Pro tier)**: `run_live.py --record` logs
  perception features alongside the WASD keys you actually press while you
  drive manually; `train_model.py` fits a small scikit-learn model on top
  of that; if a trained model exists, the Pro tier blends its steering
  suggestion in on top of the rule-based PID/lookahead output (see
  "Training your own driving" below). Throttle/brake and every safety
  behavior (e-braking, red lights, EV yield/stop, disengage) stay 100%
  rule-based regardless — a learned model can only ever nudge *how* it
  steers, never override *why* it slows or stops.
- **Speed**: the simulator can hand the pipeline its own ground truth;
  real gameplay can't, so `erlc_autopilot/perception/speed_reader.py` OCRs
  the in-game speedometer (needs the Tesseract binary installed — see
  `requirements.txt`), falling back to a rough throttle/brake-integrated
  estimate if OCR isn't available.
- **Actuation**: WASD are digital keys in Roblox, but the control policy
  wants continuous values — `KeyboardActuator` runs an independent
  50 Hz software-PWM loop converting `throttle`/`brake`/`steer` into
  held-down-a-fraction-of-the-time key presses, decoupled from the (slower)
  ~20 Hz perception rate so control latency stays low.

Frame source and actuator are both swappable behind tiny interfaces
(`erlc_autopilot/live/capture.py`, `erlc_autopilot/live/input_driver.py`)
— the *entire* rest of the stack (perception, policy, UI) is identical
whether it's driving the bundled simulator or a real ER:LC screen capture.
That's what lets the dashboard safety-test the exact code that ends up
touching your keyboard.

## Tuning

Every gain/threshold lives in `erlc_autopilot/config.py`
(`AutopilotConfig`) and can be hot-patched live from the dashboard's
"Tuning" panel, or by editing the dataclass defaults directly.

## Training your own driving (imitation learning)

```
python run_live.py --record recordings/session1.csv   # drive manually, it just watches + logs
python train_model.py                                   # trains on every recordings/*.csv
python run_live.py --tier pro                            # now blended into the Pro tier
```

1. `--record` runs the exact same perception stack (lane/object/traffic
   light/EV detection) as normal driving, but sends no input at all —
   you drive, it logs a feature snapshot (lane offset/curvature, nearest
   vehicle/pedestrian distance, light state, EV activity, speed) paired
   with whatever WASD keys you're actually holding, once per tick, to a
   CSV. Drive a variety of roads/traffic for a few minutes for anything
   useful to come out of it; Ctrl+C when done.
2. `train_model.py` loads every `recordings/*.csv` (or `--data` a specific
   glob/file) and fits a small `RandomForestRegressor` mapping those
   features to (steer, throttle, brake), reporting held-out accuracy, and
   saves it to `models/imitation_model.joblib`.
3. Only the **Pro** tier ever loads and uses it (`AutopilotConfig.imitation
   _blend_weight`, default 0.35) — and even then, only to nudge the
   rule-based *steering* output partway toward what the model learned you'd
   do. Throttle/brake and every safety behavior (e-braking, red-light
   compliance, EV yielding/stopping, disengage) stay entirely rule-based
   and are applied after the blend, exactly as before — a learned model can
   change how assertively/smoothly it steers, never whether it stops for
   something that matters.

This whole pipeline (recording → training → blended inference) is
exercised in this repo's own tests by using the rule-based policy itself
as a stand-in "demonstrator" (since there's no real human/keyboard in a
CI sandbox) — the actual `--record` capture step reads your literal
keyboard state via the same `keyboard` package the F9 kill switch uses, so
it's Windows-specific and, like a few other OS-level pieces of this
project, not independently verified by me outside that stand-in; report
back if it behaves unexpectedly on your machine.

## Development / testing without Roblox

```
python run_dashboard.py     # tune the CV/control stack against the simulator
python run_live.py --sim    # run the real app end-to-end, driving the bundled simulator
```

Both run entirely offline with no game required, which is how this whole
project is built and regression-tested in CI/sandboxes that can't run
Windows or Roblox.

## Known limitations

- The screen-capture, real keyboard injection, window auto-detection, and
  speedometer OCR paths are Windows-specific and can only be
  code-reviewed/exercised via Sim Mode outside of an actual Windows + ER:LC
  environment — if something doesn't register in-game, the most likely
  culprits are capture region calibration or the input backend
  (`pydirectinput` vs `pynput`; try switching `input_backend` in
  `~/.erlc_autopilot/calibration.json`).
- Speed OCR requires installing the Tesseract binary separately; without
  it, speed is a rough estimate, not a measurement.
- `--record`'s keyboard-state capture (what labels the imitation-learning
  dataset) is Windows-specific for the same reason the F9 kill switch is;
  it's checked up front and fails with a clear message instead of
  silently recording garbage if it's not available, but hasn't been run
  against a real human/keyboard by the author (see "Training your own
  driving" above for how the rest of that pipeline was verified instead).
- The optional Tkinter status window has not been exercised on a real
  Windows machine by the author (this project is built/tested in a Linux
  sandbox where `tkinter` isn't installable) — it's written carefully and
  kept strictly optional specifically for that reason: if it ever fails
  to open or crashes, `run_live.py` catches it and keeps driving with
  plain console output, so the autopilot itself is never blocked by it.
  Pass `--no-overlay` to skip it outright.

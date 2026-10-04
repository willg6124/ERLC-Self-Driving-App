# ERLC Autopilot

A self-driving bot for **Emergency Response: Liberty County (ER:LC)** on
Roblox. It watches your screen while you play, figures out the lane,
traffic, lights, and emergency vehicles around you with computer vision,
and drives for you by sending real keyboard input — the same way Tesla's
Autopilot (camera in, control policy, actuator out) works, just aimed at
a game instead of a real car.

It ships with two things:

1. **The real app** (`run_live.py`) — a setup wizard + small HUD overlay
   that captures your actual screen and drives actual ER:LC gameplay on
   your Windows PC.
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
python run_dashboard.py          # in-browser simulator + live tuning dashboard
python run_live.py --sim         # the *real* app's wizard/overlay UI, driving the simulator
```

**Drive real ER:LC** (Windows only):

```
python run_live.py
```

1. Launch Roblox, join ER:LC, get into a car.
2. Run `python run_live.py`. It opens the setup wizard (a normal browser
   tab unless you've installed `pywebview`, in which case it's its own
   small window).
3. Pick a **model tier** (Lite / Standard / Pro — see below).
4. Click **Detect Automatically** to find the Roblox window, or enter the
   capture region manually. Click Preview to confirm you're actually
   seeing the game.
5. Line the calibration box up over your in-game speed readout (bottom
   right by default) — this is read via OCR each frame since there's no
   other way for the app to know your real speed.
6. Read the key bindings and safety disclaimer, check the box, hit
   **Launch Autopilot**.
7. The overlay appears: mini camera preview, speed/target speed, status,
   Disengage/Honk/Hazards buttons. The car starts driving itself.

Only runs the wizard once — your calibration is saved to
`~/.erlc_autopilot/calibration.json`. Click the ⚙ in the overlay to redo
it (e.g. after changing resolution).

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
not a replacement for supervision. Hit **F9** or the overlay's
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
| **Pro** | Tighter follow distance, snappier steering gains for a more assertive drive once you trust it. Also the tier a future model trained on *your own* recorded driving would plug into. |

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
  lines), PID steering control.
- **Object detection**: a dependency-light HSV/contour heuristic by
  default; drops in a real MobileNet-SSD (`cv2.dnn`) automatically if you
  add weight files under `models/` (see `models/README.md`).
- **Traffic lights**: color-region detection + distance estimation from
  apparent size, with short-term memory so a real stop commitment doesn't
  evaporate just because the signal scrolls out of frame on final approach.
- **Emergency vehicles**: scans the roof region of every detected object
  for a flashing red+blue light-bar signature (confirmed over a few
  frames, not a single lucky one) to catch police/fire/EMS regardless of
  how the coarse shape heuristic classified the box. On a confirmed
  sighting, the car slows to a crawl and leans away from whichever side
  the lights are on — "pull over for police, move over for fire/EMS."
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

Not built yet. The planned approach: record your own ER:LC driving
(frame + the keys you actually pressed) via the same capture pipeline,
then train a small scikit-learn model mapping perception features (lane
offset/curvature, nearby object distances) to steering/throttle, and blend
it in at the **Pro** tier on top of the existing rule-based policy rather
than replacing it outright (so the safety behaviors — e-braking,
red-light compliance, EV yielding — stay rule-guaranteed regardless of
what the learned model does).

## Development / testing without Roblox

```
python run_dashboard.py     # tune the CV/control stack against the simulator
python run_live.py --sim    # test the real wizard/overlay app end-to-end against the simulator
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
- The wizard/overlay UI is an in-house redesign in the same visual
  language as the bundled dashboard, not a pixel-for-pixel clone of any
  particular reference video/screenshot.

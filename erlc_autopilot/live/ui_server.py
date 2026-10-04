"""The real app's UI: a first-run setup wizard (model tier, screen
calibration, key bindings + safety acknowledgement) and a small always-on
HUD overlay, served as local Flask pages so they're easy to style/iterate
on and can be hosted either in a normal browser tab (dev/sandbox) or a
native-feeling `pywebview` window (see run_live.py) on the user's PC.

This intentionally reuses the dashboard's dark/neon visual language so the
whole project feels like one product, not two unrelated tools -- the
dashboard is the in-sandbox *test harness*, this is the real app that
drives actual ER:LC gameplay.
"""
from __future__ import annotations

import base64
import os
import platform
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

from ..config import AutopilotConfig
from ..perception.speed_reader import SpeedReader, SpeedReaderConfig
from .calibration import Calibration
from .capture import PIPELINE_FRAME_SIZE, ScreenCaptureSource, SimFrameSource
from .input_driver import KeyboardActuator
from .runner import LiveRunner
from .window_capture import WindowRegion, find_roblox_window, primary_screen_size

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

IS_WINDOWS = platform.system() == "Windows"


class AppState:
    """Everything that outlives a single request: the loaded calibration,
    the live runner once launched, and (sandbox-only) a shared simulator
    world so the wizard's "preview" step has something to show without a
    real screen to capture."""

    def __init__(self):
        self.calibration = Calibration.load()
        self.runner: LiveRunner = None
        self.lock = threading.Lock()
        self._sim_world = None  # lazily created, sim-mode preview only

    def sim_world(self):
        if self._sim_world is None:
            from ..sim.world import World
            self._sim_world = World()
        return self._sim_world


def create_app(force_sim: bool = not IS_WINDOWS) -> Flask:
    app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
    state = AppState()
    app.config["STATE"] = state
    app.config["FORCE_SIM"] = force_sim

    @app.after_request
    def add_headers(resp):
        resp.headers.pop("X-Frame-Options", None)
        return resp

    @app.route("/")
    def index():
        if state.runner is not None:
            return render_template("redirect.html", target="/overlay")
        if state.calibration.calibrated:
            return render_template("redirect.html", target="/overlay")
        return render_template("redirect.html", target="/wizard")

    @app.route("/wizard")
    def wizard():
        return render_template(
            "wizard.html",
            calibration=state.calibration,
            is_windows=IS_WINDOWS,
            force_sim=force_sim,
        )

    @app.route("/api/setup/detect_window", methods=["POST"])
    def detect_window():
        if force_sim:
            return jsonify({"ok": True, "region": {"left": 0, "top": 0, "width": 1920, "height": 1080},
                             "note": "Sim mode: using a placeholder full-screen region. "
                                     "On Windows this detects your real Roblox window."})
        region = find_roblox_window()
        if region is None:
            region = primary_screen_size()
            if region is None:
                return jsonify({"ok": False, "error": "Could not detect a window or screen. "
                                                        "Enter the capture region manually."})
            return jsonify({"ok": True, "region": region.to_dict(),
                             "note": "Couldn't find a 'Roblox' window -- using your primary monitor instead. "
                                     "Make sure ER:LC is running, or adjust the region manually."})
        return jsonify({"ok": True, "region": region.to_dict(), "note": "Found the Roblox window."})

    @app.route("/api/setup/preview", methods=["POST"])
    def preview():
        body = request.get_json(silent=True) or {}
        try:
            if force_sim or not IS_WINDOWS:
                from ..sim.renderer import render
                frame, _meta = render(state.sim_world())
                state.sim_world().update(1 / 20, 0.0, 0.3, 0.0)
                note = "Sim-mode preview (bundled simulator, not a real screen grab)."
            else:
                import mss
                region = WindowRegion(
                    int(body.get("left", 0)), int(body.get("top", 0)),
                    int(body.get("width", 1920)), int(body.get("height", 1080)))
                with mss.mss() as sct:
                    shot = sct.grab(region.to_mss_region())
                    frame = np.array(shot)[:, :, :3]
                note = "Live screen grab of the configured region."
            frame = cv2.resize(frame, PIPELINE_FRAME_SIZE, interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ok:
                return jsonify({"ok": False, "error": "encode failed"})
            b64 = base64.b64encode(buf.tobytes()).decode("ascii")
            return jsonify({"ok": True, "image_b64": b64, "note": note})
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)})

    @app.route("/api/setup/complete", methods=["POST"])
    def complete():
        body = request.get_json(silent=True) or {}
        if not body.get("accepted_disclaimer"):
            return jsonify({"ok": False, "error": "You must acknowledge the safety disclaimer."}), 400

        cal = state.calibration
        cal.capture_left = int(body.get("left", 0))
        cal.capture_top = int(body.get("top", 0))
        cal.capture_width = int(body.get("width", 1920))
        cal.capture_height = int(body.get("height", 1080))
        cal.speed_roi = [float(x) for x in body.get("speed_roi", cal.speed_roi)]
        cal.model_tier = body.get("model_tier", "standard")
        cal.input_backend = body.get("input_backend", "auto")
        cal.calibrated = True
        cal.save()

        _launch_runner(app)
        return jsonify({"ok": True, "redirect": "/overlay"})

    @app.route("/overlay")
    def overlay():
        if state.runner is None and state.calibration.calibrated:
            _launch_runner(app)
        return render_template("overlay.html")

    def _mjpeg_generator():
        boundary = b"--frame"
        placeholder = _placeholder_jpeg()
        last_sent = 0.0
        while True:
            now = time.time()
            if now - last_sent < 1 / 20:
                time.sleep(1 / 20 - (now - last_sent))
            last_sent = time.time()
            frame = state.runner.get_frame() if state.runner is not None else b""
            if not frame:
                frame = placeholder
                time.sleep(0.05)
            yield (boundary + b"\r\nContent-Type: image/jpeg\r\nContent-Length: " +
                   str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n")

    @app.route("/video_feed")
    def video_feed():
        return Response(_mjpeg_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/api/telemetry")
    def telemetry():
        if state.runner is None:
            return jsonify({"status": "not_started"})
        t = state.runner.get_telemetry()
        t["calibrated"] = state.calibration.calibrated
        t["kill_switch_armed"] = bool(state.runner.kill_switch and state.runner.kill_switch.armed)
        return jsonify(t)

    @app.route("/api/control", methods=["POST"])
    def control():
        if state.runner is None:
            return jsonify({"ok": False, "error": "not started"}), 400
        action = (request.get_json(silent=True) or {}).get("action")
        if action == "toggle":
            state.runner.toggle_engaged()
        elif action == "disengage":
            state.runner.set_engaged(False)
        elif action == "engage":
            state.runner.set_engaged(True)
        elif action == "emergency_stop":
            state.runner.emergency_stop()
        elif action == "honk":
            state.runner.honk()
        elif action == "hazards_on":
            state.runner.set_hazards(True)
        elif action == "hazards_off":
            state.runner.set_hazards(False)
        else:
            return jsonify({"ok": False, "error": "unknown action"}), 400
        return jsonify({"ok": True})

    @app.route("/api/recalibrate", methods=["POST"])
    def recalibrate():
        if state.runner is not None:
            state.runner.stop()
            state.runner = None
        state.calibration.calibrated = False
        state.calibration.save()
        return jsonify({"ok": True, "redirect": "/wizard"})

    return app


def _launch_runner(app: Flask) -> None:
    state: AppState = app.config["STATE"]
    force_sim: bool = app.config["FORCE_SIM"]
    cal = state.calibration
    config = AutopilotConfig.for_tier(cal.model_tier)

    if force_sim or not IS_WINDOWS:
        frame_source = SimFrameSource(world=state.sim_world())
        actuator = KeyboardActuator(backend="dry_run")
        speed_reader = None
        ground_truth = lambda: frame_source.world.ego.speed  # sim mode only: the
        # one place this app is allowed to cheat and read real ground truth,
        # since there's no real speedometer to OCR off a fake screen
    else:
        region = WindowRegion(cal.capture_left, cal.capture_top, cal.capture_width, cal.capture_height)
        frame_source = ScreenCaptureSource(region)
        actuator = KeyboardActuator(backend=cal.input_backend)
        speed_reader = SpeedReader(SpeedReaderConfig(roi=cal.speed_roi))
        ground_truth = None

    runner = LiveRunner(frame_source, actuator, config=config,
                         speed_reader=speed_reader, ground_truth_speed=ground_truth)
    runner.start()
    state.runner = runner


_PLACEHOLDER_CACHE = None


def _placeholder_jpeg() -> bytes:
    global _PLACEHOLDER_CACHE
    if _PLACEHOLDER_CACHE is None:
        frame = np.zeros((540, 960, 3), dtype=np.uint8)
        frame[:] = (24, 20, 18)
        cv2.putText(frame, "waiting for autopilot...", (300, 270),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (120, 120, 120), 2, cv2.LINE_AA)
        ok, buf = cv2.imencode(".jpg", frame)
        _PLACEHOLDER_CACHE = buf.tobytes() if ok else b""
    return _PLACEHOLDER_CACHE


def main():
    app = create_app()
    port = int(os.environ.get("PORT", 8800))
    app.run(host="0.0.0.0", port=port, threaded=True)


if __name__ == "__main__":
    main()

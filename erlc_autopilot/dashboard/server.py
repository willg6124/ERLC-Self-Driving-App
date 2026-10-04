"""Flask app exposing the live autopilot dashboard: an MJPEG video feed of
the camera+CV overlay, a polled telemetry JSON endpoint, and simple control
endpoints (engage/disengage, reset, scenario triggers, live config tuning).
No websockets/eventlet needed -- keeps the dependency list small and the
whole thing trivially easy to run anywhere."""
from __future__ import annotations

import os
import time

from flask import Flask, Response, jsonify, render_template, request

from .engine import SimulationEngine

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


def create_app() -> Flask:
    app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
    engine = SimulationEngine()
    engine.start()
    app.config["ENGINE"] = engine

    @app.after_request
    def add_headers(resp):
        # keep this friendly to being embedded in an iframe preview
        resp.headers.pop("X-Frame-Options", None)
        return resp

    @app.route("/")
    def index():
        return render_template("index.html")

    def _mjpeg_generator():
        boundary = b"--frame"
        last_sent = 0.0
        while True:
            now = time.time()
            if now - last_sent < 1 / 20:
                time.sleep(1 / 20 - (now - last_sent))
            last_sent = time.time()
            frame = engine.get_frame()
            if not frame:
                time.sleep(0.05)
                continue
            yield (boundary + b"\r\nContent-Type: image/jpeg\r\nContent-Length: " +
                   str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n")

    @app.route("/video_feed")
    def video_feed():
        return Response(_mjpeg_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/api/telemetry")
    def telemetry():
        return jsonify(engine.get_telemetry())

    @app.route("/api/control", methods=["POST"])
    def control():
        action = (request.get_json(silent=True) or {}).get("action")
        if action == "toggle":
            engine.toggle_engaged()
        elif action == "reset":
            engine.reset()
        elif action == "pedestrian":
            engine.trigger_pedestrian()
        elif action == "cutin":
            engine.trigger_cutin()
        elif action == "red_light":
            engine.force_red_light()
        elif action == "ev_police_parked":
            engine.trigger_emergency_vehicle(kind="police", mode="parked")
        elif action == "ev_fire_overtaking":
            engine.trigger_emergency_vehicle(kind="fire", mode="overtaking")
        elif action == "ev_ems_overtaking":
            engine.trigger_emergency_vehicle(kind="ems", mode="overtaking")
        else:
            return jsonify({"ok": False, "error": "unknown action"}), 400
        return jsonify({"ok": True})

    @app.route("/api/config", methods=["GET", "POST"])
    def config():
        if request.method == "POST":
            patch = request.get_json(silent=True) or {}
            engine.update_config(patch)
        return jsonify(engine.config.to_dict())

    return app


def main():
    app = create_app()
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, threaded=True)


if __name__ == "__main__":
    main()

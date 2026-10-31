"""Small Flask API and web server for the existing traffic simulation."""
import json
import logging
import math
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from flask import Flask, jsonify, make_response, request

from build_dashboard import ROOT, TEMPLATE_PATH, build_payload
from src.config import Config
from src.simulate import Simulation

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024
logging.basicConfig(level=logging.INFO)

_lock = threading.RLock()
_payload = None
_job = {
    "id": None,
    "status": "idle",
    "phase": "Ready to run",
    "error": None,
    "updated_at": None,
    "settings": None,
}
_DATA_PATTERN = re.compile(r"const DATA = (\{.*?\});\s*if \(!DATA", re.DOTALL)
_DEFAULTS = {
    "delta_send": 2.0,
    "sensor_noise_std": 1.5,
    "dropout_percent": 0.3,
    "congestion_v": 42.25,
}


def _initial_payload():
    """Reuse the checked-in dashboard snapshot for a fast first page load."""
    dashboard_path = ROOT / "dashboard.html"
    if dashboard_path.exists():
        match = _DATA_PATTERN.search(dashboard_path.read_text(encoding="utf-8"))
        if match:
            payload = json.loads(match.group(1))
            if payload.get("results"):
                return payload

    results_path = ROOT / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(
            "The dashboard snapshot is missing. Run `python run.py` and "
            "`python build_dashboard.py` once to create it."
        )
    results = json.loads(results_path.read_text(encoding="utf-8"))
    return build_payload(Config(), results)


def _template_response(payload):
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
    # Keep user-controlled or dataset text from ending the inline script.
    encoded = encoded.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    html = template.replace("/*%%PAYLOAD%%*/null", encoded)
    response = make_response(html)
    response.headers["Content-Type"] = "text/html; charset=utf-8"
    response.headers["Cache-Control"] = "no-store"
    return response


def _validate_settings(body):
    if not isinstance(body, dict):
        raise ValueError("Send simulation settings as a JSON object.")

    bounds = {
        "delta_send": (0.5, 8.0, "mph"),
        "sensor_noise_std": (0.0, 5.0, "mph"),
        "dropout_percent": (0.0, 2.0, "%"),
        "congestion_v": (30.0, 55.0, "mph"),
    }
    values = {}
    for name, (minimum, maximum, unit) in bounds.items():
        value = body.get(name, _DEFAULTS[name])
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name.replace('_', ' ').title()} must be a number.")
        value = float(value)
        if not math.isfinite(value) or not minimum <= value <= maximum:
            raise ValueError(
                f"{name.replace('_', ' ').title()} must be between "
                f"{minimum:g} and {maximum:g} {unit}."
            )
        values[name] = value
    return values


def _run_simulation(job_id, settings):
    global _payload
    try:
        cfg = Config()
        cfg.delta_send = settings["delta_send"]
        cfg.sensor_noise_std = settings["sensor_noise_std"]
        cfg.dropout_prob = settings["dropout_percent"] / 100.0
        cfg.congestion_v = settings["congestion_v"]

        logging.info("Starting traffic simulation job %s", job_id)
        simulation = Simulation(cfg)
        results = simulation.run_all()
        results["alarms_sample"] = getattr(simulation, "_alarms", [])

        with _lock:
            if _job["id"] == job_id:
                _job["phase"] = "Preparing forecast and sensor views"
        payload = build_payload(cfg, results)

        with _lock:
            if _job["id"] == job_id:
                _payload = payload
                _job.update({
                    "status": "completed",
                    "phase": "Results are ready",
                    "error": None,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "settings": settings,
                })
        logging.info("Completed traffic simulation job %s", job_id)
    except Exception as exc:
        logging.exception("Traffic simulation job %s failed", job_id)
        with _lock:
            if _job["id"] == job_id:
                _job.update({
                    "status": "failed",
                    "phase": "Simulation failed",
                    "error": str(exc)[:400] or "See the server log for details.",
                })


@app.route("/", methods=["GET"])
def dashboard():
    global _payload
    try:
        if _payload is None:
            with _lock:
                if _payload is None:
                    _payload = _initial_payload()
        return _template_response(_payload)
    except Exception as exc:
        logging.exception("Unable to load the traffic dashboard")
        return make_response(
            "Dashboard data could not be loaded. Check the server log and confirm "
            f"the project data files are present. Details: {exc}",
            503,
        )


@app.route("/api/simulation", methods=["POST"])
def start_simulation():
    try:
        settings = _validate_settings(request.get_json(silent=True))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    with _lock:
        if _job["status"] == "running":
            return jsonify({"error": "A simulation is already running."}), 409
        job_id = uuid.uuid4().hex
        _job.update({
            "id": job_id,
            "status": "running",
            "phase": "Running the edge, fog, and cloud simulation",
            "error": None,
            "updated_at": None,
            "settings": settings,
        })

    worker = threading.Thread(
        target=_run_simulation, args=(job_id, settings), daemon=True,
        name=f"traffic-simulation-{job_id[:8]}",
    )
    worker.start()
    return jsonify({"id": job_id, "status": "running"}), 202


@app.route("/api/simulation/status", methods=["GET"])
def simulation_status():
    with _lock:
        return jsonify(dict(_job))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), threaded=True)

"""Build a self-contained dashboard.html from results.json and real data."""
import base64
import json
from pathlib import Path

import numpy as np

from src.config import Config
from src.data import (load_dataset, build_graph, partition_zones, split,
                      time_of_day_profile)
from src.models import build_training_matrix, make_model
from src.simulate import make_raw_stream

ROOT = Path(__file__).resolve().parent
TEMPLATE_PATH = ROOT / "dashboard_template.html"


def build_payload(cfg, res):
    """Build the dashboard data from a config and simulation result object."""
    X, A, ids = load_dataset(cfg)
    A_sym, A_hat = build_graph(A, cfg)
    zones = partition_zones(A_sym, cfg)
    X_train, _, n_train = split(X, cfg)
    prof = time_of_day_profile(X_train, cfg)

    # The graph view is schematic: adjacency is real, but METR-LA coordinates
    # are not included in the project data.
    order = np.argsort(zones, kind="stable")
    M = X[n_train:, order].T                       # (detectors, time)
    M8 = np.clip(np.rint(M * 2), 0, 180).astype(np.uint8)
    original_to_ordered = np.empty(len(order), dtype=int)
    original_to_ordered[order] = np.arange(len(order))
    edges = [
        [int(original_to_ordered[i]), int(original_to_ordered[j]),
         float(A_sym[i, j])]
        for i in range(len(ids))
        for j in range(i + 1, len(ids))
        if A_sym[i, j] > 0
    ]

    # Generate per-detector forecasts from the same training split and model
    # configuration as the simulation. Values are rounded to 1 mph for a
    # compact self-contained HTML payload.
    forecasts = {}
    for horizon in cfg.horizons:
        model = make_model(cfg, horizon).fit(X_train, A_hat, prof)
        F, _ = build_training_matrix(X, A_hat, prof, cfg, horizon)
        n_rows = X.shape[0] - horizon - (cfg.lags - 1)
        F = F.reshape(n_rows, len(ids), -1)
        F = F[n_train - (cfg.lags - 1):]
        prediction = model.predict(F.reshape(-1, F.shape[-1])).reshape(
            len(X) - n_train - horizon, len(ids)
        )
        encoded = np.clip(np.rint(prediction[:, order].T * 2), 0, 180).astype(np.uint8)
        forecasts[str(horizon * cfg.step_minutes)] = {
            "rows": int(prediction.shape[0]),
            "values": base64.b64encode(encoded.tobytes()).decode("ascii"),
        }

    raw, missing = make_raw_stream(X, cfg, np.random.default_rng(cfg.seed))
    test_raw = raw[n_train:]
    test_missing = missing[n_train:]
    test_truth = X[n_train:]
    candidate_spikes = np.abs(test_raw - test_truth) >= cfg.spike_mag * 0.5
    health = []
    for i in order:
        misses = int(test_missing[:, i].sum())
        spikes = int(candidate_spikes[:, i].sum())
        stuck_episodes = 0
        repeated = 0
        previous = None
        for t, value in enumerate(test_raw[:, i]):
            if test_missing[t, i]:
                previous = None
                repeated = 0
                continue
            if previous is not None and value == previous:
                repeated += 1
            else:
                repeated = 0
            if repeated == cfg.stuck_limit:
                stuck_episodes += 1
            previous = value
        rate = misses / max(len(test_missing), 1)
        health.append({
            "missing": misses,
            "candidate_spikes": spikes,
            "stuck_episodes": stuck_episodes,
            "status": "attention" if rate >= 0.01 or spikes >= 3 or stuck_episodes else "nominal",
        })

    payload = {
        "rows": int(M8.shape[0]),
        "cols": int(M8.shape[1]),
        "speeds": base64.b64encode(M8.tobytes()).decode("ascii"),
        "forecasts": forecasts,
        "zones": zones[order].tolist(),
        "ids": [ids[i] for i in order],
        "originalIndices": order.tolist(),
        "edges": edges,
        "health": health,
        "latestSpeeds": X[-1, order].tolist(),
        "meanSpeedMph": float(M.mean()),
        "latestCongested": int(np.sum(X[-1] < cfg.congestion_v)),
        "startStep": int(n_train),
        "stepMin": cfg.step_minutes,
        "vFree": cfg.v_free,
        "congestionV": cfg.congestion_v,
        "horizons": [h * cfg.step_minutes for h in cfg.horizons],
        "model": cfg.model_type,
        "results": res,
    }

    return payload


def main():
    cfg = Config()
    results_path = ROOT / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(
            f"{results_path.name} is missing; run run.py before building the dashboard."
        )
    res = json.loads(results_path.read_text(encoding="utf-8"))
    payload = build_payload(cfg, res)
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    html = template.replace("/*%%PAYLOAD%%*/null", json.dumps(payload))
    output_path = ROOT / "dashboard.html"
    output_path.write_text(html, encoding="utf-8")
    print(f"dashboard.html written ({len(html) / 1024:.0f} kB), "
          f"heatmap {payload['rows']}x{payload['cols']}")


if __name__ == "__main__":
    main()

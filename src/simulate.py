"""Orchestrator.

Replays the real 5-minute detector stream through three deployment
architectures and scores them all on the same ground truth:

    A. cloud_only      every reading shipped to a central server, which
                       filters, predicts and returns the signal plan
    B. edge_only       each detector decides alone, nothing coordinates
    C. edge_fog_cloud  the proposed twin

Same data, same physics, same evaluation -- only the placement of
computation changes. Two further experiments probe the design:
a sweep of the event-trigger threshold (bandwidth vs. fidelity) and a
WAN-outage test (what survives when the backhaul drops).
"""
import json
import time

import numpy as np

from .data import (load_dataset, build_graph, partition_zones,
                   time_of_day_profile, split)
from .models import (GraphRidge, GraphGBT, make_model, HistoricalAverage,
                     Persistence, build_training_matrix, assemble, LocalTrendKF)
from .infra import MessageBus, LatencyTracker, Link
from .nodes import EdgeNode, FogNode, CloudTwin
from .traffic import evaluate_control, greenshields, webster_delay


# --------------------------------------------------------------------------
def make_raw_stream(X, cfg, rng):
    """Turn the archived aggregate into a plausible raw detector feed.

    The published dataset is already cleaned, so injecting sensor noise,
    gross outliers and dropouts is what gives the edge filtering layer
    something real to do. Every architecture sees the identical feed.
    """
    noisy = X + rng.normal(0.0, cfg.sensor_noise_std, size=X.shape)
    spikes = rng.random(X.shape) < cfg.spike_prob
    noisy[spikes] += rng.choice([-1.0, 1.0], size=int(spikes.sum())) * cfg.spike_mag
    missing = rng.random(X.shape) < cfg.dropout_prob
    return np.clip(noisy, 1.0, 90.0), missing


def make_outage_mask(T, cfg, rng, duty=0.05, mean_len=8):
    """Blocky WAN outages -- backhaul failures come in bursts, not i.i.d."""
    mask = np.zeros(T, dtype=bool)
    n_out = int(T * duty / mean_len)
    for _ in range(n_out):
        s = rng.integers(0, max(T - mean_len, 1))
        mask[s:s + mean_len] = True
    return mask


def accuracy(pred, true):
    err = pred - true
    return {"MAE": float(np.mean(np.abs(err))),
            "RMSE": float(np.sqrt(np.mean(err ** 2))),
            "MAPE": float(np.mean(np.abs(err) / np.maximum(true, 1.0)) * 100.0)}


# --------------------------------------------------------------------------
class Simulation:
    def __init__(self, cfg):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.results = {}

    # ================= setup =================
    def setup(self):
        cfg = self.cfg
        t0 = time.time()
        self.X, A, self.sensor_ids = load_dataset(cfg)
        self.A_sym, self.A_hat = build_graph(A, cfg)
        self.zones = partition_zones(self.A_sym, cfg)
        self.X_train, self.X_test, self.n_train = split(self.X, cfg)
        self.prof = time_of_day_profile(self.X_train, cfg)
        self.T, self.N = self.X.shape

        self.raw, self.missing = make_raw_stream(self.X, cfg, self.rng)
        self.outage = make_outage_mask(self.T, cfg, self.rng)

        # opposing-approach demand: the cross street we never measure
        med = np.median(self.X_train, axis=0)
        _, q_med = greenshields(med, cfg)
        self.q_opp = 0.5 * q_med

        print(f"  detectors={self.N}  timesteps={self.T}  "
              f"train={self.n_train}  test={self.T - self.n_train}")
        print(f"  road links={int((self.A_sym > 0).sum())}  "
              f"fog zones={cfg.n_zones}  sizes={np.bincount(self.zones).tolist()}")
        print(f"  setup {time.time() - t0:.1f}s")

    # ================= models =================
    def train_models(self):
        cfg = self.cfg
        t0 = time.time()
        self.models = {h: make_model(cfg, h).fit(self.X_train, self.A_hat,
                                                self.prof)
                       for h in (1, 2, 3, 6)}
        self.zone_models = {}
        for z in range(cfg.n_zones):
            mask = self.zones == z
            self.zone_models[z] = {
                1: make_model(cfg, 1).fit(self.X_train, self.A_hat,
                                          self.prof, sensor_mask=mask),
                3: make_model(cfg, 3).fit(self.X_train, self.A_hat,
                                          self.prof, sensor_mask=mask),
            }
        print(f"  trained {4 + 2 * cfg.n_zones} models in {time.time() - t0:.1f}s")

    def calibrate_fusion(self, val_len=200):
        """Pick the edge/fog blend weight on a validation slice of training.

        The edge filter is hard to beat at five minutes on its own road --
        it sees every reading at full rate. The fog model is the only thing
        that sees the neighbouring roads. Neither dominates, so the twin
        fuses them and the weight is measured, not guessed. One simulated
        pass gives both component predictions; the grid search is then
        arithmetic.
        """
        lo = self.n_train - val_len
        r = self._run_efc(lo, self.n_train, 0.5, record=False)
        ve, vf, vt = r["v_edge"], r["v_fog"], r["v_true"]
        ws = np.linspace(0.0, 1.0, 21)
        maes = [float(np.mean(np.abs(w * ve + (1 - w) * vf - vt))) for w in ws]
        k = int(np.argmin(maes))
        self.fusion_w = float(ws[k])
        print(f"  fusion weight w={self.fusion_w:.2f} "
              f"(1=edge only, 0=fog only) | val MAE: edge={maes[-1]:.3f} "
              f"fog={maes[0]:.3f} fused={maes[k]:.3f}")
        self.results["fusion"] = {
            "weight": self.fusion_w,
            "val_mae_edge_only": maes[-1],
            "val_mae_fog_only": maes[0],
            "val_mae_fused": maes[k],
        }
        return self.fusion_w

    # ================= forecast benchmark =================
    def benchmark_forecasting(self):
        """Offline benchmark on ground-truth inputs, so the numbers line up
        with how METR-LA results are normally reported."""
        cfg = self.cfg
        ha = HistoricalAverage(self.prof, cfg)
        out = {}
        lo = self.n_train
        for h in cfg.horizons:
            F, _ = build_training_matrix(self.X, self.A_hat, self.prof, cfg, h)
            S = self.T - h - (cfg.lags - 1)
            F = F.reshape(S, self.N, -1)
            keep = slice(lo - (cfg.lags - 1), S)
            F = F[keep]
            ts = np.arange(lo, lo + F.shape[0])
            y_true = self.X[ts + h]                      # absolute truth

            pred_graph = self.models[h].predict(
                F.reshape(-1, F.shape[-1])).reshape(y_true.shape)
            pred_ha = self.prof[(ts[:, None] + h) % cfg.steps_per_day,
                                np.arange(self.N)[None, :]]
            pred_pers = self.X[ts]

            ridge = GraphRidge(cfg, h).fit(self.X_train, self.A_hat, self.prof)
            pred_ridge = ridge.predict(
                F.reshape(-1, F.shape[-1])).reshape(y_true.shape)

            out[f"{h * cfg.step_minutes}min"] = {
                "HistoricalAverage": accuracy(pred_ha, y_true),
                "Persistence": accuracy(pred_pers, y_true),
                "GraphRidge (lightweight)": accuracy(pred_ridge, y_true),
                "GraphGBT (twin)": accuracy(pred_graph, y_true),
            }
        self.results["forecasting"] = out
        return out

    def benchmark_congestion(self, horizon=3):
        """Can we see congestion coming 15 minutes out?

        This is where the tiers genuinely separate. Flagging a road as
        about-to-congest is a spatial question -- the queue arrives from a
        neighbouring link -- so an isolated edge node is structurally
        handicapped no matter how good its filter is.
        """
        cfg = self.cfg
        lo = self.n_train
        ts = np.arange(lo, self.T - horizon)
        thr = cfg.congestion_v
        truth = self.X[ts + horizon] < thr

        # (a) persistence: assume the next 15 minutes look like now
        pred = {"Persistence": self.X[ts] < thr}

        # (b) an edge node alone: Kalman extrapolation of its own road
        kfs = [LocalTrendKF(cfg, self.X[lo - 1, i]) for i in range(self.N)]
        kfp = np.empty((len(ts), self.N))
        for k, t in enumerate(ts):
            for i, kf in enumerate(kfs):
                z = None if self.missing[t, i] else float(self.raw[t, i])
                kf.update(z)
                kfp[k, i] = kf.forecast(horizon)
        pred["Edge Kalman (no graph)"] = kfp < thr

        # (c) fog/cloud: the graph model
        F, _ = build_training_matrix(self.X, self.A_hat, self.prof, cfg, horizon)
        S = self.T - horizon - (cfg.lags - 1)
        F = F.reshape(S, self.N, -1)[lo - (cfg.lags - 1):]
        gp = self.models[horizon].predict(
            F.reshape(-1, F.shape[-1])).reshape(len(ts), self.N)
        pred["Fog graph model"] = gp < thr

        def prf(p, t, mask=None):
            if mask is not None:
                p, t = p[mask], t[mask]
            tp = int(np.sum(p & t)); fp = int(np.sum(p & ~t))
            fn = int(np.sum(~p & t))
            pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
            return {"precision": pr, "recall": rc,
                    "f1": 2 * pr * rc / max(pr + rc, 1e-9),
                    "true_positives": tp, "false_positives": fp, "missed": fn}

        # Overall state classification is a soft target: 15 minutes out, most
        # roads are in the same state they are in now, so "assume no change"
        # scores well and tells us nothing about the architecture.
        out = {"state_classification": {n: prf(p, truth) for n, p in pred.items()}}

        # Onset is the question that matters operationally: of the roads
        # running free right now, which ones are about to seize up? By
        # construction persistence must score zero recall here -- it can only
        # ever predict "same as now" -- which is exactly the point.
        free_now = self.X[ts] >= thr
        out["congestion_onset"] = {n: prf(p, truth, free_now)
                                   for n, p in pred.items()}

        out["_meta"] = {"horizon_min": horizon * cfg.step_minutes,
                        "threshold_mph": thr,
                        "positive_rate_pct": float(100 * truth.mean()),
                        "onset_events": int(np.sum(truth & free_now))}
        self.results["congestion_detection"] = out
        for section in ("state_classification", "congestion_onset"):
            print(f"      [{section}]")
            for k, v in out[section].items():
                print(f"        {k:<24} P={v['precision']:.3f} "
                      f"R={v['recall']:.3f} F1={v['f1']:.3f}")
        return out

    # ================= architecture A: cloud only =================
    def run_cloud_only(self, outage=None):
        cfg = self.cfg
        bus = MessageBus(cfg, np.random.default_rng(cfg.seed + 1))
        lat = LatencyTracker()
        cloud = CloudTwin(cfg, self.A_hat, self.prof, self.models[2], bus,
                          self.raw[self.n_train - 1])
        for k in range(cfg.lags):
            cloud.buf[k] = self.raw[self.n_train - 1 - k].copy()

        d_ad, d_fx, err, blind = [], [], [], 0
        for t in range(self.n_train, self.T - 1):
            down = outage is not None and outage[t]
            if not down:
                bus.send(Link.EDGE_CLOUD, cfg.bytes_reading * self.N)
                lat.add(cloud.control_latency())

            v_true = self.X[t + 1]
            _, q_true = greenshields(v_true, cfg)
            if down:
                # no data up, no plan back: the controller reverts to its
                # fixed-time fallback plan across the whole city
                d = webster_delay(q_true, cfg.lam_fixed, cfg)
                d_ad.append(d); d_fx.append(d)
                err.append(np.abs(cloud.state - v_true))
                blind += 1
            else:
                # the plan acts on last interval's raw data: upload, batch,
                # central solve and return do not fit inside 5 minutes
                v_est = cloud.forecast(t - 1, horizon=2)
                a, f, _ = evaluate_control(v_est, v_true, self.q_opp, cfg)
                d_ad.append(a); d_fx.append(f)
                err.append(np.abs(v_est - v_true))

            if not down:
                cloud.ingest_state(self.raw[t])
            cloud.commit_step()

        r = self._pack("cloud_only", d_ad, d_fx, err, bus, lat)
        r["steps_uncontrolled"] = blind
        return r

    # ================= architecture B: edge only =================
    def run_edge_only(self, outage=None):
        cfg = self.cfg
        bus = MessageBus(cfg, np.random.default_rng(cfg.seed + 2))
        lat = LatencyTracker()
        edges = [EdgeNode(i, self.sensor_ids[i], self.zones[i], cfg,
                          self.X[self.n_train - 1, i], bus)
                 for i in range(self.N)]

        d_ad, d_fx, err = [], [], []
        for t in range(self.n_train, self.T - 1):
            v_est = np.empty(self.N)
            for i, e in enumerate(edges):
                z = None if self.missing[t, i] else float(self.raw[t, i])
                e.ingest(z)
                v_est[i] = e.local_forecast()
                e.since_send += 1
                if e.since_send >= cfg.heartbeat:      # monitoring only
                    e.report(v_est[i])
            lat.add(edges[0].local_latency())

            v_true = self.X[t + 1]
            a, f, _ = evaluate_control(v_est, v_true, self.q_opp, cfg)
            d_ad.append(a); d_fx.append(f)
            err.append(np.abs(v_est - v_true))

        r = self._pack("edge_only", d_ad, d_fx, err, bus, lat)
        r["steps_uncontrolled"] = 0
        return r

    # ================= architecture C: edge + fog + cloud =================
    def _run_efc(self, t_start, t_end, w, delta=None, outage=None,
                 sync_every=3, record=True):
        cfg = self.cfg
        delta = cfg.delta_send if delta is None else delta
        bus = MessageBus(cfg, np.random.default_rng(cfg.seed + 3))
        lat = LatencyTracker()

        x0 = self.X[t_start - 1].copy()
        edges = [EdgeNode(i, self.sensor_ids[i], self.zones[i], cfg,
                          x0[i], bus) for i in range(self.N)]
        for e in edges:
            e.delta = delta
        fogs = []
        for z in range(cfg.n_zones):
            members = np.where(self.zones == z)[0]
            fogs.append(FogNode(z, members, cfg, self.A_hat, self.prof,
                                self.zone_models[z][1], self.zone_models[z][3],
                                bus, x0))
        cloud = CloudTwin(cfg, self.A_hat, self.prof, self.models[6], bus, x0)
        for k in range(cfg.lags):
            frame = self.X[t_start - 1 - k].copy()
            cloud.buf[k] = frame
            for fg in fogs:
                fg.buf[k] = frame.copy()

        d_ad, d_fx, err = [], [], []
        sync_err, alarms, duty = [], [], []
        self._ve, self._vf, self._vt = [], [], []
        flagged_steps, flag_count = 0, np.zeros(self.N, int)

        for t in range(t_start, t_end):
            down = outage is not None and outage[t]

            # ---- edge tier: filter, self-check, speak only if needed ----
            n_sent = 0
            v_edge = np.empty(self.N)
            for i, e in enumerate(edges):
                z = None if self.missing[t, i] else float(self.raw[t, i])
                level = e.ingest(z)
                v_edge[i] = e.local_forecast()
                if e.should_report(level):
                    e.report(level)
                    fogs[e.zone].apply_edge_report(i, level)
                    n_sent += 1
                if e.health != "ok":
                    flagged_steps += 1
                    flag_count[i] += 1
            duty.append(n_sent / self.N)

            # ---- fog tier: sub-twin forecast + corridor coordination ----
            v_fog = np.empty(self.N)
            for fg in fogs:
                fg.commit_step()
                v_fog[fg.members] = fg.forecast(t, horizon=1)
                if record and t % sync_every == 0:
                    alarms.extend(fg.scan_congestion(t, fg.forecast(t, horizon=3)))
                    if not down:
                        fg.push_summary()
            lat.add(fogs[0].control_latency())

            # ---- hierarchical fusion: local fidelity + network context ----
            v_est = w * v_edge + (1.0 - w) * v_fog
            self._ve.append(v_edge.copy()); self._vf.append(v_fog.copy())
            self._vt.append(self.X[t + 1].copy())

            # ---- control decision, scored against reality ----
            v_true = self.X[t + 1]
            a, f, lam = evaluate_control(v_est, v_true, self.q_opp, cfg)
            for fg in fogs:                       # zone green-time budget
                lam[fg.members] = fg.coordinate(lam[fg.members])
            d_ad.append(a); d_fx.append(f); err.append(np.abs(v_est - v_true))

            # ---- twin fidelity: replica vs reality ----
            replica = np.concatenate([fg.state[fg.members] for fg in fogs])
            truth = np.concatenate([self.X[t][fg.members] for fg in fogs])
            sync_err.append(np.sqrt(np.mean((replica - truth) ** 2)))

            # ---- cloud tier: city state + inter-zone sync (WAN) ----
            city = np.empty(self.N)
            for fg in fogs:
                city[fg.members] = fg.state[fg.members]
            if not down:
                cloud.ingest_state(city)
                cloud.commit_step()
                if t % sync_every == 0:
                    for fg in fogs:
                        fg.apply_cloud_sync(cloud.state)
                        bus.send(Link.FOG_CLOUD, cfg.bytes_zone_summary)
            else:
                cloud.commit_step()

        out = {"ctrl_mae": float(np.mean(err)),
               "v_edge": np.asarray(self._ve), "v_fog": np.asarray(self._vf),
               "v_true": np.asarray(self._vt)}
        if not record:
            return out

        r = self._pack("edge_fog_cloud", d_ad, d_fx, err, bus, lat)
        r.update({
            "twin_sync_rmse_mph": float(np.mean(sync_err)),
            "uplink_duty_cycle_pct": float(100.0 * np.mean(duty)),
            "congestion_alarms": len(alarms),
            "detector_steps_flagged": int(flagged_steps),
            "persistently_faulty_detectors": int(np.sum(
                flag_count > 0.02 * (t_end - t_start))),
            "fusion_weight": w,
            "steps_uncontrolled": 0,
        })
        self._alarms = alarms[:40]
        self._cloud = cloud
        return r

    def run_edge_fog_cloud(self, outage=None):
        return self._run_efc(self.n_train, self.T - 1, self.fusion_w,
                             outage=outage)

    # ================= experiments =================
    def sweep_delta(self):
        """Bandwidth vs. twin fidelity: how loud does an edge node need to be?"""
        rows = []
        for d in (0.5, 1.0, 2.0, 4.0, 8.0):
            r = self._run_efc(self.n_train, self.T - 1, self.fusion_w, delta=d)
            rows.append({"delta_mph": d,
                         "uplink_kB": r["uplink_kB"],
                         "duty_cycle_pct": r["uplink_duty_cycle_pct"],
                         "twin_sync_rmse_mph": r["twin_sync_rmse_mph"],
                         "control_input_MAE_mph": r["control_input_MAE_mph"],
                         "mean_delay_s": r["mean_delay_adaptive_s"]})
            print(f"      delta={d:>4} mph  duty={r['uplink_duty_cycle_pct']:5.1f}%"
                  f"  sync_rmse={r['twin_sync_rmse_mph']:5.2f}"
                  f"  delay={r['mean_delay_adaptive_s']:6.2f}s")
        self.results["delta_sweep"] = rows
        return rows

    def run_resilience(self):
        """What survives a backhaul outage."""
        out = {}
        for name, fn in (("cloud_only", self.run_cloud_only),
                         ("edge_only", self.run_edge_only),
                         ("edge_fog_cloud", self.run_edge_fog_cloud)):
            r = fn(outage=self.outage)
            out[name] = {"mean_delay_adaptive_s": r["mean_delay_adaptive_s"],
                         "delay_reduction_pct": r["delay_reduction_pct"],
                         "steps_uncontrolled": r["steps_uncontrolled"]}
            print(f"      {name:<16} delay under outage="
                  f"{r['mean_delay_adaptive_s']:6.2f}s  "
                  f"uncontrolled steps={r['steps_uncontrolled']}")
        out["outage_steps"] = int(self.outage[self.n_train:].sum())
        out["outage_pct"] = float(100.0 * self.outage[self.n_train:].mean())
        self.results["resilience"] = out
        return out

    def run_what_if(self):
        cfg = self.cfg
        t = self.T - 60
        worst = int(np.argmin(self.X[t]))
        neigh = np.argsort(-self.A_sym[worst])[:3]
        inc = np.unique(np.concatenate([[worst], neigh]))
        scen = self._cloud.what_if(t, inc, drop_to=8.0, horizon=6)
        scen["description"] = ("lane-blocking incident on detector "
                               f"{self.sensor_ids[worst]} plus 3 adjacent links")
        self.results["what_if"] = scen
        return scen

    # ================= packing =================
    def _pack(self, name, d_ad, d_fx, err, bus, lat):
        d_ad = np.asarray(d_ad); d_fx = np.asarray(d_fx)
        hours = d_ad.shape[0] * self.cfg.step_minutes / 60.0
        return {
            "architecture": name,
            "mean_delay_adaptive_s": float(d_ad.mean()),
            "mean_delay_fixedtime_s": float(d_fx.mean()),
            "delay_reduction_pct": float(100.0 * (d_fx.mean() - d_ad.mean())
                                         / d_fx.mean()),
            "control_input_MAE_mph": float(np.mean(err)),
            "latency_ms": lat.summary(),
            "uplink_kB": bus.total_uplink_bytes() / 1024.0,
            "uplink_kB_per_hour": bus.total_uplink_bytes() / 1024.0 / hours,
            "messages": dict(bus.msgs),
        }

    # ================= top level =================
    def run_all(self):
        print("[1/7] loading real dataset + road graph")
        self.setup()
        print("[2/7] training tiered forecasters")
        self.train_models()
        print("[3/7] calibrating edge/fog fusion on validation slice")
        self.calibrate_fusion()
        print("[4/7] benchmarking forecast accuracy")
        self.benchmark_forecasting()
        print("[5/7] can the tiers see congestion coming? (15 min ahead)")
        self.benchmark_congestion()
        print("      simulating three deployment architectures")
        arch = {}
        for fn in (self.run_cloud_only, self.run_edge_only,
                   self.run_edge_fog_cloud):
            r = fn()
            arch[r["architecture"]] = r
            print(f"      {r['architecture']:<16} "
                  f"delay={r['mean_delay_adaptive_s']:6.2f}s  "
                  f"ctrlMAE={r['control_input_MAE_mph']:5.2f}mph  "
                  f"p95={r['latency_ms']['p95']:7.1f}ms  "
                  f"uplink={r['uplink_kB']:8.1f}kB")
        self.results["architectures"] = arch
        print("[6/7] sweeping the event-trigger threshold")
        self.sweep_delta()
        print("      --- WAN outage resilience ---")
        self.run_resilience()
        print("[7/7] what-if incident scenario")
        self.run_what_if()

        self.results["meta"] = {
            "dataset": "METR-LA / los-loop: 207 LA freeway detectors, 5-min mean speed",
            "n_detectors": int(self.N),
            "n_timesteps": int(self.T),
            "train_steps": int(self.n_train),
            "test_steps": int(self.T - self.n_train),
            "n_fog_zones": int(self.cfg.n_zones),
            "zone_sizes": np.bincount(self.zones).tolist(),
        }
        return self.results

    def save(self, path="results.json"):
        payload = dict(self.results, alarms_sample=getattr(self, "_alarms", []))
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        return path

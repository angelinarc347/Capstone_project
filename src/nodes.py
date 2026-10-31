"""The three tiers of the digital twin.

EdgeNode   one per detector. Filters, self-diagnoses, decides locally in
           milliseconds, and speaks only when it has something to say.
FogNode    one per road zone. Holds a sub-twin of its corridor, forecasts
           5-15 min ahead using neighbouring roads, coordinates the splits
           of the intersections under it.
CloudTwin  one per city. Holds the complete replica, forecasts 30 min,
           runs what-if scenarios, and keeps the fog nodes aware of each
           other.
"""
import numpy as np

from .models import LocalTrendKF, assemble
from .infra import Link
from .traffic import congestion_level, greenshields


# ==========================================================================
# EDGE
# ==========================================================================
class EdgeNode:
    def __init__(self, idx, sensor_id, zone, cfg, x0, bus):
        self.idx = idx
        self.sensor_id = sensor_id
        self.zone = zone
        self.cfg = cfg
        self.bus = bus

        self.kf = LocalTrendKF(cfg, x0)
        self.delta = cfg.delta_send      # event-trigger threshold (tunable)
        self.last_sent = x0
        self.since_send = 0
        self.last_raw = None
        self.repeat_count = 0

        self.health = "ok"
        self.n_uplinks = 0
        self.n_dropped = 0

    # -- perception -------------------------------------------------------
    def ingest(self, raw):
        """Filter one raw reading. Returns the twin-side estimate."""
        if raw is None:
            self.n_dropped += 1

        # stuck-sensor diagnosis: a real detector never repeats to the bit
        if raw is not None and self.last_raw is not None and raw == self.last_raw:
            self.repeat_count += 1
        else:
            self.repeat_count = 0
        self.last_raw = raw

        level, accepted = self.kf.update(raw)

        if self.repeat_count >= self.cfg.stuck_limit:
            self.health = "stuck"          # detector frozen on one value
        elif raw is not None and not accepted:
            self.health = "noisy"          # this reading was gated out
        else:
            self.health = "ok"
        return level

    # -- event-triggered reporting ---------------------------------------
    def should_report(self, level):
        """Speak only on significant change, or on the heartbeat.

        The fog node holds the last value it received, so silence is itself
        information: 'nothing has changed by more than delta'.
        """
        self.since_send += 1
        if abs(level - self.last_sent) >= self.delta:
            return True
        if self.since_send >= self.cfg.heartbeat:
            return True
        if self.health != "ok":
            return True
        return False

    def report(self, level):
        self.last_sent = level
        self.since_send = 0
        self.n_uplinks += 1
        return self.bus.send(Link.EDGE_FOG, self.cfg.bytes_reading)

    # -- local control ----------------------------------------------------
    def local_forecast(self):
        return self.kf.forecast(1)

    def local_latency(self):
        return (self.bus.latency(Link.EDGE_LOCAL) * 2.0) + self.cfg.cpu_edge


# ==========================================================================
# FOG
# ==========================================================================
class FogNode:
    def __init__(self, zone_id, members, cfg, A_hat, prof, model_short,
                 model_mid, bus, x0_full):
        self.zone_id = zone_id
        self.members = np.asarray(members)
        self.cfg = cfg
        self.A_hat = A_hat
        self.prof = prof
        self.model_short = model_short    # horizon 1  (5 min)  -> control
        self.model_mid = model_mid        # horizon 3  (15 min) -> planning
        self.bus = bus

        # The fog node's replica of the road state. Members are refreshed by
        # its own edge nodes; everything else arrives from the cloud sync.
        self.state = x0_full.copy()
        self.buf = [x0_full.copy() for _ in range(cfg.lags)]
        self.alarms = []

    # -- state maintenance ------------------------------------------------
    def apply_edge_report(self, idx, value):
        self.state[idx] = value

    def apply_cloud_sync(self, full_state):
        """Neighbouring zones' roads, refreshed periodically via the cloud."""
        outside = np.ones(len(self.state), dtype=bool)
        outside[self.members] = False
        self.state[outside] = full_state[outside]

    def commit_step(self):
        self.buf.insert(0, self.state.copy())
        self.buf = self.buf[: self.cfg.lags]

    # -- prediction -------------------------------------------------------
    def _features(self, t_abs, horizon):
        S = np.stack(self.buf)                     # (lags, N), [0] = newest
        n1 = S @ self.A_hat.T
        n2 = n1[0] @ self.A_hat.T
        spd = self.cfg.steps_per_day
        return assemble(list(S), list(n1), n2,
                        self.prof[t_abs % spd],
                        self.prof[(t_abs + horizon) % spd],
                        idx=self.members)

    def forecast(self, t_abs, horizon=1):
        model = self.model_short if horizon == 1 else self.model_mid
        return model.predict(self._features(t_abs, horizon))

    # -- supervision ------------------------------------------------------
    def scan_congestion(self, t_abs, pred_mid):
        """Flag roads predicted to degrade within 15 minutes."""
        self.alarms = []
        now = self.state[self.members]
        for k, idx in enumerate(self.members):
            lvl_now = congestion_level(now[k], self.cfg)
            lvl_next = congestion_level(pred_mid[k], self.cfg)
            if lvl_next in ("dense", "congested") and lvl_next != lvl_now:
                self.alarms.append({"t": int(t_abs), "sensor": int(idx),
                                    "from": lvl_now, "to": lvl_next,
                                    "v_now": float(now[k]),
                                    "v_pred": float(pred_mid[k])})
        return self.alarms

    def coordinate(self, lambdas):
        """Corridor arbitration.

        Individually-greedy intersections can all demand a long green at
        once. The fog node holds the zone to a fixed green budget, scaling
        demands back proportionally -- this is the piece an isolated edge
        node structurally cannot do.
        """
        budget = self.cfg.lam_fixed * len(lambdas)
        total = float(np.sum(lambdas))
        if total > budget:
            lambdas = lambdas * (budget / total)
        return np.clip(lambdas, self.cfg.lam_min, self.cfg.lam_max)

    def push_summary(self):
        return self.bus.send(Link.FOG_CLOUD, self.cfg.bytes_zone_summary)

    def control_latency(self):
        return (self.bus.latency(Link.EDGE_FOG) * 2.0
                + self.cfg.cpu_fog + self.cfg.cpu_edge)


# ==========================================================================
# CLOUD
# ==========================================================================
class CloudTwin:
    def __init__(self, cfg, A_hat, prof, model_long, bus, x0_full):
        self.cfg = cfg
        self.A_hat = A_hat
        self.prof = prof
        self.model_long = model_long      # horizon 6 (30 min)
        self.bus = bus
        self.state = x0_full.copy()
        self.buf = [x0_full.copy() for _ in range(cfg.lags)]
        self.history = []

    def ingest_state(self, full_state):
        self.state = full_state.copy()

    def commit_step(self):
        self.buf.insert(0, self.state.copy())
        self.buf = self.buf[: self.cfg.lags]

    def _features(self, t_abs, horizon, state_override=None):
        S = np.stack(self.buf)
        if state_override is not None:
            S = S.copy()
            S[0] = state_override
        n1 = S @ self.A_hat.T
        n2 = n1[0] @ self.A_hat.T
        spd = self.cfg.steps_per_day
        return assemble(list(S), list(n1), n2,
                        self.prof[t_abs % spd],
                        self.prof[(t_abs + horizon) % spd])

    def forecast(self, t_abs, horizon=6):
        return self.model_long.predict(self._features(t_abs, horizon))

    def control_latency(self):
        return (self.bus.latency(Link.EDGE_CLOUD) * 2.0
                + self.cfg.cpu_cloud + self.cfg.cpu_edge)

    # -- what-if ----------------------------------------------------------
    def what_if(self, t_abs, incident_sensors, drop_to=8.0, horizon=6):
        """Counterfactual: force an incident and re-run the city forecast.

        This is the capability that separates a digital twin from a
        dashboard -- the replica can be driven into a state the real network
        is not in, to see where the queue would spread and how long it takes.

        Impact is scored in travel time, not flow. Greenshields flow is
        non-monotonic in speed (the congested branch of the fundamental
        diagram carries high flow at low speed), so "flow lost" would
        report an incident as an improvement. Travel time per unit
        distance, 1/v, is monotonic and is what a driver actually pays.
        """
        base = self.forecast(t_abs, horizon)
        hyp = self.state.copy()
        hyp[incident_sensors] = drop_to
        alt = self.model_long.predict(self._features(t_abs, horizon, hyp))

        impact = base - alt                       # mph lost per road
        affected = np.where(impact > 1.0)[0]
        spill = np.setdiff1d(affected, np.atleast_1d(incident_sensors))

        tt_base = float(np.sum(1.0 / np.clip(base, 1.0, None)))
        tt_alt = float(np.sum(1.0 / np.clip(alt, 1.0, None)))

        los_base = np.array([congestion_level(v, self.cfg) for v in base])
        los_alt = np.array([congestion_level(v, self.cfg) for v in alt])

        return {
            "incident_sensors": [int(i) for i in np.atleast_1d(incident_sensors)],
            "horizon_min": horizon * self.cfg.step_minutes,
            "n_roads_affected": int(len(affected)),
            "n_roads_spillover": int(len(spill)),
            "worst_roads": [int(i) for i in np.argsort(-impact)[:8]],
            "mean_speed_loss_mph": float(impact[affected].mean()) if len(affected) else 0.0,
            "network_travel_time_increase_pct": 100.0 * (tt_alt - tt_base) / tt_base,
            "roads_degraded_a_level": int(np.sum(los_base != los_alt)),
        }

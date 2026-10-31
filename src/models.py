"""Forecasting models used at each tier of the twin.

  Edge   -> LocalTrendKF : scalar, O(1) per reading, runs on an MCU
  Fog    -> GraphRidge   : spatio-temporal, zone-scoped, retrainable on site
  Cloud  -> GraphRidge   : same structure, city-wide, longer horizon
  Baselines -> HistoricalAverage, Persistence

Feature assembly lives in one place (`assemble`) because the fog node has
to build an identical row at run time from its *reconstructed* replica.
If those two paths ever drift apart the twin degrades silently, so both
call the same function and tests/test_features.py asserts they agree.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

N_FEATURES = 14


def assemble(S, n1, n2, prof_now, prof_target, idx=None):
    """Feature matrix for a set of sensors.

    Works on a single timestep (each input (N,), result (N, 14)) and on a
    whole batch (each input (S, N), result (S, N, 14)).

    S           stack of recent states, S[0] = newest
    n1, n2      1-hop and 2-hop neighbour means (n2 at t only)
    prof_now    historical average at t
    prof_target historical average at t+h
    idx         sensor subset, or None for all
    """
    i = slice(None) if idx is None else idx
    x0, x1, x2, x3 = S[0][..., i], S[1][..., i], S[2][..., i], S[3][..., i]
    a0, a1, a2 = n1[0][..., i], n1[1][..., i], n1[2][..., i]
    b0 = n2[..., i]
    pn, pt = prof_now[..., i], prof_target[..., i]
    return np.stack([
        x0, x1, x2, x3,          # own recent history
        a0, a1, a2,              # what the connected roads are doing
        b0,                      # 2-hop: congestion arriving from further out
        pt, pn,                  # diurnal expectation
        x0 - x1,                 # local trend
        x0 - pn,                 # deviation from a normal day
        a0 - x0,                 # spatial gradient (up/downstream contrast)
        pt - pn,                 # expected diurnal change over the horizon
    ], axis=-1)


def build_training_matrix(X, A_hat, prof, cfg, horizon, t0=0, sensor_mask=None):
    """Vectorised (features, residual targets) over the whole history.

    The target is x[t+h] - x[t], not x[t+h]. Learning the *change* stops the
    regulariser from having to re-derive "the next five minutes look like the
    last five" for all 207 detectors, and it is what lifts the model clear of
    persistence at the longer horizons.
    """
    T, N = X.shape
    L = cfg.lags
    n1_all = X @ A_hat.T
    n2_all = n1_all @ A_hat.T
    spd = cfg.steps_per_day

    ts = np.arange(L - 1, T - horizon)
    S = [X[ts - l] for l in range(L)]               # each (S, N)
    n1 = [n1_all[ts - l] for l in range(3)]
    n2 = n2_all[ts]
    pn = prof[(t0 + ts) % spd]
    pt = prof[(t0 + ts + horizon) % spd]

    F = assemble(S, n1, n2, pn, pt)                 # (S, N, 14)
    Y = X[ts + horizon] - X[ts]                     # residual target

    if sensor_mask is not None:
        F = F[:, sensor_mask, :]
        Y = Y[:, sensor_mask]

    return F.reshape(-1, N_FEATURES), Y.reshape(-1)


class GraphRidge:
    """Ridge regression on graph-diffused spatio-temporal features.

    Weights are pooled across every sensor in scope, so one model serves a
    whole fog zone (or the whole city) and generalises to detectors with
    little history. Training is a closed-form solve -- under a second --
    which is what makes on-site retraining at a fog node realistic.
    """

    def __init__(self, cfg, horizon):
        self.cfg = cfg
        self.horizon = horizon
        self.model = make_pipeline(StandardScaler(),
                                   Ridge(alpha=cfg.ridge_alpha))

    def fit(self, X, A_hat, prof, t0=0, sensor_mask=None):
        F, y = build_training_matrix(X, A_hat, prof, self.cfg,
                                     self.horizon, t0, sensor_mask)
        self.model.fit(F, y)
        return self

    def predict(self, F):
        """F must come from `assemble`; column 0 is x[t], the residual base."""
        F = np.atleast_2d(F)
        delta = self.model.predict(F)
        return np.clip(F[:, 0] + delta, 1.0, self.cfg.v_free + 5.0)


class GraphGBT(GraphRidge):
    """Gradient-boosted trees on the same graph-diffused features.

    Same interface as GraphRidge, same residual target -- only the
    hypothesis class changes. Trees capture the thing a linear model
    cannot: the response to an upstream slowdown is not proportional,
    it switches on once the neighbouring road crosses into congestion.
    Training takes a few seconds and inference is a few microseconds per
    road, so this still runs at a fog node.
    """

    def __init__(self, cfg, horizon):
        self.cfg = cfg
        self.horizon = horizon
        self.model = HistGradientBoostingRegressor(
            max_iter=cfg.gbt_iters,
            max_depth=cfg.gbt_depth,
            learning_rate=0.1,
            early_stopping=False,
            random_state=cfg.seed,
        )


def make_model(cfg, horizon):
    return (GraphGBT if cfg.model_type == "gbt" else GraphRidge)(cfg, horizon)


class LocalTrendKF:
    """Two-state (level, trend) Kalman filter -- one per detector, on the edge.

    Three pieces of hardening that matter over a week of continuous running:
      * the trend state is damped, so it cannot extrapolate off to infinity
        through a run of dropped readings;
      * the innovation gate resets the filter after repeated rejections,
        because a sustained "outlier" is not an outlier -- it is a real
        regime change (an incident) and must not be gated out forever;
      * the level is clipped to physically possible speeds.
    """

    PHI = 0.85              # trend damping
    MAX_CONSEC_REJECT = 3

    def __init__(self, cfg, x0):
        self.cfg = cfg
        self.x = np.array([float(x0), 0.0])
        self.P = np.array([[cfg.kf_r, 0.0], [0.0, 1.0]])
        self.F = np.array([[1.0, self.PHI], [0.0, self.PHI]])
        self.Q = np.diag([cfg.kf_q_level, cfg.kf_q_trend])
        self.H = np.array([1.0, 0.0])
        self.R = cfg.kf_r
        self.rejected = 0
        self.consec_reject = 0

    def update(self, z):
        """Returns (filtered_level, accepted). z=None means a dropped reading."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        if z is None:
            self.x[0] = np.clip(self.x[0], 1.0, 90.0)
            return float(self.x[0]), False

        innov = z - self.H @ self.x
        S = float(self.H @ self.P @ self.H + self.R)

        if abs(innov) > self.cfg.outlier_sigma * np.sqrt(S):
            self.rejected += 1
            self.consec_reject += 1
            if self.consec_reject >= self.MAX_CONSEC_REJECT:
                # sustained disagreement: reality moved, the filter did not
                self.x = np.array([float(z), 0.0])
                self.P = np.array([[self.cfg.kf_r * 4, 0.0], [0.0, 1.0]])
                self.consec_reject = 0
                return float(self.x[0]), True
            self.x[0] = np.clip(self.x[0], 1.0, 90.0)
            return float(self.x[0]), False

        self.consec_reject = 0
        K = (self.P @ self.H) / S
        self.x = self.x + K * innov
        self.P = self.P - np.outer(K, self.H @ self.P)
        self.x[0] = np.clip(self.x[0], 1.0, 90.0)
        return float(self.x[0]), True

    def forecast(self, steps=1):
        damp = sum(self.PHI ** k for k in range(1, steps + 1))
        v = self.x[0] + self.x[1] * damp
        return float(np.clip(v, 1.0, self.cfg.v_free + 5.0))


class HistoricalAverage:
    def __init__(self, prof, cfg, t0=0):
        self.prof, self.cfg, self.t0 = prof, cfg, t0

    def predict(self, t, horizon):
        return self.prof[(self.t0 + t + horizon) % self.cfg.steps_per_day]


class Persistence:
    @staticmethod
    def predict(x_now):
        return x_now

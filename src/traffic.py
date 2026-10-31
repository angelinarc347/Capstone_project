"""Traffic-flow physics that turns a speed forecast into a control action
and a measurable outcome.

The dataset gives speed only, so the chain is:

    measured speed  --Greenshields-->  density & flow
    flow            --Webster split-->  green ratio lambda
    lambda + flow   --Webster delay-->  seconds of delay per vehicle

This matters: it is what makes prediction accuracy *cash out* as a traffic
KPI. A controller acting on a stale or noisy speed picks the wrong split and
the delay formula charges it for the mistake.
"""
import numpy as np


def greenshields(v, cfg):
    """Speed (mph) -> (density veh/mi/lane, flow veh/h/lane)."""
    v = np.clip(v, 1.0, cfg.v_free - 0.5)
    k = cfg.k_jam * (1.0 - v / cfg.v_free)
    q = k * v
    return k, q


def webster_delay(q, lam, cfg):
    """Average delay per vehicle (s) for one approach, Webster's formula."""
    lam = np.clip(lam, 0.05, 0.95)
    cap = cfg.sat_flow * lam
    X = np.clip(q / np.maximum(cap, 1e-6), 0.01, 0.95)   # degree of saturation

    uniform = cfg.cycle_s * (1.0 - lam) ** 2 / (2.0 * (1.0 - lam * X))
    q_ps = np.maximum(q, 30.0) / 3600.0
    overflow = X ** 2 / (2.0 * q_ps * (1.0 - X))
    return uniform + overflow


def adaptive_split(q_hat, q_opposing, cfg):
    """Webster's proportional split, driven by the *predicted* demand.

    q_hat is whatever estimate of near-future flow the controller actually
    had in hand at decision time -- fresh at the edge, stale in the cloud.
    """
    y = q_hat / cfg.sat_flow
    y_opp = q_opposing / cfg.sat_flow
    lam = y / np.maximum(y + y_opp, 1e-6)
    return np.clip(lam, cfg.lam_min, cfg.lam_max)


def evaluate_control(v_estimate, v_actual, q_opposing, cfg):
    """Score one control decision.

    The split is chosen from the estimate; the delay is paid on reality.
    Returns (delay_adaptive, delay_fixed_time, lambda_used).
    """
    _, q_hat = greenshields(v_estimate, cfg)
    _, q_true = greenshields(v_actual, cfg)

    lam = adaptive_split(q_hat, q_opposing, cfg)
    d_adaptive = webster_delay(q_true, lam, cfg)
    d_fixed = webster_delay(q_true, cfg.lam_fixed, cfg)
    return d_adaptive, d_fixed, lam


def congestion_level(v, cfg):
    """Level-of-service style label used by the dashboard and the fog alarms."""
    r = v / cfg.v_free
    if r >= 0.85:
        return "free"
    if r >= 0.65:
        return "stable"
    if r >= 0.45:
        return "dense"
    return "congested"

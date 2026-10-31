"""Central configuration for the Edge-Fog Digital Twin."""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # ---------------- data ----------------
    speed_csv: str = str(Path(__file__).resolve().parents[1] / "data" / "los_speed.csv")
    adj_csv: str = str(Path(__file__).resolve().parents[1] / "data" / "los_adj.csv")
    step_minutes: int = 5
    steps_per_day: int = 288
    train_frac: float = 0.70

    # ---------------- road graph ----------------
    n_zones: int = 8            # number of fog nodes
    adj_threshold: float = 0.10  # prune weak road-network links

    # ---------------- predictive models ----------------
    lags: int = 4                     # 4 x 5min = 20 min of history
    horizons: tuple = (1, 3, 6)       # 5 / 15 / 30 minutes ahead
    ridge_alpha: float = 1.0
    model_type: str = "gbt"      # "gbt" (accurate) | "ridge" (retrains in <1s)
    gbt_iters: int = 200
    gbt_depth: int = 6
    congestion_v: float = 42.25  # 0.65 * v_free -> LOS 'dense' or worse

    # ---------------- edge node ----------------
    kf_q_level: float = 1.00     # Kalman process noise (level), grid-tuned
    kf_q_trend: float = 0.01     # Kalman process noise (trend)
    kf_r: float = 4.0            # measurement noise variance, grid-tuned
    outlier_sigma: float = 4.0   # innovation gate for fault rejection
    stuck_limit: int = 8         # identical readings => stuck sensor

    # event-triggered transmission (the bandwidth saver)
    delta_send: float = 2.0      # mph deviation that forces an uplink
    heartbeat: int = 6           # ...and force one every 30 min anyway

    # ---------------- raw sensor stream ----------------
    sensor_noise_std: float = 1.5   # mph
    spike_prob: float = 0.004       # probability of a gross outlier
    spike_mag: float = 25.0
    dropout_prob: float = 0.003     # probability of a missing reading

    # ---------------- network / compute (milliseconds) ----------------
    lat_edge_local: float = 1.0
    lat_edge_fog: float = 8.0
    lat_fog_cloud: float = 45.0
    lat_edge_cloud: float = 70.0
    jitter_frac: float = 0.25
    cpu_edge: float = 1.5
    cpu_fog: float = 6.0
    cpu_cloud: float = 20.0

    bytes_reading: int = 24      # sensor_id + ts + speed + quality flag
    bytes_zone_summary: int = 96  # fog -> cloud aggregate

    # ---------------- traffic flow theory ----------------
    v_free: float = 65.0      # free-flow speed, mph (Greenshields)
    k_jam: float = 180.0      # jam density, veh/mi/lane
    sat_flow: float = 1900.0  # saturation flow, veh/h/lane
    cycle_s: float = 90.0     # signal cycle length, seconds
    lam_fixed: float = 0.50   # baseline fixed-time green ratio
    lam_min: float = 0.25
    lam_max: float = 0.70

    # ---------------- misc ----------------
    seed: int = 7

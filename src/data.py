"""Load the real LA loop-detector dataset and build the road graph.

Dataset: METR-LA / "los-loop" -- 207 inductive loop detectors on the Los
Angeles freeway network, aggregated to 5-minute mean speeds (mph) over
7 consecutive days (2016 timesteps), plus a 207x207 weighted adjacency
matrix derived from road-network distance between detectors.
"""
import numpy as np
import pandas as pd


def load_dataset(cfg):
    speed = pd.read_csv(cfg.speed_csv)
    sensor_ids = [str(c) for c in speed.columns]
    X = speed.values.astype(np.float64)              # (T, N) mph
    A = pd.read_csv(cfg.adj_csv, header=None).values.astype(np.float64)

    # Clean: zeros in this dataset mean "no vehicles detected / missing".
    X[X <= 0] = np.nan
    X = pd.DataFrame(X).ffill().bfill().values

    return X, A, sensor_ids


def build_graph(A, cfg):
    """Return (A_sym, A_hat) -- pruned symmetric adjacency and the
    row-normalised neighbour-averaging operator (self excluded)."""
    A = A.copy()
    np.fill_diagonal(A, 0.0)
    A[A < cfg.adj_threshold] = 0.0
    A_sym = np.maximum(A, A.T)              # roads are traversable both ways

    deg = A_sym.sum(axis=1, keepdims=True)
    deg[deg == 0] = 1.0
    A_hat = A_sym / deg                     # row-stochastic diffusion operator
    return A_sym, A_hat


def partition_zones(A_sym, cfg):
    """Cut the road network into `n_zones` spatially coherent fog domains.

    Spectral clustering on the road adjacency keeps physically connected
    corridors inside the same fog node, which is what makes green-wave
    coordination possible at that layer.
    """
    from sklearn.cluster import SpectralClustering

    sc = SpectralClustering(
        n_clusters=cfg.n_zones,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=cfg.seed,
    )
    labels = sc.fit_predict(A_sym + 1e-6)   # tiny epsilon keeps it connected
    return labels.astype(int)


def time_of_day_profile(X_train, cfg, t0=0):
    """Historical-average speed for each (time-of-day, sensor) pair.

    This is both a baseline forecaster and a feature: urban traffic is
    overwhelmingly diurnal, so the twin should never re-learn rush hour.
    """
    T, N = X_train.shape
    spd = cfg.steps_per_day
    prof = np.zeros((spd, N))
    cnt = np.zeros((spd, 1))
    for t in range(T):
        tod = (t0 + t) % spd
        prof[tod] += X_train[t]
        cnt[tod, 0] += 1
    cnt[cnt == 0] = 1
    prof /= cnt
    # fill any time-of-day never seen in training with the global mean
    empty = prof.sum(axis=1) == 0
    prof[empty] = X_train.mean(axis=0)
    return prof


def split(X, cfg):
    T = X.shape[0]
    n_train = int(T * cfg.train_frac)
    return X[:n_train], X[n_train:], n_train

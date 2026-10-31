"""The training path and the fog node's online path must build identical
features. If they ever diverge the twin degrades silently -- the model keeps
returning confident numbers computed from columns that no longer mean what
it was trained on. So this is asserted, not assumed.

Run:  python -m tests.test_features
"""
import numpy as np

from src.config import Config
from src.data import load_dataset, build_graph, partition_zones, \
    time_of_day_profile, split
from src.models import build_training_matrix, N_FEATURES
from src.nodes import FogNode


def main():
    cfg = Config()
    X, A, ids = load_dataset(cfg)
    A_sym, A_hat = build_graph(A, cfg)
    zones = partition_zones(A_sym, cfg)
    X_train, _, n_train = split(X, cfg)
    prof = time_of_day_profile(X_train, cfg)

    horizon = 3
    t = 500
    members = np.where(zones == 0)[0]

    # --- online path: a fog node whose replica happens to be exact ---
    fog = FogNode(0, members, cfg, A_hat, prof, None, None, None, X[t].copy())
    fog.buf = [X[t - k].copy() for k in range(cfg.lags)]
    online = fog._features(t, horizon)

    # --- training path ---
    F, _ = build_training_matrix(X, A_hat, prof, cfg, horizon)
    S = X.shape[0] - horizon - (cfg.lags - 1)
    F = F.reshape(S, X.shape[1], N_FEATURES)
    offline = F[t - (cfg.lags - 1)][members]

    assert online.shape == offline.shape, (online.shape, offline.shape)
    worst = float(np.max(np.abs(online - offline)))
    assert np.allclose(online, offline, atol=1e-9), f"max diff {worst}"

    print(f"OK  feature parity: {online.shape[0]} sensors x "
          f"{online.shape[1]} features, max abs diff {worst:.2e}")

    # a second guard: column 0 must be x[t], since GraphRidge.predict adds
    # the predicted residual back onto it
    assert np.allclose(online[:, 0], X[t][members]), \
        "column 0 is the residual base and must be x[t]"
    print("OK  residual base column is x[t]")


if __name__ == "__main__":
    main()

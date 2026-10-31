"""A tiny publish/subscribe bus that models what the network actually costs.

Everything runs in one process, but every hop is charged realistic latency
and every message is charged bytes. That accounting is the whole point: it
is how the simulation shows that pushing work to the edge buys you latency
and bandwidth, and that keeping some work in the cloud buys you context.
"""
from collections import defaultdict
import numpy as np


class Link:
    EDGE_LOCAL = "edge_local"
    EDGE_FOG = "edge_fog"
    FOG_CLOUD = "fog_cloud"
    EDGE_CLOUD = "edge_cloud"


class MessageBus:
    def __init__(self, cfg, rng):
        self.cfg = cfg
        self.rng = rng
        self.bytes_up = defaultdict(int)     # bytes per link, uplink direction
        self.msgs = defaultdict(int)
        self.base = {
            Link.EDGE_LOCAL: cfg.lat_edge_local,
            Link.EDGE_FOG: cfg.lat_edge_fog,
            Link.FOG_CLOUD: cfg.lat_fog_cloud,
            Link.EDGE_CLOUD: cfg.lat_edge_cloud,
        }

    def send(self, link, n_bytes):
        """Charge one message over `link`; returns one-way latency in ms."""
        self.bytes_up[link] += n_bytes
        self.msgs[link] += 1
        return self.latency(link)

    def latency(self, link):
        b = self.base[link]
        return float(b * (1.0 + self.rng.normal(0.0, self.cfg.jitter_frac * 0.4)))

    def total_uplink_bytes(self):
        return sum(self.bytes_up.values())

    def report(self):
        return {
            "bytes_per_link": dict(self.bytes_up),
            "messages_per_link": dict(self.msgs),
            "total_uplink_bytes": self.total_uplink_bytes(),
        }


class LatencyTracker:
    """Collects end-to-end control-loop latencies so we can report p50/p95."""

    def __init__(self):
        self.samples = []

    def add(self, ms):
        self.samples.append(ms)

    def summary(self):
        if not self.samples:
            return {"p50": 0.0, "p95": 0.0, "mean": 0.0, "max": 0.0}
        a = np.asarray(self.samples)
        return {
            "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)),
            "mean": float(a.mean()),
            "max": float(a.max()),
        }

"""Entry point: python run.py"""
from pathlib import Path

from src.config import Config
from src.simulate import Simulation

if __name__ == "__main__":
    cfg = Config()
    sim = Simulation(cfg)
    res = sim.run_all()
    p = sim.save(Path(__file__).resolve().parent / "results.json")
    print(f"\nresults written to {p}")

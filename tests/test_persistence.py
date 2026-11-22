"""Persistence round-trip including config, temperature, calibration, and unlearn counts."""

import os
import tempfile
import numpy as np
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.system import NoiseSeparatedSystem
from src.config import SystemConfig


def test_save_load_roundtrip():
    cfg = SystemConfig.small()
    s = NoiseSeparatedSystem(config=cfg, explore_prob=0.0)
    x = np.sin(np.linspace(0, 3, 16))
    for _ in range(5):
        s.observe(x, y=1.0)
    s.temperature = 1.7
    s._unlearn_counts["test_key"] = 3
    s._calib.append((2, True))
    stats_before = s.stats()

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "state.pkl")
        s.save(path)
        s2 = NoiseSeparatedSystem.load(path)

    assert s2.cfg.max_obs == cfg.max_obs
    assert abs(s2.temperature - 1.7) < 1e-6
    assert s2._unlearn_counts.get("test_key") == 3
    assert len(s2._calib) >= 1
    assert s2.stats()["step"] == stats_before["step"]
    print("persistence round-trip OK")


def test_memory_is_bounded():
    s = NoiseSeparatedSystem(explore_prob=0.0, max_obs=50, max_chaos=20)
    for i in range(200):
        s.observe(np.random.randn(8), y=float(i % 3))
    assert len(s.thought) <= 50
    assert len(s.chaos) <= 20
    print("memory bounds OK")


if __name__ == "__main__":
    test_save_load_roundtrip()
    test_memory_is_bounded()
    print("All persistence tests passed.")

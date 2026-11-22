"""Persistence and memory-bound tests."""

import os
import tempfile
import numpy as np
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.system import NoiseSeparatedSystem


def test_save_load_roundtrip():
    s = NoiseSeparatedSystem(explore_prob=0.0, max_obs=100)
    x = np.sin(np.linspace(0, 3, 16))
    for _ in range(5):
        s.observe(x, y=1.0)
    stats_before = s.stats()

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "state.pkl")
        s.save(path)
        s2 = NoiseSeparatedSystem.load(path)

    assert s2.stats()["step"] == stats_before["step"]
    assert len(s2.thought) == len(s.thought)
    pred1, _ = s.predict(x)
    pred2, _ = s2.predict(x)
    if pred1 is not None and pred2 is not None:
        assert abs(pred1 - pred2) < 1e-6


def test_memory_is_bounded():
    s = NoiseSeparatedSystem(explore_prob=0.0, max_obs=50, max_chaos=20)
    for i in range(200):
        s.observe(np.random.randn(8), y=float(i % 3))
    assert len(s.thought) <= 50
    assert len(s.chaos) <= 20


if __name__ == "__main__":
    test_save_load_roundtrip()
    test_memory_is_bounded()
    print("Persistence and scale tests passed.")

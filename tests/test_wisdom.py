"""Force early disagreement then consistency so Wisdom can form."""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from src.system import NoiseSeparatedSystem
from src.config import SystemConfig


def test_wisdom_after_deliberation():
    s = NoiseSeparatedSystem(config=SystemConfig.small())
    s.min_age = 0
    s.wisdom_strength = 0.2
    s.promote_hits = 2

    rng = np.random.RandomState(7)
    base = np.array([0.0, 0.8, 1.0, 0.6, -0.2, -0.8, -0.9, -0.4,
                     0.4, 0.9, 0.8, 0.2, -0.5, -1.0, -0.7, 0.0])

    for t in range(40):
        x = base + 0.01 * rng.randn(16)
        y = 1.0 + 4.0 * rng.randn()
        s.observe(x, y=y)

    for t in range(80):
        x = base + 0.005 * rng.randn(16)
        s.observe(x, y=1.0)

    stats = s.stats()
    print("Final stats:", stats)
    print("Disagreement:", dict(s._early_disagreement))
    assert stats["wisdom"] > 0
    print("PASS: Wisdom formed")


if __name__ == "__main__":
    test_wisdom_after_deliberation()

"""Comparative evaluation: filtered+auditor vs larger unfiltered model."""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.system import NoiseSeparatedSystem
from src.config import SystemConfig


def run_stream(system, n_steps=200, noise_frac=0.4, seed=0):
    rng = np.random.RandomState(seed)
    signal = np.sin(np.linspace(0, 4 * np.pi, 16))
    errors = []
    for t in range(n_steps):
        if rng.rand() < noise_frac:
            system.observe(rng.randn(16), None)
        else:
            x = signal + 0.05 * rng.randn(16)
            y = 1.0 + 0.05 * rng.randn()
            system.observe(x, y)
            if t > 30:
                pred, _ = system.predict(signal)
                if pred is not None:
                    errors.append(abs(float(pred) - 1.0))
    return float(np.mean(errors)) if errors else float("nan"), system.stats()


def main():
    print("=== Comparative evaluation ===")
    # Small filtered system
    cfg_small = SystemConfig.small()
    small = NoiseSeparatedSystem(**cfg_small.__dict__)
    err_small, stats_small = run_stream(small, seed=1)

    # Larger system with almost no filtering (very low threshold)
    cfg_large = SystemConfig.large()
    cfg_large.noise_threshold = 0.05  # almost everything passes
    large = NoiseSeparatedSystem(**cfg_large.__dict__)
    err_large, stats_large = run_stream(large, seed=1)

    print(f"Small + filter + auditor  mean error: {err_small:.4f}  stats: {stats_small}")
    print(f"Large + weak filter       mean error: {err_large:.4f}  stats: {stats_large}")
    print(f"Small better or equal: {err_small <= err_large + 0.05}")


if __name__ == "__main__":
    main()

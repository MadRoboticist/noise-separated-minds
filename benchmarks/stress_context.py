"""Stress-test context length and small-filtered vs large-weak-filter."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from src.system import NoiseSeparatedSystem
from src.config import SystemConfig


def run(dim, n_steps=400, noise_frac=0.4, seed=0, small=True):
    rng = np.random.RandomState(seed)
    if small:
        cfg = SystemConfig.small()
    else:
        cfg = SystemConfig.large()
        cfg.noise_threshold = 0.05
    s = NoiseSeparatedSystem(config=cfg)
    signal = np.sin(np.linspace(0, 4 * np.pi, dim))
    errors = []
    for t in range(n_steps):
        if rng.rand() < noise_frac:
            s.observe(rng.randn(dim), None)
        else:
            x = signal + 0.05 * rng.randn(dim)
            y = 1.0 + 0.05 * rng.randn()
            s.observe(x, y)
            if t > 50:
                pred, via = s.predict(signal)
                if pred is not None:
                    errors.append(abs(float(np.asarray(pred).ravel()[0]) - 1.0))
    mean_err = float(np.mean(errors)) if errors else float("nan")
    return mean_err, s.stats()


def main():
    print("=== Context-length stress + small vs large ===\n")
    dims = [8, 16, 32, 64, 128]
    results = []
    for dim in dims:
        err_s, stats_s = run(dim, seed=1, small=True)
        err_l, stats_l = run(dim, seed=1, small=False)
        ok = err_s <= err_l + 0.1
        results.append(ok)
        print(f"dim={dim:3d}  small_err={err_s:.3f}  large_err={err_l:.3f}  small_ok={ok}")
    n_ok = sum(results)
    print(f"\nSmall filtered ok in {n_ok}/{len(results)} dimensions")
    print("PASS" if n_ok >= len(results) // 2 else "FAIL")
    return 0 if n_ok >= len(results) // 2 else 1


if __name__ == "__main__":
    raise SystemExit(main())

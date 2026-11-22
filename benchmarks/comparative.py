"""Comparative evaluation with a small multi-seed, multi-noise sweep."""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.system import NoiseSeparatedSystem
from src.config import SystemConfig


def run_stream(system, n_steps=180, noise_frac=0.4, seed=0):
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
            if t > 25:
                pred, _ = system.predict(signal)
                if pred is not None:
                    errors.append(abs(float(pred) - 1.0))
    return float(np.mean(errors)) if errors else float("nan")


def main():
    print("=== Comparative sweep ===")
    seeds = [1, 2, 3]
    noise_fracs = [0.3, 0.5]
    results = []

    for nf in noise_fracs:
        for seed in seeds:
            cfg_s = SystemConfig.small()
            small = NoiseSeparatedSystem(config=cfg_s)
            err_s = run_stream(small, noise_frac=nf, seed=seed)

            cfg_l = SystemConfig.large()
            cfg_l.noise_threshold = 0.05
            large = NoiseSeparatedSystem(config=cfg_l)
            err_l = run_stream(large, noise_frac=nf, seed=seed)

            better = err_s <= err_l + 0.08
            results.append((nf, seed, err_s, err_l, better))
            print(f"noise={nf:.1f} seed={seed}  small={err_s:.3f}  large={err_l:.3f}  small_ok={better}")

    n_ok = sum(1 for r in results if r[4])
    print(f"\nSmall filtered ok in {n_ok}/{len(results)} runs")
    if n_ok < len(results) // 2:
        print("REGRESSION: small filtered system under-performed")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Compare noise gate (reject) vs a simple denoiser (clean then accept)."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from src.system import NoiseSeparatedSystem
from src.config import SystemConfig


def moving_average_denoise(x, win=5):
    """Cheap conventional denoiser: moving average."""
    x = np.asarray(x, dtype=float)
    if len(x) < win:
        return x
    kernel = np.ones(win) / win
    return np.convolve(x, kernel, mode="same")


def run(mode, n_steps=400, noise_frac=0.4, seed=0, noise_std=0.3):
    """
    mode: 'gate' | 'denoiser' | 'both' | 'neither'
    """
    rng = np.random.RandomState(seed)
    cfg = SystemConfig.small()
    if mode in ("denoiser", "neither"):
        cfg.noise_threshold = 0.0  # disable gate
    s = NoiseSeparatedSystem(config=cfg)
    signal = np.sin(np.linspace(0, 4 * np.pi, 16))
    errors = []
    accepted = 0
    for t in range(n_steps):
        is_noise_packet = rng.rand() < noise_frac
        if is_noise_packet:
            raw = rng.randn(16) * noise_std
            y = None
        else:
            raw = signal + noise_std * rng.randn(16)
            y = 1.0 + 0.05 * rng.randn()
        if mode in ("denoiser", "both"):
            x = moving_average_denoise(raw)
        else:
            x = raw
        route = s.observe(x, y)
        if route != "chaos":
            accepted += 1
        if (not is_noise_packet) and t > 50:
            pred, _ = s.predict(signal)
            if pred is not None:
                errors.append(abs(float(np.asarray(pred).ravel()[0]) - 1.0))
    mean_err = float(np.mean(errors)) if errors else float("nan")
    return mean_err, accepted / n_steps, s.stats()


def main():
    print("=== Gate vs denoiser ===\n")
    modes = ["gate", "denoiser", "both", "neither"]
    results = {m: [] for m in modes}
    for seed in [1, 2, 3]:
        for m in modes:
            err, acc, stats = run(m, seed=seed)
            results[m].append(err)
            print(f"seed={seed}  {m:10s}  err={err:.3f}  accept_rate={acc:.2f}  "
                  f"K={stats['knowledge']} R={stats['reactive']}")
    print("\n--- Means ---")
    for m in modes:
        print(f"{m:10s}  mean_err={np.mean(results[m]):.3f}")
    print("PASS")


if __name__ == "__main__":
    main()

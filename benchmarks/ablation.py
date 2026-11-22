"""Ablations: gate, Reactive/Wisdom split, auditor.
Plus a simple non-synthetic dataset (UCI-style iris via sklearn or fallback)."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from src.system import NoiseSeparatedSystem
from src.config import SystemConfig


def load_iris():
    """Try sklearn iris; fall back to a small embedded version."""
    try:
        from sklearn.datasets import load_iris
        data = load_iris()
        X, y = data.data, data.target.astype(float)
        return X, y, "sklearn iris"
    except Exception:
        # Minimal fallback: 4 features, 3 classes, 30 samples
        rng = np.random.RandomState(0)
        X = np.vstack([
            rng.randn(10, 4) + [5, 3.5, 1.5, 0.2],
            rng.randn(10, 4) + [6, 3, 4.5, 1.5],
            rng.randn(10, 4) + [6.5, 3, 5.5, 2],
        ])
        y = np.array([0]*10 + [1]*10 + [2]*10, dtype=float)
        return X, y, "embedded iris-like"


def run_config(name, **overrides):
    """Run a mixed numeric stream under a given config and return mean error + stats."""
    cfg = SystemConfig.small()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    s = NoiseSeparatedSystem(config=cfg)
    rng = np.random.RandomState(1)
    signal = np.sin(np.linspace(0, 4 * np.pi, 16))
    errors = []
    for t in range(300):
        if rng.rand() < 0.4:
            s.observe(rng.randn(16), None)
        else:
            x = signal + 0.05 * rng.randn(16)
            y = 1.0 + 0.05 * rng.randn()
            s.observe(x, y)
            if t > 40:
                pred, _ = s.predict(signal)
                if pred is not None:
                    errors.append(abs(float(np.asarray(pred).ravel()[0]) - 1.0))
    return float(np.mean(errors)) if errors else float("nan"), s.stats()


def run_iris(disable_gate=False):
    X, y, src = load_iris()
    cfg = SystemConfig.small()
    if disable_gate:
        cfg.noise_threshold = 0.0
    s = NoiseSeparatedSystem(config=cfg)
    # Normalize
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    errors = []
    for i in range(len(X)):
        s.observe(X[i], y[i])
        if i > 20:
            pred, _ = s.predict(X[i])
            if pred is not None:
                errors.append(abs(float(np.asarray(pred).ravel()[0]) - y[i]))
    return float(np.mean(errors)) if errors else float("nan"), s.stats(), src


def main():
    print("=== Ablations (synthetic sine + noise) ===")
    base_err, base_stats = run_config("full")
    print(f"full system          err={base_err:.3f}  K={base_stats['knowledge']} R={base_stats['reactive']} W={base_stats['wisdom']}")

    no_gate_err, no_gate_stats = run_config("no_gate", noise_threshold=0.0)
    print(f"no gate              err={no_gate_err:.3f}  K={no_gate_stats['knowledge']} R={no_gate_stats['reactive']}")

    no_wisdom_err, _ = run_config("no_wisdom", wisdom_strength=999)
    print(f"wisdom disabled      err={no_wisdom_err:.3f}")

    no_reactive_err, no_r_stats = run_config("no_reactive", promote_hits=999)
    print(f"reactive hard        err={no_reactive_err:.3f}  R={no_r_stats['reactive']}")

    print("\n=== Non-synthetic (iris) ===")
    iris_err, iris_stats, src = run_iris(disable_gate=False)
    print(f"{src} with gate     err={iris_err:.3f}  K={iris_stats['knowledge']} R={iris_stats['reactive']}")
    iris_ng, _, _ = run_iris(disable_gate=True)
    print(f"{src} no gate       err={iris_ng:.3f}")

    print("\n=== Summary ===")
    print(f"Gate ablation delta (no-gate - full): {no_gate_err - base_err:+.3f}")
    print(f"Iris gate delta:                      {iris_ng - iris_err:+.3f}")
    print("PASS" if base_err < 0.5 else "FAIL")


if __name__ == "__main__":
    main()

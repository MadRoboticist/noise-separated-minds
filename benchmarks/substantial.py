"""Substantial benchmark: multiple patterns, numeric + non-numeric, longer runs."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from src.system import NoiseSeparatedSystem
from src.config import SystemConfig


NUMERIC_PATTERNS = [
    np.sin(np.linspace(0, 2 * np.pi, 16)),
    np.sin(np.linspace(0, 4 * np.pi, 16)),
    np.cos(np.linspace(0, 3 * np.pi, 16)),
    np.linspace(-1, 1, 16),
]

TEXT_PATTERNS = [
    ("the sky is blue", 1.0),
    ("cats like warm places", 0.5),
    ("two plus two is four", 4.0),
    ("water freezes at zero", 0.0),
    ("fire is hot", 2.0),
]


def run_numeric(n_steps=800, noise_frac=0.35, seed=0):
    rng = np.random.RandomState(seed)
    s = NoiseSeparatedSystem(config=SystemConfig.small())
    errors = []
    for t in range(n_steps):
        if rng.rand() < noise_frac:
            s.observe(rng.randn(16), None)
        else:
            idx = rng.randint(0, len(NUMERIC_PATTERNS))
            base = NUMERIC_PATTERNS[idx]
            x = base + 0.04 * rng.randn(16)
            y = float(idx) + 0.05 * rng.randn()
            s.observe(x, y)
            if t > 100:
                pred, via = s.predict(base)
                if pred is not None:
                    errors.append(abs(float(pred) - float(idx)))
    return float(np.mean(errors)) if errors else float("nan"), s.stats()


def run_text(n_steps=600, noise_frac=0.3, seed=0):
    rng = np.random.RandomState(seed)
    s = NoiseSeparatedSystem(config=SystemConfig.small())
    noise_phrases = ["xqz plm", "asdf jkl", "random tokens", "zzz yyy"]
    errors = []
    for t in range(n_steps):
        if rng.rand() < noise_frac:
            s.observe_context(rng.choice(noise_phrases), None)
        else:
            phrase, y = TEXT_PATTERNS[rng.randint(0, len(TEXT_PATTERNS))]
            s.observe_context(phrase, y)
            if t > 80:
                pred, via = s.predict_context(phrase)
                if pred is not None:
                    errors.append(abs(float(pred) - y))
    return float(np.mean(errors)) if errors else float("nan"), s.stats()


def main():
    print("=== Substantial benchmark ===\n")

    print("-- Numeric (4 patterns, 800 steps) --")
    num_results = []
    for seed in [1, 2, 3]:
        err, stats = run_numeric(seed=seed)
        num_results.append(err)
        print(f"seed={seed}  err={err:.3f}  K={stats['knowledge']}  R={stats['reactive']}  W={stats['wisdom']}  chaos={stats['chaos']}")

    print("\n-- Non-numeric (5 phrases, 600 steps) --")
    txt_results = []
    for seed in [1, 2, 3]:
        err, stats = run_text(seed=seed)
        txt_results.append(err)
        print(f"seed={seed}  err={err:.3f}  K={stats['knowledge']}  R={stats['reactive']}  W={stats['wisdom']}  chaos={stats['chaos']}")

    print(f"\nNumeric mean error: {np.mean(num_results):.3f}")
    print(f"Text mean error:    {np.mean(txt_results):.3f}")
    print("PASS" if np.mean(num_results) < 0.5 and np.mean(txt_results) < 0.5 else "FAIL")


if __name__ == "__main__":
    main()

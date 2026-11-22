"""Benchmark non-numeric contexts through the encoder."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from src.system import NoiseSeparatedSystem
from src.config import SystemConfig


STRUCTURED = [
    "the sky is blue",
    "cats like warm places",
    "two plus two is four",
    "water freezes at zero",
]
NOISE = [
    "xqz plm vbn",
    "asdf jkl qwer",
    "zzz yyy xxx",
    "random tokens here",
]


def run(n_steps=200, noise_frac=0.4, seed=0):
    rng = np.random.RandomState(seed)
    s = NoiseSeparatedSystem(config=SystemConfig.small())
    errors = []
    residual = 0
    for t in range(n_steps):
        if rng.rand() < noise_frac:
            s.observe_context(rng.choice(NOISE), y=None)
        else:
            phrase = rng.choice(STRUCTURED)
            # label is deterministic from the phrase
            y = float(hash(phrase) % 5)
            s.observe_context(phrase, y=y)
            if t > 40:
                pred, via = s.predict_context(phrase)
                if pred is not None:
                    errors.append(abs(float(pred) - y))
                if via == "chaos":
                    residual += 1
    mean_err = float(np.mean(errors)) if errors else float("nan")
    res_rate = residual / max(n_steps, 1)
    return mean_err, res_rate, s.stats()


def main():
    print("=== Non-numeric context benchmark ===")
    results = []
    for seed in [1, 2, 3]:
        for nf in [0.3, 0.5]:
            err, res, stats = run(noise_frac=nf, seed=seed)
            results.append((nf, seed, err, res, stats.get("reactive", 0), stats.get("wisdom", 0)))
            print(f"noise={nf:.1f} seed={seed}  err={err:.3f}  residual={res:.3f}  "
                  f"reactive={stats.get('reactive')}  wisdom={stats.get('wisdom')}")

    avg_err = float(np.mean([r[2] for r in results]))
    avg_res = float(np.mean([r[3] for r in results]))
    print(f"\nAverage error: {avg_err:.3f}")
    print(f"Average residual-noise rate: {avg_res:.3f}")
    if avg_res < 0.15 and avg_err < 1.0:
        print("PASS")
        return 0
    print("FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

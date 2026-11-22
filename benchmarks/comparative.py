"""Comparative evaluation with residual-noise and unlearn-rate tracking."""

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
    residual_noise_steps = 0
    unlearn_snapshots = []

    for t in range(n_steps):
        if rng.rand() < noise_frac:
            system.observe(rng.randn(16), None)
        else:
            x = signal + 0.05 * rng.randn(16)
            y = 1.0 + 0.05 * rng.randn()
            system.observe(x, y)
            if t > 30:
                pred, via = system.predict(signal)
                if pred is not None:
                    errors.append(abs(float(pred) - 1.0))
                if via == "chaos":
                    residual_noise_steps += 1
        if t % 40 == 0 and t > 0:
            unlearn_snapshots.append(system.stats().get("unlearns", 0))

    mean_err = float(np.mean(errors)) if errors else float("nan")
    residual_rate = residual_noise_steps / max(n_steps, 1)
    final_unlearns = system.stats().get("unlearns", 0)
    return mean_err, residual_rate, final_unlearns, unlearn_snapshots


def main():
    print("=== Comparative sweep with rates ===")
    seeds = [1, 2, 3, 4]
    noise_fracs = [0.3, 0.5]
    results = []

    for nf in noise_fracs:
        for seed in seeds:
            cfg_s = SystemConfig.small()
            small = NoiseSeparatedSystem(config=cfg_s)
            err_s, res_s, unl_s, snaps_s = run_stream(small, noise_frac=nf, seed=seed)

            cfg_l = SystemConfig.large()
            cfg_l.noise_threshold = 0.05
            large = NoiseSeparatedSystem(config=cfg_l)
            err_l, res_l, unl_l, _ = run_stream(large, noise_frac=nf, seed=seed)

            better = err_s <= err_l + 0.08
            results.append({
                "nf": nf, "seed": seed,
                "err_s": err_s, "err_l": err_l,
                "res_s": res_s, "unl_s": unl_s,
                "better": better,
            })
            print(
                f"noise={nf:.1f} seed={seed}  "
                f"small_err={err_s:.3f} large_err={err_l:.3f}  "
                f"residual={res_s:.3f} unlearns={unl_s}  ok={better}"
            )

    n_ok = sum(1 for r in results if r["better"])
    avg_residual = float(np.mean([r["res_s"] for r in results]))
    avg_unlearns = float(np.mean([r["unl_s"] for r in results]))

    print(f"\nSmall filtered ok in {n_ok}/{len(results)} runs")
    print(f"Average residual-noise rate (small): {avg_residual:.3f}")
    print(f"Average unlearns (small): {avg_unlearns:.1f}")

    # Regression guards
    if n_ok < len(results) // 2:
        print("REGRESSION: small filtered under-performed on accuracy")
        return 1
    if avg_residual > 0.25:
        print("REGRESSION: residual noise rate too high")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

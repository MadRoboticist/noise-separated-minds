"""Long-horizon evaluation: residual noise, Wisdom reliability, promotion quality."""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.system import NoiseSeparatedSystem


def run(n_steps: int = 300, seed: int = 0):
    rng = np.random.RandomState(seed)
    system = NoiseSeparatedSystem(
        noise_threshold=0.35,
        promote_hits=3,
        min_age=5,
        explore_prob=0.02,
        max_obs=200,
    )

    signal = np.sin(np.linspace(0, 4 * np.pi, 16))
    errors = []
    noise_rejected = 0
    structured_accepted = 0

    for t in range(n_steps):
        if rng.rand() < 0.4:
            # pure noise
            x = rng.randn(16)
            y = None
            route = system.observe(x, y)
            if route == "chaos":
                noise_rejected += 1
        else:
            # structured
            x = signal + 0.05 * rng.randn(16)
            y = 1.0 + 0.1 * rng.randn()
            route = system.observe(x, y)
            structured_accepted += 1
            if t > 20:
                pred, _ = system.predict(signal)
                if pred is not None:
                    errors.append(abs(pred - 1.0))

        # periodic audit
        if t % 25 == 0 and t > 0:
            findings = system.auditor.inspect(system)
            system.auditor.apply(system, findings)

    stats = system.stats()
    mean_err = float(np.mean(errors)) if errors else float("nan")
    print("=== Long-horizon evaluation ===")
    print(f"Steps: {n_steps}")
    print(f"Noise rejected: {noise_rejected}")
    print(f"Structured seen: {structured_accepted}")
    print(f"Mean |pred error| on signal: {mean_err:.4f}")
    print(f"Final stats: {stats}")
    print(f"Auditor findings: {len(system.auditor.findings)}")
    return stats, mean_err


if __name__ == "__main__":
    run()

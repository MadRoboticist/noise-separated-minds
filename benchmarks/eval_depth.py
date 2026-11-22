"""Deeper evaluation: noise sweeps, gate error rates, Wisdom reliability."""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.system import NoiseSeparatedSystem


def gate_error_rates(noise_fracs=(0.2, 0.4, 0.6), n_steps=200, seed=1):
    """False-positive (structured routed to chaos) and false-negative (noise accepted)."""
    print("\n=== Gate error rates ===")
    results = {}
    signal = np.sin(np.linspace(0, 4 * np.pi, 16))
    for frac in noise_fracs:
        rng = np.random.RandomState(seed)
        system = NoiseSeparatedSystem(noise_threshold=0.35, max_obs=150, explore_prob=0.01)
        fp = fn = tp = tn = 0
        for t in range(n_steps):
            is_noise = rng.rand() < frac
            if is_noise:
                x = rng.randn(16)
                y = None
            else:
                x = signal + 0.05 * rng.randn(16)
                y = 1.0
            route = system.observe(x, y)
            predicted_noise = route == "chaos"
            if is_noise and predicted_noise:
                tn += 1
            elif is_noise and not predicted_noise:
                fn += 1
            elif not is_noise and predicted_noise:
                fp += 1
            else:
                tp += 1
        results[frac] = {"fp": fp, "fn": fn, "tp": tp, "tn": tn}
        print(f"noise_frac={frac:.1f}  FP={fp}  FN={fn}  TP={tp}  TN={tn}")
    return results


def wisdom_reliability(n_steps=250, seed=2):
    """Check whether high-confidence Wisdom entries are actually accurate."""
    print("\n=== Wisdom reliability ===")
    rng = np.random.RandomState(seed)
    system = NoiseSeparatedSystem(promote_hits=3, min_age=4, max_obs=150)
    signal = np.sin(np.linspace(0, 4 * np.pi, 16))
    true_y = 1.0
    for t in range(n_steps):
        if rng.rand() < 0.3:
            system.observe(rng.randn(16), None)
        else:
            x = signal + 0.04 * rng.randn(16)
            system.observe(x, true_y + 0.05 * rng.randn())
    # Inspect wisdom entries
    reliable = unreliable = 0
    for e in system.wisdom.entries:
        if e.confidence >= 0.8:
            err = abs(float(e.y) - true_y)
            if err < 0.3:
                reliable += 1
            else:
                unreliable += 1
    print(f"High-confidence Wisdom: reliable={reliable}  unreliable={unreliable}")
    print(f"Total Wisdom entries: {len(system.wisdom)}")
    return reliable, unreliable


def residual_noise_guard(n_steps=180, max_allowed_fn_rate=0.35, seed=3):
    """Simple regression guard: fail if too much noise reaches Thought."""
    print("\n=== Residual noise guard ===")
    rng = np.random.RandomState(seed)
    system = NoiseSeparatedSystem(noise_threshold=0.35, max_obs=100, explore_prob=0.0)
    noise_to_thought = 0
    noise_total = 0
    for t in range(n_steps):
        if rng.rand() < 0.5:
            route = system.observe(rng.randn(16), None)
            noise_total += 1
            if route != "chaos":
                noise_to_thought += 1
        else:
            system.observe(np.sin(np.linspace(0, 3, 16)) + 0.05 * rng.randn(16), 1.0)
    rate = noise_to_thought / max(noise_total, 1)
    print(f"Noise that reached Thought: {noise_to_thought}/{noise_total} = {rate:.2f}")
    ok = rate <= max_allowed_fn_rate
    print("PASS" if ok else "FAIL")
    return ok, rate


if __name__ == "__main__":
    gate_error_rates()
    wisdom_reliability()
    residual_noise_guard()

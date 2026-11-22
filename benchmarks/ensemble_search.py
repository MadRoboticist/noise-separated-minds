"""Benchmark different subsets of the 7 analytical perspectives.

Runs a simple regression task on structured signals + noise,
measures MSE for every non-empty subset (or a sampled set of subsets),
and reports the best performing combination.
"""

import itertools
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.system import NoiseSeparatedSystem
from src.minds import AnalyticalMind


def make_dataset(n: int = 80, dim: int = 16, noise: float = 0.05, seed: int = 0):
    rng = np.random.RandomState(seed)
    # Structured: y = sin(mean(x)) + small noise
    X = rng.randn(n, dim)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    y = np.sin(X.mean(axis=1) * 3) + noise * rng.randn(n)
    return X, y


def evaluate_subset(names: list, X_train, y_train, X_test, y_test) -> float:
    system = NoiseSeparatedSystem(noise_threshold=0.2)  # lenient so structure passes
    system.analytical.set_enabled(names)
    system.analytical.blender_mode = "weighted"

    for x, yi in zip(X_train, y_train):
        system.observe(x, y=float(yi))

    preds = []
    for x in X_test:
        pred, route = system.predict(x)
        preds.append(pred if pred is not None else 0.0)
    preds = np.array(preds)
    mse = float(np.mean((preds - y_test) ** 2))
    return mse


def main(max_subsets: int = 64):
    print("Generating dataset...")
    X, y = make_dataset()
    split = int(0.7 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    mind = AnalyticalMind()
    all_names = mind.names
    print("Perspectives:", all_names)

    # All non-empty subsets, or sample if too many
    subsets = []
    for r in range(1, len(all_names) + 1):
        subsets.extend(itertools.combinations(all_names, r))
    if len(subsets) > max_subsets:
        rng = np.random.RandomState(1)
        idx = rng.choice(len(subsets), max_subsets, replace=False)
        subsets = [subsets[i] for i in idx]
        print(f"Sampled {max_subsets} subsets out of {2**len(all_names)-1}")

    results = []
    for i, subset in enumerate(subsets):
        mse = evaluate_subset(list(subset), X_train, y_train, X_test, y_test)
        results.append((mse, subset))
        if (i + 1) % 10 == 0:
            print(f"  evaluated {i+1}/{len(subsets)}")

    results.sort(key=lambda t: t[0])
    print("\nTop 5 combinations (lower MSE better):")
    for mse, subset in results[:5]:
        print(f"  MSE={mse:.4f}  {subset}")

    print("\nWorst 3:")
    for mse, subset in results[-3:]:
        print(f"  MSE={mse:.4f}  {subset}")

    best_mse, best = results[0]
    print(f"\nBest: {best}  (MSE={best_mse:.4f})")


if __name__ == "__main__":
    main()

"""Minimal demo of the noise-separated system."""

import numpy as np
import sys
sys.path.insert(0, "..")

from src.system import NoiseSeparatedSystem


def main():
    system = NoiseSeparatedSystem(noise_threshold=0.35, promote_hits=3)

    # Pure noise → chaos
    noise = np.random.randn(32)
    print("Noise ->", system.observe(noise))

    # Structured signal observed several times → eventually reactive
    signal = np.sin(np.linspace(0, 4 * np.pi, 32))
    for i in range(6):
        result = system.observe(signal + 0.005 * np.random.randn(32), y=float(i % 3))
        print(f"Signal pass {i} -> {result}")

    # Query the same signal
    pred, route = system.predict(signal)
    print("Prediction:", pred, "via", route)

    # Slightly different structured input should hit analytical
    variant = np.sin(np.linspace(0.1, 4.1 * np.pi, 32))
    pred2, route2 = system.predict(variant)
    print("Variant prediction:", pred2, "via", route2)

    print("Stats:", system.stats())


if __name__ == "__main__":
    main()

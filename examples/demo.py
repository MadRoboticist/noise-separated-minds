"""Demo of full feedback: promotion, demotion, chaos re-entry."""

import numpy as np
import sys
sys.path.insert(0, "..")

from src.system import NoiseSeparatedSystem


def main():
    system = NoiseSeparatedSystem(
        noise_threshold=0.3,
        promote_hits=3,
        wisdom_strength=1.3,
        explore_prob=0.0,  # disable explore for clean demo
    )

    print("=== Noise ===")
    print(system.observe(np.random.randn(16)))

    print("\n=== Promote a signal ===")
    signal = np.sin(np.linspace(0, 3 * np.pi, 16))
    for i in range(6):
        route = system.observe(signal, y=1.0)
        print(f"  {i}: {route}")

    print("Stats after promotion:", system.stats())

    print("\n=== Contradict it (should demote) ===")
    for i in range(4):
        route = system.observe(signal, y=5.0)  # large error
        print(f"  contradict {i}: {route}")

    print("Stats after contradiction:", system.stats())
    pred, route = system.predict(signal)
    print(f"Prediction now: {pred} via {route}")


if __name__ == "__main__":
    main()

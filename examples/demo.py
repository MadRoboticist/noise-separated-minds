"""Demo of Thought / Knowledge / Wisdom consolidation."""

import numpy as np
import sys
sys.path.insert(0, "..")

from src.system import NoiseSeparatedSystem


def main():
    system = NoiseSeparatedSystem(noise_threshold=0.3, promote_hits=3, wisdom_strength=1.2)

    print("=== Noise ===")
    print(system.observe(np.random.randn(16)))

    print("\n=== Repeated structured signal ===")
    signal = np.sin(np.linspace(0, 3 * np.pi, 16))
    for i in range(8):
        route = system.observe(signal + 0.01 * np.random.randn(16), y=1.0)
        print(f"  pass {i}: {route}  | stats={system.stats()}")

    print("\n=== Query ===")
    pred, route = system.predict(signal)
    print(f"Prediction {pred} via {route}")

    print("\n=== After fading ===")
    for _ in range(30):
        system.knowledge.fade()
    print("Knowledge remaining:", len(system.knowledge))
    print("Wisdom entries (permanent):", len(system.wisdom))
    print("Final stats:", system.stats())


if __name__ == "__main__":
    main()

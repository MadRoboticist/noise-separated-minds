"""Demo including Wisdom confidence checks."""

import numpy as np
import sys
sys.path.insert(0, "..")

from src.system import NoiseSeparatedSystem


def main():
    system = NoiseSeparatedSystem(
        noise_threshold=0.3,
        promote_hits=3,
        wisdom_strength=1.2,
        explore_prob=0.0,
    )

    signal = np.sin(np.linspace(0, 3 * np.pi, 16))

    print("=== Promote to Wisdom ===")
    for i in range(7):
        print(i, system.observe(signal, y=1.0))
    print(system.stats())

    print("\n=== Contradict Wisdom ===")
    for i in range(5):
        print(i, system.observe(signal, y=9.0))
    print(system.stats())

    pred, route = system.predict(signal)
    print(f"After contradiction: {pred} via {route}")
    print("Wisdom entries still exist:", len(system.wisdom),
          "active:", system.wisdom.active_count)


if __name__ == "__main__":
    main()

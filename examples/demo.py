"""Minimal demo of the noise-separated system."""

import numpy as np
import sys
sys.path.insert(0, "..")

from src.system import NoiseSeparatedSystem


def main():
    system = NoiseSeparatedSystem(noise_threshold=0.4)

    # Pure noise
    noise = np.random.randn(32)
    print("Noise ->", system.observe(noise))

    # Structured signal (repeating pattern)
    signal = np.sin(np.linspace(0, 4 * np.pi, 32))
    for i in range(5):
        result = system.observe(signal + 0.01 * np.random.randn(32), y=i)
        print(f"Signal pass {i} -> {result}")

    # Query
    pred, route = system.predict(signal)
    print("Prediction route:", route)


if __name__ == "__main__":
    main()

"""Tests for the adaptive randomness gate."""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.randomness_gate import RandomnessGate


def test_rejects_white_noise():
    gate = RandomnessGate()
    noise = np.random.randn(128)
    assert gate.score(noise) < 0.55


def test_accepts_sinusoid():
    gate = RandomnessGate()
    signal = np.sin(np.linspace(0, 8 * np.pi, 128))
    assert gate.score(signal) > 0.5


def test_threshold_adapts():
    gate = RandomnessGate(base_threshold=0.4, adapt_rate=0.15)
    for _ in range(50):
        gate.update(0.15, was_rejected=True)
    low = gate.threshold
    for _ in range(50):
        gate.update(0.85, was_rejected=False)
    assert gate.threshold > low


def test_residual_rate():
    gate = RandomnessGate()
    for _ in range(10):
        gate.update(0.2, True)
    for _ in range(10):
        gate.update(0.8, False)
    rate = gate.residual_noise_rate()
    assert 0.4 < rate < 0.6


if __name__ == "__main__":
    test_rejects_white_noise()
    test_accepts_sinusoid()
    test_threshold_adapts()
    test_residual_rate()
    print("Adaptive gate tests passed.")

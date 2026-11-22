"""Tests for the strengthened randomness gate."""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.randomness_gate import RandomnessGate


def test_rejects_white_noise():
    gate = RandomnessGate()
    noise = np.random.randn(128)
    assert gate.is_noise(noise, threshold=0.45)
    assert gate.score(noise) < 0.5


def test_accepts_sinusoid():
    gate = RandomnessGate()
    signal = np.sin(np.linspace(0, 8 * np.pi, 128))
    assert not gate.is_noise(signal, threshold=0.45)
    assert gate.score(signal) > 0.55


def test_accepts_smooth_trend():
    gate = RandomnessGate()
    trend = np.linspace(-1, 1, 64) + 0.05 * np.random.randn(64)
    assert gate.score(trend) > gate.score(np.random.randn(64))


def test_detail_keys():
    gate = RandomnessGate()
    d = gate.detail(np.random.randn(32))
    for k in ("spectral_flatness", "histogram_entropy", "autocorr_noise", "structure"):
        assert k in d


if __name__ == "__main__":
    test_rejects_white_noise()
    test_accepts_sinusoid()
    test_accepts_smooth_trend()
    test_detail_keys()
    print("Gate tests passed.")

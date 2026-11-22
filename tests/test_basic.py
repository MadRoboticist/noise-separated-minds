"""Basic tests for the noise-separated system."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.system import NoiseSeparatedSystem
from src.randomness_gate import RandomnessGate


def test_gate_rejects_noise():
    gate = RandomnessGate()
    noise = np.random.randn(64)
    signal = np.sin(np.linspace(0, 6 * np.pi, 64))
    assert gate.is_noise(noise, threshold=0.4)
    assert not gate.is_noise(signal, threshold=0.4)


def test_system_routes_noise_to_chaos():
    system = NoiseSeparatedSystem(noise_threshold=0.4)
    noise = np.random.randn(32)
    route = system.observe(noise)
    assert route == "chaos"
    assert len(system.chaos) == 1
    assert len(system.analytical) == 0


def test_system_promotes_consistent_signal():
    system = NoiseSeparatedSystem(noise_threshold=0.3, promote_hits=3, promote_score=0.6)
    signal = np.sin(np.linspace(0, 4 * np.pi, 32))
    routes = []
    for i in range(6):
        routes.append(system.observe(signal + 1e-4 * np.random.randn(32), y=float(i)))
    assert "promoted_reactive" in routes
    assert len(system.reactive) >= 1


def test_predict_prefers_reactive():
    system = NoiseSeparatedSystem(noise_threshold=0.3, promote_hits=2)
    signal = np.ones(16)
    for _ in range(4):
        system.observe(signal, y=42.0)
    pred, route = system.predict(signal)
    assert route == "reactive"
    assert pred == 42.0


if __name__ == "__main__":
    test_gate_rejects_noise()
    test_system_routes_noise_to_chaos()
    test_system_promotes_consistent_signal()
    test_predict_prefers_reactive()
    print("All basic tests passed.")

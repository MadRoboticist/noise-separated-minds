"""Top-level unsupervised system: gate → short-term → clerk → long-term minds."""

import numpy as np
from .randomness_gate import RandomnessGate
from .short_term_buffer import ShortTermBuffer
from .minds import ReactiveMind, AnalyticalMind, ChaosPile


class NoiseSeparatedSystem:
    def __init__(self, noise_threshold: float = 0.35):
        self.gate = RandomnessGate()
        self.short_term = ShortTermBuffer()
        self.reactive = ReactiveMind()
        self.analytical = AnalyticalMind()
        self.chaos = ChaosPile()
        self.noise_threshold = noise_threshold

    def observe(self, x: np.ndarray, y=None):
        """Ingest one observation. Pure noise is rejected immediately."""
        x = np.asarray(x)
        score = self.gate.score(x)

        if score < self.noise_threshold:
            self.chaos.reject(x)
            return "chaos"

        # Structured enough to enter short-term
        existing = self.short_term.find_similar(x)
        if existing is not None and existing["hits"] >= 3 and score > 0.8:
            # High consistency → candidate for reactive
            self.reactive.store(x, y if y is not None else existing.get("output"))
            return "promoted_reactive"

        self.short_term.add(x, score, y)
        self.analytical.observe(x, y)
        return "analytical"

    def predict(self, x: np.ndarray):
        """Route a query."""
        x = np.asarray(x)
        # Try reactive first
        hit = self.reactive.lookup(x)
        if hit is not None:
            return hit, "reactive"

        score = self.gate.score(x)
        if score < self.noise_threshold:
            return self.chaos.sample(x.shape), "chaos"

        return self.analytical.sample(x), "analytical"

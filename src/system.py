"""Top-level unsupervised system: gate → short-term → clerk → long-term minds."""

import numpy as np
from .randomness_gate import RandomnessGate
from .short_term_buffer import ShortTermBuffer
from .minds import ReactiveMind, AnalyticalMind, ChaosPile


class NoiseSeparatedSystem:
    def __init__(
        self,
        noise_threshold: float = 0.35,
        promote_hits: int = 4,
        promote_score: float = 0.75,
        demote_uncertainty: float = 0.8,
    ):
        self.gate = RandomnessGate()
        self.short_term = ShortTermBuffer()
        self.reactive = ReactiveMind()
        self.analytical = AnalyticalMind()
        self.chaos = ChaosPile()
        self.noise_threshold = noise_threshold
        self.promote_hits = promote_hits
        self.promote_score = promote_score
        self.demote_uncertainty = demote_uncertainty

    def observe(self, x: np.ndarray, y=None):
        """Ingest one observation. Pure noise is rejected immediately."""
        x = np.asarray(x)
        score = self.gate.score(x)

        if score < self.noise_threshold:
            self.chaos.reject(x)
            return "chaos"

        # Structured enough for short-term / analytical
        existing = self.short_term.find_similar(x)
        if (
            existing is not None
            and existing["hits"] >= self.promote_hits
            and score >= self.promote_score
        ):
            # High consistency + high structure → promote to reactive
            stored_y = y if y is not None else existing.get("output")
            self.reactive.store(x, stored_y)
            return "promoted_reactive"

        self.short_term.add(x, score, y)
        self.analytical.observe(x, y)
        return "analytical"

    def predict(self, x: np.ndarray):
        """Route a query. Demote reactive hits that now look uncertain."""
        x = np.asarray(x)

        # Reactive first
        hit = self.reactive.lookup(x)
        if hit is not None:
            # Optional demotion check: if analytical now sees high uncertainty, demote
            unc = self.analytical.uncertainty(x)
            if unc > self.demote_uncertainty:
                # Could remove from reactive here; for now just fall through
                pass
            else:
                return hit, "reactive"

        score = self.gate.score(x)
        if score < self.noise_threshold:
            return self.chaos.sample(getattr(x, "shape", (1,))), "chaos"

        return self.analytical.sample(x), "analytical"

    def stats(self) -> dict:
        return {
            "reactive": len(self.reactive),
            "analytical": len(self.analytical),
            "chaos": len(self.chaos),
            "short_term": len(self.short_term.buffer),
        }

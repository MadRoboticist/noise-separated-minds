"""Top-level system with Thought / Knowledge / Wisdom consolidation."""

import numpy as np
from .randomness_gate import RandomnessGate
from .short_term_buffer import ShortTermBuffer
from .minds import ReactiveMind, ChaosPile
from .layers import ThoughtLayer, KnowledgeLayer, WisdomLayer


class NoiseSeparatedSystem:
    def __init__(
        self,
        noise_threshold: float = 0.35,
        promote_hits: int = 4,
        promote_score: float = 0.75,
        knowledge_boost: float = 0.4,
        wisdom_strength: float = 1.5,
    ):
        self.gate = RandomnessGate()
        self.short_term = ShortTermBuffer()
        self.reactive = ReactiveMind()
        self.thought = ThoughtLayer()          # figuring / can be lost
        self.knowledge = KnowledgeLayer()      # can fade
        self.wisdom = WisdomLayer()            # only grows
        self.chaos = ChaosPile()

        self.noise_threshold = noise_threshold
        self.promote_hits = promote_hits
        self.promote_score = promote_score
        self.knowledge_boost = knowledge_boost
        self.wisdom_strength = wisdom_strength

    def observe(self, x: np.ndarray, y=None):
        """Ingest. Noise → chaos. Structured → Thought. Promote on consistency."""
        x = np.asarray(x)
        score = self.gate.score(x)

        if score < self.noise_threshold:
            self.chaos.reject(x)
            return "chaos"

        # Always enter Thought (analytical ensemble)
        self.thought.observe(x, y)
        self.short_term.add(x, score, y)

        # Consistency check for promotion
        existing = self.short_term.find_similar(x)
        if existing is not None and existing["hits"] >= self.promote_hits and score >= self.promote_score:
            stored_y = y if y is not None else existing.get("output")
            # Promote toward Knowledge
            self.knowledge.add_or_reinforce(x, stored_y, boost=self.knowledge_boost)
            # If already strong in Knowledge, consider Wisdom
            for item in self.knowledge.strongest(3):
                if np.allclose(item.x, np.asarray(x).ravel(), atol=0.15) and item.strength >= self.wisdom_strength:
                    self.wisdom.add(x, stored_y, confidence=item.strength)
                    self.reactive.store(x, stored_y)  # also make reactive
                    return "promoted_wisdom"
            return "promoted_knowledge"

        # Periodic fade of Knowledge
        if len(self.thought) % 10 == 0:
            self.knowledge.fade()

        return "thought"

    def predict(self, x: np.ndarray):
        """Query order: Wisdom → Reactive → Knowledge → Thought → Chaos."""
        x = np.asarray(x)

        # Wisdom first (permanent)
        w = self.wisdom.lookup(x)
        if w is not None:
            return w, "wisdom"

        # Reactive
        r = self.reactive.lookup(x)
        if r is not None:
            return r, "reactive"

        # Knowledge (may have faded)
        k = self.knowledge.lookup(x)
        if k is not None:
            return k, "knowledge"

        score = self.gate.score(x)
        if score < self.noise_threshold:
            return self.chaos.sample(getattr(x, "shape", (1,))), "chaos"

        # Thought (ensemble)
        return self.thought.sample(x), "thought"

    def stats(self) -> dict:
        return {
            "thought": len(self.thought),
            "knowledge": len(self.knowledge),
            "wisdom": len(self.wisdom),
            "reactive": len(self.reactive),
            "chaos": len(self.chaos),
            "short_term": len(self.short_term.buffer),
        }

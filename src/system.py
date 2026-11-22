"""Top-level system with full feedback: promotion, demotion, chaos re-entry, exploratory routing."""

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
        explore_prob: float = 0.05,
        demote_error_tol: float = 0.5,
    ):
        self.gate = RandomnessGate()
        self.short_term = ShortTermBuffer()
        self.reactive = ReactiveMind()
        self.thought = ThoughtLayer()
        self.knowledge = KnowledgeLayer()
        self.wisdom = WisdomLayer()
        self.chaos = ChaosPile()

        self.noise_threshold = noise_threshold
        self.promote_hits = promote_hits
        self.promote_score = promote_score
        self.knowledge_boost = knowledge_boost
        self.wisdom_strength = wisdom_strength
        self.explore_prob = explore_prob
        self.demote_error_tol = demote_error_tol

        self._step = 0

    def observe(self, x: np.ndarray, y=None):
        """Ingest with exploratory routing and full promotion path."""
        x = np.asarray(x)
        self._step += 1
        score = self.gate.score(x)

        # Occasional exploratory send to chaos even if structured
        if score >= self.noise_threshold and np.random.rand() < self.explore_prob:
            self.chaos.reject(x)
            # Still allow a copy into thought for comparison
            self.thought.observe(x, y)
            return "explored_chaos"

        if score < self.noise_threshold:
            self.chaos.reject(x)
            return "chaos"

        # Normal path: Thought
        self.thought.observe(x, y)
        self.short_term.add(x, score, y)

        # Promotion check
        existing = self.short_term.find_similar(x)
        if existing is not None and existing["hits"] >= self.promote_hits and score >= self.promote_score:
            stored_y = y if y is not None else existing.get("output")
            self.knowledge.add_or_reinforce(x, stored_y, boost=self.knowledge_boost)

            # Wisdom if strong enough
            for item in self.knowledge.strongest(5):
                if np.allclose(item.x, np.asarray(x).ravel(), atol=0.15) and item.strength >= self.wisdom_strength:
                    self.wisdom.add(x, stored_y, confidence=item.strength)
                    self.reactive.store(x, stored_y)
                    return "promoted_wisdom"
            return "promoted_knowledge"

        # Periodic maintenance
        if self._step % 8 == 0:
            self.knowledge.fade()
            self._try_chaos_reentry()
            self._check_demotions(x, y)

        return "thought"

    def _try_chaos_reentry(self):
        """If a previously rejected item now looks structured, pull it into Thought."""
        if not self.chaos.rejected:
            return
        # Sample a few recent rejects
        candidates = self.chaos.rejected[-5:]
        for cx in candidates:
            if self.gate.score(cx) >= self.noise_threshold + 0.1:
                # Re-enter as thought
                self.thought.observe(cx, y=None)
                # Remove from chaos (simple: just mark by not re-adding; full removal optional)
                return "chaos_reentry"

    def _check_demotions(self, x, y):
        """Demote reactive/knowledge when prediction error is high."""
        if y is None:
            return
        # Check reactive
        r = self.reactive.lookup(x)
        if r is not None and abs(float(r) - float(y)) > self.demote_error_tol:
            # Demote: remove from reactive, weaken knowledge
            key = self.reactive._key(x)
            if key in self.reactive.table:
                del self.reactive.table[key]
            self.knowledge.add_or_reinforce(x, y, boost=-0.5)  # weaken
            self.thought.observe(x, y)  # force back into figuring
            return "demoted_reactive"

        # Check knowledge
        k = self.knowledge.lookup(x)
        if k is not None and abs(float(k) - float(y)) > self.demote_error_tol:
            self.knowledge.add_or_reinforce(x, y, boost=-0.4)
            return "demoted_knowledge"

    def predict(self, x: np.ndarray):
        x = np.asarray(x)

        w = self.wisdom.lookup(x)
        if w is not None:
            return w, "wisdom"

        r = self.reactive.lookup(x)
        if r is not None:
            return r, "reactive"

        k = self.knowledge.lookup(x)
        if k is not None:
            return k, "knowledge"

        score = self.gate.score(x)
        if score < self.noise_threshold:
            return self.chaos.sample(getattr(x, "shape", (1,))), "chaos"

        return self.thought.sample(x), "thought"

    def stats(self) -> dict:
        return {
            "thought": len(self.thought),
            "knowledge": len(self.knowledge),
            "wisdom": len(self.wisdom),
            "reactive": len(self.reactive),
            "chaos": len(self.chaos),
            "short_term": len(self.short_term.buffer),
            "step": self._step,
        }

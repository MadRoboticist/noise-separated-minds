"""System with agreement-gated promotion and accuracy-aware Thought."""

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
        min_agreement: float = 0.6,
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
        self.min_agreement = min_agreement
        self.knowledge_boost = knowledge_boost
        self.wisdom_strength = wisdom_strength
        self.explore_prob = explore_prob
        self.demote_error_tol = demote_error_tol
        self._step = 0

    def observe(self, x: np.ndarray, y=None):
        x = np.asarray(x)
        self._step += 1
        score = self.gate.score(x)

        if y is not None:
            if not self.wisdom.check(x, y, tol=self.demote_error_tol):
                self.thought.observe(x, y)
                self.knowledge.add_or_reinforce(x, y, boost=-0.3)

        if score >= self.noise_threshold and np.random.rand() < self.explore_prob:
            self.chaos.reject(x)
            self.thought.observe(x, y)
            return "explored_chaos"

        if score < self.noise_threshold:
            self.chaos.reject(x)
            return "chaos"

        self.thought.observe(x, y)
        self.short_term.add(x, score, y)

        existing = self.short_term.find_similar(x)
        agreement = self.thought.mind.agreement(x)
        if (
            existing is not None
            and existing["hits"] >= self.promote_hits
            and score >= self.promote_score
            and agreement >= self.min_agreement
        ):
            stored_y = y if y is not None else existing.get("output")
            self.knowledge.add_or_reinforce(x, stored_y, boost=self.knowledge_boost)

            for item in self.knowledge.strongest(5):
                if np.allclose(item.x, np.asarray(x).ravel(), atol=0.15) and item.strength >= self.wisdom_strength:
                    self.wisdom.add(x, stored_y, confidence=item.strength)
                    self.reactive.store(x, stored_y)
                    return "promoted_wisdom"
            return "promoted_knowledge"

        if self._step % 8 == 0:
            self.knowledge.fade()
            self._try_chaos_reentry()
            self._check_demotions(x, y)

        return "thought"

    def _try_chaos_reentry(self):
        if not self.chaos.rejected:
            return
        for cx in self.chaos.rejected[-5:]:
            if self.gate.score(cx) >= self.noise_threshold + 0.1:
                self.thought.observe(cx, y=None)
                return

    def _check_demotions(self, x, y):
        if y is None:
            return
        r = self.reactive.lookup(x)
        if r is not None and abs(float(r) - float(y)) > self.demote_error_tol:
            key = self.reactive._key(x)
            self.reactive.table.pop(key, None)
            self.knowledge.add_or_reinforce(x, y, boost=-0.5)
            self.thought.observe(x, y)
            return
        k = self.knowledge.lookup(x)
        if k is not None and abs(float(k) - float(y)) > self.demote_error_tol:
            self.knowledge.add_or_reinforce(x, y, boost=-0.4)

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
        if self.gate.score(x) < self.noise_threshold:
            return self.chaos.sample(getattr(x, "shape", (1,))), "chaos"
        return self.thought.sample(x), "thought"

    def stats(self) -> dict:
        return {
            "thought": len(self.thought),
            "knowledge": len(self.knowledge),
            "wisdom": len(self.wisdom),
            "wisdom_active": self.wisdom.active_count,
            "reactive": len(self.reactive),
            "chaos": len(self.chaos),
            "short_term": len(self.short_term.buffer),
            "step": self._step,
        }

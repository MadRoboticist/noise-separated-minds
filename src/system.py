"""System with generative hypotheses influencing promotion."""

import numpy as np
from .randomness_gate import RandomnessGate
from .short_term_buffer import ShortTermBuffer
from .minds import ReactiveMind, ChaosPile
from .layers import ThoughtLayer, KnowledgeLayer, WisdomLayer
from .persistence import save as _save, load as _load
from .auditor import Auditor
from .hypotheses import HypothesisTracker


class NoiseSeparatedSystem:
    def __init__(
        self,
        noise_threshold: float = 0.35,
        promote_hits: int = 4,
        promote_score: float = 0.75,
        min_agreement: float = 0.6,
        min_age: int = 8,
        knowledge_boost: float = 0.4,
        wisdom_strength: float = 1.5,
        explore_prob: float = 0.05,
        explore_min: float = 0.01,
        explore_max: float = 0.2,
        demote_error_tol: float = 0.5,
        max_obs: int = 2_000,
        max_chaos: int = 5_000,
        chaos_promote_budget: float = 0.15,
        revalidate_interval: int = 40,
        audit_every: int = 20,
    ):
        self.gate = RandomnessGate(base_threshold=noise_threshold)
        self.short_term = ShortTermBuffer()
        self.reactive = ReactiveMind()
        self.thought = ThoughtLayer()
        for p in self.thought.mind.perspectives:
            p.max_obs = max_obs
        self.knowledge = KnowledgeLayer()
        self.wisdom = WisdomLayer()
        self.chaos = ChaosPile(max_rejected=max_chaos)
        self.auditor = Auditor(error_tol=demote_error_tol)
        self.hypotheses = HypothesisTracker()

        self.promote_hits = promote_hits
        self.promote_score = promote_score
        self.min_agreement = min_agreement
        self.min_age = min_age
        self.knowledge_boost = knowledge_boost
        self.wisdom_strength = wisdom_strength
        self.explore_prob = explore_prob
        self.explore_min = explore_min
        self.explore_max = explore_max
        self.demote_error_tol = demote_error_tol
        self.chaos_promote_budget = chaos_promote_budget
        self.revalidate_interval = revalidate_interval
        self.audit_every = audit_every
        self._step = 0
        self._reentries = 0
        self._chaos_promotions = 0
        self._recent_agreements = []

    def save(self, path: str):
        _save(self, path)

    @staticmethod
    def load(path: str):
        return _load(path)

    def _adapt_exploration(self, agreement: float):
        self._recent_agreements.append(agreement)
        if len(self._recent_agreements) > 20:
            self._recent_agreements.pop(0)
        avg_agree = float(np.mean(self._recent_agreements)) if self._recent_agreements else 0.5
        if avg_agree < 0.5:
            self.explore_prob = min(self.explore_max, self.explore_prob * 1.05 + 0.005)
        else:
            self.explore_prob = max(self.explore_min, self.explore_prob * 0.98)
        if self._reentries > 0 and self._step % 30 == 0:
            self.explore_prob = min(self.explore_max, self.explore_prob + 0.01)
            self._reentries = max(0, self._reentries - 1)

    def _chaos_budget_ok(self) -> bool:
        if self._step < 10:
            return True
        return (self._chaos_promotions / max(self._step, 1)) < self.chaos_promote_budget

    def _propose_hypothesis(self, x):
        """Draw a generative sample and register it as a hypothesis."""
        y_hat = self.thought.mind.sample_generative(x)
        if y_hat is not None:
            self.hypotheses.propose(x, y_hat, self._step)

    def observe(self, x: np.ndarray, y=None):
        x = np.asarray(x)
        self._step += 1
        score = self.gate.score(x)
        is_noise = score < self.gate.threshold
        self.gate.update(score, is_noise)

        # Update existing hypotheses with this observation
        self.hypotheses.observe(x, y, self._step)

        if y is not None:
            if not self.wisdom.check(x, y, tol=self.demote_error_tol, step=self._step):
                self.thought.observe(x, y)
                self.knowledge.add_or_reinforce(x, y, boost=-0.3, step=self._step)

        agreement = self.thought.mind.agreement(x) if len(self.thought) > 2 else 0.5
        self._adapt_exploration(agreement)

        thr = self.gate.threshold
        if score >= thr and np.random.rand() < self.explore_prob:
            self.chaos.reject(x)
            self.thought.observe(x, y)
            self._propose_hypothesis(x)
            return "explored_chaos"

        if is_noise:
            self.chaos.reject(x)
            return "chaos"

        self.thought.observe(x, y)
        self.short_term.add(x, score, y)
        # Occasionally propose a generative hypothesis
        if self._step % 5 == 0:
            self._propose_hypothesis(x)

        existing = self.short_term.find_similar(x)
        item = self.knowledge.find_item(x)
        age_ok = item is not None and (self._step - item.first_seen) >= self.min_age
        hyp_score = self.hypotheses.confirmation_score(x)

        # Hypothesis confirmation can relax the hit requirement slightly
        hits_needed = self.promote_hits
        if hyp_score > 0.5:
            hits_needed = max(2, self.promote_hits - 1)

        if (
            existing is not None
            and existing["hits"] >= hits_needed
            and score >= self.promote_score
            and agreement >= self.min_agreement
            and self._chaos_budget_ok()
        ):
            stored_y = y if y is not None else existing.get("output")
            boost = self.knowledge_boost + 0.2 * hyp_score
            self.knowledge.add_or_reinforce(x, stored_y, boost=boost, step=self._step)
            if age_ok:
                item = self.knowledge.find_item(x)
                if item and item.strength >= self.wisdom_strength:
                    self.wisdom.add(x, stored_y, confidence=item.strength, step=self._step)
                    self.reactive.store(x, stored_y)
                    return "promoted_wisdom"
            return "promoted_knowledge"

        if self._step % 6 == 0:
            self.knowledge.fade()
            if self._active_chaos_probe():
                self._reentries += 1
                self._chaos_promotions += 1
            self._check_demotions(x, y)
            self._revalidate_wisdom()

        if self._step % self.audit_every == 0:
            findings = self.auditor.inspect(self)
            self.auditor.apply(self, findings)

        return "thought"

    def _active_chaos_probe(self) -> bool:
        candidates = self.chaos.probe(n=4)
        for cx in candidates:
            if self.gate.score(cx) >= self.gate.threshold + 0.02:
                self.thought.observe(cx, y=None)
                self.chaos.record_probe_success()
                return True
        return False

    def _revalidate_wisdom(self):
        due = self.wisdom.due_for_revalidation(self._step, self.revalidate_interval)
        for e in due[:2]:
            self.thought.observe(e.x, e.y)
            e.last_checked = self._step

    def _check_demotions(self, x, y):
        if y is None:
            return
        r = self.reactive.lookup(x)
        if r is not None and abs(float(r) - float(y)) > self.demote_error_tol:
            key = self.reactive._key(x)
            self.reactive.table.pop(key, None)
            self.knowledge.add_or_reinforce(x, y, boost=-0.5, step=self._step)
            self.thought.observe(x, y)
            return
        k = self.knowledge.lookup(x)
        if k is not None and abs(float(k) - float(y)) > self.demote_error_tol:
            self.knowledge.add_or_reinforce(x, y, boost=-0.4, step=self._step)

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
        if self.gate.score(x) < self.gate.threshold:
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
            "hypotheses_active": self.hypotheses.active_count(),
            "auditor_findings": len(self.auditor.findings),
            "explore_prob": round(self.explore_prob, 4),
            "gate_threshold": round(self.gate.threshold, 3),
            "step": self._step,
        }

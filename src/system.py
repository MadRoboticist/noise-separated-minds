"""File clerk with trace log for the procedural auditor."""

import numpy as np
from typing import Dict, List, Any, Optional
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
        audit_every: int = 25,
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
        self.auditor = Auditor()
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
        # Trace: key -> list of events
        self._traces: Dict[str, List[Dict[str, Any]]] = {}
        self._recent_inputs: List[np.ndarray] = []

    def _trace_key(self, x: np.ndarray) -> str:
        return str(np.round(np.asarray(x).ravel()[:4], 3).tolist())

    def _record(self, x: np.ndarray, event: Dict[str, Any]):
        key = self._trace_key(x)
        self._traces.setdefault(key, []).append({"step": self._step, **event})
        if len(self._traces[key]) > 20:
            self._traces[key] = self._traces[key][-20:]
        self._recent_inputs.append(np.asarray(x))
        if len(self._recent_inputs) > 100:
            self._recent_inputs.pop(0)

    def get_trace(self, key: str) -> List[Dict[str, Any]]:
        return self._traces.get(key, [])

    # --- re-file methods the auditor may call ---
    def refile_demote(self, key: str):
        # Remove matching reactive entries and weaken knowledge
        # (simplified: clear a few reactive keys)
        for k in list(self.reactive.table.keys())[:3]:
            self.reactive.table.pop(k, None)

    def refile_weaken_wisdom(self, key: str):
        for e in self.wisdom.entries:
            e.confidence = max(0.0, e.confidence - 0.3)

    def refile_weaken_knowledge(self, key: str):
        self.knowledge.fade()

    def refile_to_thought(self, key: str):
        # No-op placeholder; real re-injection would need the original x
        pass

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

    def _chaos_budget_ok(self) -> bool:
        if self._step < 10:
            return True
        return (self._chaos_promotions / max(self._step, 1)) < self.chaos_promote_budget

    def _propose_hypothesis(self, x):
        y_hat = self.thought.mind.sample_generative(x)
        if y_hat is not None:
            self.hypotheses.propose(x, y_hat, self._step)

    def observe(self, x: np.ndarray, y=None):
        x = np.asarray(x)
        self._step += 1
        score = self.gate.score(x)
        is_noise = score < self.gate.threshold
        self.gate.update(score, is_noise)

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
            self._record(x, {"kind": "explored_chaos"})
            return "explored_chaos"

        if is_noise:
            self.chaos.reject(x)
            self._record(x, {"kind": "routed_chaos", "score": score})
            return "chaos"

        self.thought.observe(x, y)
        self.short_term.add(x, score, y)
        self._record(x, {"kind": "to_thought", "score": score, "y": y})

        if self._step % 5 == 0:
            self._propose_hypothesis(x)

        existing = self.short_term.find_similar(x)
        item = self.knowledge.find_item(x)
        age_ok = item is not None and (self._step - item.first_seen) >= self.min_age
        hyp_score = self.hypotheses.confirmation_score(x)
        hits_needed = self.promote_hits - (1 if hyp_score > 0.5 else 0)

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
            self._record(x, {"kind": "promoted_knowledge", "y": stored_y})
            if age_ok:
                item = self.knowledge.find_item(x)
                if item and item.strength >= self.wisdom_strength:
                    self.wisdom.add(x, stored_y, confidence=item.strength, step=self._step)
                    self.reactive.store(x, stored_y)
                    self._record(x, {"kind": "promoted_wisdom", "y": stored_y})
                    return "promoted_wisdom"
            return "promoted_knowledge"

        if self._step % 6 == 0:
            self.knowledge.fade()
            if self._active_chaos_probe():
                self._reentries += 1
                self._chaos_promotions += 1
            self._check_demotions(x, y)
            self._revalidate_wisdom()

        # Procedural audit on a recent input
        if self._step % self.audit_every == 0 and self._recent_inputs and y is not None:
            suspect = self._recent_inputs[-1]
            finding = self.auditor.audit_input(self, suspect, observed_y=y)
            if finding:
                self.auditor.apply(self, finding)

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
            self._record(x, {"kind": "predict", "pred": w, "via": "wisdom"})
            return w, "wisdom"
        r = self.reactive.lookup(x)
        if r is not None:
            self._record(x, {"kind": "predict", "pred": r, "via": "reactive"})
            return r, "reactive"
        k = self.knowledge.lookup(x)
        if k is not None:
            self._record(x, {"kind": "predict", "pred": k, "via": "knowledge"})
            return k, "knowledge"
        if self.gate.score(x) < self.gate.threshold:
            return self.chaos.sample(getattr(x, "shape", (1,))), "chaos"
        pred = self.thought.sample(x)
        self._record(x, {"kind": "predict", "pred": pred, "via": "thought"})
        return pred, "thought"

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
            "traces": len(self._traces),
            "step": self._step,
        }

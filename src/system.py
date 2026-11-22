"""File clerk with richer audit export."""

import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict
from .randomness_gate import RandomnessGate
from .short_term_buffer import ShortTermBuffer
from .minds import ReactiveMind, ChaosPile
from .layers import ThoughtLayer, KnowledgeLayer, WisdomLayer
from .persistence import save as _save, load as _load
from .auditor import Auditor
from .hypotheses import HypothesisTracker
from .config import SystemConfig


class NoiseSeparatedSystem:
    def __init__(self, config: Optional[SystemConfig] = None, **kwargs):
        cfg = config or SystemConfig()
        for k, v in kwargs.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        self.cfg = cfg

        self.gate = RandomnessGate(base_threshold=cfg.noise_threshold)
        self.short_term = ShortTermBuffer()
        self.reactive = ReactiveMind()
        self.thought = ThoughtLayer()
        for p in self.thought.mind.perspectives:
            p.max_obs = cfg.max_obs
        self.knowledge = KnowledgeLayer()
        self.wisdom = WisdomLayer()
        self.chaos = ChaosPile(max_rejected=cfg.max_chaos)
        self.auditor = Auditor()
        self.hypotheses = HypothesisTracker()

        self.promote_hits = cfg.promote_hits
        self.promote_score = cfg.promote_score
        self.min_agreement = cfg.min_agreement
        self.min_age = cfg.min_age
        self.knowledge_boost = cfg.knowledge_boost
        self.wisdom_strength = cfg.wisdom_strength
        self.explore_prob = cfg.explore_prob
        self.explore_min = cfg.explore_min
        self.explore_max = cfg.explore_max
        self.demote_error_tol = cfg.demote_error_tol
        self.chaos_promote_budget = cfg.chaos_promote_budget
        self.revalidate_interval = cfg.revalidate_interval
        self.audit_every = cfg.audit_every

        self._step = 0
        self._reentries = 0
        self._chaos_promotions = 0
        self._recent_agreements = []
        self._traces: Dict[str, List[Dict[str, Any]]] = {}
        self._recent_good: List[np.ndarray] = []
        self.event_log: List[Dict[str, Any]] = []
        self._calib: List[tuple] = []
        self.temperature = 1.0
        self._wisdom_rel: List[tuple] = []
        self._unlearn_details: List[Dict[str, Any]] = []

    def _trace_key(self, x: np.ndarray) -> str:
        return str(np.round(np.asarray(x).ravel()[:6], 3).tolist())

    def _log(self, kind: str, **kwargs):
        entry = {"step": self._step, "kind": kind, **kwargs}
        self.event_log.append(entry)
        if len(self.event_log) > 2000:
            self.event_log = self.event_log[-2000:]
        return entry

    def _record(self, x: np.ndarray, event: Dict[str, Any]):
        key = self._trace_key(x)
        full = {"step": self._step, **event}
        self._traces.setdefault(key, []).append(full)
        if len(self._traces[key]) > 40:
            self._traces[key] = self._traces[key][-40:]
        self._log(**event, key=key)

    def get_trace(self, key: str) -> List[Dict[str, Any]]:
        return self._traces.get(key, [])

    def unlearn(self, key: str):
        dropped_reactive = []
        to_drop = [k for k in list(self.reactive.table.keys()) if key[:12] in str(k)]
        for k in to_drop[:5]:
            dropped_reactive.append(str(k)[:40])
            self.reactive.table.pop(k, None)

        weakened_k = 0
        for item in self.knowledge.items:
            if self._trace_key(item.x) == key:
                item.strength *= 0.3
                weakened_k += 1

        weakened_w = 0
        for e in self.wisdom.entries:
            if self._trace_key(e.x) == key:
                e.confidence = max(0.0, e.confidence - 0.6)
                weakened_w += 1

        deactivated_h = 0
        for h in self.hypotheses.hypotheses:
            if h.active and self._trace_key(h.x) == key:
                h.active = False
                h.rejections += 3
                deactivated_h += 1

        self.short_term.buffer = [
            b for b in self.short_term.buffer
            if self._trace_key(b.get("x", np.zeros(1))) != key
        ]

        detail = {
            "step": self._step,
            "key": key,
            "dropped_reactive": dropped_reactive,
            "weakened_knowledge": weakened_k,
            "weakened_wisdom": weakened_w,
            "deactivated_hypotheses": deactivated_h,
        }
        self._unlearn_details.append(detail)
        self._log("unlearn_surgical", **detail)

    def refile_to_thought(self, key: str):
        for x in reversed(self._recent_good):
            if self._trace_key(x) == key:
                self.thought.observe(x, y=None)
                self._record(x, {"kind": "refiled_to_thought"})
                return

    def record_calibration(self, uncertainty: float, correct: bool):
        bin_id = int(np.clip(uncertainty * 5, 0, 4))
        self._calib.append((bin_id, correct))
        if len(self._calib) > 500:
            self._calib = self._calib[-500:]
        if len(self._calib) > 30 and self._step % 15 == 0:
            summary = self.calibration_summary()
            high = summary.get(4, summary.get(3, 0.5))
            if high < 0.4:
                self.temperature = min(2.5, self.temperature * 1.05)
            elif high > 0.7:
                self.temperature = max(0.5, self.temperature * 0.98)

    def calibration_summary(self) -> dict:
        if not self._calib:
            return {}
        bins = defaultdict(list)
        for b, c in self._calib:
            bins[b].append(c)
        return {b: round(float(np.mean(v)), 3) for b, v in sorted(bins.items())}

    def calibrated_uncertainty(self, raw: float) -> float:
        return float(raw * self.temperature)

    def record_wisdom_reliability(self, confidence: float, correct: bool):
        bin_id = int(np.clip(confidence, 0, 4))
        self._wisdom_rel.append((bin_id, correct))
        if len(self._wisdom_rel) > 300:
            self._wisdom_rel = self._wisdom_rel[-300:]

    def wisdom_reliability(self) -> dict:
        if not self._wisdom_rel:
            return {}
        bins = defaultdict(list)
        for b, c in self._wisdom_rel:
            bins[b].append(c)
        return {b: round(float(np.mean(v)), 3) for b, v in sorted(bins.items())}

    def export_audit_trail(self, path: str):
        with open(path, "w") as f:
            f.write("# Audit trail\n\n")
            f.write("## Findings\n")
            for finding in self.auditor.findings[-40:]:
                f.write(f"- key={finding.input_key} at={finding.disconnect_at}: {finding.detail}\n")
            f.write("\n## Surgical unlearns\n")
            for d in self._unlearn_details[-30:]:
                f.write(
                    f"step {d['step']} key={d['key']} "
                    f"reactive_dropped={len(d['dropped_reactive'])} "
                    f"knowledge_weakened={d['weakened_knowledge']} "
                    f"wisdom_weakened={d['weakened_wisdom']} "
                    f"hypotheses_deactivated={d['deactivated_hypotheses']}\n"
                )

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
            self._log("hypothesis_proposed", y=y_hat)

    def observe(self, x: np.ndarray, y=None):
        x = np.asarray(x)
        self._step += 1
        score = self.gate.score(x)
        is_noise = score < self.gate.threshold
        self.gate.update(score, is_noise)

        active_names = self.thought.mind.active_names
        self.hypotheses.observe(x, y, self._step)

        if y is not None:
            for h in self.hypotheses.hypotheses:
                if not h.active and h.rejections >= 2 and self._trace_key(h.x) == self._trace_key(x):
                    finding = self.auditor.audit_input(self, x, observed_y=y)
                    if finding:
                        self.auditor.apply(self, finding)
                        self._log("hypothesis_triggered_audit")

        if y is not None:
            for e in self.wisdom.entries:
                if np.linalg.norm(e.x - np.asarray(x).ravel()) < 0.2:
                    correct = abs(float(e.y) - float(y)) < self.demote_error_tol
                    self.record_wisdom_reliability(e.confidence, correct)
            if not self.wisdom.check(x, y, tol=self.demote_error_tol, step=self._step):
                self.thought.observe(x, y)
                self.knowledge.add_or_reinforce(x, y, boost=-0.3, step=self._step)

        agreement = self.thought.mind.agreement(x) if len(self.thought) > 2 else 0.5
        self._adapt_exploration(agreement)

        if y is not None and len(self.thought) > 3:
            unc = self.calibrated_uncertainty(self.thought.uncertainty(x))
            pred = self.thought.sample(x)
            if pred is not None:
                self.record_calibration(unc, abs(float(pred) - float(y)) < self.demote_error_tol)

        thr = self.gate.threshold
        if not is_noise:
            self._recent_good.append(x.copy())
            if len(self._recent_good) > 50:
                self._recent_good.pop(0)

        context = {
            "score": round(score, 3),
            "agreement": round(agreement, 3),
            "active": active_names[:3],
            "temp": round(self.temperature, 2),
        }

        if score >= thr and np.random.rand() < self.explore_prob:
            self.chaos.reject(x)
            self.thought.observe(x, y)
            self._propose_hypothesis(x)
            self._record(x, {"kind": "explored_chaos", **context})
            return "explored_chaos"

        if is_noise:
            self.chaos.reject(x)
            self._record(x, {"kind": "routed_chaos", **context})
            return "chaos"

        self.thought.observe(x, y)
        self.short_term.add(x, score, y)
        self._record(x, {"kind": "to_thought", "y": y, **context})

        if self._step % 5 == 0:
            self._propose_hypothesis(x)

        existing = self.short_term.find_similar(x)
        item = self.knowledge.find_item(x)
        age_ok = item is not None and (self._step - item.first_seen) >= self.min_age
        hyp_score = self.hypotheses.confirmation_score(x)
        hits_needed = max(2, self.promote_hits - (1 if hyp_score > 0.5 else 0))

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
            self._record(x, {"kind": "promoted_knowledge", "y": stored_y, **context})
            if age_ok:
                item = self.knowledge.find_item(x)
                if item and item.strength >= self.wisdom_strength:
                    self.wisdom.add(x, stored_y, confidence=item.strength, step=self._step)
                    self.reactive.store(x, stored_y)
                    self._record(x, {"kind": "promoted_wisdom", "y": stored_y, **context})
                    return "promoted_wisdom"
            return "promoted_knowledge"

        if self._step % 6 == 0:
            self.knowledge.fade()
            if self._active_chaos_probe():
                self._reentries += 1
                self._chaos_promotions += 1
            self._check_demotions(x, y)
            self._revalidate_wisdom()

        if self._step % self.audit_every == 0 and self._recent_good and y is not None:
            suspect = self._recent_good[-1]
            finding = self.auditor.audit_input(self, suspect, observed_y=y)
            if finding:
                self.auditor.apply(self, finding)
                self._log("audit_unlearn", disconnect=finding.disconnect_at)

        return "thought"

    def _active_chaos_probe(self) -> bool:
        if not self.chaos.rejected:
            return False
        scored = []
        for cx in self.chaos.rejected[-30:]:
            s = self.gate.score(cx)
            scored.append((abs(s - self.gate.threshold), cx, s))
        scored.sort(key=lambda t: t[0])
        for _, cx, s in scored[:4]:
            if s >= self.gate.threshold - 0.05:
                self.thought.observe(cx, y=None)
                self.chaos.record_probe_success()
                self._log("chaos_probe_success", score=round(s, 3))
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
            self._log("demote_reactive")
            return
        k = self.knowledge.lookup(x)
        if k is not None and abs(float(k) - float(y)) > self.demote_error_tol:
            self.knowledge.add_or_reinforce(x, y, boost=-0.4, step=self._step)
            self._log("weaken_knowledge")

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
            "unlearns": len(self._unlearn_details),
            "event_log_len": len(self.event_log),
            "calibration": self.calibration_summary(),
            "wisdom_reliability": self.wisdom_reliability(),
            "temperature": round(self.temperature, 3),
            "step": self._step,
        }

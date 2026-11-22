"""File clerk. Reactive is muscle memory; Wisdom is later consistency after deliberation."""

import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict
from .randomness_gate import RandomnessGate, NoiseProfile
from .short_term_buffer import ShortTermBuffer
from .minds import ReactiveMind, TraumaStore, ChaosPile
from .mind_graph import MindGraph
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
        self.noise_profile = NoiseProfile()
        self.short_term = ShortTermBuffer()
        self.reactive = ReactiveMind()
        self.trauma = TraumaStore()
        self.graph_mode = 'blend'
        self.graph = MindGraph()  # cascade | blend | thought_consults
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
        self._unlearn_counts: Dict[str, int] = defaultdict(int)
        # Track early disagreement so we know what required deliberation
        self._early_disagreement: Dict[str, int] = defaultdict(int)
        self._recent_scores: List[float] = []
        self._recent_raw: List[np.ndarray] = []
        self.gate_bypass = False

    def _trace_key(self, x: np.ndarray) -> str:
        return str(np.round(np.asarray(x).ravel()[:6], 2).tolist())

    def _coarse_key(self, x: np.ndarray) -> str:
        return str(np.round(np.asarray(x).ravel()[:4], 1).tolist())

    def _log(self, kind: str, **kwargs):
        entry = {"step": self._step, "kind": kind, **kwargs}
        self.event_log.append(entry)
        if len(self.event_log) > 2000:
            self.event_log = self.event_log[-2000:]
        return entry

    def _record(self, x: np.ndarray, event: Dict[str, Any]):
        key = self._trace_key(x)
        full = {"step": self._step, "x": np.asarray(x, dtype=float).ravel().copy(), **event}
        self._traces.setdefault(key, []).append(full)
        if len(self._traces[key]) > 40:
            self._traces[key] = self._traces[key][-40:]
        self._log(**event, key=key)

    def get_trace(self, key: str) -> List[Dict[str, Any]]:
        return self._traces.get(key, [])

    def unlearn(self, key: str):
        self._unlearn_counts[key] += 1
        dropped_reactive = []
        to_drop = [k for k in list(self.reactive.table.keys()) if key[:12] in str(k)]
        for k in to_drop[:5]:
            dropped_reactive.append(str(k)[:40])
            self.reactive.table.pop(k, None)

        knowledge_before_after = []
        for item in self.knowledge.items:
            if self._trace_key(item.x) == key:
                before = item.strength
                item.strength *= 0.3
                knowledge_before_after.append((round(before, 3), round(item.strength, 3)))

        wisdom_before_after = []
        for e in self.wisdom.entries:
            if self._trace_key(e.x) == key:
                before = e.confidence
                e.confidence = max(0.0, e.confidence - 0.6)
                wisdom_before_after.append((round(before, 3), round(e.confidence, 3)))

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
            "times": self._unlearn_counts[key],
            "dropped_reactive": dropped_reactive,
            "knowledge_before_after": knowledge_before_after,
            "wisdom_before_after": wisdom_before_after,
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
                    f"step {d['step']} key={d['key']} times={d.get('times', 1)}\n"
                    f"  reactive_dropped={len(d['dropped_reactive'])}\n"
                    f"  knowledge before->after: {d.get('knowledge_before_after', [])}\n"
                    f"  wisdom before->after: {d.get('wisdom_before_after', [])}\n"
                    f"  hypotheses_deactivated={d['deactivated_hypotheses']}\n"
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

    def _is_trauma(self, x, y, existing) -> bool:
        """Large sudden error on a structured input can lock muscle memory."""
        if y is None or existing is None:
            return False
        pred = self.thought.sample(x)
        if pred is None:
            return False
        return abs(float(pred) - float(y)) > self.demote_error_tol * 2.5

    def observe(self, x: np.ndarray, y=None):
        x = np.asarray(x)
        self._step += 1
        # Audit before any early return
        if self._step % self.audit_every == 0 and self._recent_good and y is not None:
            suspect = self._recent_good[-1]
            finding = self.auditor.audit_input(self, suspect, observed_y=y)
            if finding:
                self.auditor.apply(self, finding)
                self._log("audit_unlearn", disconnect=finding.disconnect_at)
        score = self.gate.score(x)
        self._recent_raw.append(np.asarray(x).ravel().copy())
        if len(self._recent_raw) > 20:
            self._recent_raw.pop(0)
        # Bypass if recent raw inputs are highly similar to each other (clean repeated signal)
        if len(self._recent_raw) >= 12:
            base = self._recent_raw[-1]
            dists = [np.linalg.norm(r - base) for r in self._recent_raw[-12:]]
            self.gate_bypass = float(np.median(dists)) < 0.15
        self._recent_scores.append(score)
        if len(self._recent_scores) > 40:
            self._recent_scores.pop(0)
        is_noise = (not self.gate_bypass) and (score < self.gate.threshold)
        self.gate.update(score, is_noise)

        active_names = self.thought.mind.active_names
        self.hypotheses.observe(x, y, self._step)

        if y is not None:
            for h in self.hypotheses.hypotheses:
                if h.active and h.confirmations >= 1 and self._trace_key(h.x) == self._trace_key(x):
                    self.thought.mind.boost_generative(0.1)
                    break
            self.thought.mind.decay_generative_bonus()

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

        # Record early disagreement for later wisdom eligibility
        if agreement < 0.55:
            self._early_disagreement[self._coarse_key(x)] += 1

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

        score = self.gate.score(x)
        if is_noise:
            self.chaos.reject(x)
            self._record(x, {"kind": "routed_chaos", **context})
            self.noise_profile.observe_rejected(x, score)
            # Adapt threshold toward the learned separation
            self.gate.threshold = self.noise_profile.suggested_threshold(self.gate.threshold)
            return "chaos"

        self.thought.observe(x, y)
        self.noise_profile.observe_accepted(x, score)
        self.gate.threshold = self.noise_profile.suggested_threshold(self.gate.threshold)
        if y is not None:
            pred = self.thought.sample(x)
            if pred is not None:
                if not hasattr(self, "_recent_residuals"):
                    self._recent_residuals = []
                self._recent_residuals.append(abs(float(pred) - float(y)))
                if len(self._recent_residuals) > 50:
                    self._recent_residuals.pop(0)
        self.short_term.add(x, score, y)
        self._record(x, {"kind": "to_thought", "y": y, **context})

        if self._step % 5 == 0:
            self._propose_hypothesis(x)

        existing = self.short_term.find_similar(x)
        item = self.knowledge.find_item(x)
        age_ok = item is not None and (self._step - item.first_seen) >= self.min_age
        hyp_score = self.hypotheses.confirmation_score(x)
        hits_needed = max(2, self.promote_hits - (1 if hyp_score > 0.5 else 0))
        key = self._trace_key(x)

        # Always land structured repeats in Knowledge first
        if existing is not None and existing["hits"] >= hits_needed and score >= self.promote_score:
            stored_y = y if y is not None else existing.get("output")
            self.knowledge.add_or_reinforce(x, stored_y, boost=self.knowledge_boost, step=self._step)
            self._record(x, {"kind": "to_knowledge", "y": stored_y, **context})

        # --- Muscle memory (Reactive): requires many repetitions (slow to form) ---
        # Thought is immediate; reactive is earned through sustained repetition.
        reactive_hits_needed = max(200, hits_needed * 50)
        if existing is not None and y is not None:
            high_repetition = existing["hits"] >= reactive_hits_needed and score >= self.promote_score
            trauma = self._is_trauma(x, y, existing)
            if trauma and y is not None:
                # Trauma is fast and separate; it is the primary audit target
                err = abs(float(y) - float(existing.get("output", y))) if existing.get("output") is not None else 1.0
                self.trauma.store(x, y, error=err, step=self._step)
                self._record(x, {"kind": "trauma", "y": y, "error": err, **context})
            if high_repetition:
                self.reactive.store(x, y)
                self._record(x, {"kind": "muscle_memory", "via": "repetition", **context})
                # Do not early-return; let knowledge and wisdom still be considered

        # --- Knowledge (still the medium-term store) ---
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

            # --- Wisdom: self-observation across layers ---
            # Forms only when thought, knowledge, and (if present) reactive
            # agree on the same output after sustained consistency.
            # Chaos involvement is recorded but does not block.
            thought_pred = self.thought.sample(x)
            know_item = self.knowledge.find_item(x)
            react_val = self.reactive.lookup(x, tol=0.4)
            layers_agree = False
            if thought_pred is not None and know_item is not None and stored_y is not None:
                t_ok = abs(float(thought_pred) - float(stored_y)) < 0.3
                k_ok = abs(float(know_item.y) - float(stored_y)) < 0.3
                r_ok = (react_val is None) or (abs(float(react_val) - float(stored_y)) < 0.3)
                # Wisdom is the hardest: many hits, knowledge already strong,
                # and the pattern has been seen across a long span.
                sustained = existing["hits"] >= max(400, hits_needed * 100)
                know_strong = know_item is not None and know_item.strength >= 3.0
                layers_agree = t_ok and k_ok and r_ok and sustained and age_ok and know_strong
            if layers_agree:
                conf = know_item.strength if know_item else 1.0
                self.wisdom.add(x, stored_y, confidence=conf, step=self._step)
                self._record(x, {"kind": "promoted_wisdom", "y": stored_y,
                                 "via": "self_observation", **context})
                return "promoted_wisdom"
            return "promoted_knowledge"

        if self._step % 6 == 0:
            self.knowledge.fade()
            if self._active_chaos_probe():
                self._reentries += 1
                self._chaos_promotions += 1
            self._check_demotions(x, y)
            self._revalidate_wisdom()

        return "thought"

    def _similarity_to_good(self, x: np.ndarray) -> float:
        if not self._recent_good:
            return 0.0
        x = np.asarray(x).ravel()
        goods = self._recent_good[-12:]
        n = len(goods)
        weighted = 0.0
        wsum = 0.0
        for i, g in enumerate(goods):
            recency = (i + 1) / n
            dist = np.linalg.norm(g.ravel() - x)
            sim = 1.0 / (1.0 + dist)
            weighted += recency * sim
            wsum += recency
        return float(weighted / (wsum + 1e-12))

    def _active_chaos_probe(self) -> bool:
        if not self.chaos.rejected:
            return False
        scored = []
        for cx in self.chaos.rejected[-40:]:
            s = self.gate.score(cx)
            thr_dist = abs(s - self.gate.threshold)
            sim = self._similarity_to_good(cx)
            priority = thr_dist - 0.5 * sim
            scored.append((priority, cx, s, sim))
        scored.sort(key=lambda t: t[0])
        for _, cx, s, sim in scored[:5]:
            if s >= self.gate.threshold - 0.08:
                self.thought.observe(cx, y=None)
                self.chaos.record_probe_success()
                self._log("chaos_probe_success", score=round(s, 3), sim=round(sim, 3))
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
        mode = getattr(self, "graph_mode", "blend")

        # Raw signals from each store
        raw = {}
        r = self.reactive.lookup(x)
        if r is not None:
            raw["reactive"] = float(r)
        t = self.trauma.lookup(x)
        if t is not None:
            raw["trauma"] = float(t)
        w = self.wisdom.lookup(x)
        if w is not None:
            raw["wisdom"] = float(w)
        k = self.knowledge.lookup(x)
        if k is not None:
            raw["knowledge"] = float(k)
        thought_pred = self.thought.sample(x)
        if thought_pred is not None:
            raw["thought"] = float(thought_pred)
        # Chaos contributes only if the gate says this point is noise
        if self.gate.score(x) < self.gate.threshold:
            raw["chaos"] = 0.0  # no structured prediction
        # Auditor node: if recent findings exist, it signals the clerk
        if self.auditor.findings:
            raw["auditor"] = 1.0  # presence of a finding

        # Propagate: each node is a weighted mix of its raw value and its inputs
        def mix(node, raw_val):
            incoming = self.graph.informed_by(node)
            num = (raw_val if raw_val is not None else 0.0)
            den = (1.0 if raw_val is not None else 0.0)
            for src, w in incoming.items():
                if src in raw and src != node:
                    num += w * raw[src]
                    den += w
            return (num / den) if den > 0 else raw_val

        votes = {}
        for node in ("thought", "knowledge", "reactive", "trauma", "wisdom", "chaos"):
            if node in raw or self.graph.informed_by(node):
                v = mix(node, raw.get(node))
                if v is not None:
                    votes[node] = v

        if mode == "cascade":
            for name in ("reactive", "wisdom", "knowledge", "thought"):
                if name in votes:
                    self._record(x, {"kind": "predict", "pred": votes[name], "via": name})
                    return votes[name], name
            return None, "none"

        if mode == "thought_consults":
            # Thought is primary but shifts toward agreeing layers
            base = votes.get("thought")
            if base is None:
                base = next(iter(votes.values()), None)
            if base is None:
                return None, "none"
            others = [v for n, v in votes.items() if n != "thought"]
            if others:
                base = 0.6 * base + 0.4 * float(np.mean(others))
            self._record(x, {"kind": "predict", "pred": base, "via": "thought_consults"})
            return base, "thought_consults"

        # Default: full graph blend (every edge weighted)
        if not votes:
            return None, "none"
        # Apply any learned edge overrides
        if hasattr(self, "edge_weights"):
            for name, w in self.edge_weights.items():
                self.graph.set_weight(name, "output", w)
        pred = self.graph.compute_output(votes)
        via = "graph:" + "+".join(sorted(votes.keys()))
        self._record(x, {"kind": "predict", "pred": pred, "via": via})
        return pred, via

    def stats(self) -> dict:
        recurrent = sum(1 for c in self._unlearn_counts.values() if c > 1)
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
            "recurrent_unlearns": recurrent,
            "event_log_len": len(self.event_log),
            "calibration": self.calibration_summary(),
            "wisdom_reliability": self.wisdom_reliability(),
            "temperature": round(self.temperature, 3),
            "step": self._step,
        }

    def observe_context(self, context, y=None):
        """Accept numeric, string, categorical, or mixed context. Encodes then observes."""
        if not hasattr(self, "encoder"):
            from .encoder import ContextEncoder
            self.encoder = ContextEncoder(dim=32)
        x = self.encoder.encode(context)
        return self.observe(x, y)

    def predict_context(self, context):
        """Encode then predict."""
        if not hasattr(self, "encoder"):
            from .encoder import ContextEncoder
            self.encoder = ContextEncoder(dim=32)
        x = self.encoder.encode(context)
        return self.predict(x)

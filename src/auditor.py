"""Procedural auditor.

Disconnects irrational behavior from good inputs and unlearns the
reinforcements (locks, repeated promotions, supporting hypotheses)
that kept the bad behavior in place.
"""

from typing import List, Optional
import numpy as np


class AuditFinding:
    def __init__(self, input_key: str, disconnect_at: str, detail: str, action: str, distance: float = None, via: str = None):
        self.input_key = input_key
        self.disconnect_at = disconnect_at
        self.detail = detail
        self.action = action
        self.distance = distance
        self.via = via


class Auditor:
    def __init__(self, max_per_window: int = 5):
        self.max_per_window = max_per_window
        self._window_count = 0
        self._window_start = 0
        self.findings: List[AuditFinding] = []


    def _nearest_distance(self, clerk, x):
        """Distance from x to the nearest stored pattern in any layer."""
        x = np.asarray(x, dtype=float).ravel()
        best = None
        for layer in (clerk.knowledge, clerk.wisdom):
            items = getattr(layer, "items", None) or getattr(layer, "entries", [])
            for item in items:
                xi = getattr(item, "x", None)
                if xi is not None and xi.shape == x.shape:
                    d = float(np.linalg.norm(xi - x))
                    if best is None or d < best:
                        best = d
        for k in clerk.reactive.table:
            try:
                xi = np.frombuffer(k, dtype=float)
                if xi.shape == x.shape:
                    d = float(np.linalg.norm(xi - x))
                    if best is None or d < best:
                        best = d
            except Exception:
                continue
        return best

    def audit_input(self, clerk, x: np.ndarray, observed_y: Optional[float] = None) -> Optional[AuditFinding]:
        # Rate limit: at most max_per_window findings per 50 steps
        if clerk._step - self._window_start > 50:
            self._window_start = clerk._step
            self._window_count = 0
        if self._window_count >= self.max_per_window:
            return None
        score = clerk.gate.score(x)
        # No longer require the point to be above threshold; look for any prior
        # neighborhood that now contradicts a later observation.

        # Neighborhood match instead of exact key
        trace = None
        key = None
        x = np.asarray(x, dtype=float).ravel()
        for k, events in list(clerk._traces.items())[-40:]:
            # recover a representative vector from the first event if stored
            for e in events:
                if "x" in e:
                    cand = np.asarray(e["x"], dtype=float).ravel()
                    if cand.shape == x.shape and np.linalg.norm(cand - x) < 0.35:
                        trace = events
                        key = k
                        break
            if trace is not None:
                break
        if not trace:
            # fall back to exact key
            key = clerk._trace_key(x)
            trace = clerk.get_trace(key)
        if not trace:
            return None

        # Only audit if the stored y appears more than once (real reinforcement)
        y_counts = {}
        for e in trace:
            yy = e.get("y")
            if yy is not None:
                y_counts[round(float(yy), 2)] = y_counts.get(round(float(yy), 2), 0) + 1
        if not any(c >= 2 for c in y_counts.values()):
            return None

        # Only look at recent events so ordinary drift is not treated as contradiction
        recent = [e for e in trace if clerk._step - e.get("step", 0) < 30]
        if not recent:
            recent = trace[-5:]

        for event in reversed(recent):
            kind = event.get("kind")

            if kind == "routed_chaos":
                finding = AuditFinding(
                    key, "gate",
                    "Structured input was routed to chaos",
                    "sever_and_unlearn",
                    distance=self._nearest_distance(clerk, x)
                )
                self.findings.append(finding)
                self._window_count += 1
                return finding

            if kind == "predict" and observed_y is not None:
                pred = event.get("pred")
                if pred is not None and abs(float(pred) - float(observed_y)) > clerk.demote_error_tol * 1.5:
                    finding = AuditFinding(
                        key, "prediction",
                        f"Irrational prediction {pred} on structured input",
                        "sever_and_unlearn",
                        distance=self._nearest_distance(clerk, x),
                        via=event.get("via")
                    )
                    self.findings.append(finding)
                    return finding

            if kind in ("trauma", "promoted_wisdom", "promoted_knowledge", "to_knowledge", "muscle_memory") and observed_y is not None:
                promoted = event.get("y")
                if promoted is not None:
                    recent_ys = [float(e.get("y")) for e in recent if e.get("y") is not None]
                    if len(recent_ys) >= 3:
                        rmin, rmax = min(recent_ys), max(recent_ys)
                        # only flag if the new y is well outside the recent range
                        outside = (kind == 'trauma') or (float(observed_y) < rmin - 1.0 or float(observed_y) > rmax + 1.0)
                    else:
                        outside = abs(float(promoted) - float(observed_y)) > 1.5
                    if outside:
                        finding = AuditFinding(
                            key, kind,
                            "Stored behavior contradicted later good observation",
                            "sever_and_unlearn",
                            distance=self._nearest_distance(clerk, x)
                        )
                        self.findings.append(finding)
                        return finding

        return None

    def apply(self, clerk, finding: AuditFinding):
        if finding is None:
            return
        # Only sever if we have a concrete better observation
        # (the finding already encodes the contradiction)
        # If the contradicted prediction came from reactive, raise the bar
        # so the short path is taken less often. That is where error lives.
        # Quarantine the contradicted entry so it is not used for a while.
        # More contradictions -> longer quarantine. Capped so it recovers.
        if hasattr(clerk, "quarantine_steps"):
            clerk.quarantine_steps = min(40, clerk.quarantine_steps + 2)
            # Mark the nearest knowledge item as quarantined
            if hasattr(finding, "distance") and finding.distance is not None:
                for item in clerk.knowledge.items:
                    clerk._quarantine[id(item)] = clerk._step + clerk.quarantine_steps
                    break
        clerk.unlearn(finding.input_key)
        # Immediately reinforce the current good observation so the correction sticks
        if hasattr(finding, "correct_y") and finding.correct_y is not None:
            # pull a representative x from the trace
            trace = clerk._traces.get(finding.input_key, [])
            if trace and "x" in trace[-1]:
                x = trace[-1]["x"]
                clerk.knowledge.add_or_reinforce(x, finding.correct_y, boost=0.5, step=clerk._step)
                clerk.reactive.store(x, finding.correct_y)
        clerk.refile_to_thought(finding.input_key)

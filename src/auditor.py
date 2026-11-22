"""Procedural auditor.

Disconnects irrational behavior from good inputs and unlearns the
reinforcements (locks, repeated promotions, supporting hypotheses)
that kept the bad behavior in place.
"""

from typing import List, Optional
import numpy as np


class AuditFinding:
    def __init__(self, input_key: str, disconnect_at: str, detail: str, action: str):
        self.input_key = input_key
        self.disconnect_at = disconnect_at
        self.detail = detail
        self.action = action


class Auditor:
    def __init__(self):
        self.findings: List[AuditFinding] = []

    def audit_input(self, clerk, x: np.ndarray, observed_y: Optional[float] = None) -> Optional[AuditFinding]:
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
                    if cand.shape == x.shape and np.linalg.norm(cand - x) < 0.8:
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

        for event in reversed(trace):
            kind = event.get("kind")

            if kind == "routed_chaos":
                finding = AuditFinding(
                    key, "gate",
                    "Structured input was routed to chaos",
                    "sever_and_unlearn"
                )
                self.findings.append(finding)
                return finding

            if kind == "predict" and observed_y is not None:
                pred = event.get("pred")
                if pred is not None and abs(float(pred) - float(observed_y)) > clerk.demote_error_tol * 1.5:
                    finding = AuditFinding(
                        key, "prediction",
                        f"Irrational prediction {pred} on structured input",
                        "sever_and_unlearn"
                    )
                    self.findings.append(finding)
                    return finding

            if kind in ("promoted_wisdom", "promoted_knowledge") and observed_y is not None:
                promoted = event.get("y")
                if promoted is not None and abs(float(promoted) - float(observed_y)) > clerk.demote_error_tol:
                    finding = AuditFinding(
                        key, kind,
                        "Promotion contradicted later good observation",
                        "sever_and_unlearn"
                    )
                    self.findings.append(finding)
                    return finding

        return None

    def apply(self, clerk, finding: AuditFinding):
        if finding is None:
            return
        # Sever + unlearn all reinforcements, then re-file the good input
        clerk.unlearn(finding.input_key)
        clerk.refile_to_thought(finding.input_key)

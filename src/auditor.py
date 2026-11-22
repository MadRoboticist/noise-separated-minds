"""Procedural auditor.

Works only with the file clerk. Given a suspect input it recalls the
recorded trace and walks backward until it finds the disconnect, then
asks the clerk to re-file.
"""

from typing import List, Optional, Dict, Any
import numpy as np


class AuditFinding:
    def __init__(self, input_key: str, disconnect_at: str, detail: str, action: str):
        self.input_key = input_key
        self.disconnect_at = disconnect_at  # e.g. "gate", "promotion", "wisdom", "reactive"
        self.detail = detail
        self.action = action


class Auditor:
    """Purely procedural. Never touches minds directly."""

    def __init__(self):
        self.findings: List[AuditFinding] = []

    def audit_input(self, clerk, x: np.ndarray, observed_y: Optional[float] = None) -> Optional[AuditFinding]:
        """Recall the trace for this input and walk backward to the disconnect."""
        key = clerk._trace_key(x)
        trace = clerk.get_trace(key)
        if not trace:
            return None

        # Walk from most recent event backward
        for event in reversed(trace):
            kind = event.get("kind")
            if kind == "predict":
                pred = event.get("pred")
                if observed_y is not None and pred is not None:
                    if abs(float(pred) - float(observed_y)) > clerk.demote_error_tol:
                        finding = AuditFinding(
                            key, "prediction",
                            f"Prediction {pred} disagreed with {observed_y}",
                            "demote_and_rethink"
                        )
                        self.findings.append(finding)
                        return finding
            elif kind == "promoted_wisdom":
                if observed_y is not None and abs(float(event.get("y", 0)) - float(observed_y)) > clerk.demote_error_tol:
                    finding = AuditFinding(
                        key, "wisdom_promotion",
                        "Wisdom was promoted with a value that later failed",
                        "reduce_wisdom_confidence"
                    )
                    self.findings.append(finding)
                    return finding
            elif kind == "promoted_knowledge":
                if observed_y is not None and abs(float(event.get("y", 0)) - float(observed_y)) > clerk.demote_error_tol:
                    finding = AuditFinding(
                        key, "knowledge_promotion",
                        "Knowledge promotion later contradicted",
                        "weaken_knowledge"
                    )
                    self.findings.append(finding)
                    return finding
            elif kind == "routed_chaos" and observed_y is not None:
                # Structured observation was sent to chaos
                finding = AuditFinding(
                    key, "gate",
                    "Input with a label was routed to chaos",
                    "force_to_thought"
                )
                self.findings.append(finding)
                return finding

        return None

    def apply(self, clerk, finding: AuditFinding):
        """Ask the file clerk to re-file according to the finding."""
        if finding is None:
            return
        action = finding.action
        # The clerk exposes only re-file methods; auditor never touches layers directly
        if action == "demote_and_rethink":
            clerk.refile_demote(finding.input_key)
        elif action == "reduce_wisdom_confidence":
            clerk.refile_weaken_wisdom(finding.input_key)
        elif action == "weaken_knowledge":
            clerk.refile_weaken_knowledge(finding.input_key)
        elif action == "force_to_thought":
            clerk.refile_to_thought(finding.input_key)

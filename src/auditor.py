"""Procedural auditor.

Purpose: disconnect irrational behavior from good (structured) inputs.
Works only with the file clerk. Walks the trace of a good input backward
until it finds where the system acted irrationally, then severs that
behavior so the input can be re-filed cleanly.
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
        """Only act when the input itself looks structured (good)."""
        score = clerk.gate.score(x)
        if score < clerk.gate.threshold:
            return None  # not a good input; leave it alone

        key = clerk._trace_key(x)
        trace = clerk.get_trace(key)
        if not trace:
            return None

        # Walk backward looking for irrational acts on this good input
        for event in reversed(trace):
            kind = event.get("kind")

            # Irrational: good input was sent to chaos
            if kind == "routed_chaos":
                finding = AuditFinding(
                    key, "gate",
                    "Structured input was routed to chaos",
                    "sever_chaos_and_refile"
                )
                self.findings.append(finding)
                return finding

            # Irrational: prediction on a good input was wildly wrong
            if kind == "predict" and observed_y is not None:
                pred = event.get("pred")
                if pred is not None and abs(float(pred) - float(observed_y)) > clerk.demote_error_tol * 1.5:
                    finding = AuditFinding(
                        key, "prediction",
                        f"Irrational prediction {pred} on structured input (true {observed_y})",
                        "sever_prediction_path"
                    )
                    self.findings.append(finding)
                    return finding

            # Irrational: promoted a value that later contradicted the good observation
            if kind in ("promoted_wisdom", "promoted_knowledge") and observed_y is not None:
                promoted = event.get("y")
                if promoted is not None and abs(float(promoted) - float(observed_y)) > clerk.demote_error_tol:
                    finding = AuditFinding(
                        key, kind,
                        "Promotion contradicted later good observation",
                        "sever_promotion"
                    )
                    self.findings.append(finding)
                    return finding

        return None

    def apply(self, clerk, finding: AuditFinding):
        if finding is None:
            return
        # All actions sever the bad behavior and force the good input back to Thought
        if finding.action == "sever_chaos_and_refile":
            clerk.refile_to_thought(finding.input_key)
        elif finding.action == "sever_prediction_path":
            clerk.refile_demote(finding.input_key)
            clerk.refile_to_thought(finding.input_key)
        elif finding.action == "sever_promotion":
            clerk.refile_weaken_knowledge(finding.input_key)
            clerk.refile_weaken_wisdom(finding.input_key)
            clerk.refile_to_thought(finding.input_key)

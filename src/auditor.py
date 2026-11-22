"""Independent auditor.

Periodically inspects tracks that are misbehaving, attempts a root-cause
diagnosis, and recommends re-filing actions to the file clerk.
"""

from typing import List, Dict, Any
import numpy as np


class AuditFinding:
    def __init__(self, kind: str, target: str, detail: str, action: str):
        self.kind = kind          # e.g. "high_error", "stale_wisdom", "collapsed_perspective"
        self.target = target      # what was inspected
        self.detail = detail
        self.action = action      # recommended re-file action


class Auditor:
    def __init__(self, error_tol: float = 0.6):
        self.error_tol = error_tol
        self.findings: List[AuditFinding] = []

    def inspect(self, system) -> List[AuditFinding]:
        """Run a full audit pass. Returns new findings."""
        new = []
        new.extend(self._check_reactive(system))
        new.extend(self._check_knowledge(system))
        new.extend(self._check_wisdom(system))
        new.extend(self._check_perspectives(system))
        self.findings.extend(new)
        if len(self.findings) > 500:
            self.findings = self.findings[-500:]
        return new

    def _check_reactive(self, system) -> List[AuditFinding]:
        findings = []
        # Spot-check a few reactive entries against Thought
        items = list(system.reactive.table.items())[:5]
        for key, y in items:
            # We cannot easily invert the key; skip deep check if no recent context
            # Placeholder: flag if Thought uncertainty on recent inputs is high
            pass
        if len(system.reactive) > 0 and system.thought.mind.uncertainty(np.zeros(8)) > 2.0:
            findings.append(AuditFinding(
                "reactive_uncertainty",
                "reactive",
                "Thought reports high uncertainty near reactive region",
                "demote_suspect_reactive"
            ))
        return findings

    def _check_knowledge(self, system) -> List[AuditFinding]:
        findings = []
        for item in system.knowledge.items:
            if item.strength < 0.2 and item.hits > 3:
                findings.append(AuditFinding(
                    "weak_knowledge",
                    "knowledge",
                    f"Item strength {item.strength:.2f} after {item.hits} hits",
                    "weaken_or_drop"
                ))
        return findings

    def _check_wisdom(self, system) -> List[AuditFinding]:
        findings = []
        for e in system.wisdom.entries:
            if e.confidence < system.wisdom.min_confidence:
                findings.append(AuditFinding(
                    "low_confidence_wisdom",
                    "wisdom",
                    f"Confidence {e.confidence:.2f}, failures {e.failures}",
                    "force_revalidation"
                ))
            elif system._step - e.last_checked > system.revalidate_interval * 2:
                findings.append(AuditFinding(
                    "stale_wisdom",
                    "wisdom",
                    "Not checked recently",
                    "schedule_revalidation"
                ))
        return findings

    def _check_perspectives(self, system) -> List[AuditFinding]:
        findings = []
        for p in system.thought.mind.perspectives:
            if p.auto_disabled:
                findings.append(AuditFinding(
                    "collapsed_perspective",
                    p.name,
                    "Auto-disabled due to high error",
                    "keep_disabled_or_reset"
                ))
        return findings

    def apply(self, system, findings: List[AuditFinding]):
        """Execute recommended re-file actions."""
        for f in findings:
            if f.action == "force_revalidation":
                # already handled by system's revalidate; just log
                pass
            elif f.action == "weaken_or_drop":
                # extra fade on weak knowledge
                system.knowledge.fade()
            elif f.action == "demote_suspect_reactive":
                # clear a portion of reactive if uncertainty is high
                if len(system.reactive) > 10:
                    # drop oldest-ish
                    keys = list(system.reactive.table.keys())[:2]
                    for k in keys:
                        system.reactive.table.pop(k, None)
            elif f.action == "keep_disabled_or_reset":
                # leave auto-disable logic in charge
                pass

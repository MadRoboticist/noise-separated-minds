"""Generative hypotheses.

A generative sample is treated as a provisional hypothesis that can be
confirmed or rejected by later observations. Confirmed hypotheses
boost promotion; rejected ones are discarded.
"""

from typing import List, Optional
import numpy as np


class Hypothesis:
    def __init__(self, x: np.ndarray, y_pred: float, step: int):
        self.x = np.asarray(x, dtype=float).ravel()
        self.y_pred = float(y_pred)
        self.created = step
        self.confirmations = 0
        self.rejections = 0
        self.active = True

    def matches(self, x: np.ndarray, tol: float = 0.25) -> bool:
        return np.linalg.norm(self.x - np.asarray(x).ravel()) < tol


class HypothesisTracker:
    def __init__(self, max_active: int = 32, confirm_tol: float = 0.4):
        self.hypotheses: List[Hypothesis] = []
        self.max_active = max_active
        self.confirm_tol = confirm_tol

    def propose(self, x: np.ndarray, y_pred: float, step: int):
        if y_pred is None:
            return
        # Avoid near-duplicates
        for h in self.hypotheses:
            if h.active and h.matches(x, tol=0.15):
                return
        self.hypotheses.append(Hypothesis(x, y_pred, step))
        # Trim
        active = [h for h in self.hypotheses if h.active]
        if len(active) > self.max_active:
            active[0].active = False

    def observe(self, x: np.ndarray, y: Optional[float], step: int):
        if y is None:
            return
        for h in self.hypotheses:
            if not h.active:
                continue
            if h.matches(x):
                if abs(h.y_pred - float(y)) <= self.confirm_tol:
                    h.confirmations += 1
                else:
                    h.rejections += 1
                    if h.rejections >= 2:
                        h.active = False

    def confirmation_score(self, x: np.ndarray) -> float:
        """0..1 score of how well current hypotheses support this x."""
        best = 0.0
        for h in self.hypotheses:
            if h.active and h.matches(x) and h.confirmations > 0:
                best = max(best, min(1.0, h.confirmations / 3.0))
        return best

    def active_count(self) -> int:
        return sum(1 for h in self.hypotheses if h.active)

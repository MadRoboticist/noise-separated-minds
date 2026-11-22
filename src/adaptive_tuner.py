
"""Adaptive parameter search.

Tries combinations of gate, auditor, and ensemble settings on a
short rolling window and keeps the configuration that produces
the lowest recent error.
"""

from typing import Dict, Any, List
import itertools
import numpy as np
from .config import SystemConfig
from .system import NoiseSeparatedSystem


class AdaptiveTuner:
    def __init__(self, window: int = 120):
        self.window = window
        self.best_cfg: Dict[str, Any] = {}
        self.best_error = float("inf")
        self.history: List[Dict[str, Any]] = []

    def candidate_grid(self) -> List[Dict[str, Any]]:
        """Small grid of broad-stroke parameters."""
        grid = {
            "noise_threshold": [0.0, 0.35, 0.50],
            "audit_every": [10**9, 20, 50],
            "promote_hits": [2, 3],
            "promote_score": [0.15, 0.30],
        }
        keys = list(grid.keys())
        combos = []
        for vals in itertools.product(*[grid[k] for k in keys]):
            combos.append(dict(zip(keys, vals)))
        return combos

    def evaluate(self, X: np.ndarray, y: np.ndarray, cfg_overrides: Dict[str, Any]) -> float:
        cfg = SystemConfig.small()
        for k, v in cfg_overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        s = NoiseSeparatedSystem(config=cfg)
        if cfg_overrides.get("audit_every", 20) > 1000:
            s.audit_every = 10**9
        errors = []
        for i in range(len(X)):
            if i > 20:
                pred, _ = s.predict(X[i])
                if pred is not None:
                    errors.append(abs(float(np.asarray(pred).ravel()[0]) - y[i]))
            s.observe(X[i], y[i])
        return float(np.mean(errors)) if errors else float("inf")

    def search(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Run the grid on the given window and return the best overrides."""
        best = {}
        best_err = float("inf")
        for combo in self.candidate_grid():
            err = self.evaluate(X, y, combo)
            self.history.append({"cfg": combo, "error": err})
            if err < best_err:
                best_err = err
                best = combo
        self.best_cfg = best
        self.best_error = best_err
        return best

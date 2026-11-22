
"""Low-dimensional LQR on output-edge weights.

Fits a linear model of (edge delta -> error change) from recent steps,
then computes an LQR-optimal step that trades off error reduction
against control effort.
"""

import numpy as np
from typing import Dict, List, Optional


class EdgeLQR:
    def __init__(self, edge_names: List[str], horizon: int = 20, effort: float = 0.1):
        self.names = list(edge_names)
        self.n = len(self.names)
        self.horizon = horizon
        self.effort = effort  # control cost weight
        self._hist_dw: List[np.ndarray] = []
        self._hist_de: List[float] = []
        self._last_w: Optional[np.ndarray] = None
        self._last_err: Optional[float] = None
        # Linear model: de ≈ b·dw  (no state, just immediate effect)
        self.b = np.zeros(self.n)

    def observe(self, weights: Dict[str, float], error: float):
        w = np.array([weights.get(n, 0.0) for n in self.names])
        if self._last_w is not None and self._last_err is not None:
            dw = w - self._last_w
            de = error - self._last_err
            if np.linalg.norm(dw) > 1e-6:
                self._hist_dw.append(dw)
                self._hist_de.append(de)
                if len(self._hist_dw) > self.horizon:
                    self._hist_dw.pop(0)
                    self._hist_de.pop(0)
                self._fit()
        self._last_w = w
        self._last_err = error

    def _fit(self):
        if len(self._hist_dw) < 4:
            return
        X = np.array(self._hist_dw)
        y = np.array(self._hist_de)
        # ridge regression for b
        A = X.T @ X + 1e-3 * np.eye(self.n)
        self.b = np.linalg.solve(A, X.T @ y)

    def step(self, weights: Dict[str, float], error: float) -> Dict[str, float]:
        """Return optimal edge deltas.
        Cost ≈ (error + b·dw)^2 + effort·||dw||^2
        Optimal dw = -b * error / (||b||^2 + effort)
        """
        if np.linalg.norm(self.b) < 1e-6:
            return {n: 0.0 for n in self.names}
        denom = float(self.b @ self.b) + self.effort
        dw = -self.b * error / denom
        # clip
        dw = np.clip(dw, -0.05, 0.05)
        return {n: float(dw[i]) for i, n in enumerate(self.names)}

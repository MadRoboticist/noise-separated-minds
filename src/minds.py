"""Analytical ensemble with accuracy tracking and a simple parametric perspective."""

from typing import Any, Dict, List, Optional
import numpy as np


class ReactiveMind:
    def __init__(self):
        self.table: Dict[bytes, Any] = {}

    def _key(self, x: np.ndarray) -> bytes:
        return np.asarray(x).tobytes()

    def store(self, x: np.ndarray, y: Any):
        self.table[self._key(x)] = y

    def lookup(self, x: np.ndarray) -> Optional[Any]:
        return self.table.get(self._key(x))

    def __len__(self):
        return len(self.table)


class Perspective:
    def __init__(self, name: str):
        self.name = name
        self.enabled = True
        self.xs: List[np.ndarray] = []
        self.ys: List[Any] = []
        self.recent_errors: List[float] = []

    def observe(self, x: np.ndarray, y: Any):
        self.xs.append(np.asarray(x, dtype=float).ravel())
        self.ys.append(y)

    def record_error(self, err: float):
        self.recent_errors.append(abs(err))
        if len(self.recent_errors) > 50:
            self.recent_errors.pop(0)

    def accuracy_weight(self) -> float:
        if not self.recent_errors:
            return 1.0
        return 1.0 / (1.0 + np.mean(self.recent_errors))

    def predict(self, x: np.ndarray) -> Optional[float]:
        raise NotImplementedError

    def uncertainty(self, x: np.ndarray) -> float:
        return 1.0

    def clear(self):
        self.xs.clear()
        self.ys.clear()
        self.recent_errors.clear()


class LocalWeightedPerspective(Perspective):
    def __init__(self, bandwidth: float = 0.5):
        super().__init__("local_weighted")
        self.bandwidth = bandwidth

    def predict(self, x: np.ndarray) -> Optional[float]:
        if not self.xs:
            return None
        x = np.asarray(x, dtype=float).ravel()
        dists = np.array([np.linalg.norm(xi - x) for xi in self.xs])
        weights = np.exp(-dists ** 2 / (2 * self.bandwidth ** 2))
        weights /= weights.sum() + 1e-12
        vals = np.array([float(y) if y is not None else 0.0 for y in self.ys])
        return float(np.sum(weights * vals))

    def uncertainty(self, x: np.ndarray) -> float:
        if not self.xs:
            return 1.0
        x = np.asarray(x, dtype=float).ravel()
        dists = np.array([np.linalg.norm(xi - x) for xi in self.xs])
        return float(np.mean(np.sort(dists)[:5]))


class NearestNeighborPerspective(Perspective):
    def __init__(self, k: int = 1):
        super().__init__(f"nn_k{k}")
        self.k = k

    def predict(self, x: np.ndarray) -> Optional[float]:
        if not self.xs:
            return None
        x = np.asarray(x, dtype=float).ravel()
        dists = np.array([np.linalg.norm(xi - x) for xi in self.xs])
        idx = np.argsort(dists)[: self.k]
        vals = [float(self.ys[i]) for i in idx if self.ys[i] is not None]
        return float(np.mean(vals)) if vals else None


class OnlineLinearPerspective(Perspective):
    """Simple parametric: online ridge regression on raw features."""

    def __init__(self, dim_hint: int = 16, lr: float = 0.05, reg: float = 1e-3):
        super().__init__("online_linear")
        self.lr = lr
        self.reg = reg
        self.w = None

    def observe(self, x: np.ndarray, y: Any):
        super().observe(x, y)
        if y is None:
            return
        x = np.asarray(x, dtype=float).ravel()
        if self.w is None or len(self.w) != len(x) + 1:
            self.w = np.zeros(len(x) + 1)
        x_aug = np.append(x, 1.0)
        pred = float(x_aug @ self.w)
        err = float(y) - pred
        self.w += self.lr * (err * x_aug - self.reg * self.w)
        self.record_error(err)

    def predict(self, x: np.ndarray) -> Optional[float]:
        if self.w is None:
            return None
        x = np.asarray(x, dtype=float).ravel()
        if len(self.w) != len(x) + 1:
            return None
        return float(np.append(x, 1.0) @ self.w)


class AnalyticalMind:
    def __init__(self):
        self.perspectives: List[Perspective] = [
            LocalWeightedPerspective(),
            NearestNeighborPerspective(k=1),
            NearestNeighborPerspective(k=3),
            OnlineLinearPerspective(),
            LocalWeightedPerspective(bandwidth=1.0),  # wider
            NearestNeighborPerspective(k=5),
            OnlineLinearPerspective(lr=0.02),
        ]
        # rename the second local and second linear for uniqueness
        self.perspectives[4].name = "local_wide"
        self.perspectives[6].name = "online_linear_slow"
        self.blender_mode = "accuracy"  # accuracy, weighted, greedy, mean

    def set_enabled(self, names: List[str]):
        for p in self.perspectives:
            p.enabled = any(p.name == n or p.name.startswith(n) for n in names)

    def enable_all(self):
        for p in self.perspectives:
            p.enabled = True

    def observe(self, x: np.ndarray, y: Optional[Any] = None):
        for p in self.perspectives:
            if p.enabled:
                # record error before update when possible
                if y is not None:
                    pred = p.predict(x)
                    if pred is not None:
                        p.record_error(float(y) - pred)
                p.observe(x, y)

    def _active(self):
        return [p for p in self.perspectives if p.enabled]

    def agreement(self, x: np.ndarray, tol: float = 0.3) -> float:
        """Fraction of active perspectives that agree within tol."""
        preds = []
        for p in self._active():
            pr = p.predict(x)
            if pr is not None:
                preds.append(pr)
        if len(preds) < 2:
            return 1.0
        preds = np.array(preds)
        med = np.median(preds)
        return float(np.mean(np.abs(preds - med) <= tol))

    def sample(self, x: np.ndarray) -> Optional[float]:
        active = self._active()
        if not active:
            return None
        preds, weights = [], []
        for p in active:
            pr = p.predict(x)
            if pr is not None:
                preds.append(pr)
                if self.blender_mode == "accuracy":
                    weights.append(p.accuracy_weight())
                else:
                    weights.append(1.0 / max(p.uncertainty(x), 1e-6))
        if not preds:
            return None
        preds = np.array(preds)
        weights = np.array(weights)
        weights /= weights.sum() + 1e-12
        if self.blender_mode == "greedy":
            return float(preds[np.argmax(weights)])
        if self.blender_mode == "mean":
            return float(np.mean(preds))
        return float(np.sum(weights * preds))

    def uncertainty(self, x: np.ndarray) -> float:
        active = self._active()
        if not active:
            return 1.0
        return float(np.mean([p.uncertainty(x) for p in active]))

    def __len__(self):
        return len(self.perspectives[0].xs) if self.perspectives else 0

    @property
    def names(self):
        return [p.name for p in self.perspectives]


class ChaosPile:
    def __init__(self):
        self.rejected: List[np.ndarray] = []

    def reject(self, x: np.ndarray):
        self.rejected.append(np.asarray(x))

    def sample(self, shape) -> np.ndarray:
        return np.random.randn(*shape)

    def __len__(self):
        return len(self.rejected)

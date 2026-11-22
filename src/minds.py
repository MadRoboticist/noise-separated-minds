"""Thought ensemble: generative residual perspective, event log, auto-disable."""

from typing import Any, Dict, List, Optional
import numpy as np


class ReactiveMind:
    def __init__(self, max_entries: int = 10_000):
        self.table: Dict[bytes, Any] = {}
        self.max_entries = max_entries

    def _key(self, x: np.ndarray) -> bytes:
        return np.asarray(x).tobytes()

    def store(self, x: np.ndarray, y: Any):
        if len(self.table) >= self.max_entries:
            self.table.pop(next(iter(self.table)))
        self.table[self._key(x)] = y

    def lookup(self, x: np.ndarray) -> Optional[Any]:
        return self.table.get(self._key(x))

    def __len__(self):
        return len(self.table)


class Perspective:
    def __init__(self, name: str, max_obs: int = 2_000):
        self.name = name
        self.enabled = True
        self.auto_disabled = False
        self.max_obs = max_obs
        self.xs: List[np.ndarray] = []
        self.ys: List[Any] = []
        self.recent_errors: List[float] = []

    def observe(self, x: np.ndarray, y: Any):
        self.xs.append(np.asarray(x, dtype=float).ravel())
        self.ys.append(y)
        if len(self.xs) > self.max_obs:
            self.xs.pop(0)
            self.ys.pop(0)

    def record_error(self, err: float):
        self.recent_errors.append(abs(float(err)))
        if len(self.recent_errors) > 40:
            self.recent_errors.pop(0)

    def accuracy_weight(self) -> float:
        if not self.recent_errors:
            return 1.0
        return 1.0 / (1.0 + float(np.mean(self.recent_errors)))

    def is_collapsed(self, threshold: float = 5.0) -> bool:
        if len(self.recent_errors) < 8:
            return False
        return float(np.mean(self.recent_errors)) > threshold

    def predict(self, x: np.ndarray) -> Optional[float]:
        raise NotImplementedError

    def uncertainty(self, x: np.ndarray) -> float:
        return 1.0

    def clear(self):
        self.xs.clear()
        self.ys.clear()
        self.recent_errors.clear()
        self.auto_disabled = False


class LocalWeightedPerspective(Perspective):
    def __init__(self, bandwidth: float = 0.5, max_obs: int = 2_000):
        super().__init__("local_weighted", max_obs)
        self.bandwidth = bandwidth

    def predict(self, x: np.ndarray) -> Optional[float]:
        if not self.xs:
            return None
        x = np.asarray(x, dtype=float).ravel()
        dists = np.array([np.linalg.norm(xi - x) for xi in self.xs])
        weights = np.exp(-dists ** 2 / (2 * self.bandwidth ** 2))
        s = weights.sum()
        if s < 1e-12:
            return None
        weights /= s
        vals = np.array([float(y) if y is not None else 0.0 for y in self.ys])
        return float(np.sum(weights * vals))

    def uncertainty(self, x: np.ndarray) -> float:
        if not self.xs:
            return 1.0
        x = np.asarray(x, dtype=float).ravel()
        dists = np.array([np.linalg.norm(xi - x) for xi in self.xs])
        return float(np.mean(np.sort(dists)[: min(5, len(dists))]))


class NearestNeighborPerspective(Perspective):
    def __init__(self, k: int = 1, max_obs: int = 2_000):
        super().__init__(f"nn_k{k}", max_obs)
        self.k = k

    def predict(self, x: np.ndarray) -> Optional[float]:
        if not self.xs:
            return None
        x = np.asarray(x, dtype=float).ravel()
        dists = np.array([np.linalg.norm(xi - x) for xi in self.xs])
        idx = np.argsort(dists)[: self.k]
        vals = [float(self.ys[i]) for i in idx if self.ys[i] is not None]
        return float(np.mean(vals)) if vals else None

    def uncertainty(self, x: np.ndarray) -> float:
        if not self.xs:
            return 1.0
        x = np.asarray(x, dtype=float).ravel()
        dists = np.array([np.linalg.norm(xi - x) for xi in self.xs])
        return float(np.min(dists))


class OnlineLinearPerspective(Perspective):
    def __init__(self, lr: float = 0.05, reg: float = 1e-3, max_obs: int = 2_000):
        super().__init__("online_linear", max_obs)
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


class ResidualGenerativePerspective(Perspective):
    """Predicts local mean and tracks residual variance (simple generative)."""

    def __init__(self, bandwidth: float = 0.6, max_obs: int = 2_000):
        super().__init__("residual_gen", max_obs)
        self.bandwidth = bandwidth
        self.global_var = 1.0

    def observe(self, x: np.ndarray, y: Any):
        super().observe(x, y)
        if y is not None and len(self.ys) > 5:
            ys = np.array([float(yi) for yi in self.ys if yi is not None])
            if len(ys):
                self.global_var = float(ys.var() + 1e-3)

    def predict(self, x: np.ndarray) -> Optional[float]:
        if not self.xs:
            return None
        x = np.asarray(x, dtype=float).ravel()
        dists = np.array([np.linalg.norm(xi - x) for xi in self.xs])
        weights = np.exp(-dists ** 2 / (2 * self.bandwidth ** 2))
        s = weights.sum()
        if s < 1e-12:
            return None
        weights /= s
        vals = np.array([float(y) if y is not None else 0.0 for y in self.ys])
        return float(np.sum(weights * vals))

    def uncertainty(self, x: np.ndarray) -> float:
        # residual scale + distance
        if not self.xs:
            return self.global_var
        x = np.asarray(x, dtype=float).ravel()
        dists = np.array([np.linalg.norm(xi - x) for xi in self.xs])
        return float(np.sqrt(self.global_var) * (1.0 + np.min(dists)))

    def sample_residual(self, x: np.ndarray) -> Optional[float]:
        mu = self.predict(x)
        if mu is None:
            return None
        return float(mu + np.random.randn() * np.sqrt(self.global_var))


class AnalyticalMind:
    def __init__(self, max_obs: int = 2_000, collapse_threshold: float = 4.0):
        self.perspectives: List[Perspective] = [
            LocalWeightedPerspective(max_obs=max_obs),
            NearestNeighborPerspective(k=1, max_obs=max_obs),
            NearestNeighborPerspective(k=3, max_obs=max_obs),
            OnlineLinearPerspective(max_obs=max_obs),
            ResidualGenerativePerspective(max_obs=max_obs),
            LocalWeightedPerspective(bandwidth=1.2, max_obs=max_obs),
            NearestNeighborPerspective(k=5, max_obs=max_obs),
        ]
        self.perspectives[5].name = "local_wide"
        self.blender_mode = "accuracy"
        self.collapse_threshold = collapse_threshold
        self.events: List[str] = []  # collapse / recover log

    def set_enabled(self, names: List[str]):
        for p in self.perspectives:
            p.enabled = any(p.name == n or p.name.startswith(n) for n in names)
            p.auto_disabled = False

    def enable_all(self):
        for p in self.perspectives:
            p.enabled = True
            p.auto_disabled = False

    def _refresh_disabled(self):
        for p in self.perspectives:
            was = p.auto_disabled
            if p.enabled and p.is_collapsed(self.collapse_threshold):
                p.auto_disabled = True
            elif p.auto_disabled and not p.is_collapsed(self.collapse_threshold * 0.55):
                p.auto_disabled = False
            if p.auto_disabled and not was:
                self.events.append(f"disabled:{p.name}")
            elif was and not p.auto_disabled:
                self.events.append(f"recovered:{p.name}")
        if len(self.events) > 200:
            self.events = self.events[-200:]

    def observe(self, x: np.ndarray, y: Optional[Any] = None):
        self._refresh_disabled()
        for p in self.perspectives:
            if p.enabled and not p.auto_disabled:
                if y is not None:
                    pred = p.predict(x)
                    if pred is not None:
                        p.record_error(float(y) - pred)
                p.observe(x, y)

    def _active(self):
        return [p for p in self.perspectives if p.enabled and not p.auto_disabled]

    def agreement(self, x: np.ndarray, tol: float = 0.3) -> float:
        preds = [p.predict(x) for p in self._active()]
        preds = [pr for pr in preds if pr is not None]
        if len(preds) < 2:
            return 1.0
        preds = np.array(preds)
        med = np.median(preds)
        return float(np.mean(np.abs(preds - med) <= tol))

    def sample(self, x: np.ndarray) -> Optional[float]:
        active = self._active() or [p for p in self.perspectives if p.enabled]
        if not active:
            return None
        preds, weights = [], []
        for p in active:
            pr = p.predict(x)
            if pr is not None:
                preds.append(pr)
                weights.append(p.accuracy_weight() / max(p.uncertainty(x), 0.05))
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
        uncs = [p.uncertainty(x) for p in active]
        preds = [p.predict(x) for p in active]
        preds = [pr for pr in preds if pr is not None]
        disagree = float(np.std(preds)) if len(preds) > 1 else 0.0
        return float(np.mean(uncs) + 0.5 * disagree)

    def __len__(self):
        return len(self.perspectives[0].xs) if self.perspectives else 0

    @property
    def names(self):
        return [p.name for p in self.perspectives]

    @property
    def active_names(self):
        return [p.name for p in self._active()]


class ChaosPile:
    def __init__(self, max_rejected: int = 5_000):
        self.rejected: List[np.ndarray] = []
        self.max_rejected = max_rejected

    def reject(self, x: np.ndarray):
        self.rejected.append(np.asarray(x))
        if len(self.rejected) > self.max_rejected:
            self.rejected.pop(0)

    def sample(self, shape) -> np.ndarray:
        return np.random.randn(*shape)

    def __len__(self):
        return len(self.rejected)

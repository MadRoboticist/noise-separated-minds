"""Analytical ensemble: auto-disable collapsed perspectives, mixture model, better uncertainty."""

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
        self.recent_errors.append(abs(err))
        if len(self.recent_errors) > 40:
            self.recent_errors.pop(0)

    def accuracy_weight(self) -> float:
        if not self.recent_errors:
            return 1.0
        return 1.0 / (1.0 + float(np.mean(self.recent_errors)))

    def is_collapsed(self, threshold: float = 5.0) -> bool:
        if len(self.recent_errors) < 10:
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


class MixturePerspective(Perspective):
    """Tiny two-component mixture on a 1-D projection (mean of features)."""

    def __init__(self, max_obs: int = 2_000):
        super().__init__("mixture", max_obs)
        self.means = None
        self.vars = None
        self.weights = None

    def observe(self, x: np.ndarray, y: Any):
        super().observe(x, y)
        if y is None or len(self.ys) < 8:
            return
        # Fit on scalar projection
        proj = np.array([xi.mean() for xi in self.xs])
        ys = np.array([float(yi) if yi is not None else 0.0 for yi in self.ys])
        # Simple 2-means on proj
        c0, c1 = np.percentile(proj, [33, 66])
        for _ in range(4):
            d0 = np.abs(proj - c0)
            d1 = np.abs(proj - c1)
            a = d0 <= d1
            if a.any():
                c0 = proj[a].mean()
            if (~a).any():
                c1 = proj[~a].mean()
        self.means = np.array([c0, c1])
        self.weights = np.array([a.mean(), 1 - a.mean()])
        self.vars = np.array([
            ys[a].var() + 1e-3 if a.any() else 1.0,
            ys[~a].var() + 1e-3 if (~a).any() else 1.0,
        ])
        # store component predictions as last y means
        self._ymeans = np.array([
            ys[a].mean() if a.any() else ys.mean(),
            ys[~a].mean() if (~a).any() else ys.mean(),
        ])

    def predict(self, x: np.ndarray) -> Optional[float]:
        if self.means is None:
            return None
        p = float(np.asarray(x).mean())
        d = np.abs(p - self.means)
        w = np.exp(-d) * self.weights
        w /= w.sum() + 1e-12
        return float(np.sum(w * self._ymeans))


class AnalyticalMind:
    def __init__(self, max_obs: int = 2_000, collapse_threshold: float = 4.0):
        self.perspectives: List[Perspective] = [
            LocalWeightedPerspective(max_obs=max_obs),
            NearestNeighborPerspective(k=1, max_obs=max_obs),
            NearestNeighborPerspective(k=3, max_obs=max_obs),
            OnlineLinearPerspective(max_obs=max_obs),
            MixturePerspective(max_obs=max_obs),
            LocalWeightedPerspective(bandwidth=1.2, max_obs=max_obs),
            NearestNeighborPerspective(k=5, max_obs=max_obs),
        ]
        self.perspectives[5].name = "local_wide"
        self.blender_mode = "accuracy"
        self.collapse_threshold = collapse_threshold

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
            if p.enabled and p.is_collapsed(self.collapse_threshold):
                p.auto_disabled = True
            elif p.auto_disabled and not p.is_collapsed(self.collapse_threshold * 0.6):
                p.auto_disabled = False  # recover if errors improve

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
        active = self._active()
        if not active:
            # fall back to any enabled
            active = [p for p in self.perspectives if p.enabled]
        if not active:
            return None
        preds, weights = [], []
        for p in active:
            pr = p.predict(x)
            if pr is not None:
                preds.append(pr)
                w = p.accuracy_weight() / max(p.uncertainty(x), 0.1)
                weights.append(w)
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
        # also penalize disagreement
        preds = [p.predict(x) for p in active]
        preds = [pr for pr in preds if pr is not None]
        disagree = float(np.std(preds)) if len(preds) > 1 else 0.0
        return float(np.mean(uncs) + disagree)

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

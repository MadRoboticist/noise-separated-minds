"""The three minds: reactive (function), analytical (ensemble of perspectives), chaos (YOLO)."""

from typing import Any, Dict, List, Optional
import numpy as np


class ReactiveMind:
    """Deterministic mappings only. Near-zero residual variance."""

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
    """Base for a single analytical perspective."""

    def __init__(self, name: str):
        self.name = name
        self.enabled = True
        self.xs: List[np.ndarray] = []
        self.ys: List[Any] = []

    def observe(self, x: np.ndarray, y: Any):
        self.xs.append(np.asarray(x, dtype=float).ravel())
        self.ys.append(y)

    def predict(self, x: np.ndarray) -> Optional[float]:
        raise NotImplementedError

    def uncertainty(self, x: np.ndarray) -> float:
        return 1.0

    def clear(self):
        self.xs.clear()
        self.ys.clear()


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
    def __init__(self, k: int = 1, name: str = None):
        super().__init__(name or f"nn_k{k}")
        self.k = k

    def predict(self, x: np.ndarray) -> Optional[float]:
        if not self.xs:
            return None
        x = np.asarray(x, dtype=float).ravel()
        dists = np.array([np.linalg.norm(xi - x) for xi in self.xs])
        idx = np.argsort(dists)[:self.k]
        vals = [float(self.ys[i]) for i in idx if self.ys[i] is not None]
        return float(np.mean(vals)) if vals else None


class PrototypePerspective(Perspective):
    def __init__(self, n_prototypes: int = 8):
        super().__init__("prototype")
        self.n_prototypes = n_prototypes
        self.prototypes = None
        self.proto_ys = None

    def observe(self, x: np.ndarray, y: Any):
        super().observe(x, y)
        if len(self.xs) >= self.n_prototypes and len(self.xs) % 10 == 0:
            self._update_prototypes()

    def _update_prototypes(self):
        if len(self.xs) < 2:
            return
        data = np.stack(self.xs)
        idx = np.random.choice(len(data), min(self.n_prototypes, len(data)), replace=False)
        cents = data[idx].copy()
        for _ in range(5):
            dists = np.linalg.norm(data[:, None] - cents[None, :], axis=2)
            assign = np.argmin(dists, axis=1)
            for j in range(len(cents)):
                mask = assign == j
                if mask.any():
                    cents[j] = data[mask].mean(axis=0)
        self.prototypes = cents
        self.proto_ys = []
        dists = np.linalg.norm(data[:, None] - cents[None, :], axis=2)
        assign = np.argmin(dists, axis=1)
        for j in range(len(cents)):
            ys = [self.ys[i] for i in range(len(self.ys)) if assign[i] == j and self.ys[i] is not None]
            self.proto_ys.append(float(np.mean(ys)) if ys else 0.0)

    def predict(self, x: np.ndarray) -> Optional[float]:
        if self.prototypes is None or not len(self.prototypes):
            return None
        x = np.asarray(x, dtype=float).ravel()
        dists = np.linalg.norm(self.prototypes - x, axis=1)
        return float(self.proto_ys[np.argmin(dists)])


class LinearProjectionPerspective(Perspective):
    def __init__(self, proj_dim: int = 4):
        super().__init__("linear_projection")
        self.proj_dim = proj_dim
        self.W = None
        self.coef = None

    def observe(self, x: np.ndarray, y: Any):
        super().observe(x, y)
        if len(self.xs) >= 8 and len(self.xs) % 5 == 0:
            self._fit()

    def _fit(self):
        X = np.stack(self.xs)
        y = np.array([float(yi) if yi is not None else 0.0 for yi in self.ys])
        if self.W is None or self.W.shape[0] != X.shape[1]:
            rng = np.random.RandomState(0)
            self.W = rng.randn(X.shape[1], self.proj_dim) / np.sqrt(X.shape[1])
        Z = X @ self.W
        Z = np.c_[Z, np.ones(len(Z))]
        try:
            self.coef = np.linalg.lstsq(Z, y, rcond=None)[0]
        except Exception:
            self.coef = None

    def predict(self, x: np.ndarray) -> Optional[float]:
        if self.coef is None or self.W is None:
            return None
        x = np.asarray(x, dtype=float).ravel()
        z = np.append(x @ self.W, 1.0)
        return float(z @ self.coef)


class MedianNeighborPerspective(Perspective):
    def __init__(self, k: int = 5):
        super().__init__("median_neighbor")
        self.k = k

    def predict(self, x: np.ndarray) -> Optional[float]:
        if not self.xs:
            return None
        x = np.asarray(x, dtype=float).ravel()
        dists = np.array([np.linalg.norm(xi - x) for xi in self.xs])
        idx = np.argsort(dists)[:self.k]
        vals = [float(self.ys[i]) for i in idx if self.ys[i] is not None]
        return float(np.median(vals)) if vals else None


class ExponentialDecayPerspective(Perspective):
    def __init__(self, decay: float = 0.9):
        super().__init__("exp_decay")
        self.decay = decay

    def predict(self, x: np.ndarray) -> Optional[float]:
        if not self.xs:
            return None
        x = np.asarray(x, dtype=float).ravel()
        dists = np.array([np.linalg.norm(xi - x) for xi in self.xs])
        ages = np.arange(len(self.xs))[::-1]
        weights = (self.decay ** ages) * np.exp(-dists)
        weights /= weights.sum() + 1e-12
        vals = np.array([float(y) if y is not None else 0.0 for y in self.ys])
        return float(np.sum(weights * vals))


class ConstantMeanPerspective(Perspective):
    def __init__(self):
        super().__init__("constant_mean")

    def predict(self, x: np.ndarray) -> Optional[float]:
        vals = [float(y) for y in self.ys if y is not None]
        return float(np.mean(vals)) if vals else None


class AnalyticalMind:
    """Ensemble of 7 small perspectives on the same filtered stream.

    Perspectives can be enabled/disabled. A blender combines their outputs.
    """

    def __init__(self):
        self.perspectives: List[Perspective] = [
            LocalWeightedPerspective(),
            NearestNeighborPerspective(k=1),
            NearestNeighborPerspective(k=3),
            PrototypePerspective(n_prototypes=6),
            LinearProjectionPerspective(proj_dim=4),
            MedianNeighborPerspective(k=5),
            ExponentialDecayPerspective(decay=0.85),
        ]
        self.blender_mode = "weighted"  # "weighted", "greedy", "mean"

    def set_enabled(self, names: List[str]):
        """Enable only the named perspectives (exact or prefix match)."""
        for p in self.perspectives:
            p.enabled = any(p.name == n or p.name.startswith(n) for n in names)

    def enable_all(self):
        for p in self.perspectives:
            p.enabled = True

    def observe(self, x: np.ndarray, y: Optional[Any] = None):
        for p in self.perspectives:
            if p.enabled:
                p.observe(x, y)

    def _active(self) -> List[Perspective]:
        return [p for p in self.perspectives if p.enabled]

    def sample(self, x: np.ndarray) -> Optional[float]:
        active = self._active()
        if not active:
            return None
        preds = []
        uncs = []
        for p in active:
            pred = p.predict(x)
            if pred is not None:
                preds.append(pred)
                uncs.append(max(p.uncertainty(x), 1e-6))
        if not preds:
            return None
        preds = np.array(preds)
        uncs = np.array(uncs)

        if self.blender_mode == "greedy":
            return float(preds[np.argmin(uncs)])
        if self.blender_mode == "mean":
            return float(np.mean(preds))
        weights = 1.0 / uncs
        weights /= weights.sum()
        return float(np.sum(weights * preds))

    def uncertainty(self, x: np.ndarray) -> float:
        active = self._active()
        if not active:
            return 1.0
        uncs = [p.uncertainty(x) for p in active]
        return float(np.mean(uncs))

    def __len__(self):
        return len(self.perspectives[0].xs) if self.perspectives else 0

    @property
    def names(self):
        return [p.name for p in self.perspectives]


class ChaosPile:
    """Pure randomness. No model. Just rejected noise and exploratory draws."""

    def __init__(self):
        self.rejected: List[np.ndarray] = []

    def reject(self, x: np.ndarray):
        self.rejected.append(np.asarray(x))

    def sample(self, shape) -> np.ndarray:
        return np.random.randn(*shape)

    def __len__(self):
        return len(self.rejected)

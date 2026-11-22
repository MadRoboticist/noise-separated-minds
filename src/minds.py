"""The three minds: reactive (function), analytical (pdf), chaos (YOLO)."""

from typing import Any, Dict, List, Optional, Tuple
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


class AnalyticalMind:
    """Lightweight unsupervised conditional model.

    Stores vetted (x, y) pairs and answers by locally weighted average
    (or nearest-neighbor) over similar inputs. Only ever receives
    non-random observations.
    """

    def __init__(self, max_obs: int = 2048, bandwidth: float = 0.5):
        self.xs: List[np.ndarray] = []
        self.ys: List[Any] = []
        self.max_obs = max_obs
        self.bandwidth = bandwidth

    def observe(self, x: np.ndarray, y: Optional[Any] = None):
        x = np.asarray(x, dtype=float).ravel()
        self.xs.append(x)
        self.ys.append(y)
        if len(self.xs) > self.max_obs:
            # Drop oldest
            self.xs.pop(0)
            self.ys.pop(0)

    def _distances(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).ravel()
        if not self.xs:
            return np.array([])
        diffs = [np.linalg.norm(xi - x) for xi in self.xs]
        return np.asarray(diffs)

    def sample(self, x: np.ndarray, k: int = 5) -> Any:
        """Return a locally weighted prediction (or None if empty)."""
        if not self.xs:
            return None
        dists = self._distances(x)
        if len(dists) == 0:
            return None

        # Soft weights
        weights = np.exp(-dists ** 2 / (2 * self.bandwidth ** 2))
        weights /= weights.sum() + 1e-12

        # If outputs are numeric, return weighted mean; else return mode / nearest
        numeric = all(isinstance(y, (int, float, np.number)) for y in self.ys if y is not None)
        if numeric:
            vals = np.array([0.0 if y is None else float(y) for y in self.ys])
            return float(np.sum(weights * vals))

        # Non-numeric: return the highest-weight observed y
        idx = int(np.argmax(weights))
        return self.ys[idx]

    def uncertainty(self, x: np.ndarray) -> float:
        """Crude residual uncertainty: mean distance to neighbors."""
        dists = self._distances(x)
        if len(dists) == 0:
            return 1.0
        return float(np.mean(np.sort(dists)[:5]))

    def __len__(self):
        return len(self.xs)


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

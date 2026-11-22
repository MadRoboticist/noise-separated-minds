"""The three minds: reactive (function), analytical (pdf), chaos (YOLO)."""

from typing import Any, Callable, Dict, Optional
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


class AnalyticalMind:
    """Placeholder for a small generative model of structured uncertainty.
    Only ever trained on vetted (non-random) inputs.
    """

    def __init__(self):
        self.observations = []

    def observe(self, x: np.ndarray, y: Optional[Any] = None):
        self.observations.append((np.asarray(x), y))

    def sample(self, x: np.ndarray) -> Any:
        # Placeholder: return mean of similar past outputs or None
        return None


class ChaosPile:
    """Pure randomness. No model. Just rejected noise and exploratory draws."""

    def __init__(self):
        self.rejected = []

    def reject(self, x: np.ndarray):
        self.rejected.append(np.asarray(x))

    def sample(self, shape) -> np.ndarray:
        return np.random.randn(*shape)

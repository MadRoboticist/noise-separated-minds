"""Short-term elastic buffer.

High-plasticity provisional store. Patterns live here until the clerk decides
to promote them or let them decay.
"""

from collections import deque
from typing import Any, Deque, Optional
import numpy as np


class ShortTermBuffer:
    def __init__(self, maxlen: int = 256):
        self.buffer: Deque[dict] = deque(maxlen=maxlen)

    def add(self, x: np.ndarray, structure_score: float, provisional_output: Optional[Any] = None):
        self.buffer.append({
            "x": np.asarray(x),
            "score": structure_score,
            "output": provisional_output,
            "hits": 1,
        })

    def find_similar(self, x: np.ndarray, tol: float = 0.35) -> Optional[dict]:
        """Relaxed tolerance so noisy versions of the same pattern accumulate hits."""
        x = np.asarray(x).ravel()
        for item in reversed(self.buffer):
            if item["x"].shape == x.shape and np.allclose(item["x"], x, atol=tol):
                item["hits"] += 1
                return item
        return None

    def decay(self):
        """Simple aging: drop lowest-hit items if over capacity (handled by deque)."""
        pass

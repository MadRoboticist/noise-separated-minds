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
        """Add only if no similar entry exists; otherwise reinforce the existing one."""
        existing = self.find_similar(x, increment=False)
        if existing is not None:
            existing["hits"] += 1
            existing["score"] = structure_score
            if provisional_output is not None:
                existing["output"] = provisional_output
            return
        self.buffer.append({
            "x": np.asarray(x),
            "score": structure_score,
            "output": provisional_output,
            "hits": 1,
        })

    def find_similar(self, x: np.ndarray, tol: float = 0.35, increment: bool = True) -> Optional[dict]:
        """Relaxed tolerance so noisy versions of the same pattern accumulate hits."""
        x = np.asarray(x).ravel()
        for item in reversed(self.buffer):
            if item["x"].shape == x.shape and np.linalg.norm(item["x"] - x) < tol * np.sqrt(len(x)):
                if increment:
                    item["hits"] += 1
                return item
        return None

    def decay(self, min_hits: int = 2):
        """Drop entries that have not been reinforced."""
        from collections import deque
        kept = [item for item in self.buffer if item["hits"] >= min_hits]
        self.buffer = deque(kept, maxlen=256)

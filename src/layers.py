"""Thought / Knowledge / Wisdom layers.

- Thought: disposable, high plasticity (the analytical ensemble lives here).
- Knowledge: can fade with disuse or contradiction.
- Wisdom: only grows; append-only invariants.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from .minds import AnalyticalMind


class ThoughtLayer:
    """Short-term figuring. Can be lost entirely."""

    def __init__(self):
        self.mind = AnalyticalMind()  # the ensemble
        self.max_age = 50  # observations before forced decay consideration

    def observe(self, x: np.ndarray, y: Any = None):
        self.mind.observe(x, y)

    def sample(self, x: np.ndarray):
        return self.mind.sample(x)

    def uncertainty(self, x: np.ndarray) -> float:
        return self.mind.uncertainty(x)

    def clear(self):
        """Thoughts can be lost."""
        for p in self.mind.perspectives:
            p.clear()

    def __len__(self):
        return len(self.mind)


class KnowledgeItem:
    def __init__(self, x: np.ndarray, y: Any, strength: float = 1.0):
        self.x = np.asarray(x, dtype=float).ravel()
        self.y = y
        self.strength = strength
        self.hits = 1
        self.last_seen = 0


class KnowledgeLayer:
    """Medium-term. Strength fades with disuse; can be revised."""

    def __init__(self, fade_rate: float = 0.98, min_strength: float = 0.1):
        self.items: List[KnowledgeItem] = []
        self.fade_rate = fade_rate
        self.min_strength = min_strength
        self.tick = 0

    def add_or_reinforce(self, x: np.ndarray, y: Any, boost: float = 0.3):
        x = np.asarray(x, dtype=float).ravel()
        for item in self.items:
            if np.allclose(item.x, x, atol=0.15):
                item.hits += 1
                item.strength = min(2.0, item.strength + boost)
                item.last_seen = self.tick
                if y is not None:
                    item.y = y  # allow revision
                return
        self.items.append(KnowledgeItem(x, y, strength=1.0))

    def fade(self):
        """Knowledge can fade."""
        self.tick += 1
        alive = []
        for item in self.items:
            item.strength *= self.fade_rate
            if item.strength >= self.min_strength:
                alive.append(item)
        self.items = alive

    def lookup(self, x: np.ndarray, tol: float = 0.2) -> Optional[Any]:
        x = np.asarray(x, dtype=float).ravel()
        best = None
        best_d = tol
        for item in self.items:
            d = np.linalg.norm(item.x - x)
            if d < best_d and item.strength > self.min_strength:
                best_d = d
                best = item.y
        return best

    def strongest(self, n: int = 5) -> List[KnowledgeItem]:
        return sorted(self.items, key=lambda it: it.strength, reverse=True)[:n]

    def __len__(self):
        return len(self.items)


class WisdomLayer:
    """Long-term. Only grows. Never erased."""

    def __init__(self):
        self.entries: List[Tuple[np.ndarray, Any, float]] = []  # x, y, confidence

    def add(self, x: np.ndarray, y: Any, confidence: float = 1.0):
        """Append-only."""
        x = np.asarray(x, dtype=float).ravel()
        # Avoid exact duplicates
        for ex, ey, ec in self.entries:
            if np.allclose(ex, x, atol=1e-6):
                return  # already known
        self.entries.append((x, y, confidence))

    def lookup(self, x: np.ndarray, tol: float = 0.1) -> Optional[Any]:
        x = np.asarray(x, dtype=float).ravel()
        for ex, ey, ec in self.entries:
            if np.linalg.norm(ex - x) < tol:
                return ey
        return None

    def __len__(self):
        return len(self.entries)

    def all(self):
        return list(self.entries)

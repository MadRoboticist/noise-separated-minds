"""Thought / Knowledge / Wisdom with time-gap and re-validation."""

from typing import Any, List, Optional
import numpy as np
from .minds import AnalyticalMind


class ThoughtLayer:
    def __init__(self):
        self.mind = AnalyticalMind()

    def observe(self, x: np.ndarray, y: Any = None):
        self.mind.observe(x, y)

    def sample(self, x: np.ndarray):
        return self.mind.sample(x)

    def uncertainty(self, x: np.ndarray) -> float:
        return self.mind.uncertainty(x)

    def clear(self):
        for p in self.mind.perspectives:
            p.clear()

    def __len__(self):
        return len(self.mind)


class KnowledgeItem:
    """One Gaussian component: running mean of x, running mean of y."""
    def __init__(self, x: np.ndarray, y: Any, strength: float = 1.0, step: int = 0):
        self.x = np.asarray(x, dtype=float).ravel()  # mean of x
        self.y = float(y) if y is not None else 0.0  # mean of y
        self.y_var = 1.0
        self.n = 1
        self.strength = strength
        self.hits = 1
        self.first_seen = step
        self.last_seen = step

    def update(self, x, y):
        x = np.asarray(x, dtype=float).ravel()
        self.n += 1
        self.x += (x - self.x) / self.n
        if y is not None:
            y = float(y)
            delta = y - self.y
            self.y += delta / self.n
            self.y_var += (delta * (y - self.y) - self.y_var) / self.n
        self.hits += 1


class KnowledgeLayer:
    def __init__(self, fade_rate: float = 0.98, min_strength: float = 0.05):
        self.items: List[KnowledgeItem] = []
        self.fade_rate = fade_rate
        self.min_strength = min_strength

    def add_or_reinforce(self, x: np.ndarray, y: Any, boost: float = 0.3, step: int = 0):
        x = np.asarray(x, dtype=float).ravel()
        for item in self.items:
            if np.linalg.norm(item.x - x) < 0.08 * np.sqrt(len(x)):
                item.update(x, y)
                item.strength = max(0.0, min(3.0, item.strength + boost))
                item.last_seen = step
                return item
        if boost > 0:
            item = KnowledgeItem(x, y, strength=max(0.5, boost), step=step)
            self.items.append(item)
            return item
        return None

    def fade(self):
        alive = []
        for item in self.items:
            item.strength *= self.fade_rate
            if item.strength >= self.min_strength:
                alive.append(item)
        self.items = alive

    def lookup(self, x: np.ndarray, tol: float = None) -> Optional[Any]:
        x = np.asarray(x, dtype=float).ravel()
        if tol is None:
            tol = 0.08 * np.sqrt(len(x))
        best = None
        best_d = tol
        for item in self.items:
            if item.x.shape != x.shape:
                continue
            d = np.linalg.norm(item.x - x)
            if d < best_d and item.strength > self.min_strength:
                best_d = d
                best = item.y
        return best

    def find_item(self, x: np.ndarray, tol: float = 1.0) -> Optional[KnowledgeItem]:
        x = np.asarray(x, dtype=float).ravel()
        for item in self.items:
            if np.linalg.norm(item.x - x) < tol:
                return item
        return None

    def strongest(self, n: int = 5) -> List[KnowledgeItem]:
        return sorted(self.items, key=lambda it: it.strength, reverse=True)[:n]

    def __len__(self):
        return len(self.items)


class WisdomEntry:
    """Gaussian component for a stable pattern."""
    def __init__(self, x: np.ndarray, y: Any, confidence: float = 1.0, step: int = 0):
        self.x = np.asarray(x, dtype=float).ravel()
        self.y = float(y) if y is not None else 0.0
        self.y_var = 0.1
        self.n = 1
        self.confidence = confidence
        self.checks = 0
        self.failures = 0
        self.created = step
        self.last_checked = step

    def update(self, x, y):
        x = np.asarray(x, dtype=float).ravel()
        self.n += 1
        self.x += (x - self.x) / self.n
        if y is not None:
            y = float(y)
            delta = y - self.y
            self.y += delta / self.n
            self.y_var += (delta * (y - self.y) - self.y_var) / self.n


class WisdomLayer:
    def __init__(self, min_confidence: float = 0.3):
        self.entries: List[WisdomEntry] = []
        self.min_confidence = min_confidence

    def add(self, x: np.ndarray, y: Any, confidence: float = 1.0, step: int = 0):
        x = np.asarray(x, dtype=float).ravel()
        # Coarse match so one pattern is one entry
        for e in self.entries:
            if np.linalg.norm(e.x - x) < 0.08 * np.sqrt(len(x)):
                e.update(x, y)
                e.confidence = max(e.confidence, confidence)
                return
        self.entries.append(WisdomEntry(x, y, confidence, step))

    def lookup(self, x: np.ndarray, tol: float = None) -> Optional[Any]:
        x = np.asarray(x, dtype=float).ravel()
        if tol is None:
            tol = 0.08 * np.sqrt(len(x))
        for e in self.entries:
            if e.x.shape == x.shape and np.linalg.norm(e.x - x) < tol and e.confidence >= self.min_confidence:
                return e.y
        return None

    def check(self, x: np.ndarray, y: Any, tol: float = 0.5, step: int = 0) -> bool:
        if y is None:
            return True
        x = np.asarray(x, dtype=float).ravel()
        for e in self.entries:
            if np.linalg.norm(e.x - x) < 0.15:
                e.checks += 1
                e.last_checked = step
                if abs(float(e.y) - float(y)) > tol:
                    e.failures += 1
                    e.confidence = max(0.0, e.confidence - 0.25)
                    return False
                else:
                    e.confidence = min(2.0, e.confidence + 0.05)
                    return True
        return True

    def due_for_revalidation(self, current_step: int, interval: int = 50) -> List[WisdomEntry]:
        """Return entries that have not been checked recently."""
        return [e for e in self.entries if current_step - e.last_checked >= interval]

    def __len__(self):
        return len(self.entries)

    @property
    def active_count(self):
        return sum(1 for e in self.entries if e.confidence >= self.min_confidence)

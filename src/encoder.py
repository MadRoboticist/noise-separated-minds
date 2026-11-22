"""Turn non-numeric contexts into fixed-size vectors the rest of the system can use.

Supports:
- raw numeric arrays (pass-through)
- categorical / token strings (hashing trick)
- short text (same hashing, bag-of-tokens)
"""

from typing import Any, List, Union
import numpy as np


class ContextEncoder:
    def __init__(self, dim: int = 32, seed: int = 0):
        self.dim = dim
        self.rng = np.random.RandomState(seed)
        # Fixed random projection for hashing
        self.proj = self.rng.randn(1024, dim).astype(np.float32) / np.sqrt(dim)

    def _hash_token(self, token: str) -> int:
        return hash(token) % 1024

    def encode(self, x: Any) -> np.ndarray:
        """Return a 1-d float vector of length self.dim."""
        if isinstance(x, np.ndarray):
            v = x.astype(float).ravel()
            if len(v) == self.dim:
                return v
            # Pad or truncate
            out = np.zeros(self.dim)
            n = min(len(v), self.dim)
            out[:n] = v[:n]
            return out

        if isinstance(x, (int, float)):
            out = np.zeros(self.dim)
            out[0] = float(x)
            return out

        if isinstance(x, str):
            return self._encode_text(x)

        if isinstance(x, (list, tuple)):
            # Mix of numbers and strings
            acc = np.zeros(self.dim)
            for item in x:
                acc += self.encode(item)
            n = np.linalg.norm(acc)
            return acc / n if n > 1e-8 else acc

        # Fallback: hash the string representation
        return self._encode_text(str(x))

    def _encode_text(self, text: str) -> np.ndarray:
        tokens = text.lower().split()
        if not tokens:
            return np.zeros(self.dim)
        vec = np.zeros(1024, dtype=np.float32)
        for t in tokens:
            vec[self._hash_token(t)] += 1.0
        vec /= np.sqrt(len(tokens))
        return vec @ self.proj

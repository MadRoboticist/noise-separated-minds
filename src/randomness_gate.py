"""Unsupervised randomness / structure gate.

Detects pure noise so it can be rejected before it reaches long-term analytical memory.
"""

import numpy as np
from typing import Union


def spectral_flatness(x: np.ndarray) -> float:
    """Wiener entropy: geometric mean / arithmetic mean of power spectrum.
    Near 1.0 indicates noise-like flatness.
    """
    if len(x) < 4:
        return 1.0
    spectrum = np.abs(np.fft.rfft(x)) ** 2
    spectrum = spectrum[spectrum > 1e-12]
    if len(spectrum) == 0:
        return 1.0
    geo = np.exp(np.mean(np.log(spectrum)))
    arith = np.mean(spectrum)
    return float(geo / arith) if arith > 0 else 1.0


def compression_ratio(x: np.ndarray, n_bins: int = 32) -> float:
    """Crude compressibility proxy via histogram entropy.
    Higher values (closer to log2(n_bins)) indicate less structure.
    """
    if len(x) == 0:
        return 1.0
    hist, _ = np.histogram(x, bins=n_bins, density=True)
    hist = hist[hist > 0]
    if len(hist) == 0:
        return 1.0
    entropy = -np.sum(hist * np.log2(hist + 1e-12))
    max_ent = np.log2(n_bins)
    return float(entropy / max_ent) if max_ent > 0 else 1.0


class RandomnessGate:
    """Simple unsupervised gate. Returns structure score in [0, 1].
    Low score → pure randomness (route to chaos).
    """

    def __init__(self, flatness_weight: float = 0.6, entropy_weight: float = 0.4):
        self.flatness_weight = flatness_weight
        self.entropy_weight = entropy_weight

    def score(self, x: Union[np.ndarray, list]) -> float:
        x = np.asarray(x, dtype=float).ravel()
        if len(x) == 0:
            return 0.0
        flat = spectral_flatness(x)
        ent = compression_ratio(x)
        # Invert so higher = more structure
        structure = 1.0 - (self.flatness_weight * flat + self.entropy_weight * ent)
        return float(np.clip(structure, 0.0, 1.0))

    def is_noise(self, x: Union[np.ndarray, list], threshold: float = 0.35) -> bool:
        return self.score(x) < threshold

"""More robust unsupervised randomness / structure gate.

Combines several complementary tests so pure noise is rejected
before it can reach long-term memory.
"""

import numpy as np
from typing import Union


def spectral_flatness(x: np.ndarray) -> float:
    """Wiener entropy. Near 1.0 = noise-like."""
    if len(x) < 8:
        return 1.0
    spectrum = np.abs(np.fft.rfft(x - np.mean(x))) ** 2
    spectrum = spectrum[1:]  # drop DC
    spectrum = spectrum[spectrum > 1e-12]
    if len(spectrum) == 0:
        return 1.0
    geo = np.exp(np.mean(np.log(spectrum)))
    arith = np.mean(spectrum)
    return float(np.clip(geo / arith, 0.0, 1.0)) if arith > 0 else 1.0


def histogram_entropy(x: np.ndarray, n_bins: int = 24) -> float:
    """Normalized histogram entropy. Near 1.0 = unstructured."""
    if len(x) < 4:
        return 1.0
    hist, _ = np.histogram(x, bins=min(n_bins, max(4, len(x) // 2)), density=True)
    hist = hist[hist > 0]
    if len(hist) == 0:
        return 1.0
    ent = -np.sum(hist * np.log2(hist + 1e-12))
    max_ent = np.log2(len(hist))
    return float(np.clip(ent / max_ent, 0.0, 1.0)) if max_ent > 0 else 1.0


def autocorr_strength(x: np.ndarray, max_lag: int = 8) -> float:
    """Mean absolute autocorrelation at short lags.
    Near 0 = noise; higher = structure. Returns noise-likeness in [0,1].
    """
    if len(x) < max_lag + 2:
        return 1.0
    x = x - np.mean(x)
    var = np.dot(x, x) + 1e-12
    corrs = []
    for lag in range(1, max_lag + 1):
        c = np.dot(x[:-lag], x[lag:]) / var
        corrs.append(abs(c))
    # High autocorr = structure = low noise score
    return float(np.clip(1.0 - np.mean(corrs), 0.0, 1.0))


def diff_entropy(x: np.ndarray) -> float:
    """Entropy of first differences. Pure noise has high diff entropy."""
    if len(x) < 4:
        return 1.0
    d = np.diff(x)
    return histogram_entropy(d, n_bins=16)


def runs_noise(x: np.ndarray) -> float:
    """Simple runs-like score on sign of demeaned series.
    Random series have many runs; structured have fewer.
    Returns noise-likeness.
    """
    if len(x) < 6:
        return 1.0
    s = np.sign(x - np.median(x))
    s[s == 0] = 1
    runs = 1 + np.sum(s[1:] != s[:-1])
    # Expected runs for random ≈ n/2 + 1
    expected = len(x) / 2.0 + 1
    # More runs than expected → noisier
    ratio = runs / (expected + 1e-8)
    return float(np.clip(ratio / 1.5, 0.0, 1.0))


class RandomnessGate:
    """Multi-test structure score in [0, 1]. Higher = more structured.
    Low score → route to chaos.
    """

    def __init__(
        self,
        w_flat: float = 0.30,
        w_ent: float = 0.20,
        w_acf: float = 0.25,
        w_diff: float = 0.15,
        w_runs: float = 0.10,
    ):
        self.weights = np.array([w_flat, w_ent, w_acf, w_diff, w_runs])
        self.weights /= self.weights.sum()

    def score(self, x: Union[np.ndarray, list]) -> float:
        x = np.asarray(x, dtype=float).ravel()
        if len(x) == 0:
            return 0.0
        # Each helper returns noise-likeness in [0,1]
        noise_scores = np.array([
            spectral_flatness(x),
            histogram_entropy(x),
            autocorr_strength(x),
            diff_entropy(x),
            runs_noise(x),
        ])
        noise = float(np.dot(self.weights, noise_scores))
        return float(np.clip(1.0 - noise, 0.0, 1.0))

    def is_noise(self, x: Union[np.ndarray, list], threshold: float = 0.40) -> bool:
        return self.score(x) < threshold

    def detail(self, x: Union[np.ndarray, list]) -> dict:
        """Return individual test scores for debugging."""
        x = np.asarray(x, dtype=float).ravel()
        return {
            "spectral_flatness": spectral_flatness(x),
            "histogram_entropy": histogram_entropy(x),
            "autocorr_noise": autocorr_strength(x),
            "diff_entropy": diff_entropy(x),
            "runs_noise": runs_noise(x),
            "structure": self.score(x),
        }

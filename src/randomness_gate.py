"""Adaptive multi-test randomness gate.

Tracks recent structure scores and adjusts its decision threshold
to the observed noise floor. Also reports residual-noise statistics.
"""

import numpy as np
from typing import Union, List


def spectral_flatness(x: np.ndarray) -> float:
    if len(x) < 8:
        return 1.0
    spectrum = np.abs(np.fft.rfft(x - np.mean(x))) ** 2
    spectrum = spectrum[1:]
    spectrum = spectrum[spectrum > 1e-12]
    if len(spectrum) == 0:
        return 1.0
    geo = np.exp(np.mean(np.log(spectrum)))
    arith = np.mean(spectrum)
    return float(np.clip(geo / arith, 0.0, 1.0)) if arith > 0 else 1.0


def histogram_entropy(x: np.ndarray, n_bins: int = 24) -> float:
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
    if len(x) < max_lag + 2:
        return 1.0
    x = x - np.mean(x)
    var = np.dot(x, x) + 1e-12
    corrs = []
    for lag in range(1, max_lag + 1):
        c = np.dot(x[:-lag], x[lag:]) / var
        corrs.append(abs(c))
    return float(np.clip(1.0 - np.mean(corrs), 0.0, 1.0))


def diff_entropy(x: np.ndarray) -> float:
    if len(x) < 4:
        return 1.0
    return histogram_entropy(np.diff(x), n_bins=16)


def runs_noise(x: np.ndarray) -> float:
    if len(x) < 6:
        return 1.0
    s = np.sign(x - np.median(x))
    s[s == 0] = 1
    runs = 1 + np.sum(s[1:] != s[:-1])
    expected = len(x) / 2.0 + 1
    ratio = runs / (expected + 1e-8)
    return float(np.clip(ratio / 1.5, 0.0, 1.0))


class RandomnessGate:
    """Multi-test structure score with adaptive threshold."""

    def __init__(
        self,
        base_threshold: float = 0.40,
        adapt_rate: float = 0.02,
        window: int = 100,
    ):
        self.weights = np.array([0.30, 0.20, 0.25, 0.15, 0.10])
        self.weights /= self.weights.sum()
        self.base_threshold = base_threshold
        self.threshold = base_threshold
        self.adapt_rate = adapt_rate
        self.window = window
        self.recent_scores: List[float] = []
        self.rejected_count = 0
        self.accepted_count = 0

    def score(self, x: Union[np.ndarray, list]) -> float:
        x = np.asarray(x, dtype=float).ravel()
        if len(x) == 0:
            return 0.0
        # Spectral tests are unreliable below ~16 samples; use local consistency instead
        if len(x) < 16:
            return self._short_score(x)
        noise_scores = np.array([
            spectral_flatness(x),
            histogram_entropy(x),
            autocorr_strength(x),
            diff_entropy(x),
            runs_noise(x),
        ])
        noise = float(np.dot(self.weights, noise_scores))
        return float(np.clip(1.0 - noise, 0.0, 1.0))

    def _short_score(self, x: np.ndarray) -> float:
        """Score short vectors by how consistent they are with recent accepted inputs."""
        if not hasattr(self, "_recent_accepted"):
            self._recent_accepted = []
        if len(self._recent_accepted) < 5:
            self._recent_accepted.append(x.copy())
            if len(self._recent_accepted) > 40:
                self._recent_accepted.pop(0)
            return 0.55
        recent = np.array(self._recent_accepted[-20:])
        med = np.median(recent, axis=0)
        mad = np.median(np.abs(recent - med), axis=0) + 1e-6
        z = np.abs(x - med) / mad
        # high z => outlier relative to recent normal
        outlier_frac = float(np.mean(z > 4.0))
        s = float(np.clip(1.0 - outlier_frac, 0.0, 1.0))
        if s >= 0.4:
            self._recent_accepted.append(x.copy())
            if len(self._recent_accepted) > 40:
                self._recent_accepted.pop(0)
        return s

    def update(self, structure_score: float, was_rejected: bool):
        """Adapt threshold toward the lower tail of recent scores."""
        self.recent_scores.append(structure_score)
        if len(self.recent_scores) > self.window:
            self.recent_scores.pop(0)
        if was_rejected:
            self.rejected_count += 1
        else:
            self.accepted_count += 1
            if not hasattr(self, "_recent_accepted"):
                self._recent_accepted = []
            # only store if we have the raw x; caller will pass via score path
            # we store in score() for short vectors
            

        if len(self.recent_scores) >= 20:
            # For short-vector local scores the distribution is high and tight;
            # keep the threshold closer to base so we do not over-reject.
            p25 = float(np.percentile(self.recent_scores, 25))
            # Stronger pull toward base (0.6 instead of 0.3)
            target = 0.4 * p25 + 0.6 * self.base_threshold
            self.threshold += self.adapt_rate * (target - self.threshold)
            # Cap lower so short-vector mode stays permissive
            self.threshold = float(np.clip(self.threshold, 0.15, 0.50))

    def is_noise(self, x: Union[np.ndarray, list], threshold: float = None) -> bool:
        thr = self.threshold if threshold is None else threshold
        s = self.score(x)
        rejected = s < thr
        self.update(s, rejected)
        return rejected

    def residual_noise_rate(self) -> float:
        """Fraction of recent inputs that were rejected."""
        total = self.rejected_count + self.accepted_count
        return self.rejected_count / total if total else 0.0

    def detail(self, x: Union[np.ndarray, list]) -> dict:
        x = np.asarray(x, dtype=float).ravel()
        return {
            "spectral_flatness": spectral_flatness(x),
            "histogram_entropy": histogram_entropy(x),
            "autocorr_noise": autocorr_strength(x),
            "diff_entropy": diff_entropy(x),
            "runs_noise": runs_noise(x),
            "structure": self.score(x),
            "threshold": self.threshold,
            "residual_noise_rate": self.residual_noise_rate(),
        }


    def residual_weight(self, x, recent_residuals=None) -> float:
        """Soft weight in [0.05, 1.0] based on local residual profile.
        High residual relative to recent profile -> lower weight, not zero.
        """
        if recent_residuals is None or len(recent_residuals) < 8:
            return 1.0
        med = float(np.median(recent_residuals))
        mad = float(np.median(np.abs(np.array(recent_residuals) - med))) + 1e-6
        # Use the short score as a proxy for how structured this point is
        s = self.score(x)
        # Convert structure deficit into a soft weight
        deficit = max(0.0, self.threshold - s)
        # Also look at residual magnitude if we have it
        weight = float(np.clip(1.0 - 1.5 * deficit, 0.05, 1.0))
        return weight

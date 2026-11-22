"""Benchmark with periodic shocks: known patterns, wildly wrong outputs.

Compares the system (trauma negates reactive) against online linear,
k-NN, and a small MLP.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from src.system import NoiseSeparatedSystem
from src.config import SystemConfig


class OnlineLinear:
    def __init__(self, dim, lr=0.02):
        self.w = np.zeros(dim)
        self.b = 0.0
        self.lr = lr

    def observe(self, x, y):
        if y is None:
            return
        p = self.w @ x + self.b
        e = p - y
        self.w -= self.lr * e * x + 1e-3 * self.w
        self.b -= self.lr * e
        self.w = np.clip(self.w, -5, 5)

    def predict(self, x):
        return float(self.w @ x + self.b)


class OnlineKNN:
    def __init__(self, k=5, max_n=500):
        self.k = k
        self.xs, self.ys = [], []
        self.max_n = max_n

    def observe(self, x, y):
        if y is None:
            return
        self.xs.append(np.asarray(x, dtype=float))
        self.ys.append(float(y))
        if len(self.xs) > self.max_n:
            self.xs.pop(0)
            self.ys.pop(0)

    def predict(self, x):
        if not self.xs:
            return None
        d = np.array([np.linalg.norm(xi - x) for xi in self.xs])
        return float(np.mean([self.ys[i] for i in np.argsort(d)[: self.k]]))


class OnlineMLP:
    def __init__(self, dim, hidden=16, lr=0.01):
        rng = np.random.RandomState(0)
        self.W1 = rng.randn(hidden, dim) * 0.1
        self.b1 = np.zeros(hidden)
        self.W2 = rng.randn(hidden) * 0.1
        self.b2 = 0.0
        self.lr = lr

    def predict(self, x):
        h = np.tanh(self.W1 @ x + self.b1)
        return float(self.W2 @ h + self.b2)

    def observe(self, x, y):
        if y is None:
            return
        h = np.tanh(self.W1 @ x + self.b1)
        e = (self.W2 @ h + self.b2) - y
        dh = (1 - h ** 2) * self.W2
        self.W2 -= self.lr * e * h
        self.b2 -= self.lr * e
        self.W1 -= self.lr * e * np.outer(dh, x)
        self.b1 -= self.lr * e * dh


def make_stream(dim, n=800, n_patterns=4, shock_every=150, seed=0):
    """Discrete patterns with periodic shocks (wild output on a known pattern)."""
    rng = np.random.RandomState(seed)
    patterns = [np.zeros(dim) for _ in range(n_patterns)]
    for i, p in enumerate(patterns):
        p[i * (dim // n_patterns) : i * (dim // n_patterns) + 3] = [1.0, -0.5, 0.8]
    targets = [1.5, -0.8, 2.2, -1.1][:n_patterns]
    rows = []
    for t in range(n):
        if t > 0 and t % shock_every == 0:
            k = (t // shock_every) % n_patterns
            rows.append((patterns[k] + 0.02 * rng.randn(dim), targets[k] + 10.0, True))
        elif rng.rand() < 0.3:
            rows.append((rng.randn(dim), None, False))
        else:
            k = t % n_patterns
            rows.append((patterns[k] + 0.05 * rng.randn(dim), targets[k], False))
    return rows


def bench(dim, n=800):
    rows = make_stream(dim, n)
    s = NoiseSeparatedSystem(config=SystemConfig.small())
    lin, knn, mlp = OnlineLinear(dim), OnlineKNN(), OnlineMLP(dim)
    eo, el, ek, em = [], [], [], []
    for i, (x, y, is_shock) in enumerate(rows):
        # Score only on normal steps, not on the shocks themselves
        if i > 40 and y is not None and not is_shock:
            p, _ = s.predict(x)
            if p is not None:
                eo.append(abs(float(p) - y))
            el.append(abs(lin.predict(x) - y))
            kp = knn.predict(x)
            if kp is not None:
                ek.append(abs(kp - y))
            em.append(abs(mlp.predict(x) - y))
        s.observe(x, y)
        if y is not None:
            lin.observe(x, y)
            knn.observe(x, y)
            mlp.observe(x, y)
    return np.mean(eo), np.mean(el), np.mean(ek), np.mean(em), len(s.trauma)


if __name__ == "__main__":
    print(f"{'dim':>4}  {'ours':>7}  {'linear':>7}  {'knn':>7}  {'mlp':>7}  {'trauma':>6}")
    for dim in [32, 64, 128]:
        eo, el, ek, em, nt = bench(dim)
        print(f"{dim:4d}  {eo:7.4f}  {el:7.4f}  {ek:7.4f}  {em:7.4f}  {nt:6d}")

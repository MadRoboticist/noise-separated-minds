"""Compare NoiseSeparatedSystem against standard baselines on SKAB."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from src.system import NoiseSeparatedSystem
from src.config import SystemConfig

SKAB = "/tmp/SKAB/data"
FILES = [
    f"{SKAB}/valve1/1.csv",
    f"{SKAB}/valve1/2.csv",
    f"{SKAB}/other/1.csv",
]


def load(path):
    df = pd.read_csv(path, sep=";")
    cols = ["Accelerometer1RMS", "Accelerometer2RMS", "Current",
            "Pressure", "Temperature", "Thermocouple", "Voltage", "Volume Flow RateRMS"]
    X = df[cols].values.astype(float)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    # target: next Current (col 2), only on non-anomaly points
    y = X[:, 2].copy()
    anomaly = df["anomaly"].values.astype(int)
    return X, y, anomaly


class OnlineLinear:
    def __init__(self, dim, lr=0.01):
        self.w = np.zeros(dim)
        self.b = 0.0
        self.lr = lr
    def observe(self, x, y):
        if y is None:
            return
        pred = self.w @ x + self.b
        err = pred - y
        self.w -= self.lr * err * x
        self.b -= self.lr * err
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
        self.xs.append(x.copy())
        self.ys.append(y)
        if len(self.xs) > self.max_n:
            self.xs.pop(0)
            self.ys.pop(0)
    def predict(self, x):
        if not self.xs:
            return None
        d = [np.linalg.norm(x - xi) for xi in self.xs]
        idx = np.argsort(d)[:self.k]
        return float(np.mean([self.ys[i] for i in idx]))


class TinyMLP:
    def __init__(self, dim, h=16, lr=0.01):
        rng = np.random.RandomState(0)
        self.W1 = rng.randn(h, dim) * 0.1
        self.b1 = np.zeros(h)
        self.W2 = rng.randn(h) * 0.1
        self.b2 = 0.0
        self.lr = lr
    def observe(self, x, y):
        if y is None:
            return
        h = np.tanh(self.W1 @ x + self.b1)
        pred = self.W2 @ h + self.b2
        err = pred - y
        self.W2 -= self.lr * err * h
        dh = err * self.W2 * (1 - h**2)
        self.W1 -= self.lr * np.outer(dh, x)
        self.b1 -= self.lr * dh
        self.b2 -= self.lr * err
    def predict(self, x):
        h = np.tanh(self.W1 @ x + self.b1)
        return float(self.W2 @ h + self.b2)


def eval_model(model, X, y, anomaly, is_ours=False):
    errors = []
    for i in range(len(X)):
        if anomaly[i] == 1:
            model.observe(X[i], None)
            continue
        if i > 40:
            if is_ours:
                pred, _ = model.predict(X[i])
            else:
                pred = model.predict(X[i])
            if pred is not None:
                errors.append(abs(float(np.asarray(pred).ravel()[0]) - y[i]))
        model.observe(X[i], y[i])
    return float(np.mean(errors)) if errors else float("nan")


def main():
    print("=== SKAB: our system vs standard online baselines ===\n")
    names = ["ours", "linear", "knn", "mlp"]
    results = {n: [] for n in names}
    for path in FILES:
        if not os.path.exists(path):
            continue
        X, y, anomaly = load(path)
        dim = X.shape[1]
        label = os.path.basename(os.path.dirname(path)) + "/" + os.path.basename(path)
        print(f"-- {label} --")
        models = {
            "ours": NoiseSeparatedSystem(config=SystemConfig.small()),
            "linear": OnlineLinear(dim),
            "knn": OnlineKNN(),
            "mlp": TinyMLP(dim),
        }
        for n, m in models.items():
            err = eval_model(m, X, y, anomaly, is_ours=(n == "ours"))
            results[n].append(err)
            print(f"  {n:8s}  err={err:.3f}")
    print("\n--- Means ---")
    for n in names:
        print(f"{n:8s}  mean_err={np.mean(results[n]):.3f}")


if __name__ == "__main__":
    main()

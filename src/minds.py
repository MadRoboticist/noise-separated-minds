"""Thought ensemble. Generative perspectives can influence blender weights."""

from typing import Any, Dict, List, Optional
import numpy as np


class ReactiveMind:
    def __init__(self, max_entries: int = 10_000):
        self.table: Dict[bytes, Any] = {}
        self.max_entries = max_entries

    def _key(self, x: np.ndarray) -> bytes:
        v = np.round(np.asarray(x, dtype=float).ravel(), 1)
        return v.tobytes()

    def store(self, x: np.ndarray, y: Any, strength: float = 1.0):
        """Store with a strength that can fade."""
        # One pattern = one entry. If a nearby key exists, reinforce it instead.
        x = np.asarray(x, dtype=float).ravel()
        for k in list(self.table.keys()):
            try:
                xi = np.frombuffer(k, dtype=float)
                if xi.shape == x.shape and np.linalg.norm(xi - x) < 0.6:
                    self.table[k] = (y, strength)
                    return
            except Exception:
                continue
        if len(self.table) >= self.max_entries:
            self.table.pop(next(iter(self.table)))
        self.table[self._key(x)] = (y, strength)

    def lookup(self, x: np.ndarray, tol: float = 0.6) -> Optional[Any]:
        exact = self.table.get(self._key(x))
        if exact is not None:
            return exact[0] if isinstance(exact, tuple) else exact
        x = np.asarray(x, dtype=float).ravel()
        best, best_d = None, tol
        for k, v in self.table.items():
            # keys are raw bytes of earlier vectors; reconstruct only if length matches
            try:
                xi = np.frombuffer(k, dtype=float)
                if xi.shape != x.shape:
                    continue
                d = np.linalg.norm(xi - x)
                if d < best_d:
                    best_d, best = d, v
            except Exception:
                continue
        return best

    def fade(self, rate: float = 0.995, min_strength: float = 0.05):
        """Decay strength; drop entries that fall below min_strength."""
        dead = []
        for k, v in list(self.table.items()):
            if isinstance(v, tuple):
                y, s = v
                s *= rate
                if s < min_strength:
                    dead.append(k)
                else:
                    self.table[k] = (y, s)
        for k in dead:
            self.table.pop(k, None)

    def fade(self, rate: float = 0.99, min_strength: float = 0.05):
        """Decay trauma strength; drop weak entries."""
        dead = []
        for k, entry in list(self.table.items()):
            if len(entry) >= 5:
                s = entry[4] * rate
                if s < min_strength:
                    dead.append(k)
                else:
                    self.table[k] = entry[:4] + (s,)
        for k in dead:
            self.table.pop(k, None)

    def __len__(self):
        return len(self.table)


class Perspective:
    def __init__(self, name: str, max_obs: int = 2_000):
        self.name = name
        self.enabled = True
        self.auto_disabled = False
        self.max_obs = max_obs
        self.xs: List[np.ndarray] = []
        self.ys: List[Any] = []
        self.recent_errors: List[float] = []
        self.generative_bonus = 1.0  # can be raised by confirmed hypotheses

    def observe(self, x: np.ndarray, y: Any):
        self.xs.append(np.asarray(x, dtype=float).ravel())
        self.ys.append(y)
        if len(self.xs) > self.max_obs:
            self.xs.pop(0)
            self.ys.pop(0)

    def record_error(self, err: float):
        self.recent_errors.append(abs(float(err)))
        if len(self.recent_errors) > 40:
            self.recent_errors.pop(0)

    def accuracy_weight(self) -> float:
        if not self.recent_errors:
            return 1.0
        return 1.0 / (1.0 + float(np.mean(self.recent_errors)))

    def is_collapsed(self, threshold: float = 5.0) -> bool:
        if len(self.recent_errors) < 8:
            return False
        return float(np.mean(self.recent_errors)) > threshold

    def predict(self, x: np.ndarray) -> Optional[float]:
        raise NotImplementedError

    def uncertainty(self, x: np.ndarray) -> float:
        return 1.0

    def sample(self, x: np.ndarray) -> Optional[float]:
        return self.predict(x)

    def clear(self):
        self.xs.clear()
        self.ys.clear()
        self.recent_errors.clear()
        self.auto_disabled = False
        self.generative_bonus = 1.0


class LocalWeightedPerspective(Perspective):
    def __init__(self, bandwidth: float = 0.5, max_obs: int = 2_000):
        super().__init__("local_weighted", max_obs)
        self.bandwidth = bandwidth

    def predict(self, x: np.ndarray) -> Optional[float]:
        if not self.xs:
            return None
        x = np.asarray(x, dtype=float).ravel()
        dists = np.array([np.linalg.norm(xi - x) for xi in self.xs])
        weights = np.exp(-dists ** 2 / (2 * self.bandwidth ** 2))
        s = weights.sum()
        if s < 1e-12:
            return None
        weights /= s
        vals = np.array([float(y) if y is not None else 0.0 for y in self.ys])
        return float(np.sum(weights * vals))

    def uncertainty(self, x: np.ndarray) -> float:
        if not self.xs:
            return 1.0
        x = np.asarray(x, dtype=float).ravel()
        dists = np.array([np.linalg.norm(xi - x) for xi in self.xs])
        return float(np.mean(np.sort(dists)[: min(5, len(dists))]))


class NearestNeighborPerspective(Perspective):
    def __init__(self, k: int = 1, max_obs: int = 2_000):
        super().__init__(f"nn_k{k}", max_obs)
        self.k = k

    def predict(self, x: np.ndarray) -> Optional[float]:
        if not self.xs:
            return None
        x = np.asarray(x, dtype=float).ravel()
        dists = np.array([np.linalg.norm(xi - x) for xi in self.xs])
        idx = np.argsort(dists)[: self.k]
        vals = [float(self.ys[i]) for i in idx if self.ys[i] is not None]
        return float(np.mean(vals)) if vals else None

    def uncertainty(self, x: np.ndarray) -> float:
        if not self.xs:
            return 1.0
        x = np.asarray(x, dtype=float).ravel()
        dists = np.array([np.linalg.norm(xi - x) for xi in self.xs])
        return float(np.min(dists))


class OnlineLinearPerspective(Perspective):
    def __init__(self, lr: float = 0.05, reg: float = 1e-3, max_obs: int = 2_000):
        super().__init__("online_linear", max_obs)
        self.lr = lr
        self.reg = reg
        self.w = None

    def observe(self, x: np.ndarray, y: Any):
        super().observe(x, y)
        if y is None:
            return
        x = np.asarray(x, dtype=float).ravel()
        if self.w is None or len(self.w) != len(x) + 1:
            self.w = np.zeros(len(x) + 1)
        x_aug = np.append(x, 1.0)
        pred = float(x_aug @ self.w)
        err = float(y) - pred
        # Decaying learning rate
        step = getattr(self, "_step", 0) + 1
        self._step = step
        lr = self.lr / (1.0 + 0.001 * step)
        delta = lr * (err * x_aug - self.reg * self.w)
        self.w += np.clip(delta, -1.0, 1.0)

    def predict(self, x: np.ndarray) -> Optional[float]:
        if self.w is None:
            return None
        x = np.asarray(x, dtype=float).ravel()
        if len(self.w) != len(x) + 1:
            return None
        return float(np.append(x, 1.0) @ self.w)


class GaussianMixturePerspective(Perspective):
    def __init__(self, n_components: int = 3, max_obs: int = 2_000):
        super().__init__("gmm", max_obs)
        self.n_components = n_components
        self.means = None
        self.stds = None
        self.weights = None
        self.proj_centers = None

    def observe(self, x: np.ndarray, y: Any):
        super().observe(x, y)
        if y is None or len(self.ys) < 12:
            return
        proj = np.array([xi.mean() for xi in self.xs])
        ys = np.array([float(yi) if yi is not None else 0.0 for yi in self.ys])
        qs = np.linspace(10, 90, self.n_components)
        centers = np.percentile(proj, qs)
        for _ in range(6):
            dists = np.abs(proj[:, None] - centers[None, :])
            assign = np.argmin(dists, axis=1)
            for k in range(self.n_components):
                mask = assign == k
                if mask.any():
                    centers[k] = proj[mask].mean()
        self.proj_centers = centers
        self.means = np.zeros(self.n_components)
        self.stds = np.ones(self.n_components)
        self.weights = np.ones(self.n_components) / self.n_components
        dists = np.abs(proj[:, None] - centers[None, :])
        assign = np.argmin(dists, axis=1)
        for k in range(self.n_components):
            mask = assign == k
            if mask.any():
                self.means[k] = ys[mask].mean()
                self.stds[k] = ys[mask].std() + 0.1
                self.weights[k] = mask.mean()
        self.weights /= self.weights.sum() + 1e-12

    def _responsibilities(self, x: np.ndarray) -> Optional[np.ndarray]:
        if self.proj_centers is None:
            return None
        p = float(np.asarray(x).ravel().mean())
        d = np.abs(p - self.proj_centers)
        w = np.exp(-d) * self.weights
        return w / (w.sum() + 1e-12)

    def predict(self, x: np.ndarray) -> Optional[float]:
        r = self._responsibilities(x)
        if r is None:
            return None
        return float(np.sum(r * self.means))

    def uncertainty(self, x: np.ndarray) -> float:
        r = self._responsibilities(x)
        if r is None:
            return 1.0
        var = np.sum(r * (self.stds ** 2 + (self.means - np.sum(r * self.means)) ** 2))
        return float(np.sqrt(var + 1e-6))

    def sample(self, x: np.ndarray) -> Optional[float]:
        r = self._responsibilities(x)
        if r is None:
            return None
        k = np.random.choice(self.n_components, p=r)
        return float(self.means[k] + self.stds[k] * np.random.randn())


class LatentResidualPerspective(Perspective):
    def __init__(self, latent_dim: int = 2, max_obs: int = 2_000):
        super().__init__("latent_residual", max_obs)
        self.latent_dim = latent_dim
        self.W = None
        self.bandwidth = 0.5

    def observe(self, x: np.ndarray, y: Any):
        super().observe(x, y)
        if self.W is None and self.xs:
            d = len(self.xs[0])
            rng = np.random.RandomState(42)
            self.W = rng.randn(d, self.latent_dim) / np.sqrt(d)

    def _latent(self, x: np.ndarray) -> Optional[np.ndarray]:
        if self.W is None:
            return None
        return np.asarray(x, dtype=float).ravel() @ self.W

    def predict(self, x: np.ndarray) -> Optional[float]:
        if not self.xs or self.W is None:
            return None
        z = self._latent(x)
        zs = [xi @ self.W for xi in self.xs]
        dists = np.array([np.linalg.norm(zi - z) for zi in zs])
        weights = np.exp(-dists ** 2 / (2 * self.bandwidth ** 2))
        s = weights.sum()
        if s < 1e-12:
            return None
        weights /= s
        vals = np.array([float(y) if y is not None else 0.0 for y in self.ys])
        return float(np.sum(weights * vals))

    def sample(self, x: np.ndarray) -> Optional[float]:
        mu = self.predict(x)
        if mu is None:
            return None
        return float(mu + 0.1 * np.random.randn())



class TinyMLPPerspective(Perspective):
    """Small online MLP perspective."""
    def __init__(self, hidden=16, lr=0.01, max_obs=2_000):
        super().__init__("mlp", max_obs)
        self.hidden = hidden
        self.lr = lr
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = 0.0

    def _ensure(self, dim):
        if self.W1 is None:
            rng = np.random.RandomState(0)
            self.W1 = rng.randn(self.hidden, dim) * 0.1
            self.b1 = np.zeros(self.hidden)
            self.W2 = rng.randn(self.hidden) * 0.1

    def observe(self, x, y=None):
        super().observe(x, y)
        if y is None:
            return
        x = np.asarray(x, dtype=float).ravel()
        self._ensure(len(x))
        h = np.tanh(self.W1 @ x + self.b1)
        pred = float(self.W2 @ h + self.b2)
        err = pred - float(y)
        # L2 + gradient clip
        reg = 1e-3
        gW2 = np.clip(err * h + reg * self.W2, -1.0, 1.0)
        dh = err * self.W2 * (1 - h**2)
        gW1 = np.clip(np.outer(dh, x) + reg * self.W1, -1.0, 1.0)
        self.W2 -= self.lr * gW2
        self.W1 -= self.lr * gW1
        self.b1 -= self.lr * np.clip(dh, -1.0, 1.0)
        self.b2 -= self.lr * np.clip(err, -1.0, 1.0)

    def predict(self, x):
        if self.W1 is None:
            return None
        x = np.asarray(x, dtype=float).ravel()
        h = np.tanh(self.W1 @ x + self.b1)
        return float(self.W2 @ h + self.b2)

    def sample(self, x):
        return self.predict(x)


class AnalyticalMind:
    def __init__(self, max_obs: int = 2_000, collapse_threshold: float = 4.0):
        self.perspectives: List[Perspective] = [
            OnlineLinearPerspective(max_obs=max_obs),
            TinyMLPPerspective(hidden=2, max_obs=max_obs),
        ]
        self.perspectives[1].name = "mlp2"
        self.blender_mode = "greedy"
        self.collapse_threshold = collapse_threshold
        self.events: List[str] = []

    def set_enabled(self, names: List[str]):
        for p in self.perspectives:
            p.enabled = any(p.name == n or p.name.startswith(n) for n in names)
            p.auto_disabled = False

    def enable_all(self):
        for p in self.perspectives:
            p.enabled = True
            p.auto_disabled = False

    def _refresh_disabled(self):
        for p in self.perspectives:
            was = p.auto_disabled
            if p.enabled and p.is_collapsed(self.collapse_threshold):
                p.auto_disabled = True
            elif p.auto_disabled and not p.is_collapsed(self.collapse_threshold * 0.55):
                p.auto_disabled = False
            if p.auto_disabled and not was:
                self.events.append(f"disabled:{p.name}")
            elif was and not p.auto_disabled:
                self.events.append(f"recovered:{p.name}")
        if len(self.events) > 200:
            self.events = self.events[-200:]

    def observe(self, x: np.ndarray, y: Optional[Any] = None):
        self._refresh_disabled()
        for p in self.perspectives:
            if p.enabled and not p.auto_disabled:
                if y is not None:
                    pred = p.predict(x)
                    if pred is not None:
                        p.record_error(float(y) - pred)
                p.observe(x, y)

    def _active(self):
        return [p for p in self.perspectives if p.enabled and not p.auto_disabled]

    def agreement(self, x: np.ndarray, tol: float = 0.3) -> float:
        preds = [p.predict(x) for p in self._active()]
        preds = [pr for pr in preds if pr is not None]
        if len(preds) < 2:
            return 1.0
        preds = np.array(preds)
        med = np.median(preds)
        return float(np.mean(np.abs(preds - med) <= tol))

    def sample(self, x: np.ndarray) -> Optional[float]:
        active = self._active() or [p for p in self.perspectives if p.enabled]
        if not active:
            return None
        preds, weights = [], []
        for p in active:
            pr = p.predict(x)
            if pr is not None:
                # Generative bonus multiplies the weight
                w = p.accuracy_weight() * p.generative_bonus / max(p.uncertainty(x), 0.05)
                preds.append(pr)
                weights.append(w)
        if not preds:
            return None
        preds = np.array(preds)
        weights = np.array(weights)
        weights /= weights.sum() + 1e-12
        if self.blender_mode == "greedy":
            # Pick the perspective with the lowest recent error, not the heuristic weight
            best_i, best_err = 0, float("inf")
            for i, p in enumerate(active):
                if p.predict(x) is None:
                    continue
                e = float(np.mean(p.recent_errors)) if getattr(p, "recent_errors", None) else float("inf")
                if e < best_err:
                    best_err, best_i = e, i
            return float(preds[min(best_i, len(preds)-1)])
        if self.blender_mode == "mean":
            return float(np.mean(preds))
        return float(np.sum(weights * preds))

    def sample_generative(self, x: np.ndarray) -> Optional[float]:
        for p in self._active():
            if isinstance(p, (GaussianMixturePerspective, LatentResidualPerspective)):
                return p.sample(x)
        return self.sample(x)

    def boost_generative(self, amount: float = 0.15):
        """Raise blender weight of generative perspectives after confirmed hypotheses."""
        for p in self.perspectives:
            if isinstance(p, (GaussianMixturePerspective, LatentResidualPerspective)):
                p.generative_bonus = min(2.5, p.generative_bonus + amount)

    def decay_generative_bonus(self):
        for p in self.perspectives:
            p.generative_bonus = max(1.0, p.generative_bonus * 0.995)

    def uncertainty(self, x: np.ndarray) -> float:
        active = self._active()
        if not active:
            return 1.0
        uncs = [p.uncertainty(x) for p in active]
        preds = [p.predict(x) for p in active]
        preds = [pr for pr in preds if pr is not None]
        disagree = float(np.std(preds)) if len(preds) > 1 else 0.0
        return float(np.mean(uncs) + 0.5 * disagree)

    def __len__(self):
        return len(self.perspectives[0].xs) if self.perspectives else 0

    @property
    def names(self):
        return [p.name for p in self.perspectives]

    @property
    def active_names(self):
        return [p.name for p in self._active()]


class ChaosPile:
    """Running Gaussian model of rejected inputs. Fixed size."""
    def __init__(self, max_rejected: int = 5_000):
        self.n = 0
        self.mean = None
        self.var = None
        self.probe_successes = 0
        self.probe_attempts = 0
        # compat: empty list so old code that checks .rejected still works
        self.rejected: List[np.ndarray] = []

    def reject(self, x: np.ndarray):
        x = np.asarray(x, dtype=float).ravel()
        if self.mean is None:
            self.mean = x.copy()
            self.var = np.ones_like(x)
            self.n = 1
            return
        if self.mean.shape != x.shape:
            return
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.var += (delta * (x - self.mean) - self.var) / self.n
        self.var = np.maximum(self.var, 1e-6)

    def sample(self, shape) -> np.ndarray:
        if self.mean is None:
            return np.random.randn(*shape)
        return self.mean + np.sqrt(self.var) * np.random.randn(*self.mean.shape)

    def probe(self, n: int = 5) -> List[np.ndarray]:
        if self.mean is None:
            return []
        self.probe_attempts += 1
        return [self.sample(self.mean.shape) for _ in range(n)]

    def record_probe_success(self):
        self.probe_successes += 1

    def __len__(self):
        return self.n


class TraumaStore:
    """Fast-forming responses from high-error or shocking events.
    These are the primary targets for the auditor.
    """
    def __init__(self):
        self.table = {}  # key -> (x, y, error, step)

    def store(self, x, y, error: float, step: int):
        key = self._key(x)
        self.table[key] = (np.asarray(x, dtype=float).ravel().copy(), y, float(error), step, 1.0)

    def lookup(self, x, tol: float = 0.5):
        x = np.asarray(x, dtype=float).ravel()
        best, best_d = None, tol
        for k, entry in self.table.items():
            xi, y, err, step = entry[0], entry[1], entry[2], entry[3]
            if xi.shape != x.shape:
                continue
            d = np.linalg.norm(xi - x)
            if d < best_d:
                best_d, best = d, y
        return best

    def _key(self, x):
        return np.asarray(x, dtype=float).ravel().tobytes()

    def fade(self, rate: float = 0.995, min_strength: float = 0.05):
        """Decay strength; drop entries that fall below min_strength."""
        dead = []
        for k, v in list(self.table.items()):
            if isinstance(v, tuple):
                y, s = v
                s *= rate
                if s < min_strength:
                    dead.append(k)
                else:
                    self.table[k] = (y, s)
        for k in dead:
            self.table.pop(k, None)

    def fade(self, rate: float = 0.99, min_strength: float = 0.05):
        """Decay trauma strength; drop weak entries."""
        dead = []
        for k, entry in list(self.table.items()):
            if len(entry) >= 5:
                s = entry[4] * rate
                if s < min_strength:
                    dead.append(k)
                else:
                    self.table[k] = entry[:4] + (s,)
        for k in dead:
            self.table.pop(k, None)

    def __len__(self):
        return len(self.table)


class FeedbackLinear:
    """Linear model on the feedback path.
    Learns to predict the residual from the input, then supplies a correction
    that is subtracted from the graph output before it is fed back.
    """
    def __init__(self, lr: float = 0.02, reg: float = 1e-3):
        self.lr = lr
        self.reg = reg
        self.w = None
        self._step = 0

    def observe(self, x, residual):
        if residual is None:
            return
        x = np.asarray(x, dtype=float).ravel()
        if self.w is None or len(self.w) != len(x) + 1:
            self.w = np.zeros(len(x) + 1)
        x_aug = np.append(x, 1.0)
        pred = float(x_aug @ self.w)
        err = float(residual) - pred
        self._step += 1
        lr = self.lr / (1.0 + 0.001 * self._step)
        delta = lr * (err * x_aug - self.reg * self.w)
        self.w += np.clip(delta, -0.5, 0.5)

    def correct(self, x) -> float:
        if self.w is None:
            return 0.0
        x = np.asarray(x, dtype=float).ravel()
        if len(self.w) != len(x) + 1:
            return 0.0
        return float(np.append(x, 1.0) @ self.w)


class NextInputLinear:
    """Feedforward model: predicts x_{t+1} from x_t.
    One linear map, online, with L2 and a clipped update.
    """
    def __init__(self, lr: float = 0.02, reg: float = 1e-3):
        self.lr = lr
        self.reg = reg
        self.W = None  # (dim, dim+1) including bias
        self._step = 0
        self.last_error = None

    def observe(self, x_t, x_next):
        if x_next is None:
            return
        x_t = np.asarray(x_t, dtype=float).ravel()
        x_next = np.asarray(x_next, dtype=float).ravel()
        if x_t.shape != x_next.shape:
            return
        d = len(x_t)
        if self.W is None or self.W.shape != (d, d + 1):
            self.W = np.zeros((d, d + 1))
        x_aug = np.append(x_t, 1.0)
        pred = self.W @ x_aug
        err = x_next - pred
        self.last_error = float(np.mean(np.abs(err)))
        self._step += 1
        lr = self.lr / (1.0 + 0.001 * self._step)
        # clipped gradient step
        grad = np.outer(err, x_aug) - self.reg * self.W
        self.W += np.clip(lr * grad, -0.5, 0.5)

    def predict(self, x_t):
        if self.W is None:
            return None
        x_t = np.asarray(x_t, dtype=float).ravel()
        if self.W.shape[1] != len(x_t) + 1:
            return None
        return self.W @ np.append(x_t, 1.0)

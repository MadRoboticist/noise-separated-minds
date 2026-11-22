"""Adaptive parameter search over the full configuration suite.

Samples combinations of gate, auditor, knowledge, wisdom, and
ensemble-related parameters and keeps the set that produces the
lowest recent error.
"""

from typing import Dict, Any, List, Optional
import itertools
import numpy as np
from .config import SystemConfig
from .system import NoiseSeparatedSystem


# Full suite: every knob that affects routing, permanence, or audit
PARAM_SPACE = {
    # gate / noise
    "noise_threshold": [0.0, 0.25, 0.35, 0.50, 0.65],
    # promotion into knowledge / wisdom
    "promote_hits": [2, 3, 4, 6],
    "promote_score": [0.10, 0.20, 0.30, 0.42, 0.55],
    "min_agreement": [0.3, 0.5, 0.6, 0.75],
    "min_age": [2, 4, 8, 16],
    "knowledge_boost": [0.1, 0.3, 0.4, 0.7, 1.0],
    "wisdom_strength": [0.5, 1.0, 1.5, 2.5],
    # exploration / chaos
    "explore_prob": [0.0, 0.02, 0.05, 0.10],
    "chaos_promote_budget": [0.05, 0.15, 0.30],
    # demotion / audit
    "demote_error_tol": [0.2, 0.35, 0.5, 0.8],
    "audit_every": [10**9, 15, 30, 60],
    "revalidate_interval": [20, 40, 80],
    "graph_mode": ["cascade", "blend", "thought_consults"],
}


class AdaptiveTuner:
    def __init__(self, n_samples: int = 40, seed: int = 0):
        self.n_samples = n_samples
        self.rng = np.random.RandomState(seed)
        self.best_cfg: Dict[str, Any] = {}
        self.best_error = float("inf")
        self.history: List[Dict[str, Any]] = []

    def sample_config(self) -> Dict[str, Any]:
        """Draw one random configuration from the full suite."""
        cfg = {}
        for k, choices in PARAM_SPACE.items():
            cfg[k] = choices[self.rng.randint(0, len(choices))]
        return cfg

    def evaluate(self, X: np.ndarray, y: np.ndarray, overrides: Dict[str, Any]) -> float:
        cfg = SystemConfig.small()
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        s = NoiseSeparatedSystem(config=cfg)
        if "graph_mode" in overrides:
            s.graph_mode = overrides["graph_mode"]
        if "edges" in overrides:
            for k, w in overrides["edges"].items():
                src, dst = k.split("->")
                s.graph.set_weight(src, dst, w)
        s.edge_weights = {
            "thought": overrides.get("edge_w_thought", 1.0),
            "knowledge": overrides.get("edge_w_knowledge", 1.2),
            "wisdom": overrides.get("edge_w_wisdom", 1.5),
            "reactive": overrides.get("edge_w_reactive", 1.0),
            "trauma": overrides.get("edge_w_trauma", 0.4),
        }
        if overrides.get("audit_every", 20) > 1000:
            s.audit_every = 10**9
        errors = []
        for i in range(len(X)):
            if i > 15:
                pred, _ = s.predict(X[i])
                if pred is not None:
                    errors.append(abs(float(np.asarray(pred).ravel()[0]) - float(y[i])))
            s.observe(X[i], y[i])
        return float(np.mean(errors)) if errors else float("inf")

    def search(self, X: np.ndarray, y: np.ndarray, n_samples: Optional[int] = None) -> Dict[str, Any]:
        """Random search over the full suite. Returns best overrides."""
        n = n_samples or self.n_samples
        best = {}
        best_err = float("inf")
        for i in range(n):
            combo = self.sample_config()
            err = self.evaluate(X, y, combo)
            self.history.append({"cfg": combo, "error": err})
            if err < best_err:
                best_err = err
                best = combo
                print(f"  [{i+1}/{n}] new best {err:.4f}  thr={combo['noise_threshold']} audit={combo['audit_every']} hits={combo['promote_hits']}")
        self.best_cfg = best
        self.best_error = best_err
        return best

    def top(self, k: int = 5) -> List[Dict[str, Any]]:
        return sorted(self.history, key=lambda h: h["error"])[:k]


    def sample_edges(self) -> dict:
        """Sample a weight for every edge in the graph."""
        from .mind_graph import DEFAULT_EDGES
        edges = {}
        for (src, dst) in DEFAULT_EDGES:
            # sample around the default, clipped to [0, 2.5]
            base = DEFAULT_EDGES[(src, dst)]
            w = float(np.clip(base + self.rng.uniform(-0.4, 0.4), 0.0, 1.0))
            edges[f"{src}->{dst}"] = round(w, 2)
        return edges

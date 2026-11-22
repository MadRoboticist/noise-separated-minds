
"""Weighted graph of minds.

Every edge has a learnable weight. Nodes compute their output
as a weighted combination of the nodes that feed them.
"""

from typing import Dict, Optional
import numpy as np


# Edges: (src, dst) -> default weight
DEFAULT_EDGES = {
    # Thought is informed by clerk, knowledge, wisdom, trauma
    ("clerk", "thought"): 1.0,
    ("knowledge", "thought"): 0.8,
    ("wisdom", "thought"): 0.6,
    ("trauma", "thought"): 0.3,
    # Knowledge is informed by thought and experience (output)
    ("thought", "knowledge"): 0.7,
    ("output", "knowledge"): 0.5,
    # Reactive (muscle) is informed by experience and knowledge
    ("output", "reactive"): 0.6,
    ("knowledge", "reactive"): 0.8,
    # Trauma is fed by clerk and chaos
    ("clerk", "trauma"): 0.9,
    ("chaos", "trauma"): 0.4,
    # Chaos is fed by clerk and trauma
    ("clerk", "chaos"): 0.7,
    ("trauma", "chaos"): 0.5,
    # Wisdom is fed by knowledge, experience, chaos, trauma
    ("knowledge", "wisdom"): 0.8,
    ("output", "wisdom"): 0.5,
    ("chaos", "wisdom"): 0.3,
    ("trauma", "wisdom"): 0.4,
    # Output is fed by all
    ("thought", "output"): 1.0,
    ("knowledge", "output"): 1.2,
    ("reactive", "output"): 0.8,
    ("trauma", "output"): 0.2,
    ("wisdom", "output"): 0.9,
    ("chaos", "output"): 0.1,
}


class MindGraph:
    def __init__(self, edges: Optional[Dict] = None):
        self.edges = dict(DEFAULT_EDGES)
        if edges:
            self.edges.update(edges)

    def weight(self, src: str, dst: str) -> float:
        return float(self.edges.get((src, dst), 0.0))

    def set_weight(self, src: str, dst: str, w: float):
        self.edges[(src, dst)] = float(w)

    def compute_output(self, signals: Dict[str, Optional[float]]) -> Optional[float]:
        """Blend all nodes that feed output, using edge weights.
        signals: node_name -> value (or None if no vote)
        """
        num, den = 0.0, 0.0
        for src in ("thought", "knowledge", "reactive", "trauma", "wisdom", "chaos"):
            v = signals.get(src)
            if v is None:
                continue
            w = self.weight(src, "output")
            num += w * float(v)
            den += abs(w)
        if den == 0:
            return None
        return num / den

    def informed_by(self, node: str) -> Dict[str, float]:
        """Return {src: weight} for every edge into node."""
        return {src: w for (src, dst), w in self.edges.items() if dst == node}

"""Tests for Thought / Knowledge / Wisdom asymmetries and feedback."""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.system import NoiseSeparatedSystem


def test_thought_can_be_cleared():
    s = NoiseSeparatedSystem(explore_prob=0.0)
    x = np.ones(8)
    s.observe(x, y=1.0)
    assert len(s.thought) > 0
    s.thought.clear()
    assert len(s.thought) == 0


def test_knowledge_fades():
    s = NoiseSeparatedSystem(explore_prob=0.0, promote_hits=2, min_agreement=0.0)
    x = np.sin(np.linspace(0, 2, 8))
    for _ in range(5):
        s.observe(x, y=1.0)
    assert len(s.knowledge) > 0
    for _ in range(100):
        s.knowledge.fade()
    assert len(s.knowledge) == 0


def test_wisdom_never_shrinks_but_confidence_can_fall():
    s = NoiseSeparatedSystem(explore_prob=0.0, promote_hits=2, wisdom_strength=0.8, min_agreement=0.0)
    x = np.ones(8)
    for _ in range(6):
        s.observe(x, y=1.0)
    n_before = len(s.wisdom)
    assert n_before >= 1
    # contradict
    for _ in range(6):
        s.observe(x, y=10.0)
    assert len(s.wisdom) == n_before  # count never shrinks
    assert s.wisdom.active_count <= n_before


def test_demotion_removes_reactive():
    s = NoiseSeparatedSystem(explore_prob=0.0, promote_hits=2, min_agreement=0.0)
    x = np.zeros(8)
    for _ in range(5):
        s.observe(x, y=1.0)
    assert len(s.reactive) >= 1 or len(s.knowledge) >= 1
    for _ in range(3):
        s.observe(x, y=5.0)
    # reactive should have been cleared on error
    assert s.reactive.lookup(x) is None


if __name__ == "__main__":
    test_thought_can_be_cleared()
    test_knowledge_fades()
    test_wisdom_never_shrinks_but_confidence_can_fall()
    test_demotion_removes_reactive()
    print("Layer asymmetry tests passed.")

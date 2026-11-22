# Noise-Separated Minds

Unsupervised architecture that separates pure randomness from structured signal, then consolidates across three timescales: Thought, Knowledge, and Wisdom.

## Core Idea

1. **Randomness Gate**: Reject pure noise before it can skew any distribution.
2. **Thought**: Short-term figuring (analytical ensemble). Can be lost.
3. **Knowledge**: Medium-term patterns. Can fade with disuse.
4. **Wisdom**: Long-term invariants. Only grows.
5. **Reactive**: Deterministic functions promoted from strong knowledge/wisdom.
6. **Chaos**: Residual pure randomness (no model).

## Why

Pure randomness cannot be absorbed into a pdf without skewing it. Filter first. Then let thoughts be disposable, knowledge provisional, and wisdom cumulative.

## Current Implementation

- Randomness gate
- Thought layer: 7-perspective analytical ensemble (toggleable, blended)
- Knowledge layer: strength that fades; can be revised
- Wisdom layer: append-only
- Reactive mind: exact deterministic table
- Chaos pile
- Promotion path: Thought → Knowledge → Wisdom (+ Reactive)
- Demo, basic tests, ensemble benchmark

## Run

```bash
pip install -r requirements.txt
python examples/demo.py
python tests/test_basic.py
python benchmarks/ensemble_search.py
```

## Status

Prototype with Thought / Knowledge / Wisdom consolidation. Wisdom is permanent; Knowledge fades; Thought is the live ensemble.

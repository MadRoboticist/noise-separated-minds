# Noise-Separated Minds

Unsupervised architecture that separates pure randomness from structured signal, then consolidates across three timescales with full feedback.

## Core Idea

1. **Randomness Gate**: Reject pure noise before it can skew any distribution.
2. **Thought**: Short-term figuring (analytical ensemble). Can be lost.
3. **Knowledge**: Medium-term patterns. Can fade or be weakened.
4. **Wisdom**: Long-term. Count only grows; confidence is checked and can fall.
5. **Reactive**: Deterministic functions.
6. **Chaos**: Residual pure randomness, with re-entry path.

## Feedback loops

- Promotion requires hit-count, structure score, **and cross-perspective agreement**
- Demotion on high error (Reactive removed, Knowledge weakened)
- Wisdom confidence checked on every matching observation (never deleted)
- Chaos re-entry and light exploratory routing
- Blender uses recent per-perspective accuracy

## Perspectives inside Thought

Local weighted, nearest-neighbor (k=1,3,5), online linear (two learning rates), wider local. Toggleable.

## Run

```bash
pip install -r requirements.txt
python examples/demo.py
python tests/test_layers.py
python benchmarks/ensemble_search.py
```

## Status

Prototype with agreement-gated promotion, accuracy-weighted blending, parametric perspectives, and asymmetry tests.

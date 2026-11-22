# Noise-Separated Minds

Unsupervised architecture that separates pure randomness from structured signal, then consolidates across three timescales with full feedback.

## Core Idea

1. **Randomness Gate**: Multi-test detector (spectral flatness, histogram entropy, autocorrelation, difference entropy, runs) that rejects pure noise.
2. **Thought**: Short-term figuring (analytical ensemble). Can be lost.
3. **Knowledge**: Medium-term patterns. Can fade or be weakened.
4. **Wisdom**: Long-term. Count only grows; confidence is checked and can fall.
5. **Reactive**: Deterministic functions.
6. **Chaos**: Residual pure randomness, with re-entry path.

## Feedback loops

- Promotion requires hit-count, structure score, **and cross-perspective agreement**
- Demotion on high error
- Wisdom confidence checked on every matching observation (never deleted)
- Adaptive exploration rate
- Chaos re-entry
- Accuracy-weighted blender

## Run

```bash
pip install -r requirements.txt
python examples/demo.py
python tests/test_gate.py
python tests/test_layers.py
```

## Status

Prototype with robust multi-test randomness gate, agreement-gated promotion, adaptive exploration, and asymmetry tests.

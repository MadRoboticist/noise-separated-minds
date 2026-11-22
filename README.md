# Noise-Separated Minds

Unsupervised architecture that separates pure randomness from structured signal, then consolidates across three timescales with full feedback.

## Core Idea

1. **Randomness Gate**: Multi-test detector that rejects pure noise.
2. **Thought**: Short-term figuring (analytical ensemble). Bounded, clearable, auto-prunes collapsed perspectives.
3. **Knowledge**: Medium-term. Can fade or be weakened.
4. **Wisdom**: Long-term. Count only grows; confidence can fall.
5. **Reactive**: Deterministic functions (bounded).
6. **Chaos**: Residual randomness (bounded), with re-entry.

## Thought ensemble

Perspectives: local weighted, nn (k=1/3/5), online linear, mixture, wider local.
- Blender uses accuracy × inverse uncertainty
- Perspectives with sustained high error are auto-disabled and can recover
- Uncertainty includes both local distance and cross-perspective disagreement

## Persistence and scale

- `save` / `load`
- Bounded observation lists, chaos pile, and reactive table

## Feedback loops

Agreement-gated promotion, demotion, Wisdom checks, adaptive exploration, chaos re-entry.

## Run

```bash
pip install -r requirements.txt
python examples/demo.py
python tests/test_persistence.py
python tests/test_gate.py
python tests/test_layers.py
```

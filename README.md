# Noise-Separated Minds

Unsupervised architecture that separates pure randomness from structured signal, then consolidates across three timescales with full feedback.

## Core Idea

1. **Randomness Gate**: Multi-test detector that rejects pure noise.
2. **Thought**: Short-term figuring (analytical ensemble). Bounded and clearable.
3. **Knowledge**: Medium-term. Can fade or be weakened.
4. **Wisdom**: Long-term. Count only grows; confidence can fall.
5. **Reactive**: Deterministic functions (bounded).
6. **Chaos**: Residual randomness (bounded), with re-entry.

## Persistence and scale

- `system.save(path)` / `NoiseSeparatedSystem.load(path)` (pickle)
- Observation lists inside Thought capped (`max_obs`, default 2000)
- Chaos pile capped (`max_chaos`, default 5000)
- Reactive table capped

## Feedback loops

- Agreement-gated promotion
- Demotion on error
- Wisdom confidence checks
- Adaptive exploration
- Chaos re-entry
- Accuracy-weighted blender

## Run

```bash
pip install -r requirements.txt
python examples/demo.py
python tests/test_persistence.py
python tests/test_gate.py
python tests/test_layers.py
```

## Status

Prototype with persistence, bounded memory, robust gate, and full feedback loops.

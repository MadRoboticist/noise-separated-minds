# Noise-Separated Minds

Unsupervised architecture that separates pure randomness from structured signal, then consolidates across three timescales with full feedback.

## Core Idea

1. **Randomness Gate**: Reject pure noise before it can skew any distribution.
2. **Thought**: Short-term figuring (analytical ensemble). Can be lost.
3. **Knowledge**: Medium-term patterns. Can fade or be weakened.
4. **Wisdom**: Long-term invariants. Only grows.
5. **Reactive**: Deterministic functions.
6. **Chaos**: Residual pure randomness, with re-entry path.

## Feedback loops

- Promotion: consistent Thought → Knowledge → Wisdom + Reactive
- Demotion: high prediction error weakens Knowledge and removes Reactive entries, forcing return to Thought
- Chaos re-entry: previously rejected items that later score as structured are pulled back into Thought
- Exploratory routing: small probability of sending structured inputs to Chaos

## Why

Pure randomness cannot be absorbed into a pdf without skewing it. Filter first. Then allow thoughts to be disposable, knowledge to fade or be contradicted, and wisdom to accumulate, while keeping reverse paths open.

## Run

```bash
pip install -r requirements.txt
python examples/demo.py
python tests/test_basic.py
python benchmarks/ensemble_search.py
```

## Status

Prototype with bidirectional consolidation and chaos re-entry.

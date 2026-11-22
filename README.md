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

- Promotion: consistent Thought → Knowledge → Wisdom + Reactive
- Demotion: high error weakens Knowledge and removes Reactive entries
- **Wisdom check**: every observation verifies matching Wisdom entries. Failures reduce confidence (never delete). Low-confidence Wisdom is ignored at prediction time and the case is forced back into Thought.
- Chaos re-entry and exploratory routing

## Why

Pure randomness cannot be absorbed into a pdf without skewing it. Filter first. Thoughts disposable, knowledge provisional, wisdom cumulative but still accountable.

## Run

```bash
pip install -r requirements.txt
python examples/demo.py
```

## Status

Prototype with bidirectional consolidation, chaos re-entry, and Wisdom confidence checks.

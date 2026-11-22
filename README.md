# Noise-Separated Minds

Unsupervised architecture that separates pure randomness from structured signal, then consolidates across three timescales with full feedback.

## Core Idea

1. **Randomness Gate**: Multi-test detector with adaptive threshold.
2. **Thought**: Short-term figuring (analytical ensemble with generative members).
3. **Knowledge**: Medium-term. Can fade or be weakened.
4. **Wisdom**: Long-term. Count only grows; confidence can fall.
5. **Reactive**: Deterministic functions (bounded).
6. **Chaos**: Residual randomness with **active probing** and a promotion budget.

## Chaos

- Bounded reject pile.
- `probe(n)` actively samples candidates for re-testing.
- Successful probes re-enter Thought and are counted.
- Promotion of chaos-originated material is limited by `chaos_promote_budget` (fraction of steps).
- Stats report probe attempts and successes.

## Thought ensemble

Local, nearest-neighbor, online linear, GMM, latent residual. Generative sampling, auto-disable, event log.

## Randomness gate

Adaptive threshold + residual noise rate.

## Persistence

`save` / `load`. Bounded buffers.

## Run

```bash
pip install -r requirements.txt
python examples/demo.py
```

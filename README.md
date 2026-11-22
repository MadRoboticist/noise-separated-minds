# Noise-Separated Minds

Unsupervised architecture that separates pure randomness from structured signal, then consolidates across three timescales with full feedback.

## Core Idea

1. **Randomness Gate**: Multi-test detector with adaptive threshold.
2. **Thought**: Short-term figuring (analytical ensemble with generative members).
3. **Knowledge**: Medium-term. Can fade or be weakened.
4. **Wisdom**: Long-term. Count only grows; confidence can fall.
5. **Reactive**: Deterministic functions (bounded).
6. **Chaos**: Residual randomness (bounded), with re-entry.

## Thought ensemble (richer generative capacity)

- Local weighted, nearest-neighbor, online linear
- **Gaussian mixture** (3 components on a 1-D projection) — can sample from p(y|x)
- **Latent residual** (random projection + local model) — can sample
- `sample_generative(x)` draws from a generative perspective when available
- Auto-disable of collapsed perspectives + event log
- Accuracy × inverse-uncertainty blender

## Randomness gate

Adaptive threshold + residual noise rate.

## Persistence

`save` / `load`. Bounded buffers.

## Run

```bash
pip install -r requirements.txt
python examples/demo.py
python tests/test_gate.py
```

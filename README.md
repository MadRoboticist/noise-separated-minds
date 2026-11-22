# Noise-Separated Minds

Unsupervised architecture that separates pure randomness from structured signal, then consolidates across three timescales with full feedback.

## Core Idea

1. **Randomness Gate**: Multi-test detector with **adaptive threshold**. Tracks recent scores and residual noise rate.
2. **Thought**: Short-term figuring (analytical ensemble). Bounded, clearable, auto-prunes collapsed perspectives.
3. **Knowledge**: Medium-term. Can fade or be weakened.
4. **Wisdom**: Long-term. Count only grows; confidence can fall.
5. **Reactive**: Deterministic functions (bounded).
6. **Chaos**: Residual randomness (bounded), with re-entry.

## Randomness gate

- Tests: spectral flatness, histogram entropy, autocorrelation, difference entropy, runs.
- Threshold adapts toward the lower tail of recent structure scores (clamped).
- `residual_noise_rate` reports the fraction of inputs recently rejected.
- Exposed in `system.stats()` as `gate_threshold` and `residual_noise_rate`.

## Thought ensemble

Local weighted, nn, online linear, residual generative, wider local. Auto-disable + event log. Accuracy-weighted blender.

## Persistence and scale

`save` / `load`. Bounded buffers.

## Run

```bash
pip install -r requirements.txt
python examples/demo.py
python tests/test_gate.py
python tests/test_layers.py
python tests/test_persistence.py
```

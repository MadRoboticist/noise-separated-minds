# Noise-Separated Minds

Unsupervised architecture that separates pure randomness from structured signal, then consolidates across three timescales with full feedback.

## Consolidation

- Promotion to Knowledge requires hits + structure score + agreement.
- Promotion to Wisdom additionally requires the pattern to have aged at least `min_age` steps.
- Wisdom entries are periodically re-validated (forced back through Thought) if they have not been checked recently.
- Confidence can still fall on contradiction; count never shrinks.

## Chaos

Active probing + promotion budget.

## Thought

Generative ensemble (GMM, latent residual, local, linear) with auto-disable.

## Gate

Adaptive multi-test threshold + residual noise rate.

## Persistence

`save` / `load`. Bounded buffers.

## Run

```bash
pip install -r requirements.txt
python examples/demo.py
```

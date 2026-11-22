# Noise-Separated Minds

Unsupervised architecture that separates pure randomness from structured signal, then consolidates across three timescales with full feedback.

## Generative integration

- Thought can draw samples from GMM or latent residual perspectives.
- Samples are registered as **hypotheses** (`HypothesisTracker`).
- Later observations confirm or reject them.
- Confirmed hypotheses lower the hit threshold and boost Knowledge strength on promotion.
- Active hypothesis count appears in `stats()`.

## Auditor

Independent auditor inspects misbehaving tracks and re-files.

## Consolidation

Time-gap before Wisdom, re-validation, confidence checks.

## Chaos

Active probing + promotion budget.

## Run

```bash
pip install -r requirements.txt
python examples/demo.py
python benchmarks/long_horizon.py
```

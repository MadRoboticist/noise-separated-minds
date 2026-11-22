# Noise-Separated Minds

Unsupervised architecture that separates pure randomness from structured signal, then consolidates across three timescales with full feedback.

## Auditor

Independent auditor runs every `audit_every` steps.
- Inspects reactive, knowledge, wisdom, and collapsed perspectives.
- Produces findings with recommended re-file actions.
- Applies safe corrections (extra fade, limited reactive demotion, etc.).
- Findings are retained and counted in `stats()`.

## Consolidation

Time-gap before Wisdom, periodic re-validation, confidence checks.

## Chaos

Active probing + promotion budget.

## Thought

Generative ensemble with auto-disable.

## Evaluation

```bash
python benchmarks/long_horizon.py
```

Measures residual noise rejection, prediction error on structured signals, and auditor activity over hundreds of steps.

## Run

```bash
pip install -r requirements.txt
python examples/demo.py
python benchmarks/long_horizon.py
```

# Noise-Separated Minds

Unsupervised architecture that separates pure randomness from structured signal, then consolidates across three timescales with full feedback.

## Configuration

`SystemConfig` (presets `small()` / `large()`).

## Auditor

Surgical unlearn. `export_audit_trail(path)` now includes the input key and counts of reactive drops, knowledge/wisdom weakenings, and hypothesis deactivations.

## Calibration

Thought temperature scaling + Wisdom confidence reliability bins.

## Comparative evaluation

```bash
python benchmarks/comparative.py
```

Multi-seed, multi-noise sweep. Exits non-zero if the small filtered system under-performs on more than half the runs.

## Run

```bash
pip install -r requirements.txt
python examples/demo.py
python benchmarks/comparative.py
```

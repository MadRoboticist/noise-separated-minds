# Noise-Separated Minds

Unsupervised architecture that separates pure randomness from structured signal, then consolidates across three timescales with full feedback.

## Configuration

`SystemConfig` (presets `small()` / `large()`).

## Auditor

Surgical unlearn with before	o after values and recurrence count.

## Generative blender

Confirmed hypotheses raise a bonus on GMM / latent-residual weights.

## Calibration

Thought temperature scaling + Wisdom confidence reliability.

## Chaos probing

Threshold proximity + recency-weighted similarity to good inputs.

## Comparative evaluation

```bash
python benchmarks/comparative.py
```

Tracks mean error, residual-noise rate, and unlearn count across seeds and noise levels.
Exits non-zero if the small filtered system under-performs on accuracy or residual-noise rate exceeds the guard.

## Persistence

```bash
python tests/test_persistence.py
```

## Run

```bash
pip install -r requirements.txt
python examples/demo.py
```

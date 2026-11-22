# Noise-Separated Minds

Unsupervised architecture that separates pure randomness from structured signal, then consolidates across three timescales with full feedback.

## Configuration

`SystemConfig` (presets `small()` / `large()`).

## Auditor

Surgical unlearn with recurrence counter (`stats()["recurrent_unlearns"]`).

## Generative blender

Confirmed hypotheses raise a bonus on GMM / latent-residual weights. Bonus decays.

## Calibration

Thought temperature scaling + Wisdom confidence reliability.

## Chaos probing

Threshold proximity + similarity to recent good inputs.

## Persistence

`save` / `load` round-trips config, temperature, calibration bins, and unlearn counts.

```bash
python tests/test_persistence.py
```

## Comparative evaluation

```bash
python benchmarks/comparative.py
```

## Run

```bash
pip install -r requirements.txt
python examples/demo.py
```

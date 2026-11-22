# Noise-Separated Minds

Unsupervised architecture that separates pure randomness from structured signal, then consolidates across three timescales with full feedback.

## Configuration

`SystemConfig` (presets `small()` / `large()`).

## Auditor

Surgical unlearn. Recurrence counter tracks how many times the same key has been unlearned (`stats()["recurrent_unlearns"]`).

## Generative blender

Confirmed hypotheses raise a bonus weight on GMM and latent-residual perspectives inside the ensemble blender. The bonus slowly decays.

## Calibration

Thought temperature scaling + Wisdom confidence reliability.

## Chaos probing

Threshold proximity + similarity to recent good inputs.

## Comparative evaluation

```bash
python benchmarks/comparative.py
```

## Run

```bash
pip install -r requirements.txt
python examples/demo.py
```

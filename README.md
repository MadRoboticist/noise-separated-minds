# Noise-Separated Minds

Unsupervised architecture that separates pure randomness from structured signal, then consolidates across three timescales with full feedback.

## Reactive vs Wisdom

- **Reactive** = muscle memory. Acquired by straight mechanical repetition or by trauma (a large sudden error on a structured input). Fast lookup. Does not automatically become Wisdom.
- **Wisdom** = later consistency after earlier deliberation. An item becomes eligible only if it previously showed disagreement, required hypotheses, or spent time in Knowledge, and is now consistent. Not the same path as Reactive.

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

## Persistence

```bash
python tests/test_persistence.py
```

## Run

```bash
pip install -r requirements.txt
python examples/demo.py
```

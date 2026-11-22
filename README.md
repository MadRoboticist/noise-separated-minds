# Noise-Separated Minds

Unsupervised architecture that separates pure randomness from structured signal, then consolidates across three timescales with full feedback.

## Configuration

`SystemConfig` (presets `small()` / `large()`).

## Auditor

Surgical unlearn with detailed export.

## Calibration

Thought temperature scaling + Wisdom confidence reliability.

## Chaos probing

Prefers rejects that are both close to the current threshold **and** similar to recent good (structured) inputs.

## Comparative evaluation

```bash
python benchmarks/comparative.py
```

Multi-seed, multi-noise sweep with pass/fail.

## Run

```bash
pip install -r requirements.txt
python examples/demo.py
python benchmarks/comparative.py
```

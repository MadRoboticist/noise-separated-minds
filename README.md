# Noise-Separated Minds

Unsupervised architecture that separates pure randomness from structured signal, then consolidates across three timescales with full feedback.

## Configuration

`SystemConfig` (and presets `small()` / `large()`) holds every knob. Pass it to `NoiseSeparatedSystem(config=...)`.

## Auditor

Surgical unlearn, hypothesis-triggered audits, `export_audit_trail`.

## Calibration

- Thought uncertainty: online temperature scaling.
- Wisdom confidence: reliability bins (`stats()["wisdom_reliability"]`).

## Comparative evaluation

```bash
python benchmarks/comparative.py
```

Compares a small filtered+auditor system against a larger weakly-filtered system on the same stream.

## Run

```bash
pip install -r requirements.txt
python examples/demo.py
python benchmarks/comparative.py
```

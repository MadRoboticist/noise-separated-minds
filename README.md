# Noise-Separated Minds

Unsupervised architecture that separates pure randomness from structured signal, then consolidates across three timescales with full feedback.

## Auditor

Procedural, file-clerk-only. Surgical unlearn of matching items only.
Contradicted hypotheses can trigger an audit.
`export_audit_trail(path)` writes a human-readable summary of recent findings and unlearn events.

## Calibration

Uncertainty is binned against correctness. A temperature scale is adjusted online so that high-uncertainty bins that are often wrong become less trusted. Current temperature and bin accuracies appear in `stats()`.

## Chaos probing

Prefers rejects closest to the current threshold.

## Run

```bash
pip install -r requirements.txt
python examples/demo.py
```

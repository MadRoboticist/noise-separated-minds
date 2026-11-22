# Noise-Separated Minds

Unsupervised architecture that separates pure randomness from structured signal, then consolidates across three timescales with full feedback.

## Auditor

Procedural, file-clerk-only.
- Severs irrational behavior from good inputs.
- **Surgical unlearn**: only reactive keys, knowledge items, wisdom entries, and hypotheses that match the audited input are weakened or removed.
- Contradicted hypotheses can themselves trigger an audit.

## Calibration

Thought uncertainty is binned and compared against actual correctness. Summary available in `stats()["calibration"]`.

## Unified event log

Routings, promotions, demotions, hypotheses, probes, and surgical unlearns.

## Chaos probing

Prefers rejects closest to the current threshold.

## Run

```bash
pip install -r requirements.txt
python examples/demo.py
python benchmarks/eval_depth.py
```

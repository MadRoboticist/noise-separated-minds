# Noise-Separated Minds

Unsupervised architecture that separates pure randomness from structured signal, then consolidates across three timescales with full feedback.

## Auditor

Procedural, file-clerk-only. Disconnects irrational behavior from good inputs.
Traces now include gate score, agreement, and active perspectives at the moment of each decision.

## Unified event log

`system.event_log` is a single chronological record of routings, promotions, demotions, hypothesis proposals, chaos-probe successes, and auditor severances.

## Chaos probing

Prefers rejects whose structure score is closest to the current threshold (instead of pure random).

## Generative integration

Hypotheses from generative samples; confirmation influences promotion.

## Consolidation

Time-gap before Wisdom, re-validation, confidence checks.

## Run

```bash
pip install -r requirements.txt
python examples/demo.py
python benchmarks/eval_depth.py
```

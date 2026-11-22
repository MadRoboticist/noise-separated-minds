# Noise-Separated Minds

Unsupervised architecture that separates pure randomness from structured signal, then consolidates across three timescales with full feedback.

## Auditor (procedural)

Works only with the file clerk.
- The clerk records a short trace of every routing, promotion, and prediction.
- The auditor is given a suspect input and its observed outcome.
- It walks the trace backward until it finds the disconnect (bad prediction, contradicted promotion, mis-routed chaos, etc.).
- It then asks the clerk to re-file (demote, weaken knowledge/wisdom, force back to Thought).
- Never touches the minds directly.

## Generative integration

Hypotheses from generative samples; confirmation influences promotion.

## Consolidation

Time-gap before Wisdom, re-validation, confidence checks.

## Chaos

Active probing + promotion budget.

## Evaluation

```bash
python benchmarks/eval_depth.py
```

## Run

```bash
pip install -r requirements.txt
python examples/demo.py
```

# Noise-Separated Minds

Unsupervised architecture that separates pure randomness from structured signal, then consolidates across three timescales with full feedback.

## Generative integration

Hypotheses from GMM / latent residual samples; confirmation influences promotion.

## Auditor

Independent auditor inspects and re-files misbehaving tracks.

## Consolidation

Time-gap before Wisdom, re-validation, confidence checks.

## Chaos

Active probing + promotion budget.

## Evaluation

```bash
python benchmarks/long_horizon.py          # single long stream
python benchmarks/eval_depth.py            # noise sweeps, gate FP/FN, Wisdom reliability, residual-noise guard
```

`eval_depth.py` reports:
- False-positive / false-negative counts for the gate across noise fractions
- Whether high-confidence Wisdom entries match ground truth
- A simple pass/fail guard on residual noise reaching Thought

## Run

```bash
pip install -r requirements.txt
python examples/demo.py
python benchmarks/eval_depth.py
```

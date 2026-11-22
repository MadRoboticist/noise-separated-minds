# Noise-Separated Minds

Unsupervised architecture that separates pure randomness from structured signal, then consolidates across three timescales with full feedback.

## Auditor

Procedural and file-clerk-only. Its job is to **disconnect irrational behavior from good inputs**.
- Only examines inputs the gate already scored as structured.
- Walks the trace of that input backward.
- Detects irrational acts: good input sent to chaos, wildly wrong prediction, promotion later contradicted.
- Severs the bad behavior (demote reactive, weaken knowledge/wisdom) and forces the good input back into Thought for clean re-filing.
- Never touches the minds directly.

## Generative integration

Hypotheses from generative samples; confirmation influences promotion.

## Consolidation

Time-gap before Wisdom, re-validation, confidence checks.

## Chaos

Active probing + promotion budget.

## Run

```bash
pip install -r requirements.txt
python examples/demo.py
python benchmarks/eval_depth.py
```

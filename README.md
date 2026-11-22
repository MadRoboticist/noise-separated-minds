# Noise-Separated Minds

An online system that separates structureless noise from structured signal, then consolidates the signal across three timescales: knowledge, wisdom, and reactive (muscle memory).

Full write-up: [ARCHITECTURE.md](ARCHITECTURE.md)
Results: [RESULTS.md](RESULTS.md)

## Quick start

```bash
pip install -r requirements.txt
python examples/demo.py
```

## Decision order

1. **Reactive** — short path, used only at very high confidence
2. **Wisdom** — if well established
3. **Knowledge** — the default; easiest to fill
4. **Randomness** — a chaotic attempt when nothing is known

## Stores

- **Knowledge** — medium-term, easiest to form, handles most decisions
- **Wisdom** — harder to form; sustained consistency across layers
- **Reactive** — muscle memory; hardest to form (many repetitions or a trauma event)
- **Chaos** — running Gaussian over rejected inputs; not a learned predictor
- **Trauma** — fast store for high-error events; seeds reactive, negatively weighted into wisdom

A randomness gate rejects structureless inputs before they reach the stores. Match tolerances scale with √d so the same logic holds at high dimension.

## Benchmarks

On a discrete-pattern task (four sparse patterns, fixed targets, 30% noise) the system is exact and matches k-NN. Linear and MLP cannot represent the piecewise relationship.

On a continuous signal with 40% noise bursts the system is slightly ahead of linear, k-NN, and a small MLP, and remains stable at dimensions where linear diverges.

See [RESULTS.md](RESULTS.md) for numbers.

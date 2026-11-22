# Noise-Separated Minds

Unsupervised architecture that separates pure randomness from structured signal, then consolidates across three timescales with full feedback.

## Non-numeric contexts

`observe_context` / `predict_context` accept strings, categories, mixed lists, or numbers.

## Benchmarks

```bash
python benchmarks/comparative.py      # small filtered vs large weak filter
python benchmarks/context_bench.py    # non-numeric phrases
python benchmarks/substantial.py      # multi-pattern numeric + text, longer runs
```

## Reactive / Knowledge / Wisdom

- **Knowledge** receives every structured repeat (can fade).
- **Reactive** = muscle memory (repetition or trauma).
- **Wisdom** = later consistency after earlier deliberation.

## Run

```bash
pip install -r requirements.txt
python examples/demo.py
```

# Noise-Separated Minds

Unsupervised architecture that separates pure randomness from structured signal, then consolidates across three timescales with full feedback.

## Non-numeric contexts

`observe_context` / `predict_context` accept strings, categories, mixed lists, or numbers. A hashing encoder turns them into fixed vectors before the gate and minds.

```python
s.observe_context("the sky is blue", y=1.0)
s.observe_context(["cat", 3, "warm"], y=0.5)
pred, via = s.predict_context("the sky is blue")
```

Benchmark:
```bash
python benchmarks/context_bench.py
```

## Reactive vs Wisdom

- **Reactive** = muscle memory (repetition or trauma).
- **Wisdom** = later consistency after earlier deliberation.

## Comparative evaluation

```bash
python benchmarks/comparative.py
```

## Run

```bash
pip install -r requirements.txt
python examples/demo.py
```

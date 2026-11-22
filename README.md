# Noise-Separated Minds

Unsupervised online system that tries to keep pure randomness out of memory, then consolidates structured signal across three timescales.

Full write-up: [ARCHITECTURE.md](ARCHITECTURE.md)

## Quick start

```bash
pip install -r requirements.txt
python examples/demo.py
```

## Layers

- **Thought** — short-term ensemble (linear, k-NN, MLP, noise-path linear)
- **Knowledge** — medium-term, can fade
- **Reactive** — muscle memory (repetition or trauma)
- **Wisdom** — later consistency after earlier deliberation

## Benchmarks

```bash
python benchmarks/comparative.py
python benchmarks/vs_baselines.py      # vs linear / k-NN / MLP on SKAB
python benchmarks/skab_compare.py
```

On the SKAB industrial streams the system is not competitive with simple online baselines. See the architecture note for details.

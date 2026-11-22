# Architecture

Noise-Separated Minds is an unsupervised online system that tries to keep pure randomness out of its memory, then consolidates structured signal across three timescales.

## Design thesis

Neural nets estimate nonlinear maps or conditional distributions. Pure noise can drag coefficients without bound and skew those distributions. The claim here is that irreducible randomness should be excluded before learning, not fitted.

## Components

### Randomness gate
Scores each input with simple structure tests (spectral flatness, histogram entropy, autocorrelation, runs).  
If the score is below an adaptive threshold the observation is treated as noise.  
A repetition-based bypass turns the gate off when recent raw inputs are highly similar (already-clean repeated signal).

### Chaos pile
Rejected observations are stored here and do not enter the main memory.  
Optionally a dedicated noise-path model (currently a linear perspective) also sees them.

### File clerk
Routes each accepted observation and records a unified event log.

### Thought (analytical ensemble)
Short-term, elastic ensemble. Current members:
- online linear
- 5-nearest-neighbor
- small MLP
- noise-path linear

Blender modes: accuracy-weighted average (default), greedy (pick highest weight), or plain mean.

### Knowledge
Medium-term store. Structured repeats are written here and can fade.

### Reactive
Muscle memory. Acquired by high repetition or by a single large “trauma” error. Fast lookup.

### Wisdom
Longer-term store that only accepts patterns which first showed disagreement (deliberation) and later became consistent. Append-only in spirit; confidence-checked.

### Auditor
Procedural trace-walker that works through the file clerk. Can surgically unlearn a mapping and count recurrences.

## Data flow

```
input → gate
          ├─ noise → chaos (+ optional noise-path model)
          └─ structured → Thought
                              → Knowledge (on enough repeats)
                              → Reactive (repetition or trauma)
                              → Wisdom (only after early disagreement + later consistency)
```

Prediction order: Reactive → Wisdom → Knowledge → Thought.

## Non-numeric inputs
`observe_context` / `predict_context` accept strings, categories, or mixed lists. A hashing encoder projects them to a fixed vector before the gate.

## What the experiments showed

On synthetic streams with injected pure noise the gate can keep residual-noise rate near zero and a small filtered system can match a larger weakly-filtered one once dimension is adequate.

On the real SKAB industrial sensor streams the same system is substantially worse than simple online baselines (linear, k-NN, small MLP). Disabling the gate and using greedy blending improves the system but does not close the gap. Routing rejected packets through an extra linear or MLP member also does not.

The architecture is a coherent research prototype. On the data tried so far it is not a competitive general predictor.

## Layout

- `src/` — system, gate, layers, minds, auditor, encoder, config
- `benchmarks/` — comparative, stress, ablation, SKAB, gate-vs-denoiser, vs-baselines
- `tests/` — unit and persistence tests
- `examples/demo.py` — short end-to-end demo

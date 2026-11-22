# Architecture

## Overview

A file clerk routes each input. A randomness gate first decides whether the input is structureless. Structured inputs are consolidated across three timescales. Structureless inputs update a noise distribution and are not stored as patterns.

## Decision order

When predicting:

1. **Reaction.** If reactive has a match with confidence ≥ 0.98, return it. This is the short path.
2. **Wisdom.** If no reaction, and wisdom has a well-reinforced match (n ≥ 180, confidence ≥ 2.9), return it.
3. **Knowledge.** If neither of the above, and knowledge has a match, return it. This is the default.
4. **Randomness.** If nothing is known, sample from the chaos distribution.
5. **Fallback.** The analytical (thought) estimate.

## Formation difficulty

Knowledge is easiest to fill (a few consistent hits). Wisdom requires sustained consistency (many hits, agreement across layers). Reactive is hardest — it requires many repetitions, or a trauma event that seeds it directly.

On a long stream the decision counts follow this order: knowledge handles the majority, wisdom the middle, reactive the rare high-confidence cases.

## Stores

### Knowledge
Running Gaussian components. Each component holds a mean of x and a mean and variance of y. New observations update the nearest component if within tolerance; otherwise a new component is opened. Fades over time if not reinforced.

### Wisdom
Same structure as knowledge, but forms only after sustained consistency. Used only when well reinforced. Chaos and trauma are negatively weighted into wisdom — a shocking or structureless outcome suppresses the stable estimate rather than reinforcing it.

### Reactive (muscle memory)
Keyed store. On reinforcement, the stored y is updated as a running mean and a hit count increments. Confidence is a function of strength and hits; it saturates slowly (needs ~200 hits to reach 1.0). Used only at confidence ≥ 0.98. Trauma events seed reactive directly.

### Chaos
A running Gaussian (mean and diagonal variance) over rejected inputs. Fixed size regardless of how many points are rejected. Not a learned predictor — it is a model of what noise looks like. Sampled only when no other store has a match.

### Trauma
Fast store for high-error events on structured inputs. Negates reactive: pushes the stored reflex away from the bad output and drops its strength. Negatively weighted into wisdom. On a stream with periodic shocks this isolation is what keeps normal predictions accurate while linear, kNN, and MLP absorb the bad values.

## Randomness gate

Scores each input on spectral flatness, histogram entropy, autocorrelation, and difference entropy. Inputs below threshold are rejected into chaos. The threshold adapts from a noise profile that tracks accepted and rejected scores separately. On a stream with 30% noise bursts, the gate rejects approximately 31%.

There is also a bypass: if recent inputs are all close together (median pairwise distance < 0.15), the gate is skipped on the assumption the signal is clean.

## Match tolerances

All match tolerances scale with √d:

- Reactive, wisdom: 0.08 × √d
- Knowledge: 0.08 × √d (add), 0.2 × √d was tried and merged distinct patterns; 0.08 keeps them separate

At dim=128 with 0.05 per-dimension noise, the expected distance is ~0.57. The scaled tolerance covers this without merging patterns that live in different subspaces.

## Graph

Edges connect the stores. Each edge has magnitude 1. Chaos → wisdom and trauma → wisdom are −1 (suppression). All others are +1. The graph mix is the fallback when no store has a direct match.

## What was removed

The following were added during development and did not improve prediction. They have been removed:

- Next-input (feedforward) predictor
- Feedback linear corrector
- LQR parameter controller
- Residual-gradient edge nudging
- Raw observation storage in parametric models
- Input-uncertainty scaling

## What the system is for

The system wins on tasks with discrete, repeated patterns and a target that is not a linear function of the input. It is competitive with k-NN on those tasks and slightly ahead of linear, k-NN, and a small MLP on continuous signals with noise bursts. It is not competitive with a plain linear model on low-dimensional continuous regression where the relationship is already linear.

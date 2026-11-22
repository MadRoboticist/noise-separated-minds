# Noise-Separated Minds — Results

## Architecture

A file clerk routes each input to one of several stores:

- **Reactive** (muscle memory). Short path to output. Forms only after many repetitions or a trauma event. Used only when confidence is very high.
- **Thought.** The analytical path. Consults wisdom, then knowledge, then falls back to a linear estimate.
- **Wisdom.** Stable patterns confirmed over a long span. Harder to form than knowledge.
- **Knowledge.** Medium-term store. Easiest to fill. The default source of decisions.
- **Chaos.** A running Gaussian over rejected (structureless) inputs. Not a learned predictor.
- **Trauma.** Fast store for high-error events. Seeds reactive. Negatively weighted into wisdom.

Decision order: reaction → wisdom → knowledge → randomness.

Formation difficulty: knowledge (easy) < wisdom (harder) < reactive (hardest).

Match tolerances scale with √d so the same logic holds from low to high dimension.

## Benchmarks

Two synthetic tasks, compared against online linear regression (L2 + clip), k-nearest neighbors (k=5), and a small online MLP (16 hidden units).

### Discrete patterns

Four sparse patterns in different subspaces, each with a fixed target, plus 30% pure noise. A linear model cannot represent the piecewise relationship.

| dim | ours  | linear | knn   | mlp   |
|-----|-------|--------|-------|-------|
| 32  | 0.000 | 0.362  | 0.000 | 0.587 |
| 64  | 0.000 | 0.366  | 0.000 | 0.550 |
| 128 | 0.000 | 0.359  | 0.000 | 0.542 |

The system and kNN are exact. Linear fails because the target is not a linear function of the input. MLP fails because a smooth approximator does not memorize sparse patterns.

### Repeated signal with noise bursts

One continuous signal plus 40% structureless noise.

| dim | ours  | linear | knn   | mlp   |
|-----|-------|--------|-------|-------|
| 32  | 0.041 | 0.044  | 0.044 | 0.063 |
| 64  | 0.042 | 0.050  | 0.045 | 0.046 |
| 128 | 0.041 | 0.071  | 0.046 | 0.045 |

The system is slightly ahead at every dimension. The gap against linear grows with dimension. At dim=256 linear diverges; the system holds near 0.04.

## Decision mix

On a 2000-step discrete stream at dim=64:

```
knowledge: 777
wisdom:    459
reactive:  130
```

Knowledge handles the majority. Wisdom is used for well-reinforced patterns. Reactive fires only after confidence has fully built. This matches the intended difficulty order.

## What was removed

The following were added during development and did not improve results. They have been removed:

- Next-input (feedforward) predictor
- Feedback linear corrector
- LQR parameter controller
- Residual-gradient edge nudging
- Raw observation storage in parametric models
- Input-uncertainty scaling (depended on the removed next-input path)

## What remains

The file clerk, the five stores, the scaled match tolerances, the decision order, and the trauma → reactive / trauma ⊣ wisdom wiring. Chaos is a fixed-size Gaussian. Knowledge, wisdom, and reactive store running means, not raw samples.

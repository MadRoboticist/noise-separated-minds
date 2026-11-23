# Noise-Separated Minds — Results

## Architecture

A file clerk routes each input to one of several stores:

- **Reactive** (muscle memory). Short path to output. Forms only after many repetitions. Used only when confidence is very high. Trauma negates it — a bad shock pushes the reflex away from the bad value.
- **Thought.** The analytical path. Consults wisdom, then knowledge, then falls back to a linear estimate.
- **Wisdom.** Stable patterns confirmed over a long span. Harder to form than knowledge. Chaos and trauma are negatively weighted into it.
- **Knowledge.** Medium-term store. Easiest to fill. The default source of decisions.
- **Chaos.** A running Gaussian over rejected (structureless) inputs. Not a learned predictor.
- **Trauma.** Fast store for high-error events. Negates reactive. Negatively weighted into wisdom.

Decision order: reaction → wisdom → knowledge → randomness.

Formation difficulty: knowledge (easy) < wisdom (harder) < reactive (hardest).

Match tolerances scale with √d so the same logic holds from low to high dimension.

## Benchmarks

Three synthetic tasks, compared against online linear regression (L2 + clip), k-nearest neighbors (k=5), and a small online MLP (16 hidden units).

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

### Shocks

The discrete patterns with periodic shocks: a known pattern paired with a wildly wrong output (+10) every 150 steps. Scored only on the normal steps.

| dim | ours  | linear | knn   | mlp   | trauma |
|-----|-------|--------|-------|-------|--------|
| 32  | 0.072 | 0.338  | 0.859 | 0.489 | 13     |
| 64  | 0.074 | 0.335  | 1.178 | 0.459 | 10     |
| 128 | 0.070 | 0.344  | 1.160 | 0.474 | 9      |

The system is well ahead. kNN is the worst — it stores the shock samples and returns them as neighbors. Linear and MLP absorb the bad outputs into their weights. The system's trauma path isolates the shock: it negates reactive (pushes the reflex away from the bad value) and does not let the bad output become the stored pattern.


### Noisy target with shocks

A continuous signal with a noisy target (1.0 + 0.05 noise), 40% structureless bursts, and periodic shocks (output +10). The target jitters, so zero error is impossible. The noise floor is about 0.04.

| dim | ours  | linear | knn   | mlp   |
|-----|-------|--------|-------|-------|
| 32  | 0.042 | 0.181  | 2.045 | 0.158 |
| 64  | 0.043 | 0.210  | 3.916 | 0.169 |
| 128 | 0.043 | 0.353  | 4.282 | 0.174 |

The system stays at the noise floor. Without shocks it was about 0.041; with shocks it is 0.042-0.043. Linear rises to 0.18-0.35. kNN collapses to 2.0-4.3 because it stores the shock samples and returns them as neighbors. MLP rises to 0.16-0.17.

The shocks are rejected before they enter any store, so the noisy signal is tracked and the bad values are not absorbed.

## Decision mix

On a 2000-step discrete stream at dim=64:

```
knowledge: 777
wisdom:    459
reactive:  130
```

Knowledge handles the majority. Wisdom is used for well-reinforced patterns. Reactive fires only after confidence has fully built.

## What was removed

The following were added during development and did not improve results:

- Next-input (feedforward) predictor
- Feedback linear corrector
- LQR parameter controller
- Residual-gradient edge nudging
- Raw observation storage in parametric models
- Input-uncertainty scaling

## What remains

The file clerk, the five stores, the scaled match tolerances, the decision order, the randomness gate, and the trauma wiring (negates reactive, negatively weighted into wisdom). Chaos is a fixed-size Gaussian. Knowledge, wisdom, and reactive store running means, not raw samples.

# Noise-Separated Minds

Unsupervised architecture that separates pure randomness (chaos) from structured uncertainty (analytical pdf) and deterministic mappings (reactive function).

## Core Idea

1. **Randomness Gate**: Detect and reject pure noise before it can skew any distribution.
2. **Short-term elastic buffer**: Fast, overwritable memory for provisional patterns.
3. **Consolidation clerk**: Promote only vetted, repeatable structure.
4. **Long-term stores**:
   - Reactive: near-zero-variance deterministic functions
   - Analytical: modest generative model of structured conditional distributions
   - Chaos: residual pure randomness (no model)

## Why

Pure randomness cannot be accounted for inside a pdf without dragging coefficients and skewing the distribution away from truth. Filter it first (as in analog signal chains), then model only the remaining signal. This also reduces the required model capacity.

## Status

Initial scaffolding. Implementation in progress.

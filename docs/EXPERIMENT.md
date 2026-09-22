# Experiment design and scope

## Question

Does a temporal JEPA learn a representation `z=fθ(x)` and a linear predictor
`M` such that `Mz_t≈z_{t+1}` and, on the active latent span, `MA≈AK`?

The objective is to characterize JEPA's learned dynamics, not to optimize a
generic forecasting score.

## Theoretical motivation

[Ruiz-Morales et al. (AAAI 2026)](https://doi.org/10.1609/aaai.v40i30.39708)
show that, under idealized assumptions, a time-series JEPA can represent
dynamical-regime indicators. These indicators are Koopman eigenfunctions with
$\lambda=1$, and a near-identity linear predictor acts as an inductive bias for
selecting this interpretable invariant solution.

The present experiment asks whether the same predictive architecture can move
beyond invariants. Its target is a finite-dimensional active subspace with
non-trivial eigenvalues: $\{-1,i,-i\}$ for a four-phase cycle and
$\rho\{-1,i,-i\}$ for continuously varying persistence.

## Controlled setup

- Hidden state: four phases; centered dynamic rank 3.
- Observation: length-128 windows with random amplitude, offset, temporal
  jitter, and noise.
- Central control: every condition reuses the same current and future window
  banks; only present-future pairing changes.
- Train/validation: 1,024/256 pairs per seed and condition.
- Model: `1→16→32` CNN, flattening, 3D latent, linear `3×3` predictor without
  bias.
- Training: 60 epochs, EMA 0.90, predictor learning rate `4×`, encoder frozen
  after epoch 3, and mean/variance/covariance regularization.

The correct latent dimension and early encoder freeze are assumptions of the
reported result.

## Preserved evidence

| Evidence | Data | Repetitions | Result |
|---|---|---:|---|
| Mathematical oracle | true phase, cyclic law | deterministic | exact to numerical precision |
| Observation audit | three temporal pairings | 1,024 pairs each | identical single-window marginals |
| Three dynamics | static/cyclic/independent | 10 seeds each | 10/10 by action and spectrum |
| Continuous family | $\rho\in\{0,.25,.5,.75,1\}$ | 10 seeds each | modulus MAE 0.027; $R^2=0.9998$ |

## Evaluation

The action metric is

`||MA-AK||_F / ||A||_F`.

The spectrum is computed after restricting `M` to the span of the latent phase
centroids. Discrete identification requires at least 8/10 correct seeds in each
condition. Training loss does not enter the decision.

In the continuous family, the strict discrete criterion does not pass at
`rho=1`, which obtains 7/10. Continuous calibration remains close to linear.
Both facts are reported without introducing a post-hoc threshold.

## Final interpretation

This JEPA learns a non-invariant latent dynamics: changing only temporal pairing
changes the action and active spectrum of its linear predictor. The predictor
recovers a nontrivial cyclic spectrum and tracks continuous dynamic persistence.
Its slight contraction at high persistence and its accumulating rollout error
are the main observed limitations.

The claim is restricted to this observation process, latent dimension, and
training recipe. It does not imply universal Koopman recovery or robustness to
different observations and state spaces.

# Paper-faithful replication specification

## Status

- Audit date: 2026-09-19
- Paper: *Koopman Invariants as Drivers of Emergent Time-Series Clustering in
  Joint-Embedding Predictive Architectures*
- Authors: Pablo Ruiz-Morales, Dries Vanoost, Davy Pissoort, Mathias Verbeke
- Conference version: AAAI 2026
- Extended version: arXiv v2, 2026-01-23
- Implementation status: dataset generator in progress (12 of 18 regimes)

Primary sources:

- [AAAI paper and metadata](https://ojs.aaai.org/index.php/AAAI/article/view/39708)
- [AAAI conference PDF](https://ojs.aaai.org/index.php/AAAI/article/download/39708/43669)
- [Extended arXiv HTML](https://arxiv.org/html/2511.09783)
- [Extended arXiv PDF](https://arxiv.org/pdf/2511.09783)

No official implementation or dataset repository was linked from the paper or
its arXiv record at the time of this audit. GitHub searches for the exact title,
the paper identifier, author/title combinations, and unique regime names did
not locate author code. This is evidence of non-availability, not proof that no
private or unindexed implementation exists.

## Replication claim we can support

The paper does not report enough implementation and training detail for a bitwise
or strictly exact reproduction. Without author code or clarification, the
strongest defensible target is:

> A paper-faithful, independently implemented reproduction using every
> published parameter and explicitly recording all assumptions required by
> missing or internally inconsistent details.

The existing reduced Phase 0 experiment remains a pilot. Its results must not be
mixed with the paper-faithful results.

## Confirmed dataset specification

### Dataset size and splitting

| Parameter | Published value |
|---|---:|
| Number of regimes | 18 |
| Master sequences per regime | 10,000 |
| Total context-target pairs | 180,000 |
| Master sequence length | 1,024 |
| Context window | samples `[0:768]` |
| Target window | samples `[256:1024]` |
| Prediction horizon | 256 |
| Context-target overlap | 512 samples |
| Train split | 70% = 126,000 pairs |
| Validation split | 20% = 36,000 pairs |
| Test split | 10% = 18,000 pairs |

Each master sequence yields exactly one context-target pair. Splitting is
deterministic at the sequence level and proportionally represents every regime.
The complete master sequence is standardized independently before extracting
the two windows.

Deterministic signals use zero additive observation noise. AR, MA, and ARMA
signals retain their intrinsic process noise. The appendix says this innovation
noise is typically standard normal before the `statsmodels` process parameters
are applied.

### Shared periodic parameters

With `L = 1024`:

| Name | Cycles | Angular frequency |
|---|---:|---:|
| low | 7 | `2*pi*7/L` |
| medium | 10 | `2*pi*10/L` |
| high | 15 | `2*pi*15/L` |

Periodic amplitudes are 1 unless overridden. Each sinusoidal master sequence
receives a random phase distributed as `Normal(0, pi^2)`, meaning standard
deviation `pi` if the second argument denotes variance as written.

The local implementation adopts that interpretation: it samples phase with
standard deviation `pi` and does not explicitly wrap it. Since all published
periodic signals depend on phase through a periodic function, wrapping would
not change their generated values. Deterministic periodic regimes receive no
additional observation noise.

### Published regimes

1. `Sine_LowFreq`: low frequency.
2. `Sine_MedFreq`: medium frequency.
3. `Sine_HighFreq`: high frequency.
4. `Sine_LowAmp`: medium frequency, amplitude 0.3.
5. `Sine_Harmonics`:
   `0.7*sin(f_med*t + phase) + 0.3*sin(3*f_med*t + phase_prime)`.
6. `Trend_Up`: base slope 1.5.
7. `Trend_Down`: base slope -1.5.
8. `AR_PosStrong`: AR(1), coefficient 0.9.
9. `AR_PosWeak`: AR(1), coefficient 0.3.
10. `AR_Neg`: AR(1), coefficient -0.7.
11. `MA_Pos`: MA(1), coefficient 0.7.
12. `ARMA_Mixed`: ARMA(1,1), AR coefficient 0.5 and MA coefficient -0.4.
13. `Square_LowFreq`: period `L/7`.
14. `Square_HighFreq`: period `L/15`.
15. `Sawtooth_MedFreq`: period `L/10`, randomized initial phase.
16. `Pulses_Sparse`: approximately five pulses, width `L/50`, amplitude 2.0.
17. `Sine_Trend`: `0.8*sin(f_med*t + phase)` plus a trend with base slope 1.0.
18. `Sine_HighNoise`: medium-frequency sinusoid with internal noise standard
    deviation approximately three times that of the ARMA processes.

The AR, MA, and ARMA processes are generated with
`statsmodels.tsa.arima_process.ArmaProcess`.

### Dataset details not specified precisely

The following choices cannot be recovered exactly from the paper:

- random-number generator and all seeds;
- deterministic split algorithm and split seed;
- whether phase values are wrapped after sampling;
- exact square-wave convention at zero crossings;
- exact sawtooth implementation and orientation;
- pulse count distribution, placement policy, overlap policy, and pulse shape;
- exact high-noise standard deviation (`approximately 3x` is not executable);
- ARMA burn-in, initial conditions, and exact `generate_sample` arguments;
- exact interpretation of trend slope randomization and time scaling;
- exact trend intercept distribution implementation;
- standardization convention (`ddof=0` or `ddof=1`) and numerical epsilon.

These items require either author clarification or explicitly versioned local
assumptions followed by sensitivity checks.

### Local trend assumptions

The implementation follows the appendix literally where possible: each trend
slope is `base_slope + Normal(0, 1)`, and each intercept is
`Normal(0, pi^2)`, interpreted as standard deviation `pi`. The paper does not
define the time axis used by its trend generator. The local baseline uses
`linspace(0, 1, L)`. This keeps a base slope of order one commensurate with the
published sinusoidal amplitudes; using sample indices `0, ..., L-1` would make
the trend in `Sine_Trend` roughly three orders of magnitude larger than its
amplitude-0.8 sinusoid.

This choice is an explicit reconstruction assumption, not a recovered paper
parameter, and requires a later sensitivity run. It also has two important
consequences that the dataset audit must show:

- per-sequence standardization removes the intercept and the magnitude of a
  pure affine trend, leaving essentially only its direction;
- literal `Normal(0, 1)` slope variation around base slopes `+/-1.5` gives each
  pure-trend class a small probability of reversing direction, creating label
  overlap unless the unpublished implementation clipped or constrained signs.

### Local non-smooth waveform assumptions

The paper specifies only the square-wave periods, a randomized initial phase
for the sawtooth, and approximately five positive pulses of width `L/50` and
amplitude `2.0`. The independently implemented baseline therefore uses these
named conventions:

- square waves use `scipy.signal.square` with zero initial phase, so a zero
  crossing takes value `+1`; no random square-wave phase is introduced;
- the sawtooth is a rising `scipy.signal.sawtooth(..., width=1)` with initial
  phase sampled uniformly from `[0, 2*pi)`;
- sparse pulses are exactly five positive rectangular pulses, each with width
  `round(L/50)` and amplitude `2.0`; starts are sampled uniformly without
  replacement from all valid start positions, but pulse intervals may overlap.

These conventions require sensitivity checks because the paper does not state
them. Per-sequence standardization also removes the absolute pulse amplitude;
the number, width, position, and overlap pattern remain observable.

## Confirmed model specification

### Encoder

The online and EMA target encoders share the following published convolutional
trunk for an input of shape `1 x 768`:

| Layer | Channels | Kernel | Stride | Padding | Output length |
|---|---:|---:|---:|---:|---:|
| Conv1D + ReLU | 1 -> 16 | 7 | 2 | 3 | 384 |
| Conv1D + ReLU | 16 -> 32 | 5 | 2 | 2 | 192 |
| Conv1D + ReLU | 32 -> 64 | 3 | 2 | 1 | 96 |
| Conv1D + ReLU | 64 -> 128 | 3 | 2 | 1 | 48 |
| Flatten | - | - | - | - | 6,144 |

The published latent dimension is `k = 32`.

### Predictor variants

Linear predictor:

- one `32 x 32` linear layer;
- no bias;
- identity initialization for the primary operator analysis;
- an unspecified standard random initialization for the control.

MLP predictor as shown in the appendix table:

1. Linear `k -> 2k`, followed by ReLU;
2. Linear `2k -> 2k`, followed by ReLU;
3. Linear `2k -> k`.

The target encoder begins as a copy of the online encoder and is updated by EMA
with decay `alpha = 0.996`.

The stated training objective is squared Euclidean prediction error between the
online prediction and the EMA target embedding. No variance, covariance,
contrastive, whitening, or other anti-collapse term is described.

### Architectural inconsistencies in the paper

Two contradictions must be resolved before implementation is called exact:

1. The main text says the flattened convolutional output is projected to a
   latent vector in `R^k`, with `k = 32`. Appendix C instead lists a linear
   output of `2k`, and does not list a subsequent `2k -> k` encoder layer.
2. The MLP table contains two hidden `Linear + ReLU` layers, while the text
   immediately below says the number of hidden layers was one.

The most plausible encoder interpretations are therefore:

- `6144 -> 32`, following the main text; or
- `6144 -> 64 -> 32`, assuming the appendix omitted a final projection.

Neither choice should be silently selected. Both should be implemented as named
sensitivity variants unless the authors clarify the intended architecture.

## Training details absent from the paper

The conference paper, extended PDF, HTML, and TeX source do not specify:

- optimizer;
- learning rate;
- batch size;
- number of epochs or training steps;
- weight decay;
- learning-rate schedule;
- gradient clipping;
- loss reduction and any embedding normalization;
- data-loader shuffling and worker configuration;
- initialization of convolutional and projection layers;
- the random linear predictor initialization scheme;
- training, split, K-means, or t-SNE seeds;
- checkpoint selection or early stopping;
- number of independent runs used for reported means;
- numerical precision, device, or deterministic backend settings.

These omissions prevent an exact optimization-path reproduction even if the
dataset generator is reconstructed faithfully.

The autoencoder control is also underspecified: the decoder architecture,
reconstruction objective, output activation, and training hyperparameters are
not published.

## Published evaluation protocol and reference values

All reported evaluations use the held-out test split.

### Clustering experiment

- Embeddings shown with t-SNE.
- K-means uses `K = 18`.
- JEPA with the nonlinear MLP predictor: mean purity `65.48%`.
- Conventional autoencoder with an identical encoder: mean purity `38.81%`.

Unspecified evaluation details include t-SNE parameters, K-means initialization,
`n_init`, evaluation seeds, number of runs behind the word `mean`, and the exact
purity implementation.

### Identity-initialized linear predictor

Published operator diagnostics:

| Diagnostic | Definition | Reported value |
|---|---|---:|
| Relative identity error | `||M-I||_F / ||M||_F` | 2.34% |
| Relative skew norm | `||M-M^T||_F / ||M||_F` | 2.06% |
| Mean centroid action error | mean `||M c_i-c_i||_2 / ||c_i||_2` | 0.80% |

The paper also reports `r = 18` dominant eigenvalues near 1.0, but gives no
machine-readable values or numerical tolerance.

The centroid analysis uses K-means centroids, not ground-truth-label centroids.

### Randomly initialized linear predictor

The paper states qualitatively that random initialization reaches similarly low
loss and retains clear clustering, but produces a dense, non-identity matrix.
No corresponding numerical table is published.

## Provisional reproduction criteria

These are local decision rules, not thresholds claimed by the paper. They must
be reported as such.

### Dataset fidelity gate

- all 18 regimes implemented;
- 10,000 masters per regime in the full run;
- exact `1024/768/256` window geometry;
- per-master standardization before window extraction;
- deterministic, balanced `70/20/10` split;
- visual and statistical audit notebook approved before model training.

### MLP clustering gate

- reproduce K-means purity in the neighborhood of the reported `65.48%`;
- report mean, standard deviation, and individual seeds rather than a single
  best run;
- demonstrate a substantial advantage over the matched autoencoder control;
- do not use t-SNE appearance as a pass criterion.

A provisional numerical neighborhood of `60-70%` purity is useful for early
debugging, but it is not a formal equivalence interval.

### Linear identity gate

- relative identity error below 5%;
- relative skew norm below 5%;
- mean K-means-centroid action error below 2%;
- at least 18 dominant eigenvalues visibly concentrated near 1;
- low prediction loss without total representation collapse.

### Random-initialization control

- use the same dataset realization, split, encoder initialization, batches, and
  training schedule as the identity condition;
- change only the initialization of `M`;
- expect comparable prediction and clustering quality;
- expect substantially worse identity and symmetry metrics and a dense matrix.

## Required experiment matrix

| Experiment | Predictor | Purpose |
|---|---|---|
| JEPA-MLP | Published MLP | Reproduce clustering purity |
| Autoencoder | Matched encoder | Reproduce clustering baseline |
| JEPA-linear-identity | Linear, `M_0=I` | Reproduce interpretable Koopman action |
| JEPA-linear-random | Linear, random `M_0` | Test basis-selection claim |

The clustering number and the linear-operator numbers come from different
predictor experiments in the paper. They must not be presented as measurements
from one shared checkpoint.

## Questions for the authors

1. Is the encoder projection `6144 -> 32` or `6144 -> 64 -> 32`?
2. Does the MLP predictor have one or two hidden layers?
3. Which optimizer, learning rate, batch size, schedule, and number of epochs
   produced the published figures?
4. Which seeds and how many independent runs produced the reported means?
5. What exact random initialization was used for the linear-predictor control?
6. Can the exact generator code clarify pulses, trends, high noise, ARMA burn-in,
   RNGs, and deterministic splitting?
7. What decoder and objective were used for the autoencoder control?
8. Is an official code release available or planned?

No author contact should be made without explicit user approval.

## Next implementation step

Complete the remaining six stochastic regimes and then build the dataset-audit
notebook. The five sinusoidal, three trend-containing, and four non-smooth
waveform regimes are implemented. They are covered by frequency, amplitude,
harmonic-content, deterministic randomization, affine standardization,
transition-count, pulse-policy, and zero-observation-noise tests. Use named
assumptions for every unresolved item, and keep the reduced Phase 0 generator
unchanged. Do not implement or train the paper-faithful model until the dataset
audit is reviewed.

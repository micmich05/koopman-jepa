# Paper-faithful replication specification

## Status

- Audit date: 2026-09-19
- Paper: *Koopman Invariants as Drivers of Emergent Time-Series Clustering in
  Joint-Embedding Predictive Architectures*
- Authors: Pablo Ruiz-Morales, Dries Vanoost, Davy Pissoort, Mathias Verbeke
- Conference version: AAAI 2026
- Extended version: arXiv v2, 2026-01-23
- Implementation status: dataset audited, preprocessing conditions frozen,
  model variants implemented, and both single-run development gates passed;
  the five-seed stability gate failed, so test evaluation remains locked

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

### Local ARMA assumptions

The five AR, MA, and ARMA regimes use
`statsmodels.tsa.arima_process.ArmaProcess`, as specified by the paper. The AR
polynomial follows the statsmodels convention `[1, -phi_1, ...]`, while the MA
polynomial is `[1, theta_1, ...]`. Innovations are independent standard-normal
samples drawn from the stable per-sequence RNG.

The paper does not report burn-in or initial conditions. The local baseline
uses the `generate_sample` default `burnin=0`, which corresponds to a zero-state
filter before the first innovation. This minimizes deviations from the
documented library default but means the beginning of a finite sample is not
drawn exactly from the stationary marginal distribution. The dataset notebook
must compare early and late empirical variance, and a positive-burn-in variant
must be included in the sensitivity analysis. Per-sequence standardization
removes the innovation scale but does not remove this possible transient.

### Local high-noise assumption

The paper describes `Sine_HighNoise` only as a medium-frequency sinusoid with
internal process-noise standard deviation approximately three times that of
the ARMA processes. The local baseline interprets this as independent Gaussian
noise with standard deviation `3.0`, added to a unit-amplitude sinusoid whose
phase follows the same `Normal(0, pi^2)` convention as the other sinusoids.

This gives a pre-standardization RMS signal-to-noise ratio of
`(1/sqrt(2))/3`, approximately `0.236` or `-12.55 dB`. Per-sequence
standardization preserves that ratio. Both the noise family and the exact
factor require sensitivity checks because “approximately 3x” is not a complete
executable specification.

## Executed dataset-audit findings

The executed notebook `notebooks/paper_dataset_audit.ipynb` records the figures,
metrics, and written interpretation. Its principal findings are:

- the mechanical gate passes after using float64 accumulation for numerically
  stable per-sequence standardization;
- `Sine_MedFreq` and `Sine_LowAmp` become the same observable distribution
  after per-sequence standardization, because their only difference is a
  positive multiplicative factor;
- literal slope randomization reverses 6.80% of sampled `Trend_Up` sequences
  and 6.00% of sampled `Trend_Down` sequences in the deterministic audit;
- the local pulse-overlap policy produces fewer than five visible connected
  events in 33.50% of sampled sequences;
- for standardized AR(0.9), the mean early/late variance ratio is 0.899 with
  zero burn-in and 1.046 with a 256-step burn-in;
- using sample indices rather than normalized time reduces the bin-10 spectral
  power fraction of `Sine_Trend` from 0.731 to 0.006278.

The first item passes software/data mechanics. The second is a critical
identifiability failure for the claim that all 18 labels correspond to distinct
observable regimes under the published preprocessing. It is a property of the
published specification, not an implementation defect.

## Frozen preprocessing conditions

The dataset audit led to two explicitly named conditions:

1. `paper_literal` is the primary reproduction condition. It preserves the
   published per-master standardization and therefore also preserves the exact
   observational equivalence between `Sine_MedFreq` and `Sine_LowAmp`.
2. `amplitude_preserving` is a sensitivity condition. It applies one scalar
   mean and standard deviation fitted across raw training masters only, then
   reuses those statistics for validation and test. This prevents split leakage
   and preserves the published `0.3:1.0` amplitude ratio.

The global-statistics fitter is streaming and does not materialize the full
training set. A reduced fit may be used for smoke tests only; reported full-run
results must fit all 7,000 training masters per regime and persist the resulting
statistics with the run artifacts.

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

The local implementation represents this objective as

```text
mean_over_batch(sum_over_latent((prediction - target_embedding)^2))
```

This is the direct minibatch reduction of the squared Euclidean norm written in
the paper. It is not PyTorch's default element-wise `MSELoss`, which would also
divide by the latent dimension and would therefore be smaller by a factor of
32. That constant does not change the minimizer, but it does change gradient
scale and its interaction with any reconstructed learning rate.

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

### Frozen local model variants

The implementation in `src/koopman_jepa/paper_model.py` maps the published and
ambiguous cases to explicit names:

| Option | Local name | Implemented architecture |
|---|---|---|
| Main-text encoder | `direct` | `6144 -> 32` |
| Reconciled appendix encoder | `two_stage` | `6144 -> 64 -> 32`, with an intermediate ReLU |
| Linear predictor, primary | `linear` + `identity` | bias-free `32 -> 32`, initialized to the identity |
| Linear predictor, control | `linear` + `xavier_uniform` | bias-free `32 -> 32`, Xavier-uniform initialization |
| MLP prose reading | `mlp` + `one_hidden` | `32 -> 64 -> 32`, with one hidden ReLU |
| MLP table reading | `mlp` + `two_hidden` | `32 -> 64 -> 64 -> 32`, with two hidden ReLUs |

`direct` is the primary encoder condition because it follows the main text and
produces the stated 32-dimensional representation directly. `two_stage` is a
sensitivity reconstruction, not an architecture explicitly written in full in
the paper. Likewise, Xavier-uniform is a reproducible local choice for the
random-initialization control; the paper does not identify its random
initialization distribution.

The online and target encoders start with identical parameters. Target
parameters are frozen with respect to gradient updates and are moved after each
optimizer step using the published EMA decay `0.996`. Structural tests lock the
convolutional geometry, projection shapes, predictor variants, target freezing,
and EMA calculation before any result-producing training run.

### One-batch training gate

`src/koopman_jepa/paper_training.py` implements the smallest complete training
transition:

1. clear gradients;
2. encode context with the online encoder and target with the frozen target
   encoder;
3. minimize only the squared embedding prediction error;
4. update the online encoder and predictor with a caller-provided optimizer;
5. update the target encoder from the new online parameters using EMA.

A deterministic CPU smoke test confirms that the online encoder and predictor
receive nonzero gradients and change, target parameters never receive
gradients, and every target parameter follows the exact `0.996 / 0.004` EMA
calculation after the optimizer step. The test uses SGD only as a diagnostic
instrument. This does not select SGD for the paper reproduction: optimizer and
learning rate remain unpublished experimental variables and must be frozen in
the next protocol step.

### Frozen fixed-batch development gate

Before any multi-epoch run, `configs/paper_overfit_smoke.yaml` freezes a local
debugging condition. It is not attributed to the paper and its measurements
must not be compared with published clustering or operator results.

| Parameter | Development value |
|---|---:|
| Training sequences | 1 per regime, 18 total |
| Normalization | `per_sequence` |
| Batch | one fixed balanced batch of 18 |
| Encoder | `direct`, latent dimension 32 |
| Predictor | linear, identity initialization |
| EMA decay | 0.996 |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0 |
| Steps | 100 |
| Seed | 0 |
| Device | CPU |

Because predictive loss can decrease through representational collapse, the
run records mean coordinate-wise embedding dispersion and the entropy-based
effective rank in addition to loss and gradient norms. With 18 samples, the
centered batch can have rank at most 17 even though the latent dimension is 32.

The gate is evaluated on the first and last ten steps using criteria fixed
before execution:

- every recorded value must be finite;
- final mean loss must be at most 25% of initial mean loss;
- final mean embedding dispersion must retain at least 10% of its initial
  value;
- final mean effective rank must be at least 2.

Passing this gate only demonstrates that the implementation can optimize a
memorization-scale batch without complete collapse under one plausible local
setup. It provides no evidence of held-out performance or paper reproduction.

The executed notebook `notebooks/paper_overfit_smoke.ipynb` passed all four
criteria:

| Diagnostic | Executed result | Gate |
|---|---:|---:|
| Initial ten-step mean loss | 0.110840 | reference |
| Final ten-step mean loss | 0.00002798 | - |
| Final/initial loss ratio | 0.000252 | at most 0.25 |
| Final/initial embedding-dispersion ratio | 0.9913 | at least 0.10 |
| Final mean effective rank | 12.078 | at least 2.0 |
| All recorded values finite | yes | required |

The curves are not monotone. Loss peaks at `0.548645` on step 2 before its
sustained decline; embedding dispersion reaches its minimum at step 19, and
effective rank reaches its minimum of `7.727` at step 26. Both representation
diagnostics then recover. The transients therefore deserve monitoring in the
next longer run even though this gate shows no complete collapse.

### Frozen train/validation development gate

`configs/paper_train_validation_smoke.yaml` defines the next condition. Like
the fixed-batch gate, this is a local debugging protocol rather than a recovered
paper configuration.

| Parameter | Development value |
|---|---:|
| Training sequences | 32 per regime, 576 total |
| Validation sequences | 8 per regime, 144 total |
| Test sequences | 8 per regime, reserved and unused |
| Normalization | `per_sequence` |
| Encoder | `direct`, latent dimension 32 |
| Predictor | linear, identity initialization |
| EMA decay | 0.996 |
| Optimizer | AdamW |
| Learning rate | 0.0003 |
| Weight decay | 0 |
| Batch size | 64 |
| Epochs | 10 |
| Seed | 0 |
| Device | CPU |

The smaller learning rate is a conservative response to the early transient in
the fixed-batch curves; it is not attributed to the paper. Training batches are
shuffled reproducibly. Epoch 0 evaluates the untrained model, and every trained
epoch is followed by complete, unshuffled train and validation evaluations.
The test split remains untouched.

The final two epochs are averaged and compared with the epoch-0 baseline. The
criteria, frozen before execution, require:

- every recorded loss, gradient norm, dispersion, and effective rank to be
  finite;
- final validation loss at most 50% of untrained validation loss;
- final validation/train loss ratio at most 4;
- final validation embedding dispersion at least 10% of its untrained value;
- final validation effective rank at least 4.

Passing would show that the prediction objective improves on unseen sequences
without complete representational collapse under this small condition. It
would still not establish downstream clustering quality, reproduce the paper,
or justify using these unpublished optimization settings for a final result.

The executed notebook `notebooks/paper_train_validation_smoke.ipynb` passes the
preregistered aggregate gate:

| Diagnostic | Executed result | Gate |
|---|---:|---:|
| Untrained validation loss | 0.0067694 | reference |
| Final two-epoch train loss | 0.00043125 | - |
| Final two-epoch validation loss | 0.00166635 | - |
| Final/untrained validation loss ratio | 0.2462 | at most 0.50 |
| Final validation/train loss ratio | 3.864 | at most 4.0 |
| Validation-dispersion retention | 0.6695 | at least 0.10 |
| Final validation effective rank | 20.934 | at least 4.0 |
| All recorded values finite | yes | required |

The PASS has an important warning. Validation first crosses the loss threshold
at epoch 3 and reaches its best observed value, `0.00165277`, at epoch 10, but
its improvement from epoch 9 to 10 is only 1.6% while train improves 24.9%.
Consequently, the individual validation/train ratio rises monotonically after
epoch 2 and reaches `4.467` at epoch 10, above the gate threshold. The
preregistered gate still passes because it computes the ratio between the mean
losses of epochs 9 and 10, yielding `3.864`; that rule is not changed after
observing the result.

This is evidence of increasing overfit, not collapse: validation dispersion
retains 67.0% of its initial value, and effective rank bottoms at `17.656` on
epoch 6 before recovering. Test evaluation remains closed until checkpoint
selection and multi-seed stability are specified without reference to test.

### Frozen checkpoint and seed-stability gate

`configs/paper_seed_stability_smoke.yaml` reuses the exact same dataset,
architecture, optimizer, and ten-epoch schedule. The raw dataset seed remains
fixed at 0, while training seeds `0, 1, 2, 3, 4` vary model initialization and
minibatch order. This isolates optimization variability from dataset-sampling
variability. Test remains unused.

Checkpoint selection is constrained rather than simply choosing the last epoch
or the numerically smallest validation loss. An epoch is eligible only if:

- validation loss is at most 50% of its epoch-0 baseline;
- its individual validation/train loss ratio is at most 4;
- validation embedding dispersion retains at least 10% of its baseline;
- validation effective rank is at least 4;
- all involved metrics are finite.

Among eligible epochs, the checkpoint epoch is the one with the lowest
validation loss; exact ties select the earlier epoch. Applied retrospectively
to the already executed seed-0 history, this policy selects epoch 9
(`validation loss = 0.00167994`, gap `3.411`) rather than epoch 10, whose lower
validation loss comes with an ineligible gap of `4.467`. This observation
motivates the policy but does not evaluate any test sample.

The five-seed stability gate, frozen before running seeds 1–4, requires:

- every requested seed to produce an eligible checkpoint;
- every recorded checkpoint metric to be finite;
- the worst validation/baseline loss ratio to be at most 0.50;
- the worst validation/train loss ratio to be at most 4.0;
- the coefficient of variation of selected validation losses to be at most
  0.25;
- the worst validation-dispersion retention to be at least 0.10;
- the worst validation effective rank to be at least 4.0.

Passing will establish repeatability across a small set of optimization seeds
on one fixed development dataset. It will not establish robustness to dataset
sampling, and it will not by itself authorize a paper-level reproduction claim.

The executed notebook `notebooks/paper_seed_stability_smoke.ipynb` produces an
eligible checkpoint for every seed, but the global gate fails:

| Diagnostic | Executed result | Gate |
|---|---:|---:|
| Selected epochs for seeds 0–4 | 9, 10, 10, 10, 8 | all required |
| CV of selected absolute validation loss | 0.2879 | at most 0.25 |
| Worst validation/baseline loss ratio | 0.2777 | at most 0.50 |
| Worst validation/train loss ratio | 3.671 | at most 4.0 |
| Worst validation-dispersion retention | 0.6321 | at least 0.10 |
| Worst validation effective rank | 17.296 | at least 4.0 |
| All checkpoint metrics finite | yes | required |

The checkpoint constraint behaves as intended: it rejects the final epochs for
seeds 0 and 4 because their individual validation/train gaps exceed 4, then
selects earlier eligible epochs. All per-seed improvement and representation
criteria pass. The only failed criterion is variability of the *absolute*
validation losses, whose maximum is 2.34 times their minimum.

Absolute JEPA prediction loss is sensitive to latent scale because this model
does not normalize or otherwise fix embedding magnitude. As a post-result
diagnostic, the coefficient of variation of the scale-relative
validation/baseline ratios is `0.210`, below the absolute-loss CV of `0.288`.
This scale-free statistic was not preregistered and therefore does not change
the FAIL. It motivates a separately versioned stability protocol whose metric
is explicitly invariant to latent scale. Test remains untouched meanwhile.

### Executed scale-invariant stability protocol

`configs/paper_seed_stability_scale_invariant.yaml` freezes the follow-up
protocol before observing its outcomes. It keeps the dataset realization,
architecture, optimizer, ten-epoch schedule, checkpoint-selection rule, and
all worst-case thresholds unchanged. It uses fresh training seeds
`5, 6, 7, 8, 9`, so the same seed results that motivated the revision cannot
also validate it. Test remains unused.

The only changed aggregate criterion is the definition of cross-seed
variability. For each seed `s`, let

```text
r_s = selected validation loss_s / epoch-0 validation loss_s.
```

The v2 gate requires the population coefficient of variation of the five
`r_s` values to be at most `0.25`. This quantity is invariant to multiplying
all embeddings from one seed by a seed-specific constant, because both losses
for that seed scale by the same squared constant. The CV of the selected
absolute validation losses is still recorded, but it is diagnostic and cannot
make the v2 gate pass or fail.

The gate passes only if all of the following hold simultaneously:

- every seed produces one constraint-eligible checkpoint;
- all checkpoint metrics are finite;
- the worst validation/baseline ratio is at most `0.50`;
- the worst validation/train ratio is at most `4.0`;
- the CV of validation/baseline ratios is at most `0.25`;
- the worst validation-dispersion retention is at least `0.10`;
- the worst validation effective rank is at least `4.0`.

The executed notebook
`notebooks/paper_seed_stability_scale_invariant.ipynb` passes the frozen v2
gate on the fresh seeds:

| Diagnostic | Executed result | Gate |
|---|---:|---:|
| Selected epochs for seeds 5–9 | 10, 10, 10, 8, 10 | all required |
| CV of selected validation/baseline ratios | 0.2388 | at most 0.25 |
| CV of selected absolute validation loss | 0.2508 | diagnostic only |
| Worst validation/baseline loss ratio | 0.3374 | at most 0.50 |
| Worst validation/train loss ratio | 3.510 | at most 4.0 |
| Worst validation-dispersion retention | 0.6648 | at least 0.10 |
| Worst validation effective rank | 18.609 | at least 4.0 |
| All checkpoint metrics finite | yes | required |

Seed 8 exceeds the validation/train limit after epoch 8, so the constrained
policy rejects its later epochs and selects epoch 8. The other four seeds
remain eligible at epoch 10. None of the selected representations is close to
the preregistered collapse limits.

The variability result is a narrow pass: `0.2388` is only `0.0112` below the
`0.25` cutoff. Moreover, scale normalization reduces the observed CV by only
`0.0120` relative to the absolute-loss diagnostic. The result is compatible
with latent-scale sensitivity and confirms the revised rule on seeds not used
to design it, but it is not strong evidence that scaling explains all
cross-seed variability.

This revision establishes limited repeatability of *relative predictive
improvement* under optimization seeds on one fixed development dataset. It
does not prove that the latent coordinate system itself is stable, does not
measure dataset-sampling variability, and does not reproduce the paper's
clustering or operator diagnostics. Test was not instantiated or consulted.

### Executed linear-identity held-out protocol

`configs/paper_linear_identity_heldout_smoke.yaml` freezes the first held-out
evaluation before any test sample is instantiated. This is a local smoke
evaluation on 8 test masters per regime, not the paper-scale run with 1,000
test masters per regime.

The five training seeds and their constraint-selected epochs are fixed in
sweep order as `(5:10, 6:10, 7:10, 8:8, 9:10)`. Each model must first be
replayed using train and validation only. Before test is constructed, the
implementation must verify all of the following:

- the selected epoch exactly matches the frozen epoch for that seed;
- the captured state contains online encoder, EMA target encoder, and linear
  predictor parameters;
- loading the captured state reproduces its validation loss, embedding
  dispersion, and effective rank within absolute tolerance `1e-8`;
- no test dataset or test loader is created during checkpoint replay.

Failure of any replay check aborts the held-out run. It does not trigger a new
checkpoint choice.

Checkpoint capture and replay are implemented in
`src/koopman_jepa/paper_training.py`. The shared training loop snapshots the
complete model state after each trained epoch, retains only the state selected
by the existing validation constraints, reloads it strictly, and recomputes
validation diagnostics. Unit tests also verify that enabling capture does not
alter the original training history and that a wrong expected epoch fails the
replay gate. This implementation does not construct a test dataset.

For each verified checkpoint, test contexts are embedded by the online
encoder. K-means is fitted to those raw 32-dimensional embeddings with
`K = 18`, `n_init = 20`, and `random_state = 0`. Test labels are not used to
fit K-means. Let `M` be the learned `32 × 32` linear predictor and let `c_i`
be the 18 K-means centroids. The preregistered diagnostics are:

```text
relative identity error = ||M - I||_F / ||M||_F
relative skew norm      = ||M - M^T||_F / ||M||_F
centroid action error   = mean_i ||M c_i - c_i||_2 / ||c_i||_2
near-identity count     = #{lambda in eig(M): |lambda - 1| <= 0.05}
```

The local gate requires **every one** of the five checkpoints to satisfy:

- relative identity error at most `0.05`;
- relative skew norm at most `0.05`;
- mean centroid action error at most `0.02`;
- at least 18 eigenvalues within complex distance `0.05` of 1;
- test embedding effective rank at least `4.0`;
- all reported values finite.

The first three thresholds are the provisional linear-identity criteria
already stated in this specification. The eigenvalue tolerance makes the
previous phrase “visibly concentrated near 1” executable; it is a local rule,
not a tolerance reported by the authors. Test effective rank is an explicit
anti-collapse guard. Per-seed values plus mean and standard deviation will be
reported; the aggregate decision is a conjunction, so averaging cannot hide a
failed seed.

K-means label purity may be reported as a non-gating diagnostic, but it cannot
be compared directly with the paper's `65.48%` result because that number comes
from the separate nonlinear-MLP predictor experiment. No threshold may be
changed after opening test.

The frozen operator diagnostics and aggregate conjunction are implemented in
`src/koopman_jepa/paper_evaluation.py`. The implementation uses the predictor
matrix in the same column-vector convention as the equations (`M c`), which is
`centroids @ M.T` for row-major NumPy centroids. Its effective-rank calculation
matches the entropy-of-squared-singular-values definition used during
training. Synthetic unit tests cover the exact identity, a nonsymmetric matrix
with closed-form norms, missing and failed seeds, non-finite metrics, and input
shape validation. These tests were completed before any held-out embedding was
computed.

The executed notebook
`notebooks/paper_linear_identity_heldout_smoke.ipynb` implements the complete
control flow. It constructs only train and validation first, reproduces all
five checkpoints, and sets the held-out authorization flag only when every
replay result passes. The later test-construction cell asserts that flag before
instantiating `PaperRegimeDataset(..., "test", ...)`. Its result section is
already parameterized to report every gate, failed seed, per-seed plot,
spectral plot, mean and dispersion, and the limitations of the small sample.

All five checkpoint replays pass before test construction. The selected epochs
match `(10, 10, 10, 8, 10)`, and every replayed validation loss, embedding
dispersion, and effective rank matches its captured value with absolute error
zero. The 144 test samples are disjoint from train and validation.

| Seed | Identity error | Skew norm | Centroid action | Eigenvalues near 1 | Test rank |
|---:|---:|---:|---:|---:|---:|
| 5 | 1.4006% | 0.6868% | 1.1804% | 31 | 21.221 |
| 6 | 1.6948% | 0.6027% | 1.9567% | 31 | 20.133 |
| 7 | 1.8526% | 0.7399% | 0.6499% | 31 | 21.324 |
| 8 | 1.4427% | 0.6415% | 1.1865% | 31 | 17.963 |
| 9 | 1.5166% | 0.6545% | 0.8075% | 31 | 18.645 |

The aggregate held-out gate passes every frozen criterion for every seed. The
closest result is seed 6 centroid action at `1.9567%`, only `0.0433` percentage
points below the `2%` cutoff. Identity, skew, rank, and the local spectral count
have substantially larger margins.

The spectral plot shows 31 of 32 eigenvalues inside the `0.05` disk around 1
for every seed, plus one real outlier around `0.90–0.93`. This exceeds the local
minimum of 18 but does not establish the paper's claimed structure of 18
dominant invariant modes. In this short identity-initialized smoke condition,
31 near-identity modes may also indicate that most of the full predictor has
barely moved from initialization.

Across seeds, population mean ± standard deviation is `1.581% ± 0.169%` for
identity error, `0.665% ± 0.046%` for skew norm, `1.156% ± 0.452%` for centroid
action, and `19.86 ± 1.35` for test effective rank. These results qualitatively
support near-identity action without representation collapse. They are not a
formal numerical reproduction because the smoke test uses only 8 held-out
masters per regime, the paper does not report seed dispersion, and the active
spectral-rank definition is underspecified. This test split is now consumed for
the frozen smoke protocol.

### Frozen randomly initialized linear control (not yet executed)

`configs/paper_linear_random_control_smoke.yaml` freezes the paired control for
the paper's claim that random predictor initialization reaches similarly low
prediction loss while producing a dense, non-identity matrix. Its first stage
uses train and validation only; it does not instantiate test.

For each seed `5–9`, the identity and random conditions reset Python, NumPy,
and PyTorch to the same seed before model construction. Both constructors build
the online encoder and copy the EMA target before applying their named linear
predictor initialization. A unit test verifies exact equality of every online
and target encoder tensor and inequality of the predictor matrices. The data
realization, minibatch generator, optimizer, schedule, and validation
checkpoint constraints are unchanged. Thus the intended treatment difference
is only `M_0 = I` versus Xavier-uniform `M_0`.

The development stage first replays the committed identity checkpoints at
epochs `(10, 10, 10, 8, 10)`. It then trains and selects the random condition
without consulting the identity test metrics. The random checkpoints must pass
the same scale-invariant stability gate:

- every seed produces a constraint-eligible checkpoint;
- worst validation/baseline ratio at most `0.50`;
- worst validation/train ratio at most `4.0`;
- CV of validation/baseline ratios at most `0.25`;
- worst dispersion retention at least `0.10`;
- worst effective rank at least `4.0`.

For paired predictive comparability, define for each seed

```text
q_s = (random selected validation / random epoch-0 validation)
      / (identity selected validation / identity epoch-0 validation).
```

Every `q_s` must be at most `2.0`. This compares relative improvement and is
invariant to a separate global latent rescaling of either condition. Paired
absolute validation-loss ratios will still be reported as diagnostics but do
not decide the gate.

The structural control is evaluated from predictor weights only, without any
held-out data. Every random checkpoint must have:

- relative identity error `||M-I||_F / ||M||_F` at least `0.50`;
- off-diagonal fraction `||M-diag(M)||_F / ||M||_F` at least `0.50`.

These are local operational definitions of the paper's qualitative words
“non-identity” and “dense”; the authors report no corresponding thresholds.
Passing this development protocol will freeze the random checkpoints before
any paired evaluation on the already-consumed smoke test. Such a later
comparison can still test the predeclared control, but it will not constitute a
second independently blind use of that split.

The paired metrics are implemented in
`src/koopman_jepa/paper_evaluation.py`. The structural calculation separates
the diagonal explicitly and normalizes both identity distance and off-diagonal
mass by `||M||_F`. The aggregate result retains per-seed comparisons and
requires exact seed coverage, finite values, predictive comparability,
non-identity structure, and density simultaneously. Synthetic tests verify
that the relative-improvement gate is unaffected by a fourfold absolute-loss
difference, that an identity matrix fails both structural controls, and that
missing seeds or non-finite matrices cannot pass. No random-control training or
held-out evaluation has been run.

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

Prepare an unexecuted paired train/validation notebook. It must replay the
identity checkpoints, train the Xavier-initialized models with identical
encoders and batches, apply both the stability and paired-comparison gates, and
avoid any construction of test. If the development gate passes, freeze its
selected random checkpoint epochs before any comparison on the already-
consumed smoke split. Keep the reduced Phase 0 pipeline unchanged.

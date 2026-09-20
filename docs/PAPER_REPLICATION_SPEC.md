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

### Executed randomly initialized linear development control

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
Passing this development protocol freezes the random checkpoints before any
paired evaluation on the already-consumed smoke test. Such a later
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
missing seeds or non-finite matrices cannot pass.

The executed notebook
`notebooks/paper_linear_random_control_smoke.ipynb` implements the paired
development run. It first reproduces all five identity checkpoints, stores
their pre-training online and target encoder states, resets each seed, and
requires exact equality with the corresponding Xavier model before training.
It then captures and reloads the random checkpoint selected by the unchanged
validation constraints, evaluates both gates, plots all trajectories and
paired diagnostics, and generates a written interpretation. It has no
test-dataset construction and did not instantiate or consult test.

All five identity replays and all five random checkpoint replays pass. Initial
online and target encoder tensors are exactly paired for every seed. The random
condition selects epoch 10 for seeds `5–9`.

| Seed | Val/base | Val/train | Std/base | Rank | q random/identity | Absolute loss factor | Identity error | Off-diagonal |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.041 | 1.602 | 0.704 | 18.29 | 0.151 | 1.291 | 142.03% | 98.94% |
| 6 | 0.045 | 1.784 | 0.810 | 15.23 | 0.133 | 1.141 | 138.38% | 98.62% |
| 7 | 0.061 | 2.019 | 0.769 | 18.48 | 0.204 | 1.165 | 142.70% | 98.55% |
| 8 | 0.042 | 2.059 | 0.862 | 12.29 | 0.229 | 1.129 | 140.61% | 98.20% |
| 9 | 0.034 | 1.992 | 0.826 | 14.78 | 0.180 | 1.253 | 140.87% | 98.52% |

Both frozen development gates pass. The validation/baseline CV is `0.204`
against the `0.25` maximum; the worst validation/baseline ratio is `0.061`, the
worst validation/train ratio is `2.059`, the minimum dispersion retention is
`0.704`, and the minimum effective rank is `12.29`. No collapse criterion is
close to failing.

Every paired relative-improvement factor is below one, with a worst value of
`0.229` against the `2.0` cutoff. This means the Xavier condition reduces loss
more strongly relative to its own epoch-0 baseline. It does **not** mean that
its selected absolute validation loss is lower: the random/identity absolute
factor ranges from `1.129` to `1.291`. That difference is consistent with the
two independently learned latent coordinate systems having different scales,
which is why the absolute comparison was predeclared as diagnostic only.

The structural treatment check is unambiguous: minimum relative distance to
identity is `138.4%` and minimum off-diagonal norm fraction is `98.2%`, both far
past their `50%` thresholds. Thus low normalized predictive error is compatible
in development with a dense non-identity operator. This supports the paper's
qualitative basis-selection mechanism, but still supplies no held-out evidence
for the random control and no full-scale numerical reproduction.

### Executed paired held-out random control

`configs/paper_linear_random_heldout_smoke.yaml` freezes the next protocol
before any random-control test embedding is computed. It retains the same
dataset realization, five seeds, optimizer, schedule, checkpoint policy, and
model architecture. The identity checkpoint epochs are `(10, 10, 10, 8, 10)`;
the random checkpoint epochs are `(10, 10, 10, 10, 10)`. Both sets must replay
with absolute metric tolerance `1e-8`. Test construction is unauthorized unless
all ten replays pass.

After authorization, both conditions use the same 144 test pairs and labels.
For condition `c` and seed `s`, the scale-normalized prediction error is

```text
e[c,s] = sqrt(
    sum_i ||M[c,s] z_online[c,s,i] - z_target[c,s,i]||_2^2
    / sum_i ||z_target[c,s,i]||_2^2
).
```

The square root keeps the quantity on the embedding scale, while division by
target energy makes the paired comparison insensitive to a separate global
rescaling of either learned latent system. A random condition can pass only if
`e[random,s] / e[identity,s] <= 2.0` for every seed.

Clustering uses raw online test embeddings with `K = 18`, `n_init = 20`, and
`random_state = 0`. Purity is `sum_k max_j n[k,j] / N`, where `n[k,j]` counts
ground-truth regime `j` inside cluster `k`. For every seed, random purity must
be at least `0.50` and at least `0.90` times identity purity. The absolute rule
prevents two equally poor clusterings from passing a purely relative gate; the
relative rule operationalizes the paper's qualitative claim that clear
clustering is retained. Matched cluster accuracy will be reported as a
diagnostic but is not a gate because the paper reports purity.

Non-collapse is checked by the entropy effective rank already used throughout
the smoke pipeline. Random test rank must be at least `4.0` and at least `0.50`
times the paired identity rank for every seed. The structural properties of
the random matrices will still be reported, but need not be re-gated because
the exact same frozen checkpoints already passed the data-independent
non-identity and off-diagonal controls in development.

The paired held-out gate passes only if every seed satisfies all five criteria
and every reported quantity is finite. These are local criteria chosen before
the random test run; they are not thresholds published by the authors. The
identity side of this test split has already been inspected, so the experiment
can evaluate the predeclared random-control hypothesis but cannot restore an
independently blind test claim.

The metric implementation is complete in
`src/koopman_jepa/paper_evaluation.py`. Synthetic tests cover exact predictor
recovery, perfectly separated clusters, invalid labels and shapes, each gate
family, missing seeds, and non-finite metrics.

The executed notebook
`notebooks/paper_linear_random_heldout_smoke.ipynb` implements the frozen
flow. It constructs only train and validation, recreates the five identity and
five Xavier checkpoints, verifies paired initial encoder tensors and distinct
predictor initializations, and sets a single authorization flag. Exactly one
later code cell can instantiate `PaperRegimeDataset(..., "test", ...)`, and it
asserts that flag first. The notebook then evaluates both conditions on the
same ordered test loader, reports every seed, plots all absolute and relative
gates, and generates a parameterized written interpretation. A source-level
test protects the test-construction cell and its ordering.

All ten checkpoint replays and all five initialization-pairing checks pass
before the 144 test pairs are constructed. Train, validation, and test sample
keys are disjoint. The test is now consumed for both linear conditions.

| Seed | Prediction I | Prediction R | R/I | Purity I | Purity R | R/I | Rank I | Rank R | R/I |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.201 | 0.296 | 1.470 | 45.83% | 50.69% | 1.106 | 21.22 | 18.85 | 0.888 |
| 6 | 0.244 | 0.285 | 1.169 | 49.31% | 50.00% | 1.014 | 20.13 | 14.32 | 0.711 |
| 7 | 0.234 | 0.207 | 0.886 | 46.53% | 47.92% | 1.030 | 21.32 | 17.99 | 0.844 |
| 8 | 0.244 | 0.281 | 1.151 | 52.78% | 51.39% | 0.974 | 17.96 | 12.25 | 0.682 |
| 9 | 0.225 | 0.326 | 1.445 | 52.78% | 51.39% | 0.974 | 18.64 | 14.67 | 0.787 |

The preregistered aggregate gate is **FAIL**. Seed 7 random purity is `47.92%`
or 69 of 144 assignments, three assignments below the `50%` threshold. This is
the only failed criterion and the only failed seed. The threshold is not
changed after observing test.

The failure is narrow but cannot be relabeled as PASS. All predictive ratios
pass, with a worst value of `1.470` against `2.0`. Every relative-purity ratio
passes, with a minimum of `0.974` against `0.90`; random mean purity is `50.28%`
versus `49.44%` for identity. All rank gates pass, with minimum random rank
`12.25` and minimum retention `0.682`. Thus the observed random condition does
not degrade clustering relative to identity and remains non-collapsed, but it
does not satisfy the separately frozen requirement that every random seed
exceed `50%` absolute purity. Notably, three identity seeds are also below that
absolute level, which makes the smoke threshold stringent relative to this
small sample without invalidating its predeclared decision rule.

The result is consistent with the qualitative basis-selection mechanism:
dense non-identity matrices retain paired predictive error, relative
clustering, and effective rank. It is not a successful formal smoke gate, and
it is not a paper-scale reproduction.

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

## Official-source re-audit (2026-09-20)

The current official sources were checked again before defining another
optimization sensitivity:

- the AAAI publication page and proceedings PDF:
  <https://ojs.aaai.org/index.php/AAAI/article/view/39708>;
- arXiv v2, last revised on 2026-01-23, including its HTML appendices and TeX
  source entry: <https://arxiv.org/abs/2511.09783>.

Neither official landing page links an implementation, dataset artifact,
training configuration, or checkpoint. The v2 experimental section and
appendices still specify the dataset geometry, EMA decay, convolutional blocks,
latent-width prose, and predictor tables, but not the optimizer, learning rate,
batch size, number of epochs or updates, learning-rate schedule, gradient
clipping, checkpoint rule, model seeds, or embedding normalization. Therefore,
the local AdamW recipe cannot be described as the paper's recipe.

The appendix ambiguity is also unchanged: it states `k = 32`, while the encoder
table ends with a linear output of `2k`; its MLP table and prose disagree about
whether there are one or two hidden layers. Both locally coherent encoder
readings already have executed validation results below: the reconciled
`6144 -> 64 -> 32` reading performs worse than the direct `6144 -> 32` reading.
No further width reinterpretation is justified before new primary evidence.

## Next implementation step

The primary-source re-audit found no released optimization recipe. A repeated
medium-scale seed-10 diagnostic, using the same validation-only condition,
measured gradient norms alongside the already observed divergence. At epoch 2,
online/predictor gradient norms were `0.037/0.076` and validation loss was
`0.015x` its untrained baseline. At epoch 10 they were `11.929/16.234` with
loss `2.334x`; at epoch 20 they reached `1655.663/3774.823` with loss
`1014.353x`. This is direct gradient explosion, not merely a late increase in
validation error.

The next sensitivity will add global gradient-norm clipping at `1.0` while
holding the direct one-hidden condition fixed. This threshold is predeclared,
standard, and does not affect the early epoch-2 gradients; it targets the later
runaway dynamics. Because clipping is absent from the paper, the condition is
an explicit local stabilization experiment rather than a literal reproduction.
It must be evaluated on train/validation only; do not construct test.

An exploratory diagnostic notebook was executed at
`notebooks/paper_linear_random_heldout_diagnostic.ipynb`. It reconstructs the
same checkpoints and consumed test only to localize regime confusions, measure
K-means random-state sensitivity, and isolate the observationally equivalent
sinusoid pair. It contains no aggregate gate and cannot revise the recorded
FAIL.

The notebook reproduces every fixed purity count exactly. Across 20 descriptive
K-means random states, each still using `n_init = 20`, observed purity spans
`43.75–55.56%` for identity and `44.44–56.94%` for random over the five model
seeds. Seed 7 random has mean `47.67% ± 1.46%`, compared with its frozen value
`47.92%`. The algorithmic variation is material relative to the three-sample
gap that triggered the gate, but the predeclared `random_state = 0` remains the
only value used for the recorded decision.

The label-aligned confusion matrices show heterogeneous regime difficulty.
Random mean recall is `100%` for both trends and both square waves, but only
`10.0%` for strong positive AR, `17.5%` for positive MA, and `22.5%` for sparse
pulses. Relative to identity, random improves weak positive AR by 15 percentage
points and high-frequency sine by 12.5 points, while reducing strong positive
AR by 15 points and positive MA and negative AR by 12.5 points each. The model
is redistributing which regimes are separated rather than uniformly degrading
or improving clustering.

For `Sine_MedFreq` and `Sine_LowAmp`, the mean fraction assigned to either
member of the indistinguishable pair is `52.5%` under identity and `53.8%`
under random. Exact observational equivalence therefore contributes to label
ambiguity, but almost half of those samples are mapped outside the pair after
Hungarian alignment; it is not the sole source of low purity.

This post-hoc evidence diagnoses small-sample and clustering-instability issues
but does not change any threshold, seed, or conclusion. The paired smoke result
remains FAIL.

### Executed MLP clustering development protocol

`configs/paper_mlp_clustering_development.yaml` freezes the first direct step
toward the paper's main `65.48%` MLP-purity result. This is a development
condition, not the final paper-scale test. It uses a fresh synthetic realization
with `base_seed = 1`, model seeds `10–14`, 64 train and 32 validation masters per
regime, and 20 epochs. The test allocation is present only to freeze future
split geometry and must not be instantiated during development.

The primary local architecture follows the appendix-table reading:
`32 -> 64 -> 64 -> 32`, with ReLU after both hidden layers. The direct
`6144 -> 32` encoder projection, AdamW settings, EMA decay, checkpoint
constraints, and per-sequence normalization remain unchanged. The contradictory
one-hidden-layer prose reading remains a later sensitivity rather than a
simultaneous tuning choice.

For each selected model checkpoint, validation clustering uses `K = 18` and
the arithmetic mean across K-means `random_state = 0..19`, each with
`n_init = 20`. This aggregation is fixed before training because the consumed
linear diagnostic showed that one K-means seed can move purity by several
percentage points. Ground-truth labels score clusters but do not train the
encoder or select the checkpoint.

The development clustering gate requires all of the following:

- overall mean validation purity across model seeds at least `0.60`;
- worst model-seed mean purity at least `0.55`;
- coefficient of variation of model-seed mean purities at most `0.10`;
- worst within-model-seed purity standard deviation across K-means states at
  most `0.03`.

The unchanged scale-invariant prediction, validation/train-gap, embedding
spread, and effective-rank gates must also pass. These are local readiness
criteria for deciding whether to scale, not an equivalence test against the
paper's `65.48%` held-out value.

The aggregation implementation is complete in
`src/koopman_jepa/paper_evaluation.py`. Synthetic tests cover perfectly
separated embeddings across multiple K-means states, aggregate mean failure,
within-seed instability, missing and duplicate model seeds, and non-finite
metrics. The executed notebook
`notebooks/paper_mlp_clustering_development.ipynb` implements the frozen
train/validation workflow without constructing test, and a source-level test
enforces that boundary.

The predictive prerequisite produced `FAIL` before clustering. Seeds 10, 13,
and 14 selected epochs 5, 4, and 5; seeds 11 and 12 had no eligible checkpoint.
At their closest epochs both rejected seeds passed loss improvement,
validation/train gap, and embedding-spread constraints, but effective ranks
`2.71` and `3.48` remained below the local minimum `4.0`. Their corresponding
validation/baseline loss ratios were `0.018` and `0.026`, and their gaps were
`1.052` and `1.044`. The failure is therefore localized to low-dimensional
anisotropy rather than constant collapse or overfitting.

K-means was deliberately omitted: reporting only the three accepted seeds
would make the clustering result conditional on a post-hoc subset. Test was
neither constructed nor consulted. The error curves reach their minima around
epochs 4–5 and then rise while embedding scale grows rapidly, so simply adding
epochs is not the next planned response. The next clean development sensitivity
uses the contradictory prose reading of the published predictor,
`32 -> 64 -> 32`, while retaining every other frozen choice and the rank gate.

That sensitivity is now frozen in
`configs/paper_mlp_one_hidden_development.yaml`. A configuration-level test
compares its parsed dataclass against the primary condition and requires exact
equality after replacing only `model.mlp_depth`. The executed notebook
`notebooks/paper_mlp_one_hidden_development.ipynb` is validation-only and
handles a failed predictive prerequisite without reporting partial clustering.

The executed sensitivity also produced `FAIL` before clustering, but improved
checkpoint coverage from three to four of five seeds. Seeds 10, 12, 13, and 14
selected epochs 6, 5, 6, and 6 with effective ranks `8.66`, `7.66`, `4.25`, and
`6.33`. Seed 11's closest epoch was 7: validation/baseline loss `0.008`, gap
`1.081`, dispersion ratio `6.693`, and effective rank `2.10`. The failure is
again localized to a predictive low-rank solution. The selected-seed loss-ratio
CV was also `0.318` against the local maximum `0.25`. K-means and test were not
run.

The executed diagnostic notebook
`notebooks/paper_mlp_one_hidden_clustering_diagnostic.ipynb` makes the post-hoc
selection change explicit by setting the selection-only rank floor to `1.0`.
All clustering settings remain frozen, and a source test prevents construction
of the test split.

The diagnostic result is a stable but insufficient clustering level. Overall
mean validation purity is `50.76%`, compared with the local `60%` development
floor and the paper's descriptive `65.48%`. Per-seed means range from `47.60%`
to `53.59%`, so all five miss the local per-seed floor of `55%`. Seed-mean CV is
`0.049` and the worst within-seed K-means standard deviation is `1.35%`; both
stability criteria pass. Seed 11, despite effective rank `2.10`, has the
second-highest purity at `52.62%`, while seed 12 has rank `7.66` and the lowest
purity at `47.60%`. The rank guard therefore did not conceal a successful
clustering reproduction. Test remains untouched.

The encoder sensitivity is frozen in
`configs/paper_mlp_two_stage_one_hidden_development.yaml`. A configuration test
requires parsed equality with the direct one-hidden condition after replacing
only `model.encoder_projection` with `two_stage`. The executed notebook
`notebooks/paper_mlp_two_stage_one_hidden_development.ipynb` implements the
validation-only run and skips clustering if the predictive prerequisite fails.

The executed formal run produced only one eligible checkpoint: seed 10 at
epoch 4, validation/baseline ratio `0.004`, validation/train gap `1.022`, and
effective rank `4.53`. The minimum-loss epochs for seeds 11–14 had ranks `1.22`,
`2.91`, `1.48`, and `1.89`. The two-stage projection therefore worsened
low-dimensional concentration relative to the direct one-hidden condition.
Clustering and test were not run.

The executed notebook
`notebooks/paper_mlp_two_stage_clustering_diagnostic.ipynb` implements the
descriptive five-seed clustering run. A source test verifies the explicit
selection-only rank floor and prevents construction of test.

The descriptive two-stage result is worse than the direct encoder. Overall
purity is `47.90%`, compared with `50.76%` for direct plus one-hidden and
`65.48%` in the paper. Per-seed means range from `45.23%` to `51.22%`; seed-mean
CV is `0.040` and worst K-means standard deviation is `1.38%`. All minimum-loss
checkpoints have effective rank below 4 (`1.22–3.36`). The reconciled appendix
encoder is therefore not a plausible explanation for the missing clustering
quality under the current optimization. Test remains untouched.

The optimization sensitivity is frozen in
`configs/paper_mlp_low_lr_development.yaml`. A configuration test requires
parsed equality with the direct one-hidden condition after replacing only
`train.learning_rate` with `1e-4`. The executed notebook
`notebooks/paper_mlp_low_lr_development.ipynb` reports the formal predictive
gate and the five-seed validation clustering result in one deterministic run.

Lower learning rate does not materially close the gap. Minimum-loss epochs move
from roughly 5–7 to 7–10, but only seed 10 exceeds effective rank 4; the other
ranks are `1.60–3.92`. Validation-loss-ratio CV is `0.287` against the local
`0.25` maximum. Overall purity is `51.13%`, only `0.37` points above the `3e-4`
direct one-hidden result; per-seed means range from `48.19%` to `53.95%`.
Seed-mean CV is `0.041` and worst K-means standard deviation is `1.36%`. The
formal result and clustering gate both remain FAIL, and test remains untouched.

The medium-scale pilot is frozen in
`configs/paper_mlp_medium_scale_development.yaml`. A configuration test requires
equality with the direct one-hidden baseline after replacing only
`train_per_regime` and `val_per_regime` with 256 and 64. Validation sequence IDs
necessarily shift because split offsets follow the enlarged train range, so
the result will be interpreted as a scaling trend rather than a paired
sample-for-sample comparison.

The executed notebook `notebooks/paper_mlp_medium_scale_development.ipynb`
implements the 4,608-sample train and 1,152-sample validation run. It reports
formal predictive readiness and five-seed clustering in one execution, with a
source test preventing test construction.

The pilot yields `49.92%` overall purity, versus `50.76%` in the smaller direct
one-hidden condition. Because split offsets changed, the `−0.84` point
difference is not paired, but there is no large positive scaling trend. Per-seed
means range from `45.40%` to `53.40%`; seed-mean CV is `0.058` and worst K-means
standard deviation is `1.00%`. Four seeds exceed effective rank 4; seed 11 is
`1.95`, and validation-loss-ratio CV fails at `0.343`.

More importantly, minimum-loss checkpoints move to epochs 1–2 with 72 batches
per epoch, corresponding to roughly 72–144 updates. The small condition's
minima at epochs 5–7 with 18 batches per epoch correspond to roughly 90–126
updates. Later medium-scale validation loss explodes by hundreds of times its
initial value. This step-aligned instability suggests a missing optimization
detail rather than a simple data shortage. Test remains untouched.

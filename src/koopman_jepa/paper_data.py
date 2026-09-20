from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import yaml
from scipy import signal as scipy_signal
from statsmodels.tsa.arima_process import ArmaProcess
from torch.utils.data import Dataset

PAPER_REGIME_NAMES = (
    "sine_low_freq",
    "sine_med_freq",
    "sine_high_freq",
    "sine_low_amp",
    "sine_harmonics",
    "trend_up",
    "trend_down",
    "ar_pos_strong",
    "ar_pos_weak",
    "ar_neg",
    "ma_pos",
    "arma_mixed",
    "square_low_freq",
    "square_high_freq",
    "sawtooth_med_freq",
    "pulses_sparse",
    "sine_trend",
    "sine_high_noise",
)

PAPER_PERIODIC_CYCLES = {
    "low": 7,
    "medium": 10,
    "high": 15,
}
PAPER_PHASE_STD = np.pi
PAPER_PULSE_COUNT = 5
PAPER_PULSE_AMPLITUDE = 2.0
PAPER_HIGH_NOISE_STD = 3.0
PAPER_ARMA_COEFFICIENTS = {
    7: ((0.9,), ()),
    8: ((0.3,), ()),
    9: ((-0.7,), ()),
    10: ((), (0.7,)),
    11: ((0.5,), (-0.4,)),
}

Split = Literal["train", "val", "test"]
NormalizationMode = Literal["per_sequence", "global_train"]
MasterGenerator = Callable[[int, int, np.random.Generator], np.ndarray]


@dataclass(frozen=True, slots=True)
class PaperDataConfig:
    master_length: int = 1024
    context_length: int = 768
    shift: int = 256
    train_per_regime: int = 7_000
    val_per_regime: int = 2_000
    test_per_regime: int = 1_000
    base_seed: int = 0
    normalization: NormalizationMode = "per_sequence"

    def validate(self) -> None:
        if self.master_length < 1:
            raise ValueError("master_length must be positive")
        if self.context_length < 1:
            raise ValueError("context_length must be positive")
        if self.shift < 1:
            raise ValueError("shift must be positive")
        if self.context_length + self.shift > self.master_length:
            raise ValueError("master_length must cover the shifted target window")
        if min(self.train_per_regime, self.val_per_regime, self.test_per_regime) < 1:
            raise ValueError("every split must contain at least one sample per regime")
        if self.base_seed < 0:
            raise ValueError("base_seed must be non-negative")
        if self.normalization not in {"per_sequence", "global_train"}:
            raise ValueError(f"unknown normalization mode: {self.normalization}")

    def samples_per_regime(self, split: Split) -> int:
        return {
            "train": self.train_per_regime,
            "val": self.val_per_regime,
            "test": self.test_per_regime,
        }[split]

    def split_offset(self, split: Split) -> int:
        return {
            "train": 0,
            "val": self.train_per_regime,
            "test": self.train_per_regime + self.val_per_regime,
        }[split]


@dataclass(frozen=True, slots=True)
class PaperSampleKey:
    regime_id: int
    sequence_id: int


@dataclass(frozen=True, slots=True)
class PaperNormalizationStats:
    mean: float
    std: float
    count: int

    def validate(self) -> None:
        if not np.isfinite(self.mean):
            raise ValueError("normalization mean must be finite")
        if not np.isfinite(self.std) or self.std <= 0.0:
            raise ValueError("normalization std must be positive and finite")
        if self.count < 1:
            raise ValueError("normalization count must be positive")


def load_paper_data_config(path: str | Path) -> PaperDataConfig:
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    config = PaperDataConfig(**raw.get("data", {}))
    config.validate()
    return config


def fit_global_normalization(
    config: PaperDataConfig,
    generate_master: MasterGenerator,
    sequences_per_regime: int | None = None,
) -> PaperNormalizationStats:
    """Fit scalar normalization statistics using only raw training masters."""

    config.validate()
    sequence_count = config.train_per_regime
    if sequences_per_regime is not None:
        if sequences_per_regime < 1 or sequences_per_regime > config.train_per_regime:
            raise ValueError("sequences_per_regime must be within the training split")
        sequence_count = sequences_per_regime

    count = 0
    mean = 0.0
    sum_squared_deviations = 0.0

    for regime_id in range(len(PAPER_REGIME_NAMES)):
        for sequence_id in range(sequence_count):
            seed_sequence = np.random.SeedSequence(
                [config.base_seed, regime_id, sequence_id]
            )
            rng = np.random.default_rng(seed_sequence)
            master = np.asarray(
                generate_master(regime_id, config.master_length, rng),
                dtype=np.float64,
            )
            if master.shape != (config.master_length,):
                raise ValueError(
                    f"generator returned shape {master.shape}; "
                    f"expected {(config.master_length,)}"
                )
            if not np.isfinite(master).all():
                raise ValueError("generator returned non-finite values")

            batch_count = master.size
            batch_mean = float(master.mean())
            batch_deviations = master - batch_mean
            batch_sum_squared_deviations = float(batch_deviations @ batch_deviations)
            combined_count = count + batch_count
            delta = batch_mean - mean
            sum_squared_deviations += (
                batch_sum_squared_deviations
                + delta**2 * count * batch_count / combined_count
            )
            mean += delta * batch_count / combined_count
            count = combined_count

    stats = PaperNormalizationStats(
        mean=mean,
        std=float(np.sqrt(sum_squared_deviations / count)),
        count=count,
    )
    stats.validate()
    return stats


def generate_paper_master(
    regime_id: int,
    length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate one raw master sequence from a published paper regime.

    This function is being implemented one regime family at a time. Regimes
    that have not yet been audited fail explicitly instead of silently using a
    placeholder signal.
    """

    if length < 1:
        raise ValueError("length must be positive")
    if regime_id < 0 or regime_id >= len(PAPER_REGIME_NAMES):
        raise ValueError(f"unknown regime_id: {regime_id}")
    if regime_id <= 4:
        signal = _generate_sinusoid(regime_id, length, rng)
    elif regime_id == 5:
        signal = _generate_linear_trend(1.5, length, rng)
    elif regime_id == 6:
        signal = _generate_linear_trend(-1.5, length, rng)
    elif regime_id <= 11:
        ar_coefficients, ma_coefficients = PAPER_ARMA_COEFFICIENTS[regime_id]
        signal = _generate_arma(ar_coefficients, ma_coefficients, length, rng)
    elif regime_id == 12:
        signal = _generate_square("low", length)
    elif regime_id == 13:
        signal = _generate_square("high", length)
    elif regime_id == 14:
        signal = _generate_sawtooth(length, rng)
    elif regime_id == 15:
        signal = _generate_sparse_pulses(length, rng)
    elif regime_id == 16:
        signal = _generate_sine_trend(length, rng)
    else:  # regime_id == 17
        signal = _generate_high_noise_sine(length, rng)

    return signal.astype(np.float32)


def _angular_frequency(band: str, length: int) -> float:
    cycles = PAPER_PERIODIC_CYCLES[band]
    return 2.0 * np.pi * cycles / length


def _generate_sinusoid(
    regime_id: int,
    length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    time = np.arange(length, dtype=np.float64)
    phase = rng.normal(loc=0.0, scale=PAPER_PHASE_STD)

    if regime_id == 0:
        return np.sin(_angular_frequency("low", length) * time + phase)
    if regime_id == 1:
        return np.sin(_angular_frequency("medium", length) * time + phase)
    if regime_id == 2:
        return np.sin(_angular_frequency("high", length) * time + phase)
    if regime_id == 3:
        return 0.3 * np.sin(_angular_frequency("medium", length) * time + phase)

    harmonic_phase = rng.normal(loc=0.0, scale=PAPER_PHASE_STD)
    medium_frequency = _angular_frequency("medium", length)
    return 0.7 * np.sin(medium_frequency * time + phase) + 0.3 * np.sin(
        3.0 * medium_frequency * time + harmonic_phase
    )


def _generate_linear_trend(
    base_slope: float,
    length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    normalized_time = np.linspace(0.0, 1.0, length, dtype=np.float64)
    slope = base_slope + rng.normal(loc=0.0, scale=1.0)
    intercept = rng.normal(loc=0.0, scale=PAPER_PHASE_STD)
    return slope * normalized_time + intercept


def _generate_sine_trend(length: int, rng: np.random.Generator) -> np.ndarray:
    sample_time = np.arange(length, dtype=np.float64)
    phase = rng.normal(loc=0.0, scale=PAPER_PHASE_STD)
    sinusoid = 0.8 * np.sin(_angular_frequency("medium", length) * sample_time + phase)
    return sinusoid + _generate_linear_trend(1.0, length, rng)


def _generate_square(band: str, length: int) -> np.ndarray:
    sample_time = np.arange(length, dtype=np.float64)
    phase = _angular_frequency(band, length) * sample_time
    return scipy_signal.square(phase)


def _generate_sawtooth(length: int, rng: np.random.Generator) -> np.ndarray:
    sample_time = np.arange(length, dtype=np.float64)
    initial_phase = rng.uniform(0.0, 2.0 * np.pi)
    phase = _angular_frequency("medium", length) * sample_time + initial_phase
    return scipy_signal.sawtooth(phase, width=1.0)


def _generate_sparse_pulses(length: int, rng: np.random.Generator) -> np.ndarray:
    width = max(1, round(length / 50))
    valid_start_count = length - width + 1
    pulse_count = min(PAPER_PULSE_COUNT, valid_start_count)
    starts = rng.choice(valid_start_count, size=pulse_count, replace=False)
    signal = np.zeros(length, dtype=np.float64)
    for start in starts:
        signal[start : start + width] = PAPER_PULSE_AMPLITUDE
    return signal


def _generate_arma(
    ar_coefficients: tuple[float, ...],
    ma_coefficients: tuple[float, ...],
    length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    ar_polynomial = np.array((1.0, *(-value for value in ar_coefficients)))
    ma_polynomial = np.array((1.0, *ma_coefficients))
    process = ArmaProcess(ar_polynomial, ma_polynomial)
    return process.generate_sample(
        nsample=length,
        scale=1.0,
        distrvs=rng.standard_normal,
        burnin=0,
    )


def _generate_high_noise_sine(length: int, rng: np.random.Generator) -> np.ndarray:
    sample_time = np.arange(length, dtype=np.float64)
    phase = rng.normal(loc=0.0, scale=PAPER_PHASE_STD)
    sinusoid = np.sin(_angular_frequency("medium", length) * sample_time + phase)
    internal_noise = rng.normal(loc=0.0, scale=PAPER_HIGH_NOISE_STD, size=length)
    return sinusoid + internal_noise


class PaperRegimeDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Generate paper-style samples lazily from stable per-sequence RNG streams."""

    def __init__(
        self,
        config: PaperDataConfig,
        split: Split,
        generate_master: MasterGenerator,
        normalization_stats: PaperNormalizationStats | None = None,
    ) -> None:
        config.validate()
        if config.normalization == "global_train":
            if normalization_stats is None:
                raise ValueError("global_train normalization requires fitted statistics")
            normalization_stats.validate()
        elif normalization_stats is not None:
            raise ValueError("normalization statistics require global_train mode")
        self.config = config
        self.split = split
        self.generate_master = generate_master
        self.normalization_stats = normalization_stats
        self._samples_per_regime = config.samples_per_regime(split)
        self._split_offset = config.split_offset(split)

    def __len__(self) -> int:
        return len(PAPER_REGIME_NAMES) * self._samples_per_regime

    def sample_key(self, index: int) -> PaperSampleKey:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        regime_id, within_regime = divmod(index, self._samples_per_regime)
        return PaperSampleKey(
            regime_id=regime_id,
            sequence_id=self._split_offset + within_regime,
        )

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = self.sample_key(index)
        seed_sequence = np.random.SeedSequence(
            [self.config.base_seed, key.regime_id, key.sequence_id]
        )
        rng = np.random.default_rng(seed_sequence)
        master = np.asarray(
            self.generate_master(key.regime_id, self.config.master_length, rng),
            dtype=np.float32,
        )
        if master.shape != (self.config.master_length,):
            raise ValueError(
                f"generator returned shape {master.shape}; "
                f"expected {(self.config.master_length,)}"
            )
        if not np.isfinite(master).all():
            raise ValueError("generator returned non-finite values")

        if self.config.normalization == "per_sequence":
            working_master = master.astype(np.float64)
            mean = float(working_master.mean())
            std = max(float(working_master.std()), 1e-6)
            master = ((working_master - mean) / std).astype(np.float32)
        else:
            assert self.normalization_stats is not None
            master = (
                (master.astype(np.float64) - self.normalization_stats.mean)
                / self.normalization_stats.std
            ).astype(np.float32)

        context = master[: self.config.context_length]
        target = master[
            self.config.shift : self.config.shift + self.config.context_length
        ]
        return (
            torch.from_numpy(context.copy()).unsqueeze(0),
            torch.from_numpy(target.copy()).unsqueeze(0),
            torch.tensor(key.regime_id, dtype=torch.long),
        )

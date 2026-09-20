from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import yaml
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

Split = Literal["train", "val", "test"]
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
    standardize: bool = True

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


def load_paper_data_config(path: str | Path) -> PaperDataConfig:
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    config = PaperDataConfig(**raw.get("data", {}))
    config.validate()
    return config


class PaperRegimeDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Generate paper-style samples lazily from stable per-sequence RNG streams."""

    def __init__(
        self,
        config: PaperDataConfig,
        split: Split,
        generate_master: MasterGenerator,
    ) -> None:
        config.validate()
        self.config = config
        self.split = split
        self.generate_master = generate_master
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

        if self.config.standardize:
            mean = float(master.mean())
            std = max(float(master.std()), 1e-6)
            master = ((master - mean) / std).astype(np.float32, copy=False)

        context = master[: self.config.context_length]
        target = master[
            self.config.shift : self.config.shift + self.config.context_length
        ]
        return (
            torch.from_numpy(context.copy()).unsqueeze(0),
            torch.from_numpy(target.copy()).unsqueeze(0),
            torch.tensor(key.regime_id, dtype=torch.long),
        )

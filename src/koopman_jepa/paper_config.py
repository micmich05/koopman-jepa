from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

from .paper_data import PaperDataConfig
from .paper_model import PaperModelConfig

PaperOptimizer = Literal["adamw"]


@dataclass(frozen=True, slots=True)
class PaperOptimizationConfig:
    optimizer: PaperOptimizer = "adamw"
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 18
    steps: int = 100
    seed: int = 0
    device: str = "cpu"

    def validate(self) -> None:
        if self.optimizer != "adamw":
            raise ValueError(f"unknown paper optimizer: {self.optimizer}")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.steps < 1:
            raise ValueError("steps must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if not self.device:
            raise ValueError("device must not be empty")


@dataclass(frozen=True, slots=True)
class PaperOverfitGateConfig:
    comparison_window: int = 10
    max_loss_ratio: float = 0.25
    min_embedding_std_ratio: float = 0.10
    min_effective_rank: float = 2.0

    def validate(self) -> None:
        if self.comparison_window < 1:
            raise ValueError("comparison_window must be positive")
        if not 0.0 < self.max_loss_ratio < 1.0:
            raise ValueError("max_loss_ratio must be between zero and one")
        if not 0.0 < self.min_embedding_std_ratio <= 1.0:
            raise ValueError("min_embedding_std_ratio must be in (0, 1]")
        if self.min_effective_rank < 1.0:
            raise ValueError("min_effective_rank must be at least one")


@dataclass(frozen=True, slots=True)
class PaperExperimentConfig:
    data: PaperDataConfig
    model: PaperModelConfig
    optimization: PaperOptimizationConfig
    gate: PaperOverfitGateConfig

    def validate(self) -> None:
        self.data.validate()
        self.model.validate()
        self.optimization.validate()
        self.gate.validate()
        if self.gate.comparison_window > self.optimization.steps:
            raise ValueError("comparison_window must not exceed optimization steps")


def load_paper_experiment_config(path: str | Path) -> PaperExperimentConfig:
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    config = PaperExperimentConfig(
        data=PaperDataConfig(**raw.get("data", {})),
        model=PaperModelConfig(**raw.get("model", {})),
        optimization=PaperOptimizationConfig(**raw.get("optimization", {})),
        gate=PaperOverfitGateConfig(**raw.get("gate", {})),
    )
    config.validate()
    return config

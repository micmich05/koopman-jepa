from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class DataConfig:
    context_length: int = 128
    shift: int = 32
    train_per_regime: int = 600
    val_per_regime: int = 120
    test_per_regime: int = 120
    noise_std: float = 0.03
    standardize: bool = True


@dataclass(slots=True)
class ModelConfig:
    latent_dim: int = 8
    channels: list[int] = field(default_factory=lambda: [16, 32, 64])
    predictor_init: str = "identity"


@dataclass(slots=True)
class TrainConfig:
    seed: int = 0
    epochs: int = 30
    batch_size: int = 128
    learning_rate: float = 1e-3
    predictor_learning_rate_multiplier: float = 1.0
    weight_decay: float = 1e-4
    ema_momentum: float = 0.99
    mean_weight: float = 0.0
    variance_weight: float = 0.0
    covariance_weight: float = 0.0
    device: str = "auto"
    num_workers: int = 0


@dataclass(slots=True)
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    output_dir: str = "runs/phase0"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    return ExperimentConfig(
        data=DataConfig(**raw.get("data", {})),
        model=ModelConfig(**raw.get("model", {})),
        train=TrainConfig(**raw.get("train", {})),
        output_dir=raw.get("output_dir", "runs/phase0"),
    )


def validate_config(config: ExperimentConfig) -> None:
    if config.data.context_length < 16:
        raise ValueError("context_length must be at least 16")
    if config.data.shift < 1:
        raise ValueError("shift must be positive")
    if config.model.latent_dim < 1:
        raise ValueError("latent_dim must be positive")
    if config.model.predictor_init not in {"identity", "random"}:
        raise ValueError("predictor_init must be 'identity' or 'random'")
    if not 0.0 <= config.train.ema_momentum < 1.0:
        raise ValueError("ema_momentum must be in [0, 1)")
    if config.train.epochs < 1:
        raise ValueError("epochs must be positive")
    if config.train.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if config.train.predictor_learning_rate_multiplier <= 0.0:
        raise ValueError("predictor_learning_rate_multiplier must be positive")
    if min(
        config.train.mean_weight,
        config.train.variance_weight,
        config.train.covariance_weight,
    ) < 0.0:
        raise ValueError("regularization weights must be non-negative")

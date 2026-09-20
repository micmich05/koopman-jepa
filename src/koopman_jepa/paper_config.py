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
class PaperTrainConfig:
    optimizer: PaperOptimizer = "adamw"
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    batch_size: int = 64
    epochs: int = 10
    seed: int = 0
    device: str = "cpu"
    num_workers: int = 0

    def validate(self) -> None:
        if self.optimizer != "adamw":
            raise ValueError(f"unknown paper optimizer: {self.optimizer}")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if not self.device:
            raise ValueError("device must not be empty")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")


@dataclass(frozen=True, slots=True)
class PaperValidationGateConfig:
    final_window: int = 2
    max_validation_loss_ratio: float = 0.50
    max_validation_train_loss_ratio: float = 4.0
    min_validation_embedding_std_ratio: float = 0.10
    min_validation_effective_rank: float = 4.0

    def validate(self) -> None:
        if self.final_window < 1:
            raise ValueError("final_window must be positive")
        if not 0.0 < self.max_validation_loss_ratio < 1.0:
            raise ValueError("max_validation_loss_ratio must be between zero and one")
        if self.max_validation_train_loss_ratio < 1.0:
            raise ValueError("max_validation_train_loss_ratio must be at least one")
        if not 0.0 < self.min_validation_embedding_std_ratio <= 1.0:
            raise ValueError("min_validation_embedding_std_ratio must be in (0, 1]")
        if self.min_validation_effective_rank < 1.0:
            raise ValueError("min_validation_effective_rank must be at least one")


@dataclass(frozen=True, slots=True)
class PaperSeedSweepConfig:
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)

    def validate(self) -> None:
        if not self.seeds:
            raise ValueError("seed sweep must contain at least one seed")
        if any(seed < 0 for seed in self.seeds):
            raise ValueError("seed sweep values must be non-negative")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seed sweep values must be unique")


@dataclass(frozen=True, slots=True)
class PaperSeedStabilityGateConfig:
    max_worst_validation_loss_ratio: float = 0.50
    max_worst_validation_train_loss_ratio: float = 4.0
    max_validation_loss_coefficient_of_variation: float = 0.25
    min_worst_validation_embedding_std_ratio: float = 0.10
    min_worst_validation_effective_rank: float = 4.0

    def validate(self) -> None:
        if not 0.0 < self.max_worst_validation_loss_ratio < 1.0:
            raise ValueError(
                "max_worst_validation_loss_ratio must be between zero and one"
            )
        if self.max_worst_validation_train_loss_ratio < 1.0:
            raise ValueError(
                "max_worst_validation_train_loss_ratio must be at least one"
            )
        if self.max_validation_loss_coefficient_of_variation < 0.0:
            raise ValueError(
                "max_validation_loss_coefficient_of_variation must be non-negative"
            )
        if not 0.0 < self.min_worst_validation_embedding_std_ratio <= 1.0:
            raise ValueError(
                "min_worst_validation_embedding_std_ratio must be in (0, 1]"
            )
        if self.min_worst_validation_effective_rank < 1.0:
            raise ValueError("min_worst_validation_effective_rank must be at least one")


@dataclass(frozen=True, slots=True)
class PaperScaleInvariantStabilityGateConfig:
    max_worst_validation_loss_ratio: float = 0.50
    max_worst_validation_train_loss_ratio: float = 4.0
    max_validation_loss_ratio_coefficient_of_variation: float = 0.25
    min_worst_validation_embedding_std_ratio: float = 0.10
    min_worst_validation_effective_rank: float = 4.0

    def validate(self) -> None:
        if not 0.0 < self.max_worst_validation_loss_ratio < 1.0:
            raise ValueError(
                "max_worst_validation_loss_ratio must be between zero and one"
            )
        if self.max_worst_validation_train_loss_ratio < 1.0:
            raise ValueError(
                "max_worst_validation_train_loss_ratio must be at least one"
            )
        if self.max_validation_loss_ratio_coefficient_of_variation < 0.0:
            raise ValueError(
                "max_validation_loss_ratio_coefficient_of_variation "
                "must be non-negative"
            )
        if not 0.0 < self.min_worst_validation_embedding_std_ratio <= 1.0:
            raise ValueError(
                "min_worst_validation_embedding_std_ratio must be in (0, 1]"
            )
        if self.min_worst_validation_effective_rank < 1.0:
            raise ValueError("min_worst_validation_effective_rank must be at least one")


@dataclass(frozen=True, slots=True)
class PaperCheckpointReplayConfig:
    expected_epochs: tuple[int, ...] = (10, 10, 10, 8, 10)
    metric_absolute_tolerance: float = 1e-8

    def validate(self) -> None:
        if not self.expected_epochs:
            raise ValueError("expected checkpoint epochs must not be empty")
        if any(epoch < 1 for epoch in self.expected_epochs):
            raise ValueError("expected checkpoint epochs must be positive")
        if self.metric_absolute_tolerance < 0.0:
            raise ValueError("metric_absolute_tolerance must be non-negative")


@dataclass(frozen=True, slots=True)
class PaperLinearIdentityEvaluationConfig:
    kmeans_clusters: int = 18
    kmeans_n_init: int = 20
    kmeans_seed: int = 0
    eigenvalue_tolerance: float = 0.05

    def validate(self) -> None:
        if self.kmeans_clusters < 2:
            raise ValueError("kmeans_clusters must be at least two")
        if self.kmeans_n_init < 1:
            raise ValueError("kmeans_n_init must be positive")
        if self.kmeans_seed < 0:
            raise ValueError("kmeans_seed must be non-negative")
        if not 0.0 < self.eigenvalue_tolerance < 1.0:
            raise ValueError("eigenvalue_tolerance must be between zero and one")


@dataclass(frozen=True, slots=True)
class PaperLinearIdentityHeldoutGateConfig:
    max_relative_identity_error: float = 0.05
    max_relative_skew_norm: float = 0.05
    max_mean_centroid_action_error: float = 0.02
    min_near_identity_eigenvalues: int = 18
    min_test_effective_rank: float = 4.0

    def validate(self) -> None:
        for name, value in (
            ("max_relative_identity_error", self.max_relative_identity_error),
            ("max_relative_skew_norm", self.max_relative_skew_norm),
            (
                "max_mean_centroid_action_error",
                self.max_mean_centroid_action_error,
            ),
        ):
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be between zero and one")
        if self.min_near_identity_eigenvalues < 1:
            raise ValueError("min_near_identity_eigenvalues must be positive")
        if self.min_test_effective_rank < 1.0:
            raise ValueError("min_test_effective_rank must be at least one")


@dataclass(frozen=True, slots=True)
class PaperLinearRandomComparisonGateConfig:
    max_random_to_identity_validation_improvement_ratio: float = 2.0
    min_random_relative_identity_error: float = 0.50
    min_random_off_diagonal_fraction: float = 0.50

    def validate(self) -> None:
        if self.max_random_to_identity_validation_improvement_ratio < 1.0:
            raise ValueError(
                "max_random_to_identity_validation_improvement_ratio "
                "must be at least one"
            )
        if self.min_random_relative_identity_error <= 0.0:
            raise ValueError("min_random_relative_identity_error must be positive")
        if not 0.0 < self.min_random_off_diagonal_fraction <= 1.0:
            raise ValueError("min_random_off_diagonal_fraction must be in (0, 1]")


@dataclass(frozen=True, slots=True)
class PaperLinearPairedHeldoutEvaluationConfig:
    kmeans_clusters: int = 18
    kmeans_n_init: int = 20
    kmeans_seed: int = 0

    def validate(self) -> None:
        if self.kmeans_clusters < 2:
            raise ValueError("kmeans_clusters must be at least two")
        if self.kmeans_n_init < 1:
            raise ValueError("kmeans_n_init must be positive")
        if self.kmeans_seed < 0:
            raise ValueError("kmeans_seed must be non-negative")


@dataclass(frozen=True, slots=True)
class PaperLinearRandomHeldoutGateConfig:
    max_random_to_identity_prediction_error_ratio: float = 2.0
    min_random_kmeans_purity: float = 0.50
    min_random_to_identity_kmeans_purity_ratio: float = 0.90
    min_random_test_effective_rank: float = 4.0
    min_random_to_identity_effective_rank_ratio: float = 0.50

    def validate(self) -> None:
        if self.max_random_to_identity_prediction_error_ratio < 1.0:
            raise ValueError(
                "max_random_to_identity_prediction_error_ratio must be at least one"
            )
        if not 0.0 < self.min_random_kmeans_purity <= 1.0:
            raise ValueError("min_random_kmeans_purity must be in (0, 1]")
        if not 0.0 < self.min_random_to_identity_kmeans_purity_ratio <= 1.0:
            raise ValueError(
                "min_random_to_identity_kmeans_purity_ratio must be in (0, 1]"
            )
        if self.min_random_test_effective_rank < 1.0:
            raise ValueError("min_random_test_effective_rank must be at least one")
        if not 0.0 < self.min_random_to_identity_effective_rank_ratio <= 1.0:
            raise ValueError(
                "min_random_to_identity_effective_rank_ratio must be in (0, 1]"
            )


@dataclass(frozen=True, slots=True)
class PaperKMeansSweepConfig:
    clusters: int = 18
    n_init: int = 20
    random_states: tuple[int, ...] = tuple(range(20))

    def validate(self) -> None:
        if self.clusters < 2:
            raise ValueError("clusters must be at least two")
        if self.n_init < 1:
            raise ValueError("n_init must be positive")
        if not self.random_states:
            raise ValueError("random_states must not be empty")
        if any(seed < 0 for seed in self.random_states):
            raise ValueError("random_states must be non-negative")
        if len(set(self.random_states)) != len(self.random_states):
            raise ValueError("random_states must be unique")


@dataclass(frozen=True, slots=True)
class PaperMLPClusteringGateConfig:
    min_overall_mean_purity: float = 0.60
    min_worst_seed_mean_purity: float = 0.55
    max_seed_mean_purity_coefficient_of_variation: float = 0.10
    max_within_seed_purity_std: float = 0.03

    def validate(self) -> None:
        for name, value in (
            ("min_overall_mean_purity", self.min_overall_mean_purity),
            ("min_worst_seed_mean_purity", self.min_worst_seed_mean_purity),
        ):
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")
        if self.min_worst_seed_mean_purity > self.min_overall_mean_purity:
            raise ValueError(
                "min_worst_seed_mean_purity must not exceed "
                "min_overall_mean_purity"
            )
        if self.max_seed_mean_purity_coefficient_of_variation < 0.0:
            raise ValueError(
                "max_seed_mean_purity_coefficient_of_variation "
                "must be non-negative"
            )
        if not 0.0 <= self.max_within_seed_purity_std <= 1.0:
            raise ValueError("max_within_seed_purity_std must be in [0, 1]")


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


@dataclass(frozen=True, slots=True)
class PaperTrainValidationConfig:
    data: PaperDataConfig
    model: PaperModelConfig
    train: PaperTrainConfig
    gate: PaperValidationGateConfig

    def validate(self) -> None:
        self.data.validate()
        self.model.validate()
        self.train.validate()
        self.gate.validate()
        if self.gate.final_window > self.train.epochs:
            raise ValueError("final_window must not exceed training epochs")


@dataclass(frozen=True, slots=True)
class PaperSeedStabilityConfig:
    data: PaperDataConfig
    model: PaperModelConfig
    train: PaperTrainConfig
    checkpoint_gate: PaperValidationGateConfig
    sweep: PaperSeedSweepConfig
    stability_gate: PaperSeedStabilityGateConfig

    def validate(self) -> None:
        self.data.validate()
        self.model.validate()
        self.train.validate()
        self.checkpoint_gate.validate()
        self.sweep.validate()
        self.stability_gate.validate()
        if self.checkpoint_gate.final_window > self.train.epochs:
            raise ValueError("final_window must not exceed training epochs")


@dataclass(frozen=True, slots=True)
class PaperScaleInvariantSeedStabilityConfig:
    data: PaperDataConfig
    model: PaperModelConfig
    train: PaperTrainConfig
    checkpoint_gate: PaperValidationGateConfig
    sweep: PaperSeedSweepConfig
    stability_gate: PaperScaleInvariantStabilityGateConfig

    def validate(self) -> None:
        self.data.validate()
        self.model.validate()
        self.train.validate()
        self.checkpoint_gate.validate()
        self.sweep.validate()
        self.stability_gate.validate()
        if self.checkpoint_gate.final_window > self.train.epochs:
            raise ValueError("final_window must not exceed training epochs")


@dataclass(frozen=True, slots=True)
class PaperLinearIdentityHeldoutConfig:
    data: PaperDataConfig
    model: PaperModelConfig
    train: PaperTrainConfig
    checkpoint_gate: PaperValidationGateConfig
    sweep: PaperSeedSweepConfig
    replay: PaperCheckpointReplayConfig
    evaluation: PaperLinearIdentityEvaluationConfig
    gate: PaperLinearIdentityHeldoutGateConfig

    def validate(self) -> None:
        self.data.validate()
        self.model.validate()
        self.train.validate()
        self.checkpoint_gate.validate()
        self.sweep.validate()
        self.replay.validate()
        self.evaluation.validate()
        self.gate.validate()
        if self.model.predictor_kind != "linear":
            raise ValueError("held-out identity evaluation requires a linear predictor")
        if self.model.linear_initialization != "identity":
            raise ValueError("held-out identity evaluation requires identity initialization")
        if len(self.replay.expected_epochs) != len(self.sweep.seeds):
            raise ValueError("expected checkpoint epochs must align with sweep seeds")
        if any(epoch > self.train.epochs for epoch in self.replay.expected_epochs):
            raise ValueError("expected checkpoint epochs must not exceed training epochs")
        if self.gate.min_near_identity_eigenvalues > self.model.latent_dim:
            raise ValueError("min_near_identity_eigenvalues must not exceed latent_dim")
        if self.gate.min_test_effective_rank > self.model.latent_dim:
            raise ValueError("min_test_effective_rank must not exceed latent_dim")


@dataclass(frozen=True, slots=True)
class PaperLinearRandomControlConfig:
    data: PaperDataConfig
    model: PaperModelConfig
    train: PaperTrainConfig
    checkpoint_gate: PaperValidationGateConfig
    sweep: PaperSeedSweepConfig
    identity_replay: PaperCheckpointReplayConfig
    stability_gate: PaperScaleInvariantStabilityGateConfig
    comparison_gate: PaperLinearRandomComparisonGateConfig

    def validate(self) -> None:
        self.data.validate()
        self.model.validate()
        self.train.validate()
        self.checkpoint_gate.validate()
        self.sweep.validate()
        self.identity_replay.validate()
        self.stability_gate.validate()
        self.comparison_gate.validate()
        if self.model.predictor_kind != "linear":
            raise ValueError("random control requires a linear predictor")
        if self.model.linear_initialization != "xavier_uniform":
            raise ValueError("random control requires Xavier-uniform initialization")
        if len(self.identity_replay.expected_epochs) != len(self.sweep.seeds):
            raise ValueError("identity checkpoint epochs must align with sweep seeds")
        if any(
            epoch > self.train.epochs
            for epoch in self.identity_replay.expected_epochs
        ):
            raise ValueError("identity checkpoint epochs must not exceed training epochs")


@dataclass(frozen=True, slots=True)
class PaperLinearRandomHeldoutConfig:
    data: PaperDataConfig
    model: PaperModelConfig
    train: PaperTrainConfig
    checkpoint_gate: PaperValidationGateConfig
    sweep: PaperSeedSweepConfig
    identity_replay: PaperCheckpointReplayConfig
    random_replay: PaperCheckpointReplayConfig
    evaluation: PaperLinearPairedHeldoutEvaluationConfig
    gate: PaperLinearRandomHeldoutGateConfig

    def validate(self) -> None:
        self.data.validate()
        self.model.validate()
        self.train.validate()
        self.checkpoint_gate.validate()
        self.sweep.validate()
        self.identity_replay.validate()
        self.random_replay.validate()
        self.evaluation.validate()
        self.gate.validate()
        if self.model.predictor_kind != "linear":
            raise ValueError("paired held-out control requires a linear predictor")
        if self.model.linear_initialization != "xavier_uniform":
            raise ValueError(
                "paired held-out control requires Xavier-uniform initialization"
            )
        if len(self.identity_replay.expected_epochs) != len(self.sweep.seeds):
            raise ValueError("identity checkpoint epochs must align with sweep seeds")
        if len(self.random_replay.expected_epochs) != len(self.sweep.seeds):
            raise ValueError("random checkpoint epochs must align with sweep seeds")
        replay_epochs = (
            *self.identity_replay.expected_epochs,
            *self.random_replay.expected_epochs,
        )
        if any(epoch > self.train.epochs for epoch in replay_epochs):
            raise ValueError("checkpoint epochs must not exceed training epochs")
        if self.gate.min_random_test_effective_rank > self.model.latent_dim:
            raise ValueError("min_random_test_effective_rank must not exceed latent_dim")


@dataclass(frozen=True, slots=True)
class PaperMLPClusteringDevelopmentConfig:
    data: PaperDataConfig
    model: PaperModelConfig
    train: PaperTrainConfig
    checkpoint_gate: PaperValidationGateConfig
    sweep: PaperSeedSweepConfig
    stability_gate: PaperScaleInvariantStabilityGateConfig
    clustering: PaperKMeansSweepConfig
    clustering_gate: PaperMLPClusteringGateConfig

    def validate(self) -> None:
        self.data.validate()
        self.model.validate()
        self.train.validate()
        self.checkpoint_gate.validate()
        self.sweep.validate()
        self.stability_gate.validate()
        self.clustering.validate()
        self.clustering_gate.validate()
        if self.model.predictor_kind != "mlp":
            raise ValueError("MLP clustering development requires an MLP predictor")
        if self.model.mlp_depth != "two_hidden":
            raise ValueError(
                "primary MLP clustering development requires two hidden layers"
            )
        if self.checkpoint_gate.final_window > self.train.epochs:
            raise ValueError("final_window must not exceed training epochs")
        validation_samples = (
            self.data.val_per_regime * self.clustering.clusters
        )
        if validation_samples < self.clustering.clusters:
            raise ValueError("validation split must cover all configured clusters")


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


def load_paper_train_validation_config(path: str | Path) -> PaperTrainValidationConfig:
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    config = PaperTrainValidationConfig(
        data=PaperDataConfig(**raw.get("data", {})),
        model=PaperModelConfig(**raw.get("model", {})),
        train=PaperTrainConfig(**raw.get("train", {})),
        gate=PaperValidationGateConfig(**raw.get("gate", {})),
    )
    config.validate()
    return config


def load_paper_seed_stability_config(path: str | Path) -> PaperSeedStabilityConfig:
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    sweep_values = raw.get("sweep", {}).get("seeds", (0, 1, 2, 3, 4))
    config = PaperSeedStabilityConfig(
        data=PaperDataConfig(**raw.get("data", {})),
        model=PaperModelConfig(**raw.get("model", {})),
        train=PaperTrainConfig(**raw.get("train", {})),
        checkpoint_gate=PaperValidationGateConfig(**raw.get("checkpoint_gate", {})),
        sweep=PaperSeedSweepConfig(seeds=tuple(sweep_values)),
        stability_gate=PaperSeedStabilityGateConfig(**raw.get("stability_gate", {})),
    )
    config.validate()
    return config


def load_paper_scale_invariant_seed_stability_config(
    path: str | Path,
) -> PaperScaleInvariantSeedStabilityConfig:
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    sweep_values = raw.get("sweep", {}).get("seeds", (5, 6, 7, 8, 9))
    config = PaperScaleInvariantSeedStabilityConfig(
        data=PaperDataConfig(**raw.get("data", {})),
        model=PaperModelConfig(**raw.get("model", {})),
        train=PaperTrainConfig(**raw.get("train", {})),
        checkpoint_gate=PaperValidationGateConfig(**raw.get("checkpoint_gate", {})),
        sweep=PaperSeedSweepConfig(seeds=tuple(sweep_values)),
        stability_gate=PaperScaleInvariantStabilityGateConfig(
            **raw.get("stability_gate", {})
        ),
    )
    config.validate()
    return config


def load_paper_linear_identity_heldout_config(
    path: str | Path,
) -> PaperLinearIdentityHeldoutConfig:
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    sweep_values = raw.get("sweep", {}).get("seeds", (5, 6, 7, 8, 9))
    expected_epochs = raw.get("replay", {}).get(
        "expected_epochs",
        (10, 10, 10, 8, 10),
    )
    replay_raw = {
        **raw.get("replay", {}),
        "expected_epochs": tuple(expected_epochs),
    }
    config = PaperLinearIdentityHeldoutConfig(
        data=PaperDataConfig(**raw.get("data", {})),
        model=PaperModelConfig(**raw.get("model", {})),
        train=PaperTrainConfig(**raw.get("train", {})),
        checkpoint_gate=PaperValidationGateConfig(**raw.get("checkpoint_gate", {})),
        sweep=PaperSeedSweepConfig(seeds=tuple(sweep_values)),
        replay=PaperCheckpointReplayConfig(**replay_raw),
        evaluation=PaperLinearIdentityEvaluationConfig(
            **raw.get("evaluation", {})
        ),
        gate=PaperLinearIdentityHeldoutGateConfig(**raw.get("gate", {})),
    )
    config.validate()
    return config


def load_paper_linear_random_control_config(
    path: str | Path,
) -> PaperLinearRandomControlConfig:
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    sweep_values = raw.get("sweep", {}).get("seeds", (5, 6, 7, 8, 9))
    identity_epochs = raw.get("identity_replay", {}).get(
        "expected_epochs",
        (10, 10, 10, 8, 10),
    )
    identity_replay_raw = {
        **raw.get("identity_replay", {}),
        "expected_epochs": tuple(identity_epochs),
    }
    config = PaperLinearRandomControlConfig(
        data=PaperDataConfig(**raw.get("data", {})),
        model=PaperModelConfig(**raw.get("model", {})),
        train=PaperTrainConfig(**raw.get("train", {})),
        checkpoint_gate=PaperValidationGateConfig(**raw.get("checkpoint_gate", {})),
        sweep=PaperSeedSweepConfig(seeds=tuple(sweep_values)),
        identity_replay=PaperCheckpointReplayConfig(**identity_replay_raw),
        stability_gate=PaperScaleInvariantStabilityGateConfig(
            **raw.get("stability_gate", {})
        ),
        comparison_gate=PaperLinearRandomComparisonGateConfig(
            **raw.get("comparison_gate", {})
        ),
    )
    config.validate()
    return config


def load_paper_linear_random_heldout_config(
    path: str | Path,
) -> PaperLinearRandomHeldoutConfig:
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    sweep_values = raw.get("sweep", {}).get("seeds", (5, 6, 7, 8, 9))
    identity_epochs = raw.get("identity_replay", {}).get(
        "expected_epochs",
        (10, 10, 10, 8, 10),
    )
    random_epochs = raw.get("random_replay", {}).get(
        "expected_epochs",
        (10, 10, 10, 10, 10),
    )
    identity_replay_raw = {
        **raw.get("identity_replay", {}),
        "expected_epochs": tuple(identity_epochs),
    }
    random_replay_raw = {
        **raw.get("random_replay", {}),
        "expected_epochs": tuple(random_epochs),
    }
    config = PaperLinearRandomHeldoutConfig(
        data=PaperDataConfig(**raw.get("data", {})),
        model=PaperModelConfig(**raw.get("model", {})),
        train=PaperTrainConfig(**raw.get("train", {})),
        checkpoint_gate=PaperValidationGateConfig(**raw.get("checkpoint_gate", {})),
        sweep=PaperSeedSweepConfig(seeds=tuple(sweep_values)),
        identity_replay=PaperCheckpointReplayConfig(**identity_replay_raw),
        random_replay=PaperCheckpointReplayConfig(**random_replay_raw),
        evaluation=PaperLinearPairedHeldoutEvaluationConfig(
            **raw.get("evaluation", {})
        ),
        gate=PaperLinearRandomHeldoutGateConfig(**raw.get("gate", {})),
    )
    config.validate()
    return config


def load_paper_mlp_clustering_development_config(
    path: str | Path,
) -> PaperMLPClusteringDevelopmentConfig:
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    sweep_values = raw.get("sweep", {}).get("seeds", (10, 11, 12, 13, 14))
    clustering_raw = raw.get("clustering", {})
    random_states = clustering_raw.get("random_states", tuple(range(20)))
    clustering_values = {
        **clustering_raw,
        "random_states": tuple(random_states),
    }
    config = PaperMLPClusteringDevelopmentConfig(
        data=PaperDataConfig(**raw.get("data", {})),
        model=PaperModelConfig(**raw.get("model", {})),
        train=PaperTrainConfig(**raw.get("train", {})),
        checkpoint_gate=PaperValidationGateConfig(**raw.get("checkpoint_gate", {})),
        sweep=PaperSeedSweepConfig(seeds=tuple(sweep_values)),
        stability_gate=PaperScaleInvariantStabilityGateConfig(
            **raw.get("stability_gate", {})
        ),
        clustering=PaperKMeansSweepConfig(**clustering_values),
        clustering_gate=PaperMLPClusteringGateConfig(
            **raw.get("clustering_gate", {})
        ),
    )
    config.validate()
    return config

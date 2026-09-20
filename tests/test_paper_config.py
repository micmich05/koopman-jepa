from pathlib import Path

import numpy as np

from koopman_jepa.paper_config import (
    PaperOptimizationConfig,
    load_paper_experiment_config,
    load_paper_scale_invariant_seed_stability_config,
    load_paper_seed_stability_config,
    load_paper_train_validation_config,
)


def test_overfit_smoke_config_is_small_balanced_and_explicit() -> None:
    path = Path(__file__).parents[1] / "configs" / "paper_overfit_smoke.yaml"

    config = load_paper_experiment_config(path)

    assert config.data.train_per_regime == 1
    assert config.data.normalization == "per_sequence"
    assert config.model.encoder_projection == "direct"
    assert config.model.predictor_kind == "linear"
    assert config.model.linear_initialization == "identity"
    assert config.optimization.optimizer == "adamw"
    assert config.optimization.batch_size == 18
    assert config.optimization.steps == 100
    assert config.optimization.device == "cpu"
    assert config.gate.comparison_window == 10
    assert config.gate.max_loss_ratio == 0.25
    assert config.gate.min_embedding_std_ratio == 0.10
    assert config.gate.min_effective_rank == 2.0


def test_optimization_config_rejects_invalid_values() -> None:
    invalid = PaperOptimizationConfig(steps=0)

    with np.testing.assert_raises_regex(ValueError, "steps must be positive"):
        invalid.validate()


def test_train_validation_smoke_config_is_separated_and_explicit() -> None:
    path = Path(__file__).parents[1] / "configs" / "paper_train_validation_smoke.yaml"

    config = load_paper_train_validation_config(path)

    assert config.data.train_per_regime == 32
    assert config.data.val_per_regime == 8
    assert config.data.normalization == "per_sequence"
    assert config.model.encoder_projection == "direct"
    assert config.model.linear_initialization == "identity"
    assert config.train.learning_rate == 3e-4
    assert config.train.batch_size == 64
    assert config.train.epochs == 10
    assert config.gate.final_window == 2
    assert config.gate.max_validation_loss_ratio == 0.50
    assert config.gate.max_validation_train_loss_ratio == 4.0
    assert config.gate.min_validation_embedding_std_ratio == 0.10
    assert config.gate.min_validation_effective_rank == 4.0


def test_seed_stability_config_reuses_data_and_freezes_five_seeds() -> None:
    path = Path(__file__).parents[1] / "configs" / "paper_seed_stability_smoke.yaml"

    config = load_paper_seed_stability_config(path)

    assert config.data.base_seed == 0
    assert config.data.train_per_regime == 32
    assert config.data.val_per_regime == 8
    assert config.train.epochs == 10
    assert config.sweep.seeds == (0, 1, 2, 3, 4)
    assert config.checkpoint_gate.max_validation_train_loss_ratio == 4.0
    assert config.stability_gate.max_worst_validation_loss_ratio == 0.50
    assert config.stability_gate.max_worst_validation_train_loss_ratio == 4.0
    assert config.stability_gate.max_validation_loss_coefficient_of_variation == 0.25


def test_scale_invariant_stability_config_uses_fresh_seeds() -> None:
    path = (
        Path(__file__).parents[1]
        / "configs"
        / "paper_seed_stability_scale_invariant.yaml"
    )

    config = load_paper_scale_invariant_seed_stability_config(path)

    assert config.data.base_seed == 0
    assert config.sweep.seeds == (5, 6, 7, 8, 9)
    assert config.train.seed == 5
    assert config.checkpoint_gate.max_validation_train_loss_ratio == 4.0
    assert (
        config.stability_gate.max_validation_loss_ratio_coefficient_of_variation
        == 0.25
    )

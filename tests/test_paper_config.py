from dataclasses import replace
from pathlib import Path

import numpy as np

from koopman_jepa.paper_config import (
    PaperOptimizationConfig,
    load_paper_experiment_config,
    load_paper_linear_identity_heldout_config,
    load_paper_linear_random_control_config,
    load_paper_linear_random_heldout_config,
    load_paper_mlp_clustering_development_config,
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


def test_linear_identity_heldout_config_freezes_replay_and_evaluation() -> None:
    path = (
        Path(__file__).parents[1]
        / "configs"
        / "paper_linear_identity_heldout_smoke.yaml"
    )

    config = load_paper_linear_identity_heldout_config(path)

    assert config.data.base_seed == 0
    assert config.data.test_per_regime == 8
    assert config.model.predictor_kind == "linear"
    assert config.model.linear_initialization == "identity"
    assert config.sweep.seeds == (5, 6, 7, 8, 9)
    assert config.replay.expected_epochs == (10, 10, 10, 8, 10)
    assert config.replay.metric_absolute_tolerance == 1e-8
    assert config.evaluation.kmeans_clusters == 18
    assert config.evaluation.kmeans_n_init == 20
    assert config.evaluation.eigenvalue_tolerance == 0.05
    assert config.gate.max_relative_identity_error == 0.05
    assert config.gate.max_relative_skew_norm == 0.05
    assert config.gate.max_mean_centroid_action_error == 0.02
    assert config.gate.min_near_identity_eigenvalues == 18
    assert config.gate.min_test_effective_rank == 4.0


def test_linear_identity_heldout_config_rejects_random_predictor() -> None:
    path = (
        Path(__file__).parents[1]
        / "configs"
        / "paper_linear_identity_heldout_smoke.yaml"
    )
    config = load_paper_linear_identity_heldout_config(path)
    invalid = replace(
        config,
        model=replace(config.model, linear_initialization="xavier_uniform"),
    )

    with np.testing.assert_raises_regex(ValueError, "identity initialization"):
        invalid.validate()


def test_linear_random_control_config_freezes_paired_comparison() -> None:
    path = (
        Path(__file__).parents[1]
        / "configs"
        / "paper_linear_random_control_smoke.yaml"
    )

    config = load_paper_linear_random_control_config(path)

    assert config.data.base_seed == 0
    assert config.model.predictor_kind == "linear"
    assert config.model.linear_initialization == "xavier_uniform"
    assert config.sweep.seeds == (5, 6, 7, 8, 9)
    assert config.identity_replay.expected_epochs == (10, 10, 10, 8, 10)
    assert (
        config.stability_gate.max_validation_loss_ratio_coefficient_of_variation
        == 0.25
    )
    assert (
        config.comparison_gate.max_random_to_identity_validation_improvement_ratio
        == 2.0
    )
    assert config.comparison_gate.min_random_relative_identity_error == 0.50
    assert config.comparison_gate.min_random_off_diagonal_fraction == 0.50


def test_linear_random_control_config_rejects_identity_initialization() -> None:
    path = (
        Path(__file__).parents[1]
        / "configs"
        / "paper_linear_random_control_smoke.yaml"
    )
    config = load_paper_linear_random_control_config(path)
    invalid = replace(
        config,
        model=replace(config.model, linear_initialization="identity"),
    )

    with np.testing.assert_raises_regex(ValueError, "Xavier-uniform"):
        invalid.validate()


def test_linear_random_heldout_config_freezes_both_replays_and_gates() -> None:
    path = (
        Path(__file__).parents[1]
        / "configs"
        / "paper_linear_random_heldout_smoke.yaml"
    )

    config = load_paper_linear_random_heldout_config(path)

    assert config.data.base_seed == 0
    assert config.data.test_per_regime == 8
    assert config.model.predictor_kind == "linear"
    assert config.model.linear_initialization == "xavier_uniform"
    assert config.sweep.seeds == (5, 6, 7, 8, 9)
    assert config.identity_replay.expected_epochs == (10, 10, 10, 8, 10)
    assert config.random_replay.expected_epochs == (10, 10, 10, 10, 10)
    assert config.evaluation.kmeans_clusters == 18
    assert config.evaluation.kmeans_n_init == 20
    assert config.evaluation.kmeans_seed == 0
    assert config.gate.max_random_to_identity_prediction_error_ratio == 2.0
    assert config.gate.min_random_kmeans_purity == 0.50
    assert config.gate.min_random_to_identity_kmeans_purity_ratio == 0.90
    assert config.gate.min_random_test_effective_rank == 4.0
    assert config.gate.min_random_to_identity_effective_rank_ratio == 0.50


def test_linear_random_heldout_config_rejects_unfrozen_random_epochs() -> None:
    path = (
        Path(__file__).parents[1]
        / "configs"
        / "paper_linear_random_heldout_smoke.yaml"
    )
    config = load_paper_linear_random_heldout_config(path)
    invalid = replace(
        config,
        random_replay=replace(config.random_replay, expected_epochs=(10,)),
    )

    with np.testing.assert_raises_regex(ValueError, "random checkpoint epochs"):
        invalid.validate()


def test_mlp_clustering_development_uses_fresh_data_and_aggregated_kmeans() -> None:
    path = (
        Path(__file__).parents[1]
        / "configs"
        / "paper_mlp_clustering_development.yaml"
    )

    config = load_paper_mlp_clustering_development_config(path)

    assert config.data.base_seed == 1
    assert config.data.train_per_regime == 64
    assert config.data.val_per_regime == 32
    assert config.data.test_per_regime == 64
    assert config.model.predictor_kind == "mlp"
    assert config.model.mlp_depth == "two_hidden"
    assert config.train.epochs == 20
    assert config.sweep.seeds == (10, 11, 12, 13, 14)
    assert config.clustering.clusters == 18
    assert config.clustering.n_init == 20
    assert config.clustering.random_states == tuple(range(20))
    assert config.clustering_gate.min_overall_mean_purity == 0.60
    assert config.clustering_gate.min_worst_seed_mean_purity == 0.55
    assert (
        config.clustering_gate.max_seed_mean_purity_coefficient_of_variation
        == 0.10
    )
    assert config.clustering_gate.max_within_seed_purity_std == 0.03


def test_mlp_clustering_development_rejects_linear_predictor() -> None:
    path = (
        Path(__file__).parents[1]
        / "configs"
        / "paper_mlp_clustering_development.yaml"
    )
    config = load_paper_mlp_clustering_development_config(path)
    invalid = replace(
        config,
        model=replace(config.model, predictor_kind="linear"),
    )

    with np.testing.assert_raises_regex(ValueError, "requires an MLP predictor"):
        invalid.validate()

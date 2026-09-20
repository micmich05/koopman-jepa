from dataclasses import replace

import numpy as np

from koopman_jepa.paper_config import (
    PaperLinearIdentityEvaluationConfig,
    PaperLinearIdentityHeldoutGateConfig,
    PaperLinearPairedHeldoutEvaluationConfig,
    PaperLinearRandomComparisonGateConfig,
    PaperLinearRandomHeldoutGateConfig,
    PaperSeedSweepConfig,
)
from koopman_jepa.paper_evaluation import (
    PaperLinearPairedHeldoutMetrics,
    evaluate_paper_linear_identity,
    evaluate_paper_linear_identity_heldout_gate,
    evaluate_paper_linear_paired_heldout,
    evaluate_paper_linear_random_comparison_gate,
    evaluate_paper_linear_random_heldout_gate,
    evaluate_paper_linear_structure,
)
from koopman_jepa.paper_training import PaperSeedSummary


def _separated_embeddings() -> np.ndarray:
    rng = np.random.default_rng(7)
    centers = np.array(
        [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
        ]
    )
    return np.concatenate(
        [center + 0.05 * rng.standard_normal((8, 4)) for center in centers],
        axis=0,
    )


def test_identity_operator_has_zero_errors_and_full_near_identity_count() -> None:
    config = PaperLinearIdentityEvaluationConfig(
        kmeans_clusters=3,
        kmeans_n_init=10,
    )

    metrics = evaluate_paper_linear_identity(
        np.eye(4),
        _separated_embeddings(),
        config,
    )

    assert metrics.relative_identity_error == 0.0
    assert metrics.relative_skew_norm == 0.0
    assert metrics.mean_centroid_action_error == 0.0
    assert metrics.near_identity_eigenvalues == 4
    assert metrics.test_embedding_std_mean > 0.0
    assert metrics.test_effective_rank > 1.0
    assert metrics.kmeans_inertia > 0.0
    assert metrics.eigenvalue_real == (1.0, 1.0, 1.0, 1.0)
    assert metrics.eigenvalue_imag == (0.0, 0.0, 0.0, 0.0)


def test_operator_metrics_match_closed_form_for_nonsymmetric_matrix() -> None:
    matrix = np.array([[1.0, 0.1], [0.0, 1.0]])
    embeddings = np.array(
        [
            [1.0, 0.0],
            [1.1, 0.1],
            [0.0, 1.0],
            [0.1, 1.1],
        ]
    )
    config = PaperLinearIdentityEvaluationConfig(
        kmeans_clusters=2,
        kmeans_n_init=10,
    )

    metrics = evaluate_paper_linear_identity(matrix, embeddings, config)

    matrix_norm = np.linalg.norm(matrix)
    expected_identity_error = 0.1 / matrix_norm
    expected_skew_norm = np.sqrt(0.02) / matrix_norm
    expected_centroids = np.array([[1.05, 0.05], [0.05, 1.05]])
    expected_centroid_action = np.mean(
        np.linalg.norm(expected_centroids @ matrix.T - expected_centroids, axis=1)
        / np.linalg.norm(expected_centroids, axis=1)
    )
    assert np.isclose(metrics.relative_identity_error, expected_identity_error)
    assert np.isclose(metrics.relative_skew_norm, expected_skew_norm)
    assert np.isclose(metrics.mean_centroid_action_error, expected_centroid_action)
    assert metrics.near_identity_eigenvalues == 2


def test_heldout_gate_requires_every_seed_and_every_criterion() -> None:
    evaluation = PaperLinearIdentityEvaluationConfig(
        kmeans_clusters=3,
        kmeans_n_init=10,
    )
    healthy = evaluate_paper_linear_identity(
        np.eye(4),
        _separated_embeddings(),
        evaluation,
    )
    sweep = PaperSeedSweepConfig(seeds=(5, 6))
    gate_config = PaperLinearIdentityHeldoutGateConfig(
        min_near_identity_eigenvalues=4,
        min_test_effective_rank=1.0,
    )

    passed = evaluate_paper_linear_identity_heldout_gate(
        {5: healthy, 6: healthy},
        sweep,
        gate_config,
    )
    failed_metric = evaluate_paper_linear_identity_heldout_gate(
        {5: healthy, 6: replace(healthy, relative_identity_error=0.10)},
        sweep,
        gate_config,
    )
    missing_seed = evaluate_paper_linear_identity_heldout_gate(
        {5: healthy},
        sweep,
        gate_config,
    )
    nonfinite = evaluate_paper_linear_identity_heldout_gate(
        {5: healthy, 6: replace(healthy, kmeans_inertia=float("nan"))},
        sweep,
        gate_config,
    )

    assert passed.passed
    assert passed.all_seeds_present
    assert passed.failed_seeds == ()
    assert not failed_metric.passed
    assert not failed_metric.identity_passed
    assert failed_metric.failed_seeds == (6,)
    assert not missing_seed.passed
    assert not missing_seed.all_seeds_present
    assert not nonfinite.passed
    assert not nonfinite.all_finite
    assert nonfinite.failed_seeds == (6,)


def test_operator_evaluation_rejects_incompatible_shapes_and_nonfinite_data() -> None:
    config = PaperLinearIdentityEvaluationConfig(kmeans_clusters=2)

    with np.testing.assert_raises_regex(ValueError, "square"):
        evaluate_paper_linear_identity(np.ones((2, 3)), np.ones((4, 2)), config)
    with np.testing.assert_raises_regex(ValueError, "dimension"):
        evaluate_paper_linear_identity(np.eye(3), np.ones((4, 2)), config)
    with np.testing.assert_raises_regex(ValueError, "finite"):
        evaluate_paper_linear_identity(
            np.eye(2),
            np.array([[0.0, 1.0], [np.nan, 0.0]]),
            config,
        )


def test_linear_structure_distinguishes_identity_from_dense_permutation() -> None:
    identity = evaluate_paper_linear_structure(np.eye(4))
    dense_control = evaluate_paper_linear_structure(
        np.roll(np.eye(4), shift=1, axis=1)
    )

    assert identity.relative_identity_error == 0.0
    assert identity.off_diagonal_fraction == 0.0
    assert dense_control.relative_identity_error > 0.5
    assert dense_control.off_diagonal_fraction == 1.0


def _seed_summary(
    seed: int,
    *,
    validation_loss: float,
    validation_loss_ratio: float,
) -> PaperSeedSummary:
    return PaperSeedSummary(
        seed=seed,
        checkpoint_epoch=9,
        train_loss=validation_loss / 3.0,
        validation_loss=validation_loss,
        validation_loss_ratio=validation_loss_ratio,
        validation_train_loss_ratio=3.0,
        validation_embedding_std_ratio=0.7,
        validation_effective_rank=20.0,
    )


def test_random_comparison_uses_relative_improvement_and_requires_structure() -> None:
    sweep = PaperSeedSweepConfig(seeds=(5, 6))
    gate_config = PaperLinearRandomComparisonGateConfig()
    identity = [
        _seed_summary(5, validation_loss=0.0010, validation_loss_ratio=0.20),
        _seed_summary(6, validation_loss=0.0012, validation_loss_ratio=0.25),
    ]
    random = [
        _seed_summary(5, validation_loss=0.0030, validation_loss_ratio=0.30),
        _seed_summary(6, validation_loss=0.0048, validation_loss_ratio=0.40),
    ]
    dense_matrix = np.roll(np.eye(4), shift=1, axis=1)

    result = evaluate_paper_linear_random_comparison_gate(
        identity,
        random,
        {5: dense_matrix, 6: dense_matrix},
        sweep,
        gate_config,
    )
    identity_matrix_failure = evaluate_paper_linear_random_comparison_gate(
        identity,
        random,
        {5: dense_matrix, 6: np.eye(4)},
        sweep,
        gate_config,
    )
    missing_seed = evaluate_paper_linear_random_comparison_gate(
        identity,
        random[:-1],
        {5: dense_matrix},
        sweep,
        gate_config,
    )

    assert result.passed
    assert result.worst_validation_improvement_ratio == 1.6
    assert result.worst_absolute_validation_loss_ratio == 4.0
    assert result.predictive_comparability_passed
    assert result.non_identity_passed
    assert result.density_passed
    assert not identity_matrix_failure.passed
    assert identity_matrix_failure.failed_seeds == (6,)
    assert not identity_matrix_failure.non_identity_passed
    assert not identity_matrix_failure.density_passed
    assert not missing_seed.passed
    assert not missing_seed.all_seeds_present
    assert missing_seed.failed_seeds == (6,)


def test_linear_structure_rejects_nonfinite_matrix() -> None:
    with np.testing.assert_raises_regex(ValueError, "finite"):
        evaluate_paper_linear_structure(np.array([[1.0, np.nan], [0.0, 1.0]]))


def test_paired_heldout_metrics_recover_oracle_prediction_and_clustering() -> None:
    online = _separated_embeddings()
    labels = np.repeat(np.arange(3), 8)
    matrix = np.array(
        [
            [1.0, 0.2, 0.0, 0.0],
            [0.0, 1.0, 0.1, 0.0],
            [0.1, 0.0, 1.0, 0.2],
            [0.0, 0.1, 0.0, 1.0],
        ]
    )
    target = online @ matrix.T
    config = PaperLinearPairedHeldoutEvaluationConfig(
        kmeans_clusters=3,
        kmeans_n_init=10,
    )

    metrics = evaluate_paper_linear_paired_heldout(
        matrix,
        online,
        target,
        labels,
        config,
    )

    assert metrics.normalized_prediction_error == 0.0
    assert metrics.kmeans_purity == 1.0
    assert metrics.kmeans_matched_accuracy == 1.0
    assert metrics.test_embedding_std_mean > 0.0
    assert metrics.test_effective_rank > 1.0


def test_paired_heldout_metrics_reject_invalid_labels_and_shapes() -> None:
    embeddings = _separated_embeddings()
    labels = np.repeat(np.arange(3), 8)
    config = PaperLinearPairedHeldoutEvaluationConfig(kmeans_clusters=3)

    with np.testing.assert_raises_regex(ValueError, "cover every"):
        evaluate_paper_linear_paired_heldout(
            np.eye(4),
            embeddings,
            embeddings,
            np.zeros(24, dtype=np.int64),
            config,
        )
    with np.testing.assert_raises_regex(ValueError, "same shape"):
        evaluate_paper_linear_paired_heldout(
            np.eye(4),
            embeddings,
            embeddings[:-1],
            labels,
            config,
        )
    with np.testing.assert_raises_regex(ValueError, "finite"):
        invalid = embeddings.copy()
        invalid[0, 0] = np.nan
        evaluate_paper_linear_paired_heldout(
            np.eye(4),
            invalid,
            embeddings,
            labels,
            config,
        )


def _heldout_metrics(
    *,
    prediction_error: float,
    purity: float,
    rank: float,
) -> PaperLinearPairedHeldoutMetrics:
    return PaperLinearPairedHeldoutMetrics(
        normalized_prediction_error=prediction_error,
        kmeans_purity=purity,
        kmeans_matched_accuracy=purity - 0.05,
        test_embedding_std_mean=0.5,
        test_effective_rank=rank,
    )


def test_random_heldout_gate_requires_every_paired_criterion() -> None:
    sweep = PaperSeedSweepConfig(seeds=(5, 6))
    gate_config = PaperLinearRandomHeldoutGateConfig()
    identity = {
        5: _heldout_metrics(prediction_error=0.10, purity=0.70, rank=20.0),
        6: _heldout_metrics(prediction_error=0.12, purity=0.60, rank=16.0),
    }
    random = {
        5: _heldout_metrics(prediction_error=0.15, purity=0.65, rank=15.0),
        6: _heldout_metrics(prediction_error=0.18, purity=0.55, rank=10.0),
    }

    passed = evaluate_paper_linear_random_heldout_gate(
        identity,
        random,
        sweep,
        gate_config,
    )
    low_purity = evaluate_paper_linear_random_heldout_gate(
        identity,
        {
            **random,
            6: replace(random[6], kmeans_purity=0.45),
        },
        sweep,
        gate_config,
    )
    low_rank = evaluate_paper_linear_random_heldout_gate(
        identity,
        {
            **random,
            6: replace(random[6], test_effective_rank=3.0),
        },
        sweep,
        gate_config,
    )
    high_error = evaluate_paper_linear_random_heldout_gate(
        identity,
        {
            **random,
            6: replace(random[6], normalized_prediction_error=0.30),
        },
        sweep,
        gate_config,
    )
    missing_seed = evaluate_paper_linear_random_heldout_gate(
        identity,
        {5: random[5]},
        sweep,
        gate_config,
    )

    assert passed.passed
    assert passed.worst_prediction_error_ratio == 1.5
    assert np.isclose(passed.minimum_kmeans_purity_ratio, 0.55 / 0.60)
    assert passed.minimum_random_kmeans_purity == 0.55
    assert passed.minimum_random_test_effective_rank == 10.0
    assert passed.minimum_effective_rank_ratio == 0.625
    assert not low_purity.passed
    assert not low_purity.absolute_purity_passed
    assert not low_purity.relative_purity_passed
    assert low_purity.failed_seeds == (6,)
    assert not low_rank.passed
    assert not low_rank.absolute_rank_passed
    assert not low_rank.relative_rank_passed
    assert not high_error.passed
    assert not high_error.prediction_passed
    assert not missing_seed.passed
    assert not missing_seed.all_seeds_present
    assert missing_seed.failed_seeds == (6,)


def test_random_heldout_gate_rejects_nonfinite_metrics() -> None:
    sweep = PaperSeedSweepConfig(seeds=(5,))
    gate_config = PaperLinearRandomHeldoutGateConfig()
    identity = {
        5: _heldout_metrics(prediction_error=0.10, purity=0.70, rank=20.0),
    }
    random = {
        5: _heldout_metrics(prediction_error=np.nan, purity=0.65, rank=15.0),
    }

    result = evaluate_paper_linear_random_heldout_gate(
        identity,
        random,
        sweep,
        gate_config,
    )

    assert not result.passed
    assert not result.all_finite
    assert result.failed_seeds == (5,)

from dataclasses import replace

import numpy as np

from koopman_jepa.paper_config import (
    PaperLinearIdentityEvaluationConfig,
    PaperLinearIdentityHeldoutGateConfig,
    PaperSeedSweepConfig,
)
from koopman_jepa.paper_evaluation import (
    evaluate_paper_linear_identity,
    evaluate_paper_linear_identity_heldout_gate,
)


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

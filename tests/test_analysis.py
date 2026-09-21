import numpy as np

from koopman_jepa.analysis import (
    calibration_statistics,
    clustering_diagnostics,
    exact_one_sided_sign_flip_test,
    linear_probe_accuracy,
    predictor_subspace_statistics,
)


def test_identity_is_recovered_on_oracle_centroid_span() -> None:
    centroids = np.eye(4, dtype=np.float64)
    metrics = predictor_subspace_statistics(np.eye(4), centroids, centroids)

    assert metrics["active_rank"] == 4
    assert metrics["centroid_identity_error"] < 1e-12
    assert metrics["active_identity_error"] < 1e-12
    assert metrics["active_eigenvalue_one_error"] < 1e-12


def test_collapsed_centroids_report_undefined_active_metrics() -> None:
    centroids = np.zeros((3, 2), dtype=np.float64)
    metrics = predictor_subspace_statistics(np.eye(2), centroids, centroids)

    assert metrics["active_rank"] == 0
    assert metrics["active_invariance_error"] is None
    assert metrics["active_identity_error"] is None
    assert metrics["active_eigenvalue_one_error"] is None


def test_linear_probe_is_invariant_to_global_embedding_scale() -> None:
    labels = np.repeat(np.arange(3), 8)
    embeddings = np.eye(3)[labels]

    original_accuracy = linear_probe_accuracy(embeddings, labels, embeddings, labels, seed=0)
    scaled_accuracy = linear_probe_accuracy(
        embeddings * 1e-6,
        labels,
        embeddings * 1e-6,
        labels,
        seed=0,
    )

    assert original_accuracy == 1.0
    assert scaled_accuracy == original_accuracy


def test_clustering_diagnostics_aligns_permuted_cluster_ids() -> None:
    labels = np.repeat(np.arange(3), 6)
    embeddings = np.eye(3)[labels]

    diagnostics = clustering_diagnostics(
        embeddings,
        labels,
        num_regimes=3,
        seed=4,
        n_init=5,
    )

    assert diagnostics["kmeans_purity"] == 1.0
    assert diagnostics["kmeans_matched_accuracy"] == 1.0
    np.testing.assert_array_equal(
        diagnostics["aligned_confusion"],
        np.diag([6, 6, 6]),
    )
    np.testing.assert_array_equal(diagnostics["per_regime_recall"], np.ones(3))


def test_clustering_diagnostics_rejects_incomplete_labels() -> None:
    labels = np.zeros(6, dtype=np.int64)
    embeddings = np.ones((6, 2))

    with np.testing.assert_raises_regex(ValueError, "cover every regime"):
        clustering_diagnostics(
            embeddings,
            labels,
            num_regimes=2,
            seed=0,
        )


def test_calibration_statistics_penalizes_invalid_estimate_without_dropping_it() -> None:
    metrics = calibration_statistics(
        np.array([0.0, 0.5, 1.0]),
        np.array([0.0, np.nan, 0.9]),
        invalid_absolute_error=1.0,
    )

    assert np.isclose(metrics["mae"], (0.0 + 1.0 + 0.1) / 3.0)
    assert metrics["valid_count"] == 2
    assert metrics["invalid_count"] == 1
    assert np.isclose(metrics["slope"], 0.9)


def test_exact_sign_flip_test_enumerates_every_assignment() -> None:
    result = exact_one_sided_sign_flip_test(np.array([1.0, 2.0, 3.0]))

    assert result["pair_count"] == 3
    assert result["assignment_count"] == 8
    assert result["observed_mean_difference"] == 2.0
    assert result["p_value"] == 1.0 / 8.0

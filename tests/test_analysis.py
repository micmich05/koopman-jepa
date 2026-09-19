import numpy as np

from koopman_jepa.analysis import evaluate_phase0, predictor_subspace_statistics


def test_identity_is_recovered_on_oracle_centroid_span() -> None:
    centroids = np.eye(4, dtype=np.float64)
    metrics = predictor_subspace_statistics(np.eye(4), centroids, centroids)

    assert metrics["active_rank"] == 4
    assert metrics["centroid_identity_error"] < 1e-12
    assert metrics["active_identity_error"] < 1e-12
    assert metrics["active_eigenvalue_one_error"] < 1e-12


def test_oracle_regime_embeddings_score_perfectly() -> None:
    labels = np.repeat(np.arange(3), 10)
    embeddings = np.eye(3)[labels]
    metrics = evaluate_phase0(
        train_embeddings=embeddings,
        train_labels=labels,
        test_embeddings=embeddings,
        test_target_embeddings=embeddings,
        test_labels=labels,
        predictor_matrix=np.eye(3),
        num_regimes=3,
        seed=0,
    )

    assert metrics["nearest_centroid_accuracy"] == 1.0
    assert metrics["linear_probe_accuracy"] == 1.0
    assert metrics["kmeans_purity"] == 1.0


def test_collapsed_centroids_report_undefined_active_metrics() -> None:
    centroids = np.zeros((3, 2), dtype=np.float64)
    metrics = predictor_subspace_statistics(np.eye(2), centroids, centroids)

    assert metrics["active_rank"] == 0
    assert metrics["active_invariance_error"] is None
    assert metrics["active_identity_error"] is None
    assert metrics["active_eigenvalue_one_error"] is None

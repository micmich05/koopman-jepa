from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression


def regime_centroids(
    embeddings: np.ndarray,
    labels: np.ndarray,
    num_regimes: int,
) -> np.ndarray:
    centroids = []
    for regime_id in range(num_regimes):
        selected = embeddings[labels == regime_id]
        if selected.size == 0:
            raise ValueError(f"No samples for regime {regime_id}")
        centroids.append(selected.mean(axis=0))
    return np.stack(centroids, axis=0)


def covariance_statistics(embeddings: np.ndarray) -> dict[str, Any]:
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(embeddings.shape[0] - 1, 1)
    eigenvalues = np.clip(np.linalg.eigvalsh(covariance), 0.0, None)
    total = float(eigenvalues.sum())
    if total <= 1e-12:
        effective_rank = 0.0
    else:
        probabilities = eigenvalues / total
        nonzero = probabilities > 1e-12
        entropy = -(probabilities[nonzero] * np.log(probabilities[nonzero])).sum()
        effective_rank = float(np.exp(entropy))

    return {
        "embedding_std_mean": float(np.sqrt(np.diag(covariance) + 1e-12).mean()),
        "covariance_eigenvalues": eigenvalues.tolist(),
        "effective_rank": effective_rank,
    }


def separation_statistics(
    embeddings: np.ndarray,
    labels: np.ndarray,
    num_regimes: int,
) -> dict[str, float]:
    centroids = regime_centroids(embeddings, labels, num_regimes)
    global_mean = embeddings.mean(axis=0)
    within = np.mean(
        [
            np.mean(np.sum((embeddings[labels == regime_id] - centroids[regime_id]) ** 2, axis=1))
            for regime_id in range(num_regimes)
        ]
    )
    between = float(np.mean(np.sum((centroids - global_mean) ** 2, axis=1)))
    return {
        "within_regime_variance": float(within),
        "between_regime_variance": between,
        "between_within_ratio": float(between / max(float(within), 1e-12)),
    }


def nearest_centroid_accuracy(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
    num_regimes: int,
) -> float:
    centroids = regime_centroids(train_embeddings, train_labels, num_regimes)
    distances = ((test_embeddings[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    predictions = distances.argmin(axis=1)
    return float(np.mean(predictions == test_labels))


def linear_probe_accuracy(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
    seed: int,
) -> float:
    probe = LogisticRegression(max_iter=2_000, random_state=seed)
    probe.fit(train_embeddings, train_labels)
    return float(probe.score(test_embeddings, test_labels))


def clustering_scores(
    embeddings: np.ndarray,
    labels: np.ndarray,
    num_regimes: int,
    seed: int,
) -> dict[str, float]:
    assignments = KMeans(
        n_clusters=num_regimes,
        n_init=20,
        random_state=seed,
    ).fit_predict(embeddings)
    contingency = np.zeros((num_regimes, num_regimes), dtype=np.int64)
    for cluster_id, label in zip(assignments, labels, strict=True):
        contingency[cluster_id, label] += 1

    purity = float(contingency.max(axis=1).sum() / labels.size)
    rows, columns = linear_sum_assignment(-contingency)
    matched_accuracy = float(contingency[rows, columns].sum() / labels.size)
    return {
        "kmeans_purity": purity,
        "kmeans_matched_accuracy": matched_accuracy,
    }


def _complex_list(values: np.ndarray) -> list[dict[str, float]]:
    ordered = sorted(values, key=lambda value: (float(value.real), float(value.imag)))
    return [{"real": float(value.real), "imag": float(value.imag)} for value in ordered]


def predictor_subspace_statistics(
    matrix: np.ndarray,
    online_centroids: np.ndarray,
    target_centroids: np.ndarray,
    rank_tolerance: float = 1e-6,
) -> dict[str, Any]:
    source = online_centroids.T
    destination = target_centroids.T
    left_vectors, singular_values, _ = np.linalg.svd(source, full_matrices=False)
    if singular_values.size == 0 or singular_values[0] <= 1e-12:
        active_rank = 0
        basis = np.zeros((matrix.shape[0], 0), dtype=matrix.dtype)
    else:
        active_rank = int(np.sum(singular_values > rank_tolerance * singular_values[0]))
        basis = left_vectors[:, :active_rank]

    prediction_error = np.linalg.norm(matrix @ source - destination) / max(
        np.linalg.norm(destination), 1e-12
    )
    identity_error = np.linalg.norm(matrix @ source - source) / max(np.linalg.norm(source), 1e-12)
    ema_error = np.linalg.norm(source - destination) / max(np.linalg.norm(destination), 1e-12)

    if active_rank == 0:
        reduced_eigenvalues = np.array([], dtype=np.complex128)
        invariance_error = None
        reduced_identity_error = None
        eigenvalue_one_error = None
    else:
        reduced = basis.T @ matrix @ basis
        reduced_eigenvalues = np.linalg.eigvals(reduced)
        projected = basis @ basis.T
        residual = (np.eye(matrix.shape[0]) - projected) @ matrix @ basis
        invariance_error = float(
            np.linalg.norm(residual) / max(np.linalg.norm(matrix @ basis), 1e-12)
        )
        reduced_identity_error = float(
            np.linalg.norm(reduced - np.eye(active_rank)) / np.sqrt(active_rank)
        )
        eigenvalue_one_error = float(np.mean(np.abs(reduced_eigenvalues - 1.0)))

    return {
        "active_rank": active_rank,
        "centroid_singular_values": singular_values.tolist(),
        "centroid_prediction_error": float(prediction_error),
        "centroid_identity_error": float(identity_error),
        "online_target_centroid_error": float(ema_error),
        "active_invariance_error": invariance_error,
        "active_identity_error": reduced_identity_error,
        "active_eigenvalue_one_error": eigenvalue_one_error,
        "active_eigenvalues": _complex_list(reduced_eigenvalues),
        "full_eigenvalues": _complex_list(np.linalg.eigvals(matrix)),
    }


def evaluate_phase0(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    test_embeddings: np.ndarray,
    test_target_embeddings: np.ndarray,
    test_labels: np.ndarray,
    predictor_matrix: np.ndarray,
    num_regimes: int,
    seed: int,
) -> dict[str, Any]:
    online_centroids = regime_centroids(test_embeddings, test_labels, num_regimes)
    target_centroids = regime_centroids(test_target_embeddings, test_labels, num_regimes)

    metrics: dict[str, Any] = {}
    metrics.update(covariance_statistics(test_embeddings))
    metrics.update(separation_statistics(test_embeddings, test_labels, num_regimes))
    metrics.update(clustering_scores(test_embeddings, test_labels, num_regimes, seed))
    metrics["nearest_centroid_accuracy"] = nearest_centroid_accuracy(
        train_embeddings,
        train_labels,
        test_embeddings,
        test_labels,
        num_regimes,
    )
    metrics["linear_probe_accuracy"] = linear_probe_accuracy(
        train_embeddings,
        train_labels,
        test_embeddings,
        test_labels,
        seed,
    )
    metrics.update(
        predictor_subspace_statistics(
            predictor_matrix,
            online_centroids,
            target_centroids,
        )
    )
    return metrics

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


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
    probe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2_000, random_state=seed),
    )
    probe.fit(train_embeddings, train_labels)
    return float(probe.score(test_embeddings, test_labels))


def calibration_statistics(
    true_values: np.ndarray,
    estimated_values: np.ndarray,
    invalid_absolute_error: float = 1.0,
) -> dict[str, Any]:
    """Summarize a continuous calibration curve without dropping invalid runs."""

    truth = np.asarray(true_values, dtype=np.float64)
    estimates = np.asarray(estimated_values, dtype=np.float64)
    if truth.ndim != 1 or estimates.ndim != 1 or truth.shape != estimates.shape:
        raise ValueError("true and estimated values must be aligned one-dimensional arrays")
    if truth.size < 2 or not np.isfinite(truth).all():
        raise ValueError("true values must contain at least two finite observations")
    if not np.isfinite(invalid_absolute_error) or invalid_absolute_error < 0.0:
        raise ValueError("invalid_absolute_error must be finite and non-negative")

    valid = np.isfinite(estimates)
    absolute_errors = np.full(truth.shape, invalid_absolute_error, dtype=np.float64)
    absolute_errors[valid] = np.abs(estimates[valid] - truth[valid])
    slope: float | None = None
    intercept: float | None = None
    r_squared: float | None = None
    if valid.sum() >= 2 and np.unique(truth[valid]).size >= 2:
        slope, intercept = (
            float(value) for value in np.polyfit(truth[valid], estimates[valid], 1)
        )
        fitted = slope * truth[valid] + intercept
        total_variation = float(np.sum(np.square(estimates[valid] - estimates[valid].mean())))
        if total_variation > 1e-15:
            r_squared = float(
                1.0
                - np.sum(np.square(estimates[valid] - fitted)) / total_variation
            )

    return {
        "mae": float(absolute_errors.mean()),
        "absolute_errors": absolute_errors.tolist(),
        "valid_count": int(valid.sum()),
        "invalid_count": int((~valid).sum()),
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_squared,
    }


def exact_one_sided_sign_flip_test(differences: np.ndarray) -> dict[str, float | int]:
    """Test whether paired differences have positive mean by exact sign flips."""

    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("differences must be a non-empty one-dimensional array")
    if values.size > 16:
        raise ValueError("exact sign-flip enumeration is limited to 16 pairs")
    if not np.isfinite(values).all():
        raise ValueError("differences must be finite")

    observed_mean = float(values.mean())
    assignment_count = 2 ** values.size
    at_least_observed = 0
    tolerance = 1e-15 * max(1.0, abs(observed_mean))
    for assignment in range(assignment_count):
        signs = np.array(
            [1.0 if assignment & (1 << index) else -1.0 for index in range(values.size)]
        )
        permuted_mean = float(np.mean(signs * values))
        at_least_observed += permuted_mean >= observed_mean - tolerance

    return {
        "pair_count": int(values.size),
        "assignment_count": assignment_count,
        "observed_mean_difference": observed_mean,
        "p_value": at_least_observed / assignment_count,
    }


def clustering_scores(
    embeddings: np.ndarray,
    labels: np.ndarray,
    num_regimes: int,
    seed: int,
) -> dict[str, float]:
    diagnostics = clustering_diagnostics(
        embeddings,
        labels,
        num_regimes,
        seed,
    )
    return {
        "kmeans_purity": diagnostics["kmeans_purity"],
        "kmeans_matched_accuracy": diagnostics["kmeans_matched_accuracy"],
    }


def clustering_diagnostics(
    embeddings: np.ndarray,
    labels: np.ndarray,
    num_regimes: int,
    seed: int,
    *,
    n_init: int = 20,
) -> dict[str, Any]:
    """Return purity plus a label-aligned confusion matrix for diagnostics."""

    embeddings = np.asarray(embeddings, dtype=np.float64)
    labels = np.asarray(labels)
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be two-dimensional")
    if labels.ndim != 1 or labels.shape[0] != embeddings.shape[0]:
        raise ValueError("labels must be one-dimensional and align with embeddings")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("labels must be integers")
    if num_regimes < 2:
        raise ValueError("num_regimes must be at least two")
    if embeddings.shape[0] < num_regimes:
        raise ValueError("sample count must be at least num_regimes")
    if not np.array_equal(np.unique(labels), np.arange(num_regimes)):
        raise ValueError("labels must cover every regime exactly")
    if not np.isfinite(embeddings).all():
        raise ValueError("embeddings must be finite")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if n_init < 1:
        raise ValueError("n_init must be positive")

    assignments = KMeans(
        n_clusters=num_regimes,
        n_init=n_init,
        random_state=seed,
    ).fit_predict(embeddings)
    contingency = np.zeros((num_regimes, num_regimes), dtype=np.int64)
    for cluster_id, label in zip(assignments, labels, strict=True):
        contingency[cluster_id, label] += 1

    purity = float(contingency.max(axis=1).sum() / labels.size)
    rows, columns = linear_sum_assignment(-contingency)
    cluster_to_label = np.full(num_regimes, -1, dtype=np.int64)
    cluster_to_label[rows] = columns
    predicted_labels = cluster_to_label[assignments]
    aligned_confusion = np.zeros((num_regimes, num_regimes), dtype=np.int64)
    for true_label, predicted_label in zip(labels, predicted_labels, strict=True):
        aligned_confusion[true_label, predicted_label] += 1
    regime_counts = aligned_confusion.sum(axis=1)
    per_regime_recall = np.diag(aligned_confusion) / regime_counts
    matched_accuracy = float(np.mean(predicted_labels == labels))
    return {
        "kmeans_purity": purity,
        "kmeans_matched_accuracy": matched_accuracy,
        "contingency": contingency,
        "aligned_confusion": aligned_confusion,
        "per_regime_recall": per_regime_recall,
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

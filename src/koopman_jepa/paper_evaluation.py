from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans

from .paper_config import (
    PaperLinearIdentityEvaluationConfig,
    PaperLinearIdentityHeldoutGateConfig,
    PaperSeedSweepConfig,
)


@dataclass(frozen=True, slots=True)
class PaperLinearIdentityMetrics:
    relative_identity_error: float
    relative_skew_norm: float
    mean_centroid_action_error: float
    near_identity_eigenvalues: int
    test_embedding_std_mean: float
    test_effective_rank: float
    kmeans_inertia: float
    eigenvalue_real: tuple[float, ...]
    eigenvalue_imag: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class PaperLinearIdentityHeldoutGateResult:
    all_seeds_present: bool
    all_finite: bool
    worst_relative_identity_error: float
    worst_relative_skew_norm: float
    worst_mean_centroid_action_error: float
    minimum_near_identity_eigenvalues: int
    minimum_test_effective_rank: float
    failed_seeds: tuple[int, ...]
    identity_passed: bool
    skew_passed: bool
    centroid_action_passed: bool
    eigenvalues_passed: bool
    effective_rank_passed: bool
    passed: bool


def _validate_operator_inputs(
    matrix: np.ndarray,
    embeddings: np.ndarray,
    config: PaperLinearIdentityEvaluationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    config.validate()
    matrix = np.asarray(matrix, dtype=np.float64)
    embeddings = np.asarray(embeddings, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("predictor matrix must be square")
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a two-dimensional array")
    if embeddings.shape[1] != matrix.shape[0]:
        raise ValueError("embedding dimension must match predictor matrix")
    if embeddings.shape[0] < config.kmeans_clusters:
        raise ValueError("sample count must be at least kmeans_clusters")
    if not np.isfinite(matrix).all() or not np.isfinite(embeddings).all():
        raise ValueError("predictor matrix and embeddings must be finite")
    return matrix, embeddings


def _embedding_spread(embeddings: np.ndarray) -> tuple[float, float]:
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    embedding_std_mean = float(np.sqrt(np.mean(centered**2, axis=0)).mean())
    squared_singular_values = np.linalg.svd(centered, compute_uv=False) ** 2
    total = float(squared_singular_values.sum())
    if total <= 1e-12:
        return embedding_std_mean, 0.0
    probabilities = squared_singular_values / total
    probabilities = probabilities[probabilities > 1e-12]
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return embedding_std_mean, float(np.exp(entropy))


def evaluate_paper_linear_identity(
    matrix: np.ndarray,
    embeddings: np.ndarray,
    config: PaperLinearIdentityEvaluationConfig,
) -> PaperLinearIdentityMetrics:
    """Compute the frozen linear-identity diagnostics on raw embeddings."""

    matrix, embeddings = _validate_operator_inputs(matrix, embeddings, config)
    clustering = KMeans(
        n_clusters=config.kmeans_clusters,
        n_init=config.kmeans_n_init,
        random_state=config.kmeans_seed,
    ).fit(embeddings)
    centroids = clustering.cluster_centers_

    matrix_norm = max(float(np.linalg.norm(matrix)), 1e-12)
    identity = np.eye(matrix.shape[0], dtype=matrix.dtype)
    relative_identity_error = float(np.linalg.norm(matrix - identity) / matrix_norm)
    relative_skew_norm = float(np.linalg.norm(matrix - matrix.T) / matrix_norm)

    transformed_centroids = centroids @ matrix.T
    centroid_denominators = np.maximum(np.linalg.norm(centroids, axis=1), 1e-12)
    centroid_errors = np.linalg.norm(transformed_centroids - centroids, axis=1)
    mean_centroid_action_error = float(
        np.mean(centroid_errors / centroid_denominators)
    )

    eigenvalues = np.linalg.eigvals(matrix)
    near_identity_eigenvalues = int(
        np.sum(np.abs(eigenvalues - 1.0) <= config.eigenvalue_tolerance)
    )
    test_embedding_std_mean, test_effective_rank = _embedding_spread(embeddings)
    return PaperLinearIdentityMetrics(
        relative_identity_error=relative_identity_error,
        relative_skew_norm=relative_skew_norm,
        mean_centroid_action_error=mean_centroid_action_error,
        near_identity_eigenvalues=near_identity_eigenvalues,
        test_embedding_std_mean=test_embedding_std_mean,
        test_effective_rank=test_effective_rank,
        kmeans_inertia=float(clustering.inertia_),
        eigenvalue_real=tuple(float(value.real) for value in eigenvalues),
        eigenvalue_imag=tuple(float(value.imag) for value in eigenvalues),
    )


def _metrics_are_finite(metrics: PaperLinearIdentityMetrics) -> bool:
    values = (
        metrics.relative_identity_error,
        metrics.relative_skew_norm,
        metrics.mean_centroid_action_error,
        metrics.test_embedding_std_mean,
        metrics.test_effective_rank,
        metrics.kmeans_inertia,
        *metrics.eigenvalue_real,
        *metrics.eigenvalue_imag,
    )
    return all(math.isfinite(value) for value in values)


def evaluate_paper_linear_identity_heldout_gate(
    metrics_by_seed: dict[int, PaperLinearIdentityMetrics],
    sweep: PaperSeedSweepConfig,
    config: PaperLinearIdentityHeldoutGateConfig,
) -> PaperLinearIdentityHeldoutGateResult:
    """Require every frozen seed to pass every linear-identity criterion."""

    sweep.validate()
    config.validate()
    expected_seeds = set(sweep.seeds)
    all_seeds_present = set(metrics_by_seed) == expected_seeds
    all_finite = bool(metrics_by_seed) and all(
        _metrics_are_finite(metrics) for metrics in metrics_by_seed.values()
    )

    if metrics_by_seed:
        worst_relative_identity_error = max(
            metrics.relative_identity_error for metrics in metrics_by_seed.values()
        )
        worst_relative_skew_norm = max(
            metrics.relative_skew_norm for metrics in metrics_by_seed.values()
        )
        worst_mean_centroid_action_error = max(
            metrics.mean_centroid_action_error
            for metrics in metrics_by_seed.values()
        )
        minimum_near_identity_eigenvalues = min(
            metrics.near_identity_eigenvalues
            for metrics in metrics_by_seed.values()
        )
        minimum_test_effective_rank = min(
            metrics.test_effective_rank for metrics in metrics_by_seed.values()
        )
    else:
        worst_relative_identity_error = math.inf
        worst_relative_skew_norm = math.inf
        worst_mean_centroid_action_error = math.inf
        minimum_near_identity_eigenvalues = 0
        minimum_test_effective_rank = 0.0

    def seed_passes(metrics: PaperLinearIdentityMetrics) -> bool:
        return (
            _metrics_are_finite(metrics)
            and metrics.relative_identity_error
            <= config.max_relative_identity_error
            and metrics.relative_skew_norm <= config.max_relative_skew_norm
            and metrics.mean_centroid_action_error
            <= config.max_mean_centroid_action_error
            and metrics.near_identity_eigenvalues
            >= config.min_near_identity_eigenvalues
            and metrics.test_effective_rank >= config.min_test_effective_rank
        )

    failed_seeds = tuple(
        sorted(
            seed
            for seed, metrics in metrics_by_seed.items()
            if not seed_passes(metrics)
        )
    )
    identity_passed = (
        worst_relative_identity_error <= config.max_relative_identity_error
    )
    skew_passed = worst_relative_skew_norm <= config.max_relative_skew_norm
    centroid_action_passed = (
        worst_mean_centroid_action_error
        <= config.max_mean_centroid_action_error
    )
    eigenvalues_passed = (
        minimum_near_identity_eigenvalues
        >= config.min_near_identity_eigenvalues
    )
    effective_rank_passed = (
        minimum_test_effective_rank >= config.min_test_effective_rank
    )
    return PaperLinearIdentityHeldoutGateResult(
        all_seeds_present=all_seeds_present,
        all_finite=all_finite,
        worst_relative_identity_error=worst_relative_identity_error,
        worst_relative_skew_norm=worst_relative_skew_norm,
        worst_mean_centroid_action_error=worst_mean_centroid_action_error,
        minimum_near_identity_eigenvalues=minimum_near_identity_eigenvalues,
        minimum_test_effective_rank=minimum_test_effective_rank,
        failed_seeds=failed_seeds,
        identity_passed=identity_passed,
        skew_passed=skew_passed,
        centroid_action_passed=centroid_action_passed,
        eigenvalues_passed=eigenvalues_passed,
        effective_rank_passed=effective_rank_passed,
        passed=(
            all_seeds_present
            and all_finite
            and not failed_seeds
            and identity_passed
            and skew_passed
            and centroid_action_passed
            and eigenvalues_passed
            and effective_rank_passed
        ),
    )

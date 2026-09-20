from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

from .paper_config import (
    PaperLinearIdentityEvaluationConfig,
    PaperLinearIdentityHeldoutGateConfig,
    PaperLinearPairedHeldoutEvaluationConfig,
    PaperLinearRandomComparisonGateConfig,
    PaperLinearRandomHeldoutGateConfig,
    PaperSeedSweepConfig,
)
from .paper_training import PaperSeedSummary


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


@dataclass(frozen=True, slots=True)
class PaperLinearStructureMetrics:
    relative_identity_error: float
    off_diagonal_fraction: float


@dataclass(frozen=True, slots=True)
class PaperLinearRandomSeedComparison:
    seed: int
    validation_improvement_ratio: float
    absolute_validation_loss_ratio: float
    relative_identity_error: float
    off_diagonal_fraction: float
    passed: bool


@dataclass(frozen=True, slots=True)
class PaperLinearRandomComparisonGateResult:
    all_seeds_present: bool
    all_finite: bool
    worst_validation_improvement_ratio: float
    worst_absolute_validation_loss_ratio: float
    minimum_relative_identity_error: float
    minimum_off_diagonal_fraction: float
    failed_seeds: tuple[int, ...]
    predictive_comparability_passed: bool
    non_identity_passed: bool
    density_passed: bool
    comparisons: tuple[PaperLinearRandomSeedComparison, ...]
    passed: bool


@dataclass(frozen=True, slots=True)
class PaperLinearPairedHeldoutMetrics:
    normalized_prediction_error: float
    kmeans_purity: float
    kmeans_matched_accuracy: float
    test_embedding_std_mean: float
    test_effective_rank: float


@dataclass(frozen=True, slots=True)
class PaperLinearRandomHeldoutSeedComparison:
    seed: int
    identity_prediction_error: float
    random_prediction_error: float
    prediction_error_ratio: float
    identity_kmeans_purity: float
    random_kmeans_purity: float
    kmeans_purity_ratio: float
    identity_effective_rank: float
    random_effective_rank: float
    effective_rank_ratio: float
    random_kmeans_matched_accuracy: float
    passed: bool


@dataclass(frozen=True, slots=True)
class PaperLinearRandomHeldoutGateResult:
    all_seeds_present: bool
    all_finite: bool
    worst_prediction_error_ratio: float
    minimum_random_kmeans_purity: float
    minimum_kmeans_purity_ratio: float
    minimum_random_test_effective_rank: float
    minimum_effective_rank_ratio: float
    failed_seeds: tuple[int, ...]
    prediction_passed: bool
    absolute_purity_passed: bool
    relative_purity_passed: bool
    absolute_rank_passed: bool
    relative_rank_passed: bool
    comparisons: tuple[PaperLinearRandomHeldoutSeedComparison, ...]
    passed: bool


def evaluate_paper_linear_structure(
    matrix: np.ndarray,
) -> PaperLinearStructureMetrics:
    """Measure distance from identity and off-diagonal matrix mass."""

    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("predictor matrix must be square")
    if not np.isfinite(matrix).all():
        raise ValueError("predictor matrix must be finite")
    matrix_norm = max(float(np.linalg.norm(matrix)), 1e-12)
    identity = np.eye(matrix.shape[0], dtype=matrix.dtype)
    diagonal = np.diag(np.diag(matrix))
    return PaperLinearStructureMetrics(
        relative_identity_error=float(
            np.linalg.norm(matrix - identity) / matrix_norm
        ),
        off_diagonal_fraction=float(
            np.linalg.norm(matrix - diagonal) / matrix_norm
        ),
    )


def evaluate_paper_linear_paired_heldout(
    matrix: np.ndarray,
    online_embeddings: np.ndarray,
    target_embeddings: np.ndarray,
    labels: np.ndarray,
    config: PaperLinearPairedHeldoutEvaluationConfig,
) -> PaperLinearPairedHeldoutMetrics:
    """Compute frozen prediction, clustering, and rank held-out metrics."""

    config.validate()
    matrix = np.asarray(matrix, dtype=np.float64)
    online_embeddings = np.asarray(online_embeddings, dtype=np.float64)
    target_embeddings = np.asarray(target_embeddings, dtype=np.float64)
    labels = np.asarray(labels)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("predictor matrix must be square")
    if online_embeddings.ndim != 2 or target_embeddings.ndim != 2:
        raise ValueError("online and target embeddings must be two-dimensional")
    if online_embeddings.shape != target_embeddings.shape:
        raise ValueError("online and target embeddings must have the same shape")
    if online_embeddings.shape[1] != matrix.shape[0]:
        raise ValueError("embedding dimension must match predictor matrix")
    if online_embeddings.shape[0] < config.kmeans_clusters:
        raise ValueError("sample count must be at least kmeans_clusters")
    if labels.ndim != 1 or labels.shape[0] != online_embeddings.shape[0]:
        raise ValueError("labels must be one-dimensional and align with embeddings")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("labels must be integers")
    expected_labels = np.arange(config.kmeans_clusters)
    if not np.array_equal(np.unique(labels), expected_labels):
        raise ValueError("labels must cover every configured cluster exactly")
    if not (
        np.isfinite(matrix).all()
        and np.isfinite(online_embeddings).all()
        and np.isfinite(target_embeddings).all()
    ):
        raise ValueError("predictor matrix and embeddings must be finite")

    prediction = online_embeddings @ matrix.T
    residual_energy = float(np.sum((prediction - target_embeddings) ** 2))
    target_energy = max(float(np.sum(target_embeddings**2)), 1e-12)
    normalized_prediction_error = math.sqrt(residual_energy / target_energy)

    assignments = KMeans(
        n_clusters=config.kmeans_clusters,
        n_init=config.kmeans_n_init,
        random_state=config.kmeans_seed,
    ).fit_predict(online_embeddings)
    contingency = np.zeros(
        (config.kmeans_clusters, config.kmeans_clusters),
        dtype=np.int64,
    )
    for cluster_id, label in zip(assignments, labels, strict=True):
        contingency[cluster_id, label] += 1
    kmeans_purity = float(contingency.max(axis=1).sum() / labels.size)
    rows, columns = linear_sum_assignment(-contingency)
    kmeans_matched_accuracy = float(
        contingency[rows, columns].sum() / labels.size
    )
    test_embedding_std_mean, test_effective_rank = _embedding_spread(
        online_embeddings
    )
    return PaperLinearPairedHeldoutMetrics(
        normalized_prediction_error=normalized_prediction_error,
        kmeans_purity=kmeans_purity,
        kmeans_matched_accuracy=kmeans_matched_accuracy,
        test_embedding_std_mean=test_embedding_std_mean,
        test_effective_rank=test_effective_rank,
    )


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

    structure = evaluate_paper_linear_structure(matrix)
    matrix_norm = max(float(np.linalg.norm(matrix)), 1e-12)
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
        relative_identity_error=structure.relative_identity_error,
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


def _index_seed_summaries(
    summaries: list[PaperSeedSummary],
) -> tuple[dict[int, PaperSeedSummary], bool]:
    seeds = [summary.seed for summary in summaries]
    return {summary.seed: summary for summary in summaries}, len(seeds) == len(set(seeds))


def evaluate_paper_linear_random_comparison_gate(
    identity_summaries: list[PaperSeedSummary],
    random_summaries: list[PaperSeedSummary],
    random_matrices: dict[int, np.ndarray],
    sweep: PaperSeedSweepConfig,
    config: PaperLinearRandomComparisonGateConfig,
) -> PaperLinearRandomComparisonGateResult:
    """Evaluate paired predictive comparability and random-matrix structure."""

    sweep.validate()
    config.validate()
    expected_seeds = set(sweep.seeds)
    identity_by_seed, identity_unique = _index_seed_summaries(identity_summaries)
    random_by_seed, random_unique = _index_seed_summaries(random_summaries)
    all_seeds_present = (
        identity_unique
        and random_unique
        and set(identity_by_seed) == expected_seeds
        and set(random_by_seed) == expected_seeds
        and set(random_matrices) == expected_seeds
    )
    common_seeds = expected_seeds.intersection(
        identity_by_seed,
        random_by_seed,
        random_matrices,
    )

    comparisons: list[PaperLinearRandomSeedComparison] = []
    for seed in sorted(common_seeds):
        identity = identity_by_seed[seed]
        random = random_by_seed[seed]
        structure = evaluate_paper_linear_structure(random_matrices[seed])
        validation_improvement_ratio = random.validation_loss_ratio / max(
            identity.validation_loss_ratio,
            1e-12,
        )
        absolute_validation_loss_ratio = random.validation_loss / max(
            identity.validation_loss,
            1e-12,
        )
        values = (
            validation_improvement_ratio,
            absolute_validation_loss_ratio,
            structure.relative_identity_error,
            structure.off_diagonal_fraction,
        )
        passed = (
            all(math.isfinite(value) for value in values)
            and validation_improvement_ratio
            <= config.max_random_to_identity_validation_improvement_ratio
            and structure.relative_identity_error
            >= config.min_random_relative_identity_error
            and structure.off_diagonal_fraction
            >= config.min_random_off_diagonal_fraction
        )
        comparisons.append(
            PaperLinearRandomSeedComparison(
                seed=seed,
                validation_improvement_ratio=validation_improvement_ratio,
                absolute_validation_loss_ratio=absolute_validation_loss_ratio,
                relative_identity_error=structure.relative_identity_error,
                off_diagonal_fraction=structure.off_diagonal_fraction,
                passed=passed,
            )
        )

    if comparisons:
        worst_validation_improvement_ratio = max(
            comparison.validation_improvement_ratio
            for comparison in comparisons
        )
        worst_absolute_validation_loss_ratio = max(
            comparison.absolute_validation_loss_ratio
            for comparison in comparisons
        )
        minimum_relative_identity_error = min(
            comparison.relative_identity_error for comparison in comparisons
        )
        minimum_off_diagonal_fraction = min(
            comparison.off_diagonal_fraction for comparison in comparisons
        )
    else:
        worst_validation_improvement_ratio = math.inf
        worst_absolute_validation_loss_ratio = math.inf
        minimum_relative_identity_error = 0.0
        minimum_off_diagonal_fraction = 0.0

    all_finite = bool(comparisons) and all(
        all(
            math.isfinite(value)
            for value in (
                comparison.validation_improvement_ratio,
                comparison.absolute_validation_loss_ratio,
                comparison.relative_identity_error,
                comparison.off_diagonal_fraction,
            )
        )
        for comparison in comparisons
    )
    predictive_comparability_passed = (
        worst_validation_improvement_ratio
        <= config.max_random_to_identity_validation_improvement_ratio
    )
    non_identity_passed = (
        minimum_relative_identity_error
        >= config.min_random_relative_identity_error
    )
    density_passed = (
        minimum_off_diagonal_fraction
        >= config.min_random_off_diagonal_fraction
    )
    failed_seeds = tuple(
        sorted(
            (expected_seeds - common_seeds)
            | {
                comparison.seed
                for comparison in comparisons
                if not comparison.passed
            }
        )
    )
    return PaperLinearRandomComparisonGateResult(
        all_seeds_present=all_seeds_present,
        all_finite=all_finite,
        worst_validation_improvement_ratio=worst_validation_improvement_ratio,
        worst_absolute_validation_loss_ratio=worst_absolute_validation_loss_ratio,
        minimum_relative_identity_error=minimum_relative_identity_error,
        minimum_off_diagonal_fraction=minimum_off_diagonal_fraction,
        failed_seeds=failed_seeds,
        predictive_comparability_passed=predictive_comparability_passed,
        non_identity_passed=non_identity_passed,
        density_passed=density_passed,
        comparisons=tuple(comparisons),
        passed=(
            all_seeds_present
            and all_finite
            and not failed_seeds
            and predictive_comparability_passed
            and non_identity_passed
            and density_passed
        ),
    )


def _paired_heldout_metrics_are_finite(
    metrics: PaperLinearPairedHeldoutMetrics,
) -> bool:
    return all(
        math.isfinite(value)
        for value in (
            metrics.normalized_prediction_error,
            metrics.kmeans_purity,
            metrics.kmeans_matched_accuracy,
            metrics.test_embedding_std_mean,
            metrics.test_effective_rank,
        )
    )


def evaluate_paper_linear_random_heldout_gate(
    identity_metrics_by_seed: dict[int, PaperLinearPairedHeldoutMetrics],
    random_metrics_by_seed: dict[int, PaperLinearPairedHeldoutMetrics],
    sweep: PaperSeedSweepConfig,
    config: PaperLinearRandomHeldoutGateConfig,
) -> PaperLinearRandomHeldoutGateResult:
    """Evaluate every frozen paired held-out criterion for every seed."""

    sweep.validate()
    config.validate()
    expected_seeds = set(sweep.seeds)
    all_seeds_present = (
        set(identity_metrics_by_seed) == expected_seeds
        and set(random_metrics_by_seed) == expected_seeds
    )
    common_seeds = expected_seeds.intersection(
        identity_metrics_by_seed,
        random_metrics_by_seed,
    )

    comparisons: list[PaperLinearRandomHeldoutSeedComparison] = []
    for seed in sorted(common_seeds):
        identity = identity_metrics_by_seed[seed]
        random = random_metrics_by_seed[seed]
        prediction_error_ratio = random.normalized_prediction_error / max(
            identity.normalized_prediction_error,
            1e-12,
        )
        kmeans_purity_ratio = random.kmeans_purity / max(
            identity.kmeans_purity,
            1e-12,
        )
        effective_rank_ratio = random.test_effective_rank / max(
            identity.test_effective_rank,
            1e-12,
        )
        finite = _paired_heldout_metrics_are_finite(
            identity
        ) and _paired_heldout_metrics_are_finite(random)
        passed = (
            finite
            and prediction_error_ratio
            <= config.max_random_to_identity_prediction_error_ratio
            and random.kmeans_purity >= config.min_random_kmeans_purity
            and kmeans_purity_ratio
            >= config.min_random_to_identity_kmeans_purity_ratio
            and random.test_effective_rank
            >= config.min_random_test_effective_rank
            and effective_rank_ratio
            >= config.min_random_to_identity_effective_rank_ratio
        )
        comparisons.append(
            PaperLinearRandomHeldoutSeedComparison(
                seed=seed,
                identity_prediction_error=identity.normalized_prediction_error,
                random_prediction_error=random.normalized_prediction_error,
                prediction_error_ratio=prediction_error_ratio,
                identity_kmeans_purity=identity.kmeans_purity,
                random_kmeans_purity=random.kmeans_purity,
                kmeans_purity_ratio=kmeans_purity_ratio,
                identity_effective_rank=identity.test_effective_rank,
                random_effective_rank=random.test_effective_rank,
                effective_rank_ratio=effective_rank_ratio,
                random_kmeans_matched_accuracy=random.kmeans_matched_accuracy,
                passed=passed,
            )
        )

    if comparisons:
        worst_prediction_error_ratio = max(
            comparison.prediction_error_ratio for comparison in comparisons
        )
        minimum_random_kmeans_purity = min(
            comparison.random_kmeans_purity for comparison in comparisons
        )
        minimum_kmeans_purity_ratio = min(
            comparison.kmeans_purity_ratio for comparison in comparisons
        )
        minimum_random_test_effective_rank = min(
            comparison.random_effective_rank for comparison in comparisons
        )
        minimum_effective_rank_ratio = min(
            comparison.effective_rank_ratio for comparison in comparisons
        )
    else:
        worst_prediction_error_ratio = math.inf
        minimum_random_kmeans_purity = 0.0
        minimum_kmeans_purity_ratio = 0.0
        minimum_random_test_effective_rank = 0.0
        minimum_effective_rank_ratio = 0.0

    all_finite = bool(comparisons) and all(
        all(
            math.isfinite(value)
            for value in (
                comparison.identity_prediction_error,
                comparison.random_prediction_error,
                comparison.prediction_error_ratio,
                comparison.identity_kmeans_purity,
                comparison.random_kmeans_purity,
                comparison.kmeans_purity_ratio,
                comparison.identity_effective_rank,
                comparison.random_effective_rank,
                comparison.effective_rank_ratio,
                comparison.random_kmeans_matched_accuracy,
            )
        )
        for comparison in comparisons
    )
    prediction_passed = (
        worst_prediction_error_ratio
        <= config.max_random_to_identity_prediction_error_ratio
    )
    absolute_purity_passed = (
        minimum_random_kmeans_purity >= config.min_random_kmeans_purity
    )
    relative_purity_passed = (
        minimum_kmeans_purity_ratio
        >= config.min_random_to_identity_kmeans_purity_ratio
    )
    absolute_rank_passed = (
        minimum_random_test_effective_rank
        >= config.min_random_test_effective_rank
    )
    relative_rank_passed = (
        minimum_effective_rank_ratio
        >= config.min_random_to_identity_effective_rank_ratio
    )
    failed_seeds = tuple(
        sorted(
            (expected_seeds - common_seeds)
            | {
                comparison.seed
                for comparison in comparisons
                if not comparison.passed
            }
        )
    )
    return PaperLinearRandomHeldoutGateResult(
        all_seeds_present=all_seeds_present,
        all_finite=all_finite,
        worst_prediction_error_ratio=worst_prediction_error_ratio,
        minimum_random_kmeans_purity=minimum_random_kmeans_purity,
        minimum_kmeans_purity_ratio=minimum_kmeans_purity_ratio,
        minimum_random_test_effective_rank=minimum_random_test_effective_rank,
        minimum_effective_rank_ratio=minimum_effective_rank_ratio,
        failed_seeds=failed_seeds,
        prediction_passed=prediction_passed,
        absolute_purity_passed=absolute_purity_passed,
        relative_purity_passed=relative_purity_passed,
        absolute_rank_passed=absolute_rank_passed,
        relative_rank_passed=relative_rank_passed,
        comparisons=tuple(comparisons),
        passed=(
            all_seeds_present
            and all_finite
            and not failed_seeds
            and prediction_passed
            and absolute_purity_passed
            and relative_purity_passed
            and absolute_rank_passed
            and relative_rank_passed
        ),
    )

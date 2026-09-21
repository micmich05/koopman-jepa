from __future__ import annotations

from typing import Any

import numpy as np

from .analysis import covariance_statistics, linear_probe_accuracy
from .koopman import (
    PhaseDynamics,
    centered_phase_indicators,
    expected_active_spectrum,
    expected_phase_operator,
    fit_linear_operator,
    match_eigenvalues,
)


def _complex_list(values: np.ndarray) -> list[dict[str, float]]:
    ordered = sorted(values, key=lambda value: (float(value.real), float(value.imag)))
    return [{"real": float(value.real), "imag": float(value.imag)} for value in ordered]


def evaluate_phase_representation(
    train_embeddings: np.ndarray,
    train_phases: np.ndarray,
    validation_embeddings: np.ndarray,
    validation_phases: np.ndarray,
    predictor_matrix: np.ndarray,
    dynamics: PhaseDynamics,
    seed: int,
    rank_tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Measure phase recovery and Koopman closure in a learned latent space."""

    train_embeddings = np.asarray(train_embeddings, dtype=np.float64)
    validation_embeddings = np.asarray(validation_embeddings, dtype=np.float64)
    train_phases = np.asarray(train_phases, dtype=np.int64)
    validation_phases = np.asarray(validation_phases, dtype=np.int64)
    predictor_matrix = np.asarray(predictor_matrix, dtype=np.float64)
    if train_embeddings.ndim != 2 or validation_embeddings.ndim != 2:
        raise ValueError("embeddings must be two-dimensional")
    if train_embeddings.shape[1] != validation_embeddings.shape[1]:
        raise ValueError("train and validation latent dimensions must match")
    latent_dim = train_embeddings.shape[1]
    if predictor_matrix.shape != (latent_dim, latent_dim):
        raise ValueError("predictor matrix must match the latent dimension")
    if train_phases.shape != (train_embeddings.shape[0],):
        raise ValueError("train phases must align with train embeddings")
    if validation_phases.shape != (validation_embeddings.shape[0],):
        raise ValueError("validation phases must align with validation embeddings")
    num_phases = int(max(train_phases.max(), validation_phases.max()) + 1)
    expected_labels = np.arange(num_phases)
    if not np.array_equal(np.unique(train_phases), expected_labels):
        raise ValueError("train phases must cover every phase")
    if not np.array_equal(np.unique(validation_phases), expected_labels):
        raise ValueError("validation phases must cover every phase")

    train_mean = train_embeddings.mean(axis=0, keepdims=True)
    train_centered = train_embeddings - train_mean
    validation_centered = validation_embeddings - train_mean
    train_features = centered_phase_indicators(train_phases, num_phases)
    validation_features = centered_phase_indicators(validation_phases, num_phases)
    alignment = np.linalg.lstsq(train_features, train_centered, rcond=None)[0]
    validation_reconstruction = validation_features @ alignment
    alignment_error = np.linalg.norm(validation_centered - validation_reconstruction) / max(
        np.linalg.norm(validation_centered),
        1e-15,
    )

    alignment_columns = alignment.T
    left_vectors, singular_values, _ = np.linalg.svd(
        alignment_columns,
        full_matrices=False,
    )
    if singular_values.size == 0 or singular_values[0] <= 1e-12:
        active_rank = 0
        basis = np.zeros((latent_dim, 0), dtype=np.float64)
    else:
        active_rank = int(np.sum(singular_values > rank_tolerance * singular_values[0]))
        basis = left_vectors[:, :active_rank]

    expected_operator = expected_phase_operator(dynamics, num_phases)
    expected_action = alignment_columns @ expected_operator
    estimated_action = predictor_matrix @ alignment_columns
    if np.linalg.norm(expected_action) <= 1e-12:
        intertwining_error = np.linalg.norm(estimated_action) / max(
            np.linalg.norm(alignment_columns),
            1e-15,
        )
    else:
        intertwining_error = np.linalg.norm(estimated_action - expected_action) / np.linalg.norm(
            expected_action
        )

    if active_rank == 0:
        active_invariance_error = None
        estimated_spectrum = np.array([], dtype=np.complex128)
        spectral_mean_error = None
        spectral_max_error = None
    else:
        reduced_predictor = basis.T @ predictor_matrix @ basis
        estimated_spectrum = np.linalg.eigvals(reduced_predictor)
        residual = (np.eye(latent_dim) - basis @ basis.T) @ predictor_matrix @ basis
        active_invariance_error = float(
            np.linalg.norm(residual) / max(np.linalg.norm(predictor_matrix @ basis), 1e-15)
        )
        if active_rank == num_phases - 1:
            spectral_match = match_eigenvalues(
                estimated_spectrum,
                expected_active_spectrum(dynamics, num_phases),
            )
            spectral_mean_error = spectral_match["mean_absolute_error"]
            spectral_max_error = spectral_match["max_absolute_error"]
        else:
            spectral_mean_error = None
            spectral_max_error = None

    metrics: dict[str, Any] = {}
    metrics.update(covariance_statistics(validation_embeddings))
    metrics.update(
        {
            "linear_probe_accuracy": linear_probe_accuracy(
                train_embeddings,
                train_phases,
                validation_embeddings,
                validation_phases,
                seed,
            ),
            "phase_alignment_error": float(alignment_error),
            "active_rank": active_rank,
            "alignment_singular_values": singular_values.tolist(),
            "intertwining_error": float(intertwining_error),
            "active_invariance_error": active_invariance_error,
            "spectral_mean_absolute_error": spectral_mean_error,
            "spectral_max_absolute_error": spectral_max_error,
            "active_eigenvalues": _complex_list(estimated_spectrum),
        }
    )
    return metrics


def _relative_error(estimate: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.linalg.norm(estimate - reference) / max(np.linalg.norm(reference), 1e-15)
    )


def _phase_alignment_columns(
    embeddings: np.ndarray,
    phases: np.ndarray,
    num_phases: int,
) -> tuple[np.ndarray, np.ndarray]:
    mean = embeddings.mean(axis=0, keepdims=True)
    centered = embeddings - mean
    features = centered_phase_indicators(phases, num_phases)
    alignment = np.linalg.lstsq(features, centered, rcond=None)[0].T
    return alignment, centered


def evaluate_phase_operator_diagnostics(
    current_online: np.ndarray,
    future_online: np.ndarray,
    future_target: np.ndarray,
    current_phases: np.ndarray,
    future_phases: np.ndarray,
    predictor_matrix: np.ndarray,
    dynamics: PhaseDynamics,
) -> dict[str, Any]:
    """Separate online endomorphism quality from the online-to-EMA training map."""

    current_online = np.asarray(current_online, dtype=np.float64)
    future_online = np.asarray(future_online, dtype=np.float64)
    future_target = np.asarray(future_target, dtype=np.float64)
    current_phases = np.asarray(current_phases, dtype=np.int64)
    future_phases = np.asarray(future_phases, dtype=np.int64)
    predictor_matrix = np.asarray(predictor_matrix, dtype=np.float64)
    if not (
        current_online.shape == future_online.shape == future_target.shape
        and current_online.ndim == 2
    ):
        raise ValueError("current, online future, and target future embeddings must align")
    if current_phases.shape != future_phases.shape or current_phases.shape != (
        current_online.shape[0],
    ):
        raise ValueError("phase labels must align with embeddings")
    latent_dim = current_online.shape[1]
    if predictor_matrix.shape != (latent_dim, latent_dim):
        raise ValueError("predictor matrix must match the latent dimension")
    num_phases = int(max(current_phases.max(), future_phases.max()) + 1)

    current_alignment, current_centered = _phase_alignment_columns(
        current_online,
        current_phases,
        num_phases,
    )
    future_online_alignment, future_online_centered = _phase_alignment_columns(
        future_online,
        future_phases,
        num_phases,
    )
    future_target_alignment, future_target_centered = _phase_alignment_columns(
        future_target,
        future_phases,
        num_phases,
    )
    expected_operator = expected_phase_operator(dynamics, num_phases)
    predictor_action = predictor_matrix @ current_alignment
    expected_online_action = future_online_alignment @ expected_operator
    expected_target_action = future_target_alignment @ expected_operator

    posthoc_online = fit_linear_operator(current_centered, future_online_centered)
    posthoc_target = fit_linear_operator(current_centered, future_target_centered)
    expected_spectrum = expected_active_spectrum(dynamics, num_phases)
    if latent_dim == num_phases - 1:
        online_spectral_match = match_eigenvalues(
            np.linalg.eigvals(posthoc_online),
            expected_spectrum,
        )
        target_spectral_match = match_eigenvalues(
            np.linalg.eigvals(posthoc_target),
            expected_spectrum,
        )
    else:
        online_spectral_match = {"mean_absolute_error": None, "max_absolute_error": None}
        target_spectral_match = {"mean_absolute_error": None, "max_absolute_error": None}

    return {
        "online_encoder_phase_basis_error": _relative_error(
            future_online_alignment,
            current_alignment,
        ),
        "online_target_phase_basis_error": _relative_error(
            future_target_alignment,
            future_online_alignment,
        ),
        "trained_online_endomorphism_error": _relative_error(
            predictor_action,
            expected_online_action,
        ),
        "trained_cross_encoder_error": _relative_error(
            predictor_action,
            expected_target_action,
        ),
        "predictor_vs_posthoc_online_error": _relative_error(
            predictor_matrix,
            posthoc_online,
        ),
        "predictor_vs_posthoc_target_error": _relative_error(
            predictor_matrix,
            posthoc_target,
        ),
        "posthoc_online_prediction_error": _relative_error(
            current_centered @ posthoc_online.T,
            future_online_centered,
        ),
        "posthoc_target_prediction_error": _relative_error(
            current_centered @ posthoc_target.T,
            future_target_centered,
        ),
        "posthoc_online_spectral_mean_error": online_spectral_match[
            "mean_absolute_error"
        ],
        "posthoc_online_spectral_max_error": online_spectral_match[
            "max_absolute_error"
        ],
        "posthoc_target_spectral_mean_error": target_spectral_match[
            "mean_absolute_error"
        ],
        "posthoc_target_spectral_max_error": target_spectral_match[
            "max_absolute_error"
        ],
        "posthoc_online_operator": posthoc_online,
        "posthoc_target_operator": posthoc_target,
    }

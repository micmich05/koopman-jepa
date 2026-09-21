from __future__ import annotations

from typing import Any

import numpy as np

from .analysis import covariance_statistics, linear_probe_accuracy
from .koopman import (
    PhaseDynamics,
    centered_phase_indicators,
    expected_active_spectrum,
    expected_phase_operator,
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

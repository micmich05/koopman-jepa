from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .analysis import covariance_statistics, linear_probe_accuracy
from .koopman import (
    PhaseDynamics,
    centered_phase_indicators,
    decay_phase_operator,
    expected_active_spectrum,
    expected_decay_active_spectrum,
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


def evaluate_phase_operator_candidates(
    embeddings: np.ndarray,
    phases: np.ndarray,
    predictor_matrix: np.ndarray,
    candidates: Sequence[PhaseDynamics],
    rollout_horizons: Sequence[int] = (1, 2, 3, 4, 8),
    rank_tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Rank candidate phase dynamics by their action and active spectrum."""

    embeddings = np.asarray(embeddings, dtype=np.float64)
    phases = np.asarray(phases, dtype=np.int64)
    predictor_matrix = np.asarray(predictor_matrix, dtype=np.float64)
    candidate_names = tuple(candidates)
    horizons = tuple(int(horizon) for horizon in rollout_horizons)
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be two-dimensional")
    if phases.shape != (embeddings.shape[0],):
        raise ValueError("phases must align with embeddings")
    if predictor_matrix.shape != (embeddings.shape[1], embeddings.shape[1]):
        raise ValueError("predictor matrix must match the latent dimension")
    if not candidate_names or len(set(candidate_names)) != len(candidate_names):
        raise ValueError("candidates must be non-empty and unique")
    if any(name not in {"static", "cyclic", "independent"} for name in candidate_names):
        raise ValueError("unknown candidate dynamics")
    if not horizons or any(horizon < 1 for horizon in horizons):
        raise ValueError("rollout horizons must be positive")
    if rank_tolerance <= 0.0:
        raise ValueError("rank_tolerance must be positive")

    num_phases = int(phases.max() + 1)
    if not np.array_equal(np.unique(phases), np.arange(num_phases)):
        raise ValueError("phases must cover every phase")
    alignment, _ = _phase_alignment_columns(embeddings, phases, num_phases)
    alignment_norm = max(np.linalg.norm(alignment), 1e-15)
    left_vectors, singular_values, _ = np.linalg.svd(alignment, full_matrices=False)
    if singular_values.size == 0 or singular_values[0] <= 1e-12:
        active_rank = 0
        basis = np.zeros((embeddings.shape[1], 0), dtype=np.float64)
    else:
        active_rank = int(np.sum(singular_values > rank_tolerance * singular_values[0]))
        basis = left_vectors[:, :active_rank]

    action_errors: dict[str, float] = {}
    rollout_errors: dict[str, dict[int, float]] = {}
    for candidate in candidate_names:
        expected = expected_phase_operator(candidate, num_phases)
        action_errors[candidate] = float(
            np.linalg.norm(predictor_matrix @ alignment - alignment @ expected)
            / alignment_norm
        )
        rollout_errors[candidate] = {
            horizon: float(
                np.linalg.norm(
                    np.linalg.matrix_power(predictor_matrix, horizon) @ alignment
                    - alignment @ np.linalg.matrix_power(expected, horizon)
                )
                / alignment_norm
            )
            for horizon in horizons
        }

    predicted_action = min(action_errors, key=action_errors.__getitem__)
    ordered_action_errors = sorted(action_errors.values())
    action_margin = (
        float(ordered_action_errors[1] - ordered_action_errors[0])
        if len(ordered_action_errors) > 1
        else None
    )

    spectral_mean_errors: dict[str, float] | None = None
    spectral_max_errors: dict[str, float] | None = None
    predicted_spectrum: str | None = None
    spectrum_margin: float | None = None
    active_eigenvalues = np.array([], dtype=np.complex128)
    if active_rank == num_phases - 1:
        reduced_predictor = basis.T @ predictor_matrix @ basis
        active_eigenvalues = np.linalg.eigvals(reduced_predictor)
        spectral_matches = {
            candidate: match_eigenvalues(
                active_eigenvalues,
                expected_active_spectrum(candidate, num_phases),
            )
            for candidate in candidate_names
        }
        spectral_mean_errors = {
            candidate: float(match["mean_absolute_error"])
            for candidate, match in spectral_matches.items()
        }
        spectral_max_errors = {
            candidate: float(match["max_absolute_error"])
            for candidate, match in spectral_matches.items()
        }
        predicted_spectrum = min(
            spectral_mean_errors,
            key=spectral_mean_errors.__getitem__,
        )
        ordered_spectral_errors = sorted(spectral_mean_errors.values())
        if len(ordered_spectral_errors) > 1:
            spectrum_margin = float(
                ordered_spectral_errors[1] - ordered_spectral_errors[0]
            )

    return {
        "active_rank": active_rank,
        "alignment_singular_values": singular_values.tolist(),
        "action_errors": action_errors,
        "predicted_action_dynamics": predicted_action,
        "action_margin": action_margin,
        "rollout_errors": rollout_errors,
        "active_eigenvalues": _complex_list(active_eigenvalues),
        "spectral_mean_errors": spectral_mean_errors,
        "spectral_max_errors": spectral_max_errors,
        "predicted_spectral_dynamics": predicted_spectrum,
        "spectrum_margin": spectrum_margin,
    }


def evaluate_decay_operator_candidates(
    embeddings: np.ndarray,
    phases: np.ndarray,
    predictor_matrix: np.ndarray,
    candidate_rhos: Sequence[float],
    rollout_horizons: Sequence[int] = (1, 2, 3, 4, 8),
    rank_tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Rank damped cyclic operators by their action and active spectrum."""

    embeddings = np.asarray(embeddings, dtype=np.float64)
    phases = np.asarray(phases, dtype=np.int64)
    predictor_matrix = np.asarray(predictor_matrix, dtype=np.float64)
    rhos = tuple(float(rho) for rho in candidate_rhos)
    horizons = tuple(int(horizon) for horizon in rollout_horizons)
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be two-dimensional")
    if phases.shape != (embeddings.shape[0],):
        raise ValueError("phases must align with embeddings")
    if predictor_matrix.shape != (embeddings.shape[1], embeddings.shape[1]):
        raise ValueError("predictor matrix must match the latent dimension")
    if not rhos or len(set(rhos)) != len(rhos):
        raise ValueError("candidate_rhos must be non-empty and unique")
    if any(not np.isfinite(rho) or not 0.0 <= rho <= 1.0 for rho in rhos):
        raise ValueError("candidate rhos must lie in [0, 1]")
    if not horizons or any(horizon < 1 for horizon in horizons):
        raise ValueError("rollout horizons must be positive")
    if rank_tolerance <= 0.0:
        raise ValueError("rank_tolerance must be positive")

    num_phases = int(phases.max() + 1)
    if not np.array_equal(np.unique(phases), np.arange(num_phases)):
        raise ValueError("phases must cover every phase")
    alignment, _ = _phase_alignment_columns(embeddings, phases, num_phases)
    alignment_norm = max(np.linalg.norm(alignment), 1e-15)
    left_vectors, singular_values, _ = np.linalg.svd(alignment, full_matrices=False)
    if singular_values.size == 0 or singular_values[0] <= 1e-12:
        active_rank = 0
        basis = np.zeros((embeddings.shape[1], 0), dtype=np.float64)
    else:
        active_rank = int(np.sum(singular_values > rank_tolerance * singular_values[0]))
        basis = left_vectors[:, :active_rank]

    expected_operators = {
        rho: decay_phase_operator(rho, num_phases)
        for rho in rhos
    }
    action_errors = {
        rho: float(
            np.linalg.norm(predictor_matrix @ alignment - alignment @ expected)
            / alignment_norm
        )
        for rho, expected in expected_operators.items()
    }
    rollout_errors = {
        rho: {
            horizon: float(
                np.linalg.norm(
                    np.linalg.matrix_power(predictor_matrix, horizon) @ alignment
                    - alignment @ np.linalg.matrix_power(expected, horizon)
                )
                / alignment_norm
            )
            for horizon in horizons
        }
        for rho, expected in expected_operators.items()
    }
    predicted_action_rho = min(action_errors, key=action_errors.__getitem__)
    ordered_action_errors = sorted(action_errors.values())
    action_margin = (
        float(ordered_action_errors[1] - ordered_action_errors[0])
        if len(ordered_action_errors) > 1
        else None
    )

    spectral_mean_errors: dict[float, float] | None = None
    spectral_max_errors: dict[float, float] | None = None
    predicted_spectral_rho: float | None = None
    spectrum_margin: float | None = None
    mean_eigenvalue_modulus: float | None = None
    active_eigenvalues = np.array([], dtype=np.complex128)
    if active_rank == num_phases - 1:
        reduced_predictor = basis.T @ predictor_matrix @ basis
        active_eigenvalues = np.linalg.eigvals(reduced_predictor)
        mean_eigenvalue_modulus = float(np.mean(np.abs(active_eigenvalues)))
        spectral_matches = {
            rho: match_eigenvalues(
                active_eigenvalues,
                expected_decay_active_spectrum(rho, num_phases),
            )
            for rho in rhos
        }
        spectral_mean_errors = {
            rho: float(match["mean_absolute_error"])
            for rho, match in spectral_matches.items()
        }
        spectral_max_errors = {
            rho: float(match["max_absolute_error"])
            for rho, match in spectral_matches.items()
        }
        predicted_spectral_rho = min(
            spectral_mean_errors,
            key=spectral_mean_errors.__getitem__,
        )
        ordered_spectral_errors = sorted(spectral_mean_errors.values())
        if len(ordered_spectral_errors) > 1:
            spectrum_margin = float(
                ordered_spectral_errors[1] - ordered_spectral_errors[0]
            )

    return {
        "active_rank": active_rank,
        "alignment_singular_values": singular_values.tolist(),
        "action_errors": action_errors,
        "predicted_action_rho": predicted_action_rho,
        "action_margin": action_margin,
        "rollout_errors": rollout_errors,
        "active_eigenvalues": _complex_list(active_eigenvalues),
        "mean_eigenvalue_modulus": mean_eigenvalue_modulus,
        "spectral_mean_errors": spectral_mean_errors,
        "spectral_max_errors": spectral_max_errors,
        "predicted_spectral_rho": predicted_spectral_rho,
        "spectrum_margin": spectrum_margin,
    }


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

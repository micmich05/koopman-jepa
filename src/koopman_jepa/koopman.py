from __future__ import annotations

from typing import Any, Literal

import numpy as np
from scipy.optimize import linear_sum_assignment

PhaseDynamics = Literal["static", "cyclic", "independent"]


def _relative_or_absolute_error(
    estimate: np.ndarray,
    reference: np.ndarray,
    zero_tolerance: float = 1e-12,
) -> float:
    """Use relative error unless the reference is mathematically zero."""

    absolute_error = np.linalg.norm(estimate - reference)
    reference_norm = np.linalg.norm(reference)
    if reference_norm <= zero_tolerance:
        return float(absolute_error)
    return float(absolute_error / reference_norm)


def centered_phase_indicators(
    phases: np.ndarray,
    num_phases: int = 4,
) -> np.ndarray:
    """Encode discrete phases as one-hot indicators with the constant mode removed."""

    phases = np.asarray(phases)
    if phases.ndim != 1:
        raise ValueError("phases must be one-dimensional")
    if phases.size == 0:
        raise ValueError("phases must not be empty")
    if not np.issubdtype(phases.dtype, np.integer):
        raise ValueError("phases must be integers")
    if num_phases < 2:
        raise ValueError("num_phases must be at least two")
    if np.any(phases < 0) or np.any(phases >= num_phases):
        raise ValueError("phase values must lie in [0, num_phases)")

    indicators = np.eye(num_phases, dtype=np.float64)[phases]
    return indicators - 1.0 / num_phases


def cyclic_phase_operator(num_phases: int = 4) -> np.ndarray:
    """Return the column-vector operator mapping each phase to its successor."""

    if num_phases < 2:
        raise ValueError("num_phases must be at least two")
    operator = np.zeros((num_phases, num_phases), dtype=np.float64)
    for phase in range(num_phases):
        operator[(phase + 1) % num_phases, phase] = 1.0
    return operator


def balanced_phase_transitions(
    dynamics: PhaseDynamics,
    repeats_per_transition: int = 16,
    num_phases: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Build equally sized transition tables with uniform source/future marginals."""

    if dynamics not in {"static", "cyclic", "independent"}:
        raise ValueError(f"unknown phase dynamics: {dynamics}")
    if repeats_per_transition < 1:
        raise ValueError("repeats_per_transition must be positive")
    if num_phases < 2:
        raise ValueError("num_phases must be at least two")

    if dynamics == "independent":
        pairs = np.array(
            [
                (current, future)
                for current in range(num_phases)
                for future in range(num_phases)
            ],
            dtype=np.int64,
        )
        pairs = np.tile(pairs, (repeats_per_transition, 1))
        return pairs[:, 0], pairs[:, 1]

    repeats_per_phase = num_phases * repeats_per_transition
    current = np.repeat(np.arange(num_phases), repeats_per_phase)
    if dynamics == "static":
        future = current.copy()
    else:
        future = (current + 1) % num_phases
    return current, future


def expected_phase_operator(
    dynamics: PhaseDynamics,
    num_phases: int = 4,
) -> np.ndarray:
    """Return the conditional-expectation operator for centered phase indicators."""

    if dynamics == "static":
        return np.eye(num_phases, dtype=np.float64)
    if dynamics == "cyclic":
        return cyclic_phase_operator(num_phases)
    if dynamics == "independent":
        return np.zeros((num_phases, num_phases), dtype=np.float64)
    raise ValueError(f"unknown phase dynamics: {dynamics}")


def expected_active_spectrum(
    dynamics: PhaseDynamics,
    num_phases: int = 4,
) -> np.ndarray:
    """Return the expected spectrum after removing the constant phase mode."""

    if num_phases < 2:
        raise ValueError("num_phases must be at least two")
    if dynamics == "static":
        return np.ones(num_phases - 1, dtype=np.complex128)
    if dynamics == "cyclic":
        indices = np.arange(1, num_phases)
        return np.exp(2.0j * np.pi * indices / num_phases)
    if dynamics == "independent":
        return np.zeros(num_phases - 1, dtype=np.complex128)
    raise ValueError(f"unknown phase dynamics: {dynamics}")


def fit_linear_operator(
    current: np.ndarray,
    future: np.ndarray,
) -> np.ndarray:
    """Fit the column-vector operator M in z_future = M z_current."""

    current = np.asarray(current, dtype=np.float64)
    future = np.asarray(future, dtype=np.float64)
    if current.ndim != 2 or future.ndim != 2:
        raise ValueError("current and future samples must be two-dimensional")
    if current.shape != future.shape:
        raise ValueError("current and future samples must have the same shape")
    if current.shape[0] == 0 or current.shape[1] == 0:
        raise ValueError("sample and feature dimensions must be non-empty")
    if not np.isfinite(current).all() or not np.isfinite(future).all():
        raise ValueError("current and future samples must be finite")

    row_operator, *_ = np.linalg.lstsq(current, future, rcond=None)
    return row_operator.T


def sample_span_basis(
    samples: np.ndarray,
    relative_tolerance: float = 1e-10,
) -> np.ndarray:
    """Return an orthonormal feature-space basis for the span of row samples."""

    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 2:
        raise ValueError("samples must be two-dimensional")
    if samples.shape[0] == 0 or samples.shape[1] == 0:
        raise ValueError("sample and feature dimensions must be non-empty")
    if not np.isfinite(samples).all():
        raise ValueError("samples must be finite")
    if relative_tolerance <= 0.0:
        raise ValueError("relative_tolerance must be positive")

    _, singular_values, right_vectors = np.linalg.svd(samples, full_matrices=False)
    if singular_values.size == 0 or singular_values[0] <= 1e-15:
        return np.zeros((samples.shape[1], 0), dtype=np.float64)
    rank = int(np.sum(singular_values > relative_tolerance * singular_values[0]))
    return right_vectors[:rank].T


def restrict_operator(operator: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Represent an ambient column-vector operator in an orthonormal basis."""

    operator = np.asarray(operator, dtype=np.float64)
    basis = np.asarray(basis, dtype=np.float64)
    if operator.ndim != 2 or operator.shape[0] != operator.shape[1]:
        raise ValueError("operator must be square")
    if basis.ndim != 2 or basis.shape[0] != operator.shape[0]:
        raise ValueError("basis must align with the operator dimension")
    if not np.isfinite(operator).all() or not np.isfinite(basis).all():
        raise ValueError("operator and basis must be finite")
    if basis.shape[1] == 0:
        return np.zeros((0, 0), dtype=np.float64)
    gram = basis.T @ basis
    if not np.allclose(gram, np.eye(basis.shape[1]), atol=1e-10):
        raise ValueError("basis columns must be orthonormal")
    return basis.T @ operator @ basis


def match_eigenvalues(
    estimated: np.ndarray,
    expected: np.ndarray,
) -> dict[str, Any]:
    """Match two spectra with minimum total absolute error."""

    estimated = np.asarray(estimated, dtype=np.complex128)
    expected = np.asarray(expected, dtype=np.complex128)
    if estimated.ndim != 1 or expected.ndim != 1:
        raise ValueError("spectra must be one-dimensional")
    if estimated.size == 0 or estimated.shape != expected.shape:
        raise ValueError("spectra must have the same non-zero length")
    if not np.isfinite(estimated).all() or not np.isfinite(expected).all():
        raise ValueError("spectra must be finite")

    cost = np.abs(estimated[:, None] - expected[None, :])
    estimated_indices, expected_indices = linear_sum_assignment(cost)
    errors = cost[estimated_indices, expected_indices]
    return {
        "estimated": estimated[estimated_indices],
        "expected": expected[expected_indices],
        "absolute_errors": errors,
        "mean_absolute_error": float(errors.mean()),
        "max_absolute_error": float(errors.max()),
    }


def left_eigendecomposition(
    operator: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return eigenvalues and normalized left eigenvectors as matrix rows."""

    operator = np.asarray(operator, dtype=np.float64)
    if operator.ndim != 2 or operator.shape[0] != operator.shape[1]:
        raise ValueError("operator must be square")
    if not np.isfinite(operator).all():
        raise ValueError("operator must be finite")

    eigenvalues, column_vectors = np.linalg.eig(operator.T)
    left_vectors = column_vectors.T.astype(np.complex128)
    norms = np.linalg.norm(left_vectors, axis=1, keepdims=True)
    left_vectors = left_vectors / np.maximum(norms, 1e-15)
    return eigenvalues.astype(np.complex128), left_vectors


def linear_rollout(
    operator: np.ndarray,
    initial: np.ndarray,
    steps: int,
) -> np.ndarray:
    """Roll a column-vector linear system forward, including the initial state."""

    operator = np.asarray(operator, dtype=np.float64)
    initial = np.asarray(initial, dtype=np.float64)
    if operator.ndim != 2 or operator.shape[0] != operator.shape[1]:
        raise ValueError("operator must be square")
    if initial.ndim != 1 or initial.shape[0] != operator.shape[0]:
        raise ValueError("initial state must align with the operator dimension")
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if not np.isfinite(operator).all() or not np.isfinite(initial).all():
        raise ValueError("operator and initial state must be finite")

    states = [initial.copy()]
    for _ in range(steps):
        states.append(operator @ states[-1])
    return np.stack(states)


def evaluate_phase_dynamics_oracle(
    dynamics: PhaseDynamics,
    repeats_per_transition: int = 16,
    num_phases: int = 4,
) -> dict[str, Any]:
    """Fit and evaluate an oracle operator for one balanced phase dynamics."""

    current_phases, future_phases = balanced_phase_transitions(
        dynamics,
        repeats_per_transition,
        num_phases,
    )
    current = centered_phase_indicators(current_phases, num_phases)
    future = centered_phase_indicators(future_phases, num_phases)
    fitted = fit_linear_operator(current, future)
    expected = expected_phase_operator(dynamics, num_phases)
    basis = sample_span_basis(current)
    fitted_active = restrict_operator(fitted, basis)
    expected_active = restrict_operator(expected, basis)
    estimated_spectrum = np.linalg.eigvals(fitted_active)
    expected_spectrum = expected_active_spectrum(dynamics, num_phases)
    spectral_match = match_eigenvalues(estimated_spectrum, expected_spectrum)

    sample_prediction_error = np.linalg.norm(current @ fitted.T - future) / max(
        np.linalg.norm(future),
        1e-15,
    )
    phase_features = centered_phase_indicators(np.arange(num_phases), num_phases)
    empirical_conditional_means = np.stack(
        [future[current_phases == phase].mean(axis=0) for phase in range(num_phases)]
    )
    fitted_conditional_means = phase_features @ fitted.T
    conditional_mean_error = _relative_or_absolute_error(
        fitted_conditional_means,
        empirical_conditional_means,
    )

    projector = basis @ basis.T
    invariance_numerator = np.linalg.norm(
        (np.eye(num_phases) - projector) @ fitted @ basis
    )
    active_action = fitted @ basis
    if np.linalg.norm(active_action) <= 1e-12:
        active_invariance_error = float(invariance_numerator)
    else:
        active_invariance_error = float(
            invariance_numerator / np.linalg.norm(active_action)
        )
    active_operator_error = _relative_or_absolute_error(
        fitted_active,
        expected_active,
    )

    return {
        "dynamics": dynamics,
        "sample_count": int(current.shape[0]),
        "source_counts": np.bincount(current_phases, minlength=num_phases),
        "future_counts": np.bincount(future_phases, minlength=num_phases),
        "active_rank": int(basis.shape[1]),
        "sample_prediction_error": float(sample_prediction_error),
        "conditional_mean_error": conditional_mean_error,
        "active_invariance_error": active_invariance_error,
        "active_operator_error": active_operator_error,
        "spectral_mean_absolute_error": spectral_match["mean_absolute_error"],
        "spectral_max_absolute_error": spectral_match["max_absolute_error"],
        "fitted_operator": fitted,
        "expected_operator": expected,
        "basis": basis,
        "fitted_active_operator": fitted_active,
        "expected_active_operator": expected_active,
        "estimated_spectrum": estimated_spectrum,
        "expected_spectrum": expected_spectrum,
    }

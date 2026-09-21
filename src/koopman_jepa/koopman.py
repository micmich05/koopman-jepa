from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


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

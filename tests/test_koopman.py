import numpy as np

from koopman_jepa.koopman import (
    centered_phase_indicators,
    cyclic_phase_operator,
    fit_linear_operator,
    left_eigendecomposition,
    linear_rollout,
    match_eigenvalues,
    restrict_operator,
    sample_span_basis,
)


def _four_phase_oracle() -> tuple[np.ndarray, np.ndarray]:
    phases = np.tile(np.arange(4), 16)
    return (
        centered_phase_indicators(phases),
        centered_phase_indicators((phases + 1) % 4),
    )


def test_centered_phase_indicators_remove_constant_mode() -> None:
    indicators = centered_phase_indicators(np.arange(4))

    np.testing.assert_allclose(indicators.sum(axis=1), 0.0, atol=1e-12)
    np.testing.assert_allclose(indicators.mean(axis=0), 0.0, atol=1e-12)
    assert np.linalg.matrix_rank(indicators) == 3


def test_cyclic_operator_maps_centered_indicators_to_successors() -> None:
    current, future = _four_phase_oracle()
    operator = cyclic_phase_operator()

    np.testing.assert_allclose(current @ operator.T, future, atol=1e-12)


def test_least_squares_recovers_active_cyclic_spectrum() -> None:
    current, future = _four_phase_oracle()
    fitted = fit_linear_operator(current, future)
    basis = sample_span_basis(current)
    reduced = restrict_operator(fitted, basis)
    expected = np.array([-1.0, 1.0j, -1.0j])
    matching = match_eigenvalues(np.linalg.eigvals(reduced), expected)

    assert basis.shape == (4, 3)
    np.testing.assert_allclose(current @ fitted.T, future, atol=1e-12)
    assert matching["mean_absolute_error"] < 1e-12
    assert matching["max_absolute_error"] < 1e-12


def test_left_eigenvectors_satisfy_row_convention() -> None:
    current, future = _four_phase_oracle()
    fitted = fit_linear_operator(current, future)
    reduced = restrict_operator(fitted, sample_span_basis(current))
    eigenvalues, left_vectors = left_eigendecomposition(reduced)

    residual = left_vectors @ reduced - eigenvalues[:, None] * left_vectors

    np.testing.assert_allclose(residual, 0.0, atol=1e-12)


def test_fitted_operator_rollout_tracks_centered_phase_cycle() -> None:
    current, future = _four_phase_oracle()
    fitted = fit_linear_operator(current, future)
    initial = centered_phase_indicators(np.array([0]))[0]
    rollout = linear_rollout(fitted, initial, steps=8)
    expected = centered_phase_indicators(np.arange(9) % 4)

    np.testing.assert_allclose(rollout, expected, atol=1e-12)


def test_spectral_helpers_reject_invalid_shapes() -> None:
    with np.testing.assert_raises_regex(ValueError, "same shape"):
        fit_linear_operator(np.zeros((3, 2)), np.zeros((4, 2)))
    with np.testing.assert_raises_regex(ValueError, "same non-zero length"):
        match_eigenvalues(np.array([1.0]), np.array([1.0, -1.0]))
    with np.testing.assert_raises_regex(ValueError, "orthonormal"):
        restrict_operator(np.eye(2), np.ones((2, 1)))
    with np.testing.assert_raises_regex(ValueError, "non-negative"):
        linear_rollout(np.eye(2), np.zeros(2), steps=-1)

import numpy as np

from koopman_jepa.koopman import (
    balanced_phase_transitions,
    centered_phase_indicators,
    cyclic_phase_operator,
    evaluate_phase_dynamics_oracle,
    expected_active_spectrum,
    expected_phase_operator,
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


def test_balanced_dynamics_have_identical_uniform_marginals() -> None:
    transition_tables = {
        dynamics: balanced_phase_transitions(dynamics, repeats_per_transition=3)
        for dynamics in ("static", "cyclic", "independent")
    }

    for current, future in transition_tables.values():
        assert current.shape == future.shape == (48,)
        np.testing.assert_array_equal(np.bincount(current), np.repeat(12, 4))
        np.testing.assert_array_equal(np.bincount(future), np.repeat(12, 4))


def test_expected_phase_operators_have_distinct_active_spectra() -> None:
    expected = {
        "static": np.ones(3, dtype=np.complex128),
        "cyclic": np.array([1.0j, -1.0, -1.0j]),
        "independent": np.zeros(3, dtype=np.complex128),
    }

    for dynamics, spectrum in expected.items():
        operator = expected_phase_operator(dynamics)
        basis = sample_span_basis(centered_phase_indicators(np.arange(4)))
        estimated = np.linalg.eigvals(restrict_operator(operator, basis))
        matching = match_eigenvalues(estimated, expected_active_spectrum(dynamics))

        assert matching["max_absolute_error"] < 1e-12
        assert match_eigenvalues(estimated, spectrum)["max_absolute_error"] < 1e-12


def test_oracles_recover_all_three_conditional_operators() -> None:
    results = {
        dynamics: evaluate_phase_dynamics_oracle(dynamics, repeats_per_transition=4)
        for dynamics in ("static", "cyclic", "independent")
    }

    for result in results.values():
        assert result["active_rank"] == 3
        assert result["conditional_mean_error"] < 1e-12
        assert result["active_invariance_error"] < 1e-12
        assert result["active_operator_error"] < 1e-12
        assert result["spectral_max_absolute_error"] < 1e-12
        np.testing.assert_array_equal(result["source_counts"], np.repeat(16, 4))
        np.testing.assert_array_equal(result["future_counts"], np.repeat(16, 4))

    assert results["static"]["sample_prediction_error"] < 1e-12
    assert results["cyclic"]["sample_prediction_error"] < 1e-12
    assert np.isclose(results["independent"]["sample_prediction_error"], 1.0)


def test_phase_dynamics_reject_invalid_configuration() -> None:
    with np.testing.assert_raises_regex(ValueError, "unknown phase dynamics"):
        balanced_phase_transitions("unknown")  # type: ignore[arg-type]
    with np.testing.assert_raises_regex(ValueError, "positive"):
        balanced_phase_transitions("static", repeats_per_transition=0)
    with np.testing.assert_raises_regex(ValueError, "unknown phase dynamics"):
        expected_phase_operator("unknown")  # type: ignore[arg-type]

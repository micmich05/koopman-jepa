import numpy as np

from koopman_jepa.koopman import (
    balanced_decay_phase_transitions,
    balanced_phase_transitions,
    centered_phase_indicators,
    expected_phase_operator,
    fit_linear_operator,
    restrict_operator,
    sample_span_basis,
)
from koopman_jepa.phase_analysis import (
    evaluate_decay_operator_candidates,
    evaluate_phase_operator_candidates,
    evaluate_phase_operator_diagnostics,
    evaluate_phase_representation,
)


def test_oracle_cycle_has_perfect_phase_and_koopman_metrics() -> None:
    train_phases = np.tile(np.arange(4), 16)
    validation_phases = np.tile(np.arange(4), 8)
    basis = sample_span_basis(centered_phase_indicators(train_phases))
    train_embeddings = centered_phase_indicators(train_phases) @ basis
    validation_embeddings = centered_phase_indicators(validation_phases) @ basis
    predictor = restrict_operator(expected_phase_operator("cyclic"), basis)

    metrics = evaluate_phase_representation(
        train_embeddings,
        train_phases,
        validation_embeddings,
        validation_phases,
        predictor,
        dynamics="cyclic",
        seed=0,
    )

    assert metrics["linear_probe_accuracy"] == 1.0
    assert metrics["phase_alignment_error"] < 1e-12
    assert metrics["active_rank"] == 3
    assert metrics["intertwining_error"] < 1e-12
    assert metrics["active_invariance_error"] < 1e-12
    assert metrics["spectral_max_absolute_error"] < 1e-12


def test_independent_zero_predictor_has_zero_intertwining_error() -> None:
    phases = np.tile(np.arange(4), 8)
    basis = sample_span_basis(centered_phase_indicators(phases))
    embeddings = centered_phase_indicators(phases) @ basis

    metrics = evaluate_phase_representation(
        embeddings,
        phases,
        embeddings,
        phases,
        np.zeros((3, 3)),
        dynamics="independent",
        seed=0,
    )

    assert metrics["intertwining_error"] == 0.0
    assert metrics["spectral_max_absolute_error"] == 0.0


def test_phase_representation_rejects_misaligned_inputs() -> None:
    embeddings = np.ones((8, 3))
    phases = np.tile(np.arange(4), 2)

    with np.testing.assert_raises_regex(ValueError, "predictor matrix"):
        evaluate_phase_representation(
            embeddings,
            phases,
            embeddings,
            phases,
            np.eye(2),
            dynamics="static",
            seed=0,
        )
    with np.testing.assert_raises_regex(ValueError, "validation phases"):
        evaluate_phase_representation(
            embeddings,
            phases,
            embeddings,
            phases[:-1],
            np.eye(3),
            dynamics="static",
            seed=0,
        )


def test_oracle_operator_diagnostic_separates_consistent_coordinates() -> None:
    current_phases = np.tile(np.arange(4), 8)
    future_phases = (current_phases + 1) % 4
    basis = sample_span_basis(centered_phase_indicators(current_phases))
    current = centered_phase_indicators(current_phases) @ basis
    future = centered_phase_indicators(future_phases) @ basis
    predictor = restrict_operator(expected_phase_operator("cyclic"), basis)

    metrics = evaluate_phase_operator_diagnostics(
        current,
        future,
        future,
        current_phases,
        future_phases,
        predictor,
        dynamics="cyclic",
    )

    assert metrics["online_encoder_phase_basis_error"] < 1e-12
    assert metrics["online_target_phase_basis_error"] < 1e-12
    assert metrics["trained_online_endomorphism_error"] < 1e-12
    assert metrics["trained_cross_encoder_error"] < 1e-12
    assert metrics["predictor_vs_posthoc_online_error"] < 1e-12
    assert metrics["posthoc_online_spectral_max_error"] < 1e-12


def test_operator_diagnostic_rejects_misaligned_embeddings() -> None:
    phases = np.tile(np.arange(4), 2)
    embeddings = np.ones((8, 3))
    with np.testing.assert_raises_regex(ValueError, "must align"):
        evaluate_phase_operator_diagnostics(
            embeddings,
            embeddings[:-1],
            embeddings,
            phases,
            phases,
            np.eye(3),
            dynamics="static",
        )


def test_candidate_comparison_identifies_all_oracle_dynamics() -> None:
    candidates = ("static", "cyclic", "independent")
    indicator_table = centered_phase_indicators(np.arange(4), num_phases=4)
    _, _, right_vectors = np.linalg.svd(indicator_table, full_matrices=False)
    coordinates = right_vectors[:3].T

    for dynamics in candidates:
        current_phases, future_phases = balanced_phase_transitions(
            dynamics,
            repeats_per_transition=8,
            num_phases=4,
        )
        current = centered_phase_indicators(current_phases, 4) @ coordinates
        future = centered_phase_indicators(future_phases, 4) @ coordinates
        predictor = fit_linear_operator(current, future)

        result = evaluate_phase_operator_candidates(
            current,
            current_phases,
            predictor,
            candidates,
            rollout_horizons=(1, 2, 4, 8),
        )

        assert result["active_rank"] == 3
        assert result["predicted_action_dynamics"] == dynamics
        assert result["predicted_spectral_dynamics"] == dynamics
        assert result["action_errors"][dynamics] < 1e-12
        assert result["spectral_max_errors"][dynamics] < 1e-12
        assert max(result["rollout_errors"][dynamics].values()) < 1e-11


def test_candidate_comparison_with_incomplete_span_has_no_spectral_assignment() -> None:
    phases = np.tile(np.arange(4), 4)
    embeddings = (phases == 0).astype(np.float64)[:, None]

    result = evaluate_phase_operator_candidates(
        embeddings,
        phases,
        np.eye(1),
        ("static", "cyclic", "independent"),
    )

    assert result["active_rank"] == 1
    assert result["predicted_spectral_dynamics"] is None
    assert result["spectral_mean_errors"] is None
    assert result["spectral_max_errors"] is None


def test_decay_candidate_comparison_identifies_every_oracle_rho() -> None:
    candidates = (0.0, 0.25, 0.5, 0.75, 1.0)
    indicator_table = centered_phase_indicators(np.arange(4), num_phases=4)
    _, _, right_vectors = np.linalg.svd(indicator_table, full_matrices=False)
    coordinates = right_vectors[:3].T

    for rho in candidates:
        current_phases, future_phases = balanced_decay_phase_transitions(
            rho,
            repeats_per_transition=8,
        )
        current = centered_phase_indicators(current_phases, 4) @ coordinates
        future = centered_phase_indicators(future_phases, 4) @ coordinates
        predictor = fit_linear_operator(current, future)

        result = evaluate_decay_operator_candidates(
            current,
            current_phases,
            predictor,
            candidates,
            rollout_horizons=(1, 2, 4, 8),
        )

        assert result["active_rank"] == 3
        assert result["predicted_action_rho"] == rho
        assert result["predicted_spectral_rho"] == rho
        assert result["action_errors"][rho] < 1e-12
        assert result["spectral_max_errors"][rho] < 1e-12
        assert np.isclose(result["mean_eigenvalue_modulus"], rho)
        assert max(result["rollout_errors"][rho].values()) < 1e-11

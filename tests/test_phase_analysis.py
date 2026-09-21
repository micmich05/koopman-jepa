import numpy as np

from koopman_jepa.koopman import (
    centered_phase_indicators,
    expected_phase_operator,
    restrict_operator,
    sample_span_basis,
)
from koopman_jepa.phase_analysis import (
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

from dataclasses import replace

import numpy as np
import torch

from koopman_jepa.phase_data import (
    PhaseWindowConfig,
    make_decay_phase_tensor_dataset_splits,
    make_phase_tensor_dataset_splits,
    make_shared_decay_phase_observation_bundle,
    make_shared_phase_observation_bundle,
    nearest_template_phase_predictions,
    ordered_observation_marginal,
    phase_window_templates,
    validate_phase_window_config,
)


def test_phase_templates_are_standardized_circular_translations() -> None:
    config = PhaseWindowConfig(window_length=64, repeats_per_transition=2)
    templates = phase_window_templates(config)

    assert templates.shape == (4, 64)
    np.testing.assert_allclose(templates.mean(axis=1), 0.0, atol=1e-7)
    np.testing.assert_allclose(templates.std(axis=1), 1.0, atol=1e-7)
    for phase in range(1, config.num_phases):
        np.testing.assert_allclose(
            templates[phase],
            np.roll(templates[0], phase * config.window_length // config.num_phases),
            atol=1e-6,
        )


def test_shared_bundle_has_balanced_phase_transitions_and_shapes() -> None:
    config = PhaseWindowConfig(window_length=64, repeats_per_transition=3)
    bundle = make_shared_phase_observation_bundle(config, seed=17)

    expected_samples = config.num_phases**2 * config.repeats_per_transition
    for dynamics, condition in bundle.conditions.items():
        assert condition.dynamics == dynamics
        assert condition.current_windows.shape == (expected_samples, 1, 64)
        assert condition.future_windows.shape == condition.current_windows.shape
        np.testing.assert_array_equal(
            np.bincount(condition.current_phases),
            np.repeat(12, 4),
        )
        np.testing.assert_array_equal(
            np.bincount(condition.future_phases),
            np.repeat(12, 4),
        )

    static = bundle.conditions["static"]
    cyclic = bundle.conditions["cyclic"]
    independent = bundle.conditions["independent"]
    assert np.all(static.future_phases == static.current_phases)
    assert np.all(cyclic.future_phases == (cyclic.current_phases + 1) % 4)
    np.testing.assert_array_equal(
        np.bincount(4 * independent.current_phases + independent.future_phases),
        np.repeat(3, 16),
    )


def test_conditions_reuse_exactly_the_same_observation_marginals() -> None:
    config = PhaseWindowConfig(window_length=64, repeats_per_transition=3)
    bundle = make_shared_phase_observation_bundle(config, seed=23)
    reference = bundle.conditions["static"]

    for condition in bundle.conditions.values():
        np.testing.assert_array_equal(
            ordered_observation_marginal(condition, "current"),
            ordered_observation_marginal(reference, "current"),
        )
        np.testing.assert_array_equal(
            ordered_observation_marginal(condition, "future"),
            ordered_observation_marginal(reference, "future"),
        )


def test_emissions_are_reproducible_but_source_and_future_are_independent() -> None:
    config = PhaseWindowConfig(window_length=64, repeats_per_transition=2)
    first = make_shared_phase_observation_bundle(config, seed=29)
    replay = make_shared_phase_observation_bundle(config, seed=29)
    other = make_shared_phase_observation_bundle(config, seed=30)

    np.testing.assert_array_equal(
        first.conditions["cyclic"].current_windows,
        replay.conditions["cyclic"].current_windows,
    )
    assert not np.array_equal(
        first.conditions["static"].current_windows,
        first.conditions["static"].future_windows,
    )
    assert not np.array_equal(
        first.conditions["cyclic"].current_windows,
        other.conditions["cyclic"].current_windows,
    )


def test_nearest_template_control_recovers_observed_phase() -> None:
    config = PhaseWindowConfig(window_length=64, repeats_per_transition=8)
    bundle = make_shared_phase_observation_bundle(config, seed=31)
    condition = bundle.conditions["independent"]

    current_predictions = nearest_template_phase_predictions(
        condition.current_windows,
        bundle.templates,
        config.max_shift,
    )
    future_predictions = nearest_template_phase_predictions(
        condition.future_windows,
        bundle.templates,
        config.max_shift,
    )

    assert np.mean(current_predictions == condition.current_phases) > 0.99
    assert np.mean(future_predictions == condition.future_phases) > 0.99


def test_phase_window_validation_rejects_invalid_values() -> None:
    base = PhaseWindowConfig()
    invalid = (
        replace(base, num_phases=1),
        replace(base, window_length=15),
        replace(base, window_length=130),
        replace(base, repeats_per_transition=0),
        replace(base, noise_std=-0.1),
        replace(base, amplitude_low=0.0),
        replace(base, amplitude_low=1.3, amplitude_high=1.2),
        replace(base, max_shift=16),
        replace(base, pulse_width=0.0),
        replace(base, secondary_offset=0.5),
        replace(base, secondary_strength=1.1),
    )
    for config in invalid:
        with np.testing.assert_raises(ValueError):
            validate_phase_window_config(config)

    bundle = make_shared_phase_observation_bundle(base, seed=1)
    with np.testing.assert_raises_regex(ValueError, "unknown observation side"):
        ordered_observation_marginal(bundle.conditions["static"], "middle")  # type: ignore[arg-type]
    with np.testing.assert_raises_regex(ValueError, "windows must have shape"):
        nearest_template_phase_predictions(np.zeros(5), bundle.templates, base.max_shift)


def test_tensor_splits_use_distinct_seeds_and_phase_pair_labels() -> None:
    config = PhaseWindowConfig(window_length=64)
    splits = make_phase_tensor_dataset_splits(
        config,
        train_repeats_per_transition=3,
        validation_repeats_per_transition=2,
        seed=41,
    )
    replay = make_phase_tensor_dataset_splits(config, 3, 2, seed=41)

    train_context, train_future, train_labels = splits.train["cyclic"].tensors
    val_context, val_future, val_labels = splits.validation["cyclic"].tensors
    assert train_context.shape == train_future.shape == (48, 1, 64)
    assert train_labels.shape == (48, 2)
    assert val_context.shape == val_future.shape == (32, 1, 64)
    assert val_labels.shape == (32, 2)
    assert splits.train_seed == 52
    assert splits.validation_seed == 64
    assert splits.test is None
    assert splits.test_seed is None
    assert torch.equal(train_context, replay.train["cyclic"].tensors[0])
    assert not torch.equal(train_context[:32], val_context)
    assert torch.all(train_labels[:, 1] == (train_labels[:, 0] + 1) % 4)


def test_tensor_splits_materialize_independent_test_only_when_requested() -> None:
    config = PhaseWindowConfig(window_length=64)
    splits = make_phase_tensor_dataset_splits(
        config,
        train_repeats_per_transition=3,
        validation_repeats_per_transition=2,
        test_repeats_per_transition=2,
        seed=41,
    )

    assert splits.test is not None
    assert splits.test_seed == 78
    test_context, test_future, test_labels = splits.test["cyclic"].tensors
    validation_context = splits.validation["cyclic"].tensors[0]
    assert test_context.shape == test_future.shape == (32, 1, 64)
    assert test_labels.shape == (32, 2)
    assert not torch.equal(test_context, validation_context)
    assert torch.all(test_labels[:, 1] == (test_labels[:, 0] + 1) % 4)


def test_tensor_split_rejects_non_positive_repeat_counts() -> None:
    config = PhaseWindowConfig()
    with np.testing.assert_raises_regex(ValueError, "train_repeats_per_transition"):
        make_phase_tensor_dataset_splits(config, 0, 2, seed=1)
    with np.testing.assert_raises_regex(ValueError, "validation_repeats_per_transition"):
        make_phase_tensor_dataset_splits(config, 2, 0, seed=1)
    with np.testing.assert_raises_regex(ValueError, "test_repeats_per_transition"):
        make_phase_tensor_dataset_splits(
            config,
            2,
            1,
            seed=1,
            test_repeats_per_transition=0,
        )


def test_decay_conditions_reuse_identical_observation_marginals() -> None:
    rhos = (0.0, 0.25, 0.5, 0.75, 1.0)
    config = PhaseWindowConfig(window_length=64, repeats_per_transition=4)
    bundle = make_shared_decay_phase_observation_bundle(config, rhos, seed=47)
    reference = bundle.conditions[0.0]

    for rho, condition in bundle.conditions.items():
        assert condition.rho == rho
        np.testing.assert_array_equal(
            ordered_observation_marginal(condition, "current"),
            ordered_observation_marginal(reference, "current"),
        )
        np.testing.assert_array_equal(
            ordered_observation_marginal(condition, "future"),
            ordered_observation_marginal(reference, "future"),
        )


def test_decay_tensor_splits_are_reproducible_and_disjoint() -> None:
    rhos = (0.0, 0.25, 0.5, 0.75, 1.0)
    config = PhaseWindowConfig(window_length=64)
    splits = make_decay_phase_tensor_dataset_splits(config, rhos, 4, 4, seed=53)
    replay = make_decay_phase_tensor_dataset_splits(config, rhos, 4, 4, seed=53)

    assert set(splits.train) == set(rhos)
    assert set(splits.validation) == set(rhos)
    assert splits.train_seed == 64
    assert splits.validation_seed == 76
    assert torch.equal(splits.train[0.5].tensors[0], replay.train[0.5].tensors[0])
    assert not torch.equal(
        splits.train[0.5].tensors[0],
        splits.validation[0.5].tensors[0],
    )

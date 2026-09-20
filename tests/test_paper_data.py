from pathlib import Path

import numpy as np
import torch
from scipy import signal as scipy_signal

from koopman_jepa.paper_data import (
    PAPER_PERIODIC_CYCLES,
    PAPER_PULSE_AMPLITUDE,
    PAPER_PULSE_COUNT,
    PAPER_REGIME_NAMES,
    PaperDataConfig,
    PaperRegimeDataset,
    PaperSampleKey,
    generate_paper_master,
    load_paper_data_config,
)


def _dummy_master(
    regime_id: int,
    length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    trend = np.linspace(-1.0, 1.0, length, dtype=np.float32)
    noise = rng.normal(0.0, 0.01, size=length)
    return trend + regime_id + noise


def test_paper_audit_config_uses_published_geometry() -> None:
    path = Path(__file__).parents[1] / "configs" / "paper_dataset_audit.yaml"
    config = load_paper_data_config(path)
    full_config = PaperDataConfig()

    assert len(PAPER_REGIME_NAMES) == 18
    assert config.master_length == 1024
    assert config.context_length == 768
    assert config.shift == 256
    assert config.context_length + config.shift == config.master_length
    assert (
        full_config.train_per_regime
        + full_config.val_per_regime
        + full_config.test_per_regime
    ) * len(PAPER_REGIME_NAMES) == 180_000


def test_lazy_indexing_is_balanced_and_split_disjoint() -> None:
    config = PaperDataConfig(
        train_per_regime=2,
        val_per_regime=1,
        test_per_regime=1,
    )
    train = PaperRegimeDataset(config, "train", _dummy_master)
    val = PaperRegimeDataset(config, "val", _dummy_master)
    test = PaperRegimeDataset(config, "test", _dummy_master)

    assert len(train) == len(PAPER_REGIME_NAMES) * 2
    assert train.sample_key(0) == PaperSampleKey(regime_id=0, sequence_id=0)
    assert train.sample_key(2) == PaperSampleKey(regime_id=1, sequence_id=0)
    assert val.sample_key(0) == PaperSampleKey(regime_id=0, sequence_id=2)
    assert test.sample_key(0) == PaperSampleKey(regime_id=0, sequence_id=3)


def test_samples_are_deterministic_and_standardized_before_windowing() -> None:
    config = PaperDataConfig(
        train_per_regime=2,
        val_per_regime=1,
        test_per_regime=1,
    )
    dataset = PaperRegimeDataset(config, "train", _dummy_master)

    context, target, label = dataset[0]
    repeated_context, repeated_target, repeated_label = dataset[0]
    reconstructed_master = torch.cat([context.flatten(), target.flatten()[-config.shift :]])

    assert torch.equal(context, repeated_context)
    assert torch.equal(target, repeated_target)
    assert torch.equal(label, repeated_label)
    assert context.shape == (1, 768)
    assert target.shape == context.shape
    assert label.item() == 0
    assert torch.allclose(context[:, config.shift :], target[:, : -config.shift])
    assert abs(float(reconstructed_master.mean())) < 1e-6
    assert abs(float(reconstructed_master.std(unbiased=False)) - 1.0) < 1e-6


def test_different_splits_use_different_rng_streams() -> None:
    config = PaperDataConfig(
        train_per_regime=1,
        val_per_regime=1,
        test_per_regime=1,
    )
    train = PaperRegimeDataset(config, "train", _dummy_master)
    val = PaperRegimeDataset(config, "val", _dummy_master)

    train_context, _, _ = train[0]
    val_context, _, _ = val[0]

    assert not torch.equal(train_context, val_context)


def test_published_sine_regimes_have_expected_frequency_and_amplitude() -> None:
    length = 1024
    expected = {
        0: (PAPER_PERIODIC_CYCLES["low"], 1.0),
        1: (PAPER_PERIODIC_CYCLES["medium"], 1.0),
        2: (PAPER_PERIODIC_CYCLES["high"], 1.0),
        3: (PAPER_PERIODIC_CYCLES["medium"], 0.3),
    }

    for regime_id, (expected_bin, expected_amplitude) in expected.items():
        signal = generate_paper_master(regime_id, length, np.random.default_rng(7))
        spectrum = np.abs(np.fft.rfft(signal))
        dominant_bin = int(np.argmax(spectrum[1:]) + 1)
        recovered_amplitude = 2.0 * spectrum[dominant_bin] / length

        assert dominant_bin == expected_bin
        assert np.isclose(recovered_amplitude, expected_amplitude, atol=1e-6)


def test_harmonic_regime_has_published_components() -> None:
    length = 1024
    signal = generate_paper_master(4, length, np.random.default_rng(11))
    spectrum = 2.0 * np.abs(np.fft.rfft(signal)) / length
    medium_bin = PAPER_PERIODIC_CYCLES["medium"]

    assert np.isclose(spectrum[medium_bin], 0.7, atol=1e-6)
    assert np.isclose(spectrum[3 * medium_bin], 0.3, atol=1e-6)

    spectrum[[0, medium_bin, 3 * medium_bin]] = 0.0
    assert float(spectrum.max()) < 1e-6


def test_sinusoidal_generation_is_seeded_and_noise_free() -> None:
    first = generate_paper_master(1, 1024, np.random.default_rng(23))
    repeated = generate_paper_master(1, 1024, np.random.default_rng(23))
    changed_phase = generate_paper_master(1, 1024, np.random.default_rng(24))

    assert np.array_equal(first, repeated)
    assert not np.array_equal(first, changed_phase)

    time = np.arange(1024)
    basis = np.column_stack(
        [
            np.sin(2.0 * np.pi * PAPER_PERIODIC_CYCLES["medium"] * time / 1024),
            np.cos(2.0 * np.pi * PAPER_PERIODIC_CYCLES["medium"] * time / 1024),
        ]
    )
    fitted = basis @ np.linalg.lstsq(basis, first, rcond=None)[0]
    assert np.max(np.abs(first - fitted)) < 1e-6


def test_unimplemented_regimes_fail_explicitly() -> None:
    with np.testing.assert_raises_regex(NotImplementedError, "ar_pos_strong"):
        generate_paper_master(7, 1024, np.random.default_rng(0))


def test_trend_regimes_follow_documented_randomization() -> None:
    length = 1024
    normalized_time = np.linspace(0.0, 1.0, length)

    for regime_id, base_slope in ((5, 1.5), (6, -1.5)):
        expected_rng = np.random.default_rng(31)
        slope = base_slope + expected_rng.normal(0.0, 1.0)
        intercept = expected_rng.normal(0.0, np.pi)
        expected = slope * normalized_time + intercept

        actual = generate_paper_master(regime_id, length, np.random.default_rng(31))

        assert np.allclose(actual, expected, atol=1e-6)


def test_standardization_removes_affine_trend_scale_and_intercept() -> None:
    upward = generate_paper_master(5, 1024, np.random.default_rng(0))
    downward = generate_paper_master(6, 1024, np.random.default_rng(0))
    upward = (upward - upward.mean()) / upward.std()
    downward = (downward - downward.mean()) / downward.std()

    assert np.allclose(upward, -downward, atol=1e-6)
    assert np.corrcoef(np.arange(1024), upward)[0, 1] > 0.999999
    assert np.corrcoef(np.arange(1024), downward)[0, 1] < -0.999999


def test_sine_trend_matches_published_sum_under_local_time_assumption() -> None:
    length = 1024
    expected_rng = np.random.default_rng(41)
    phase = expected_rng.normal(0.0, np.pi)
    slope = 1.0 + expected_rng.normal(0.0, 1.0)
    intercept = expected_rng.normal(0.0, np.pi)
    sample_time = np.arange(length)
    normalized_time = np.linspace(0.0, 1.0, length)
    expected = (
        0.8
        * np.sin(
            2.0 * np.pi * PAPER_PERIODIC_CYCLES["medium"] * sample_time / length + phase
        )
        + slope * normalized_time
        + intercept
    )

    actual = generate_paper_master(16, length, np.random.default_rng(41))

    assert np.allclose(actual, expected, atol=1e-6)


def test_square_regimes_have_published_cycles_and_binary_levels() -> None:
    length = 1024
    for regime_id, band in ((12, "low"), (13, "high")):
        signal = generate_paper_master(regime_id, length, np.random.default_rng(0))
        spectrum = np.abs(np.fft.rfft(signal))
        dominant_bin = int(np.argmax(spectrum[1:]) + 1)
        circular_transitions = np.count_nonzero(signal != np.roll(signal, 1))

        assert set(np.unique(signal)) == {-1.0, 1.0}
        assert dominant_bin == PAPER_PERIODIC_CYCLES[band]
        assert circular_transitions == 2 * PAPER_PERIODIC_CYCLES[band]
        assert signal[0] == 1.0


def test_sawtooth_uses_random_uniform_phase_and_rising_ramps() -> None:
    length = 1024
    expected_rng = np.random.default_rng(53)
    initial_phase = expected_rng.uniform(0.0, 2.0 * np.pi)
    expected = scipy_signal.sawtooth(
        2.0
        * np.pi
        * PAPER_PERIODIC_CYCLES["medium"]
        * np.arange(length)
        / length
        + initial_phase,
        width=1.0,
    )
    actual = generate_paper_master(14, length, np.random.default_rng(53))
    differences = np.diff(actual)

    assert np.allclose(actual, expected, atol=1e-6)
    assert np.count_nonzero(differences < 0.0) == PAPER_PERIODIC_CYCLES["medium"]
    assert np.all(differences[differences > 0.0] > 0.0)


def test_sparse_pulses_follow_documented_local_policy() -> None:
    length = 1024
    width = round(length / 50)
    expected_rng = np.random.default_rng(67)
    starts = expected_rng.choice(
        length - width + 1,
        size=PAPER_PULSE_COUNT,
        replace=False,
    )
    expected = np.zeros(length, dtype=np.float32)
    for start in starts:
        expected[start : start + width] = PAPER_PULSE_AMPLITUDE

    actual = generate_paper_master(15, length, np.random.default_rng(67))

    assert np.array_equal(actual, expected)
    assert set(np.unique(actual)) <= {0.0, PAPER_PULSE_AMPLITUDE}
    assert np.count_nonzero(actual) <= PAPER_PULSE_COUNT * width

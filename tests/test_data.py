import numpy as np

from koopman_jepa.config import DataConfig
from koopman_jepa.data import REGIME_NAMES, make_phase0_datasets


def test_dataset_is_balanced_and_has_expected_shapes() -> None:
    config = DataConfig(
        context_length=64,
        shift=16,
        train_per_regime=5,
        val_per_regime=2,
        test_per_regime=2,
    )
    bundle = make_phase0_datasets(config, seed=7)
    context, target, labels = bundle.train.tensors

    assert context.shape == (len(REGIME_NAMES) * 5, 1, 64)
    assert target.shape == context.shape
    assert labels.shape == (len(REGIME_NAMES) * 5,)
    assert np.array_equal(np.bincount(labels.numpy()), np.full(len(REGIME_NAMES), 5))


def test_standardization_is_applied_before_windowing() -> None:
    config = DataConfig(
        context_length=64,
        shift=16,
        train_per_regime=2,
        val_per_regime=1,
        test_per_regime=1,
        noise_std=0.0,
        standardize=True,
    )
    bundle = make_phase0_datasets(config, seed=3)
    context, target, _ = bundle.train.tensors
    reconstructed_overlap_error = (context[:, :, 16:] - target[:, :, :-16]).abs().max()

    assert reconstructed_overlap_error.item() < 1e-6

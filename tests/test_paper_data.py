from pathlib import Path

import numpy as np
import torch

from koopman_jepa.paper_data import (
    PAPER_REGIME_NAMES,
    PaperDataConfig,
    PaperRegimeDataset,
    PaperSampleKey,
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

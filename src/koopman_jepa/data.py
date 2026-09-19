from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import TensorDataset

from .config import DataConfig

REGIME_NAMES = (
    "sine_low",
    "sine_high",
    "square",
    "sawtooth",
    "ar_positive",
    "ar_negative",
)


@dataclass(frozen=True, slots=True)
class DatasetBundle:
    train: TensorDataset
    val: TensorDataset
    test: TensorDataset
    regime_names: tuple[str, ...] = REGIME_NAMES


def _periodic_phase(time: np.ndarray, cycles: float, phase: float) -> np.ndarray:
    return 2.0 * np.pi * cycles * time / time.size + phase


def _generate_regime(
    regime_id: int,
    length: int,
    rng: np.random.Generator,
    noise_std: float,
) -> np.ndarray:
    time = np.arange(length, dtype=np.float32)
    phase = rng.uniform(0.0, 2.0 * np.pi)
    amplitude = rng.uniform(0.8, 1.2)
    offset = rng.normal(0.0, 0.25)

    if regime_id == 0:
        angle = _periodic_phase(time, cycles=4.0, phase=phase)
        signal = np.sin(angle) + 0.15 * np.sin(2.0 * angle + 0.3)
    elif regime_id == 1:
        angle = _periodic_phase(time, cycles=10.0, phase=phase)
        signal = np.sin(angle) + 0.10 * np.cos(3.0 * angle)
    elif regime_id == 2:
        angle = _periodic_phase(time, cycles=6.0, phase=phase)
        signal = np.where(np.sin(angle) >= 0.0, 1.0, -1.0)
    elif regime_id == 3:
        fractional = np.mod(5.0 * time / length + phase / (2.0 * np.pi), 1.0)
        signal = 2.0 * fractional - 1.0
    elif regime_id in {4, 5}:
        coefficient = 0.85 if regime_id == 4 else -0.70
        innovations = rng.normal(0.0, 0.45, size=length).astype(np.float32)
        signal = np.empty(length, dtype=np.float32)
        signal[0] = innovations[0]
        for index in range(1, length):
            signal[index] = coefficient * signal[index - 1] + innovations[index]
    else:
        raise ValueError(f"Unknown regime id: {regime_id}")

    observation_noise = rng.normal(0.0, noise_std, size=length)
    return (amplitude * signal + offset + observation_noise).astype(np.float32)


def _make_split(
    samples_per_regime: int,
    config: DataConfig,
    seed: int,
) -> TensorDataset:
    rng = np.random.default_rng(seed)
    master_length = config.context_length + config.shift
    contexts: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    labels: list[int] = []

    for regime_id in range(len(REGIME_NAMES)):
        for _ in range(samples_per_regime):
            master = _generate_regime(
                regime_id=regime_id,
                length=master_length,
                rng=rng,
                noise_std=config.noise_std,
            )
            if config.standardize:
                mean = float(master.mean())
                std = max(float(master.std()), 1e-6)
                master = (master - mean) / std

            contexts.append(master[: config.context_length])
            targets.append(master[config.shift : config.shift + config.context_length])
            labels.append(regime_id)

    permutation = rng.permutation(len(labels))
    context_array = np.stack(contexts, axis=0)[permutation, None, :]
    target_array = np.stack(targets, axis=0)[permutation, None, :]
    label_array = np.asarray(labels, dtype=np.int64)[permutation]

    return TensorDataset(
        torch.from_numpy(context_array),
        torch.from_numpy(target_array),
        torch.from_numpy(label_array),
    )


def make_phase0_datasets(config: DataConfig, seed: int) -> DatasetBundle:
    return DatasetBundle(
        train=_make_split(config.train_per_regime, config, seed=seed + 11),
        val=_make_split(config.val_per_regime, config, seed=seed + 23),
        test=_make_split(config.test_per_regime, config, seed=seed + 37),
    )

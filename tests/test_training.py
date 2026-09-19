import math

import torch

from koopman_jepa.config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig
from koopman_jepa.data import make_phase0_datasets
from koopman_jepa.model import TemporalJEPA
from koopman_jepa.training import train_model


def test_one_training_epoch_updates_the_predictor() -> None:
    config = ExperimentConfig(
        data=DataConfig(
            context_length=32,
            shift=8,
            train_per_regime=2,
            val_per_regime=1,
            test_per_regime=1,
        ),
        model=ModelConfig(latent_dim=3, channels=[4], predictor_init="identity"),
        train=TrainConfig(epochs=1, batch_size=6, learning_rate=1e-3),
    )
    datasets = make_phase0_datasets(config.data, seed=config.train.seed)
    model = TemporalJEPA(
        latent_dim=config.model.latent_dim,
        channels=config.model.channels,
        predictor_init=config.model.predictor_init,
    )
    initial_predictor = model.predictor.matrix.detach().clone()

    history = train_model(model, datasets.train, datasets.val, config, torch.device("cpu"))

    assert len(history) == 1
    assert math.isfinite(history[0]["train_loss"])
    assert math.isfinite(history[0]["val_loss"])
    assert not torch.equal(model.predictor.matrix, initial_predictor)

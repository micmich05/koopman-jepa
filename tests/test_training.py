import math

import torch
from torch import nn
from torch.utils.data import TensorDataset

from koopman_jepa.config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig
from koopman_jepa.data import make_phase0_datasets
from koopman_jepa.model import TemporalJEPA
from koopman_jepa.training import collect_embeddings, train_model


class _MeanEncoder(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs.mean(dim=-1)


class _EvaluationModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.online_encoder = _MeanEncoder()
        self.target_encoder = _MeanEncoder()


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


def test_collect_embeddings_encodes_the_future_target_window() -> None:
    context = torch.zeros(2, 1, 8)
    target = torch.full((2, 1, 8), 3.0)
    labels = torch.tensor([0, 1])
    dataset = TensorDataset(context, target, labels)

    online, target_embeddings, collected_labels = collect_embeddings(
        _EvaluationModel(),
        dataset,
        batch_size=2,
        device=torch.device("cpu"),
    )

    assert (online == 0.0).all()
    assert (target_embeddings == 3.0).all()
    assert (collected_labels == labels.numpy()).all()

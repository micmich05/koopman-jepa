import math

import torch
from torch import nn
from torch.utils.data import TensorDataset

from koopman_jepa.config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig
from koopman_jepa.data import make_phase0_datasets
from koopman_jepa.model import TemporalJEPA
from koopman_jepa.training import (
    collect_embeddings,
    collect_paired_embeddings,
    evaluate_model_loss,
    make_optimizer,
    train_model,
    train_model_with_validation_checkpoint,
)


class _MeanEncoder(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs.mean(dim=-1)


class _EvaluationModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.online_encoder = _MeanEncoder()
        self.target_encoder = _MeanEncoder()


def test_optimizer_can_accelerate_only_the_predictor() -> None:
    model = TemporalJEPA(latent_dim=3, channels=[4], predictor_init="random")
    config = ExperimentConfig(
        train=TrainConfig(
            learning_rate=1e-3,
            predictor_learning_rate_multiplier=4.0,
        )
    )

    optimizer = make_optimizer(model, config)

    assert [group["lr"] for group in optimizer.param_groups] == [1e-3, 4e-3]
    assert {
        id(parameter) for parameter in optimizer.param_groups[0]["params"]
    } == {id(parameter) for parameter in model.online_encoder.parameters()}
    assert {
        id(parameter) for parameter in optimizer.param_groups[1]["params"]
    } == {id(parameter) for parameter in model.predictor.parameters()}


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

    current_online, future_online, future_target, paired_labels = (
        collect_paired_embeddings(
            _EvaluationModel(),
            dataset,
            batch_size=2,
            device=torch.device("cpu"),
        )
    )
    assert (current_online == 0.0).all()
    assert (future_online == 3.0).all()
    assert (future_target == 3.0).all()
    assert (paired_labels == labels.numpy()).all()


def test_validation_checkpoint_and_loss_evaluation_are_available() -> None:
    config = ExperimentConfig(
        data=DataConfig(
            context_length=32,
            shift=8,
            train_per_regime=2,
            val_per_regime=1,
            test_per_regime=1,
        ),
        model=ModelConfig(latent_dim=3, channels=[4], predictor_init="random"),
        train=TrainConfig(
            epochs=2,
            batch_size=6,
            learning_rate=1e-3,
            mean_weight=1.0,
            variance_weight=1.0,
            covariance_weight=1.0,
        ),
    )
    datasets = make_phase0_datasets(config.data, seed=0)
    model = TemporalJEPA(latent_dim=3, channels=[4], predictor_init="random")

    baseline = evaluate_model_loss(model, datasets.val, config, torch.device("cpu"))
    callback_epochs: list[int] = []

    def callback(
        epoch: int,
        _model: TemporalJEPA,
        _row: dict[str, float],
    ) -> dict[str, float]:
        callback_epochs.append(epoch)
        return {"callback_marker": float(epoch * 10)}

    result = train_model_with_validation_checkpoint(
        model,
        datasets.train,
        datasets.val,
        config,
        torch.device("cpu"),
        epoch_callback=callback,
    )
    selected = evaluate_model_loss(model, datasets.val, config, torch.device("cpu"))

    assert math.isfinite(baseline.loss)
    assert len(result.history) == 2
    assert callback_epochs == [1, 2]
    assert result.history[-1]["callback_marker"] == 20.0
    assert result.history[-1]["train_online_gradient_norm"] > 0.0
    assert result.history[-1]["train_predictor_gradient_norm"] > 0.0
    assert result.best_epoch in {1, 2}
    assert math.isclose(selected.loss, result.best_validation_loss, rel_tol=1e-6)

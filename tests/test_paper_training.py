import math

import numpy as np
import torch

from koopman_jepa.paper_model import PaperModelConfig, PaperTemporalJEPA
from koopman_jepa.paper_training import (
    paper_train_step,
    paper_trainable_parameters,
    squared_embedding_error,
)


def test_squared_embedding_error_matches_mean_squared_l2_distance() -> None:
    prediction = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    target = torch.tensor([[0.0, 0.0], [1.0, 1.0]])

    loss = squared_embedding_error(prediction, target)

    assert loss.item() == 9.0


def test_squared_embedding_error_rejects_incompatible_embeddings() -> None:
    with np.testing.assert_raises_regex(ValueError, "same shape"):
        squared_embedding_error(torch.zeros(2, 3), torch.zeros(2, 4))


def test_trainable_parameters_exclude_ema_target() -> None:
    model = PaperTemporalJEPA(PaperModelConfig(latent_dim=4))
    trainable_ids = {id(parameter) for parameter in paper_trainable_parameters(model)}
    online_ids = {id(parameter) for parameter in model.online_encoder.parameters()}
    predictor_ids = {id(parameter) for parameter in model.predictor.parameters()}
    target_ids = {id(parameter) for parameter in model.target_encoder.parameters()}

    assert trainable_ids == online_ids | predictor_ids
    assert trainable_ids.isdisjoint(target_ids)


def test_one_batch_smoke_updates_online_predictor_then_ema_target() -> None:
    torch.manual_seed(7)
    config = PaperModelConfig(latent_dim=4, ema_decay=0.996)
    model = PaperTemporalJEPA(config)
    optimizer = torch.optim.SGD(paper_trainable_parameters(model), lr=1e-4)
    context = torch.randn(2, 1, 768)
    target = torch.randn(2, 1, 768)

    online_before = [
        parameter.detach().clone() for parameter in model.online_encoder.parameters()
    ]
    predictor_before = [
        parameter.detach().clone() for parameter in model.predictor.parameters()
    ]
    target_before = [
        parameter.detach().clone() for parameter in model.target_encoder.parameters()
    ]

    metrics = paper_train_step(model, context, target, optimizer)

    assert math.isfinite(metrics.loss)
    assert metrics.loss > 0.0
    assert metrics.online_gradient_norm > 0.0
    assert metrics.predictor_gradient_norm > 0.0
    assert any(
        not torch.equal(before, after)
        for before, after in zip(
            online_before,
            model.online_encoder.parameters(),
            strict=True,
        )
    )
    assert any(
        not torch.equal(before, after)
        for before, after in zip(
            predictor_before,
            model.predictor.parameters(),
            strict=True,
        )
    )
    assert all(parameter.grad is None for parameter in model.target_encoder.parameters())

    for before, online, ema_target in zip(
        target_before,
        model.online_encoder.parameters(),
        model.target_encoder.parameters(),
        strict=True,
    ):
        expected = config.ema_decay * before + (1.0 - config.ema_decay) * online
        assert torch.allclose(ema_target, expected, atol=1e-7)

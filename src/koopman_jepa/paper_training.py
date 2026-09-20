from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .paper_model import PaperTemporalJEPA


@dataclass(frozen=True, slots=True)
class PaperStepMetrics:
    loss: float
    online_gradient_norm: float
    predictor_gradient_norm: float


def squared_embedding_error(
    prediction: torch.Tensor,
    target_embedding: torch.Tensor,
) -> torch.Tensor:
    """Mean per-sample squared Euclidean error from the paper objective."""

    if prediction.shape != target_embedding.shape:
        raise ValueError(
            "prediction and target embedding must have the same shape; "
            f"received {tuple(prediction.shape)} and {tuple(target_embedding.shape)}"
        )
    if prediction.ndim != 2:
        raise ValueError("embeddings must be shaped (batch, latent_dim)")
    if prediction.shape[0] == 0:
        raise ValueError("the batch must contain at least one sample")
    return (prediction - target_embedding).square().sum(dim=1).mean()


def paper_trainable_parameters(model: PaperTemporalJEPA) -> tuple[nn.Parameter, ...]:
    """Return only parameters updated by gradient descent in temporal JEPA."""

    return tuple(model.online_encoder.parameters()) + tuple(model.predictor.parameters())


def _gradient_norm(parameters: tuple[nn.Parameter, ...]) -> float:
    squared_norm = sum(
        float(parameter.grad.detach().square().sum())
        for parameter in parameters
        if parameter.grad is not None
    )
    return squared_norm**0.5


def paper_train_step(
    model: PaperTemporalJEPA,
    context: torch.Tensor,
    target: torch.Tensor,
    optimizer: torch.optim.Optimizer,
) -> PaperStepMetrics:
    """Run one optimizer step followed by the published EMA target update."""

    if context.shape != target.shape:
        raise ValueError(
            "context and target must have the same shape; "
            f"received {tuple(context.shape)} and {tuple(target.shape)}"
        )
    if context.shape[0] == 0:
        raise ValueError("the batch must contain at least one sample")

    model.train()
    optimizer.zero_grad(set_to_none=True)

    _, prediction, target_embedding = model(context, target)
    loss = squared_embedding_error(prediction, target_embedding)
    loss.backward()

    online_parameters = tuple(model.online_encoder.parameters())
    predictor_parameters = tuple(model.predictor.parameters())
    online_gradient_norm = _gradient_norm(online_parameters)
    predictor_gradient_norm = _gradient_norm(predictor_parameters)

    optimizer.step()
    model.update_target()

    return PaperStepMetrics(
        loss=float(loss.detach()),
        online_gradient_norm=online_gradient_norm,
        predictor_gradient_norm=predictor_gradient_norm,
    )

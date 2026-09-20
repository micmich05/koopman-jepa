from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import fmean

import torch
from torch import nn

from .paper_config import PaperOptimizationConfig, PaperOverfitGateConfig
from .paper_model import PaperTemporalJEPA


@dataclass(frozen=True, slots=True)
class PaperStepMetrics:
    loss: float
    online_gradient_norm: float
    predictor_gradient_norm: float
    embedding_std_mean: float
    effective_rank: float


@dataclass(frozen=True, slots=True)
class PaperOverfitGateResult:
    all_finite: bool
    initial_loss: float
    final_loss: float
    loss_ratio: float
    initial_embedding_std: float
    final_embedding_std: float
    embedding_std_ratio: float
    final_effective_rank: float
    loss_passed: bool
    spread_passed: bool
    rank_passed: bool
    passed: bool


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


def make_paper_optimizer(
    model: PaperTemporalJEPA,
    config: PaperOptimizationConfig,
) -> torch.optim.Optimizer:
    """Build the explicitly local optimizer used by a development condition."""

    config.validate()
    if config.optimizer == "adamw":
        return torch.optim.AdamW(
            paper_trainable_parameters(model),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    raise ValueError(f"unknown paper optimizer: {config.optimizer}")


def _gradient_norm(parameters: tuple[nn.Parameter, ...]) -> float:
    squared_norm = sum(
        float(parameter.grad.detach().square().sum())
        for parameter in parameters
        if parameter.grad is not None
    )
    return squared_norm**0.5


@torch.no_grad()
def _embedding_spread(embeddings: torch.Tensor) -> tuple[float, float]:
    centered = embeddings.detach() - embeddings.detach().mean(dim=0, keepdim=True)
    embedding_std_mean = float(centered.square().mean(dim=0).sqrt().mean())
    squared_singular_values = torch.linalg.svdvals(centered).square()
    total = squared_singular_values.sum()
    if float(total) <= 1e-12:
        return embedding_std_mean, 0.0
    probabilities = squared_singular_values / total
    probabilities = probabilities[probabilities > 1e-12]
    entropy = -(probabilities * probabilities.log()).sum()
    return embedding_std_mean, float(entropy.exp())


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

    online_embedding, prediction, target_embedding = model(context, target)
    loss = squared_embedding_error(prediction, target_embedding)
    loss.backward()

    online_parameters = tuple(model.online_encoder.parameters())
    predictor_parameters = tuple(model.predictor.parameters())
    online_gradient_norm = _gradient_norm(online_parameters)
    predictor_gradient_norm = _gradient_norm(predictor_parameters)
    embedding_std_mean, effective_rank = _embedding_spread(online_embedding)

    optimizer.step()
    model.update_target()

    return PaperStepMetrics(
        loss=float(loss.detach()),
        online_gradient_norm=online_gradient_norm,
        predictor_gradient_norm=predictor_gradient_norm,
        embedding_std_mean=embedding_std_mean,
        effective_rank=effective_rank,
    )


def overfit_fixed_batch(
    model: PaperTemporalJEPA,
    context: torch.Tensor,
    target: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    config: PaperOptimizationConfig,
) -> list[PaperStepMetrics]:
    """Repeat one fixed batch as a development gate, not an evaluation run."""

    config.validate()
    if context.shape[0] != config.batch_size:
        raise ValueError(
            f"expected the frozen batch size {config.batch_size}; "
            f"received {context.shape[0]}"
        )
    return [
        paper_train_step(model, context, target, optimizer)
        for _ in range(config.steps)
    ]


def evaluate_overfit_gate(
    history: list[PaperStepMetrics],
    config: PaperOverfitGateConfig,
) -> PaperOverfitGateResult:
    """Evaluate preregistered loss-reduction and anti-collapse checks."""

    config.validate()
    if len(history) < config.comparison_window:
        raise ValueError("history is shorter than the comparison window")

    initial = history[: config.comparison_window]
    final = history[-config.comparison_window :]
    all_values = [
        value
        for row in history
        for value in (
            row.loss,
            row.online_gradient_norm,
            row.predictor_gradient_norm,
            row.embedding_std_mean,
            row.effective_rank,
        )
    ]
    all_finite = all(math.isfinite(value) for value in all_values)
    initial_loss = fmean(row.loss for row in initial)
    final_loss = fmean(row.loss for row in final)
    loss_ratio = final_loss / max(initial_loss, 1e-12)
    initial_embedding_std = fmean(row.embedding_std_mean for row in initial)
    final_embedding_std = fmean(row.embedding_std_mean for row in final)
    embedding_std_ratio = final_embedding_std / max(initial_embedding_std, 1e-12)
    final_effective_rank = fmean(row.effective_rank for row in final)

    loss_passed = loss_ratio <= config.max_loss_ratio
    spread_passed = embedding_std_ratio >= config.min_embedding_std_ratio
    rank_passed = final_effective_rank >= config.min_effective_rank
    return PaperOverfitGateResult(
        all_finite=all_finite,
        initial_loss=initial_loss,
        final_loss=final_loss,
        loss_ratio=loss_ratio,
        initial_embedding_std=initial_embedding_std,
        final_embedding_std=final_embedding_std,
        embedding_std_ratio=embedding_std_ratio,
        final_effective_rank=final_effective_rank,
        loss_passed=loss_passed,
        spread_passed=spread_passed,
        rank_passed=rank_passed,
        passed=all_finite and loss_passed and spread_passed and rank_passed,
    )

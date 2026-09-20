from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import fmean, pstdev

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .paper_config import (
    PaperOptimizationConfig,
    PaperOverfitGateConfig,
    PaperSeedStabilityGateConfig,
    PaperSeedSweepConfig,
    PaperTrainConfig,
    PaperValidationGateConfig,
)
from .paper_model import PaperTemporalJEPA

PaperDataset = Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]


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


@dataclass(frozen=True, slots=True)
class PaperEvaluationMetrics:
    loss: float
    embedding_std_mean: float
    effective_rank: float
    sample_count: int


@dataclass(frozen=True, slots=True)
class PaperEpochMetrics:
    epoch: int
    train_loss: float
    validation_loss: float
    train_embedding_std: float
    validation_embedding_std: float
    train_effective_rank: float
    validation_effective_rank: float
    online_gradient_norm: float
    predictor_gradient_norm: float


@dataclass(frozen=True, slots=True)
class PaperValidationGateResult:
    all_finite: bool
    initial_validation_loss: float
    final_train_loss: float
    final_validation_loss: float
    validation_loss_ratio: float
    validation_train_loss_ratio: float
    initial_validation_embedding_std: float
    final_validation_embedding_std: float
    validation_embedding_std_ratio: float
    final_validation_effective_rank: float
    loss_passed: bool
    generalization_gap_passed: bool
    spread_passed: bool
    rank_passed: bool
    passed: bool


@dataclass(frozen=True, slots=True)
class PaperCheckpointSelection:
    epoch: int
    train_loss: float
    validation_loss: float
    validation_loss_ratio: float
    validation_train_loss_ratio: float
    validation_embedding_std: float
    validation_embedding_std_ratio: float
    validation_effective_rank: float


@dataclass(frozen=True, slots=True)
class PaperSeedSummary:
    seed: int
    checkpoint_epoch: int
    train_loss: float
    validation_loss: float
    validation_loss_ratio: float
    validation_train_loss_ratio: float
    validation_embedding_std_ratio: float
    validation_effective_rank: float


@dataclass(frozen=True, slots=True)
class PaperSeedStabilityGateResult:
    all_checkpoints_selected: bool
    all_finite: bool
    mean_validation_loss: float
    validation_loss_coefficient_of_variation: float
    worst_validation_loss_ratio: float
    worst_validation_train_loss_ratio: float
    worst_validation_embedding_std_ratio: float
    worst_validation_effective_rank: float
    validation_loss_passed: bool
    generalization_gap_passed: bool
    variability_passed: bool
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
    config: PaperOptimizationConfig | PaperTrainConfig,
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
def embedding_spread(embeddings: torch.Tensor) -> tuple[float, float]:
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
    embedding_std_mean, effective_rank = embedding_spread(online_embedding)

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


def make_paper_loader(
    dataset: PaperDataset,
    config: PaperTrainConfig,
    *,
    shuffle: bool,
) -> DataLoader:
    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        generator=generator,
        drop_last=False,
    )


@torch.no_grad()
def evaluate_paper_model(
    model: PaperTemporalJEPA,
    loader: DataLoader,
    device: torch.device,
) -> PaperEvaluationMetrics:
    model.eval()
    total_loss = 0.0
    sample_count = 0
    embeddings: list[torch.Tensor] = []

    for context, target, _ in loader:
        context = context.to(device)
        target = target.to(device)
        online_embedding, prediction, target_embedding = model(context, target)
        batch_size = context.shape[0]
        total_loss += float(squared_embedding_error(prediction, target_embedding)) * batch_size
        sample_count += batch_size
        embeddings.append(online_embedding.cpu())

    if sample_count == 0:
        raise ValueError("cannot evaluate an empty dataset")
    embedding_std_mean, effective_rank = embedding_spread(torch.cat(embeddings, dim=0))
    return PaperEvaluationMetrics(
        loss=total_loss / sample_count,
        embedding_std_mean=embedding_std_mean,
        effective_rank=effective_rank,
        sample_count=sample_count,
    )


def _epoch_metrics(
    epoch: int,
    train_evaluation: PaperEvaluationMetrics,
    validation_evaluation: PaperEvaluationMetrics,
    online_gradient_norm: float,
    predictor_gradient_norm: float,
) -> PaperEpochMetrics:
    return PaperEpochMetrics(
        epoch=epoch,
        train_loss=train_evaluation.loss,
        validation_loss=validation_evaluation.loss,
        train_embedding_std=train_evaluation.embedding_std_mean,
        validation_embedding_std=validation_evaluation.embedding_std_mean,
        train_effective_rank=train_evaluation.effective_rank,
        validation_effective_rank=validation_evaluation.effective_rank,
        online_gradient_norm=online_gradient_norm,
        predictor_gradient_norm=predictor_gradient_norm,
    )


def run_paper_train_validation(
    model: PaperTemporalJEPA,
    train_dataset: PaperDataset,
    validation_dataset: PaperDataset,
    config: PaperTrainConfig,
) -> list[PaperEpochMetrics]:
    """Run a deterministic local train/validation development condition."""

    config.validate()
    device = torch.device(config.device)
    model.to(device)
    optimizer = make_paper_optimizer(model, config)
    train_loader = make_paper_loader(train_dataset, config, shuffle=True)
    train_evaluation_loader = make_paper_loader(train_dataset, config, shuffle=False)
    validation_loader = make_paper_loader(validation_dataset, config, shuffle=False)

    train_evaluation = evaluate_paper_model(model, train_evaluation_loader, device)
    validation_evaluation = evaluate_paper_model(model, validation_loader, device)
    history = [
        _epoch_metrics(
            0,
            train_evaluation,
            validation_evaluation,
            online_gradient_norm=0.0,
            predictor_gradient_norm=0.0,
        )
    ]

    for epoch in range(1, config.epochs + 1):
        gradient_sample_count = 0
        online_gradient_total = 0.0
        predictor_gradient_total = 0.0
        for context, target, _ in train_loader:
            context = context.to(device)
            target = target.to(device)
            step = paper_train_step(model, context, target, optimizer)
            batch_size = context.shape[0]
            gradient_sample_count += batch_size
            online_gradient_total += step.online_gradient_norm * batch_size
            predictor_gradient_total += step.predictor_gradient_norm * batch_size

        train_evaluation = evaluate_paper_model(model, train_evaluation_loader, device)
        validation_evaluation = evaluate_paper_model(model, validation_loader, device)
        history.append(
            _epoch_metrics(
                epoch,
                train_evaluation,
                validation_evaluation,
                online_gradient_norm=online_gradient_total / gradient_sample_count,
                predictor_gradient_norm=predictor_gradient_total / gradient_sample_count,
            )
        )

    return history


def evaluate_validation_gate(
    history: list[PaperEpochMetrics],
    config: PaperValidationGateConfig,
) -> PaperValidationGateResult:
    """Evaluate preregistered held-out improvement and anti-collapse checks."""

    config.validate()
    if len(history) < config.final_window + 1:
        raise ValueError("history must include epoch zero and the final window")

    initial = history[0]
    final = history[-config.final_window :]
    all_values = [
        value
        for row in history
        for value in (
            row.train_loss,
            row.validation_loss,
            row.train_embedding_std,
            row.validation_embedding_std,
            row.train_effective_rank,
            row.validation_effective_rank,
            row.online_gradient_norm,
            row.predictor_gradient_norm,
        )
    ]
    all_finite = all(math.isfinite(value) for value in all_values)
    final_train_loss = fmean(row.train_loss for row in final)
    final_validation_loss = fmean(row.validation_loss for row in final)
    validation_loss_ratio = final_validation_loss / max(initial.validation_loss, 1e-12)
    validation_train_loss_ratio = final_validation_loss / max(final_train_loss, 1e-12)
    final_validation_embedding_std = fmean(row.validation_embedding_std for row in final)
    validation_embedding_std_ratio = final_validation_embedding_std / max(
        initial.validation_embedding_std,
        1e-12,
    )
    final_validation_effective_rank = fmean(
        row.validation_effective_rank for row in final
    )

    loss_passed = validation_loss_ratio <= config.max_validation_loss_ratio
    generalization_gap_passed = (
        validation_train_loss_ratio <= config.max_validation_train_loss_ratio
    )
    spread_passed = (
        validation_embedding_std_ratio >= config.min_validation_embedding_std_ratio
    )
    rank_passed = (
        final_validation_effective_rank >= config.min_validation_effective_rank
    )
    return PaperValidationGateResult(
        all_finite=all_finite,
        initial_validation_loss=initial.validation_loss,
        final_train_loss=final_train_loss,
        final_validation_loss=final_validation_loss,
        validation_loss_ratio=validation_loss_ratio,
        validation_train_loss_ratio=validation_train_loss_ratio,
        initial_validation_embedding_std=initial.validation_embedding_std,
        final_validation_embedding_std=final_validation_embedding_std,
        validation_embedding_std_ratio=validation_embedding_std_ratio,
        final_validation_effective_rank=final_validation_effective_rank,
        loss_passed=loss_passed,
        generalization_gap_passed=generalization_gap_passed,
        spread_passed=spread_passed,
        rank_passed=rank_passed,
        passed=(
            all_finite
            and loss_passed
            and generalization_gap_passed
            and spread_passed
            and rank_passed
        ),
    )


def select_validation_checkpoint(
    history: list[PaperEpochMetrics],
    config: PaperValidationGateConfig,
) -> PaperCheckpointSelection | None:
    """Select the lowest validation loss among constraint-eligible epochs."""

    config.validate()
    if len(history) < 2 or history[0].epoch != 0:
        raise ValueError("history must start with untrained epoch zero")

    initial = history[0]
    candidates: list[PaperCheckpointSelection] = []
    for row in history[1:]:
        values = (
            row.train_loss,
            row.validation_loss,
            row.validation_embedding_std,
            row.validation_effective_rank,
        )
        if not all(math.isfinite(value) for value in values):
            continue
        validation_loss_ratio = row.validation_loss / max(
            initial.validation_loss,
            1e-12,
        )
        validation_train_loss_ratio = row.validation_loss / max(row.train_loss, 1e-12)
        validation_embedding_std_ratio = row.validation_embedding_std / max(
            initial.validation_embedding_std,
            1e-12,
        )
        if validation_loss_ratio > config.max_validation_loss_ratio:
            continue
        if validation_train_loss_ratio > config.max_validation_train_loss_ratio:
            continue
        if (
            validation_embedding_std_ratio
            < config.min_validation_embedding_std_ratio
        ):
            continue
        if row.validation_effective_rank < config.min_validation_effective_rank:
            continue
        candidates.append(
            PaperCheckpointSelection(
                epoch=row.epoch,
                train_loss=row.train_loss,
                validation_loss=row.validation_loss,
                validation_loss_ratio=validation_loss_ratio,
                validation_train_loss_ratio=validation_train_loss_ratio,
                validation_embedding_std=row.validation_embedding_std,
                validation_embedding_std_ratio=validation_embedding_std_ratio,
                validation_effective_rank=row.validation_effective_rank,
            )
        )

    if not candidates:
        return None
    return min(candidates, key=lambda candidate: (candidate.validation_loss, candidate.epoch))


def summarize_seed_checkpoint(
    seed: int,
    selection: PaperCheckpointSelection,
) -> PaperSeedSummary:
    if seed < 0:
        raise ValueError("seed must be non-negative")
    return PaperSeedSummary(
        seed=seed,
        checkpoint_epoch=selection.epoch,
        train_loss=selection.train_loss,
        validation_loss=selection.validation_loss,
        validation_loss_ratio=selection.validation_loss_ratio,
        validation_train_loss_ratio=selection.validation_train_loss_ratio,
        validation_embedding_std_ratio=selection.validation_embedding_std_ratio,
        validation_effective_rank=selection.validation_effective_rank,
    )


def evaluate_seed_stability_gate(
    summaries: list[PaperSeedSummary],
    sweep: PaperSeedSweepConfig,
    config: PaperSeedStabilityGateConfig,
) -> PaperSeedStabilityGateResult:
    """Evaluate whether every seed yields a stable eligible checkpoint."""

    sweep.validate()
    config.validate()
    expected_seeds = set(sweep.seeds)
    observed_seeds = [summary.seed for summary in summaries]
    all_checkpoints_selected = (
        len(observed_seeds) == len(expected_seeds)
        and len(set(observed_seeds)) == len(observed_seeds)
        and set(observed_seeds) == expected_seeds
    )
    all_values = [
        value
        for summary in summaries
        for value in (
            summary.train_loss,
            summary.validation_loss,
            summary.validation_loss_ratio,
            summary.validation_train_loss_ratio,
            summary.validation_embedding_std_ratio,
            summary.validation_effective_rank,
        )
    ]
    all_finite = bool(all_values) and all(math.isfinite(value) for value in all_values)

    if summaries:
        validation_losses = [summary.validation_loss for summary in summaries]
        mean_validation_loss = fmean(validation_losses)
        validation_loss_coefficient_of_variation = pstdev(validation_losses) / max(
            mean_validation_loss,
            1e-12,
        )
        worst_validation_loss_ratio = max(
            summary.validation_loss_ratio for summary in summaries
        )
        worst_validation_train_loss_ratio = max(
            summary.validation_train_loss_ratio for summary in summaries
        )
        worst_validation_embedding_std_ratio = min(
            summary.validation_embedding_std_ratio for summary in summaries
        )
        worst_validation_effective_rank = min(
            summary.validation_effective_rank for summary in summaries
        )
    else:
        mean_validation_loss = math.inf
        validation_loss_coefficient_of_variation = math.inf
        worst_validation_loss_ratio = math.inf
        worst_validation_train_loss_ratio = math.inf
        worst_validation_embedding_std_ratio = 0.0
        worst_validation_effective_rank = 0.0

    validation_loss_passed = (
        worst_validation_loss_ratio <= config.max_worst_validation_loss_ratio
    )
    generalization_gap_passed = (
        worst_validation_train_loss_ratio
        <= config.max_worst_validation_train_loss_ratio
    )
    variability_passed = (
        validation_loss_coefficient_of_variation
        <= config.max_validation_loss_coefficient_of_variation
    )
    spread_passed = (
        worst_validation_embedding_std_ratio
        >= config.min_worst_validation_embedding_std_ratio
    )
    rank_passed = (
        worst_validation_effective_rank
        >= config.min_worst_validation_effective_rank
    )
    return PaperSeedStabilityGateResult(
        all_checkpoints_selected=all_checkpoints_selected,
        all_finite=all_finite,
        mean_validation_loss=mean_validation_loss,
        validation_loss_coefficient_of_variation=(
            validation_loss_coefficient_of_variation
        ),
        worst_validation_loss_ratio=worst_validation_loss_ratio,
        worst_validation_train_loss_ratio=worst_validation_train_loss_ratio,
        worst_validation_embedding_std_ratio=worst_validation_embedding_std_ratio,
        worst_validation_effective_rank=worst_validation_effective_rank,
        validation_loss_passed=validation_loss_passed,
        generalization_gap_passed=generalization_gap_passed,
        variability_passed=variability_passed,
        spread_passed=spread_passed,
        rank_passed=rank_passed,
        passed=(
            all_checkpoints_selected
            and all_finite
            and validation_loss_passed
            and generalization_gap_passed
            and variability_passed
            and spread_passed
            and rank_passed
        ),
    )

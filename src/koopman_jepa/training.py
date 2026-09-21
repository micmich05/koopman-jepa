from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import ExperimentConfig
from .model import TemporalJEPA, covariance_loss, mean_loss, variance_loss


@dataclass(slots=True)
class EpochMetrics:
    loss: float
    prediction_loss: float
    mean_loss: float
    variance_loss: float
    covariance_loss: float
    online_gradient_norm: float
    predictor_gradient_norm: float


@dataclass(slots=True)
class TrainingResult:
    history: list[dict[str, float]]
    best_epoch: int
    best_validation_loss: float


EpochCallback = Callable[
    [int, TemporalJEPA, dict[str, float]],
    dict[str, float] | None,
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loader(
    dataset: TensorDataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
        drop_last=False,
    )


def _gradient_norm(parameters: list[torch.nn.Parameter]) -> float:
    squared_norm = torch.zeros((), device=parameters[0].device)
    for parameter in parameters:
        if parameter.grad is not None:
            squared_norm += parameter.grad.detach().square().sum()
    return float(torch.sqrt(squared_norm))


def _run_epoch(
    model: TemporalJEPA,
    loader: DataLoader,
    device: torch.device,
    config: ExperimentConfig,
    optimizer: torch.optim.Optimizer | None,
) -> EpochMetrics:
    training = optimizer is not None
    model.train(training)
    mse = nn.MSELoss()
    totals = np.zeros(8, dtype=np.float64)

    for context, target, _ in loader:
        context = context.to(device)
        target = target.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            online_embedding, prediction, target_embedding = model(context, target)
            prediction_term = mse(prediction, target_embedding)

            mean_term = torch.zeros((), device=device)
            variance_term = torch.zeros((), device=device)
            covariance_term = torch.zeros((), device=device)
            if (
                config.train.mean_weight > 0.0
                or config.train.variance_weight > 0.0
                or config.train.covariance_weight > 0.0
            ):
                online_target = model.online_encoder(target)
                joined = torch.cat([online_embedding, online_target], dim=0)
                mean_term = mean_loss(joined)
                variance_term = variance_loss(joined)
                covariance_term = covariance_loss(joined)

            loss = (
                prediction_term
                + config.train.mean_weight * mean_term
                + config.train.variance_weight * variance_term
                + config.train.covariance_weight * covariance_term
            )

            online_gradient_norm = 0.0
            predictor_gradient_norm = 0.0
            if training:
                loss.backward()
                online_parameters = list(model.online_encoder.parameters())
                predictor_parameters = list(model.predictor.parameters())
                online_gradient_norm = _gradient_norm(online_parameters)
                predictor_gradient_norm = _gradient_norm(predictor_parameters)
                torch.nn.utils.clip_grad_norm_(
                    online_parameters + predictor_parameters,
                    max_norm=5.0,
                )
                optimizer.step()
                model.update_target(config.train.ema_momentum)

        batch_size = context.shape[0]
        totals += np.array(
            [
                batch_size,
                float(loss.detach()),
                float(prediction_term.detach()),
                float(mean_term.detach()),
                float(variance_term.detach()),
                float(covariance_term.detach()),
                online_gradient_norm,
                predictor_gradient_norm,
            ]
        ) * np.array(
            [
                1.0,
                batch_size,
                batch_size,
                batch_size,
                batch_size,
                batch_size,
                batch_size,
                batch_size,
            ]
        )

    count = max(totals[0], 1.0)
    return EpochMetrics(
        loss=float(totals[1] / count),
        prediction_loss=float(totals[2] / count),
        mean_loss=float(totals[3] / count),
        variance_loss=float(totals[4] / count),
        covariance_loss=float(totals[5] / count),
        online_gradient_norm=float(totals[6] / count),
        predictor_gradient_norm=float(totals[7] / count),
    )


def make_optimizer(
    model: TemporalJEPA,
    config: ExperimentConfig,
) -> torch.optim.AdamW:
    """Build AdamW with an explicit time-scale control for the predictor."""

    base_learning_rate = config.train.learning_rate
    return torch.optim.AdamW(
        [
            {
                "params": model.online_encoder.parameters(),
                "lr": base_learning_rate,
            },
            {
                "params": model.predictor.parameters(),
                "lr": (
                    base_learning_rate
                    * config.train.predictor_learning_rate_multiplier
                ),
            },
        ],
        weight_decay=config.train.weight_decay,
    )


def _fit_model(
    model: TemporalJEPA,
    train_dataset: TensorDataset,
    val_dataset: TensorDataset,
    config: ExperimentConfig,
    device: torch.device,
    restore_best: bool,
    epoch_callback: EpochCallback | None = None,
) -> TrainingResult:
    train_loader = make_loader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        seed=config.train.seed,
        num_workers=config.train.num_workers,
    )
    val_loader = make_loader(
        val_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        seed=config.train.seed,
        num_workers=config.train.num_workers,
    )
    optimizer = make_optimizer(model, config)

    history: list[dict[str, float]] = []
    best_epoch = 0
    best_validation_loss = float("inf")
    best_state: dict[str, Any] | None = None
    for epoch in range(1, config.train.epochs + 1):
        train_metrics = _run_epoch(model, train_loader, device, config, optimizer)
        with torch.no_grad():
            val_metrics = _run_epoch(model, val_loader, device, config, optimizer=None)

        row = {
            "epoch": float(epoch),
            "train_loss": train_metrics.loss,
            "train_prediction_loss": train_metrics.prediction_loss,
            "train_mean_loss": train_metrics.mean_loss,
            "train_variance_loss": train_metrics.variance_loss,
            "train_covariance_loss": train_metrics.covariance_loss,
            "train_online_gradient_norm": train_metrics.online_gradient_norm,
            "train_predictor_gradient_norm": train_metrics.predictor_gradient_norm,
            "val_loss": val_metrics.loss,
            "val_prediction_loss": val_metrics.prediction_loss,
            "val_mean_loss": val_metrics.mean_loss,
            "val_variance_loss": val_metrics.variance_loss,
            "val_covariance_loss": val_metrics.covariance_loss,
        }
        if epoch_callback is not None:
            callback_metrics = epoch_callback(epoch, model, row)
            if callback_metrics is not None:
                overlap = row.keys() & callback_metrics.keys()
                if overlap:
                    raise ValueError(f"epoch callback cannot overwrite metrics: {sorted(overlap)}")
                row.update(callback_metrics)
        history.append(row)
        if val_metrics.loss < best_validation_loss:
            best_epoch = epoch
            best_validation_loss = val_metrics.loss
            best_state = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
            }

        if config.train.freeze_encoder_after_epoch == epoch:
            model.online_encoder.requires_grad_(False)

        if epoch == 1 or epoch % max(config.train.epochs // 5, 1) == 0:
            print(
                f"epoch={epoch:03d} "
                f"train={train_metrics.loss:.6f} "
                f"val={val_metrics.loss:.6f}"
            )

    if restore_best:
        if best_state is None:
            raise RuntimeError("training completed without a validation checkpoint")
        model.load_state_dict(best_state)
    return TrainingResult(
        history=history,
        best_epoch=best_epoch,
        best_validation_loss=float(best_validation_loss),
    )


def train_model(
    model: TemporalJEPA,
    train_dataset: TensorDataset,
    val_dataset: TensorDataset,
    config: ExperimentConfig,
    device: torch.device,
    epoch_callback: EpochCallback | None = None,
) -> list[dict[str, float]]:
    """Train through the fixed horizon and keep the final model state."""

    return _fit_model(
        model,
        train_dataset,
        val_dataset,
        config,
        device,
        restore_best=False,
        epoch_callback=epoch_callback,
    ).history


def train_model_with_validation_checkpoint(
    model: TemporalJEPA,
    train_dataset: TensorDataset,
    val_dataset: TensorDataset,
    config: ExperimentConfig,
    device: torch.device,
    epoch_callback: EpochCallback | None = None,
) -> TrainingResult:
    """Train and restore the checkpoint with the lowest total validation loss."""

    return _fit_model(
        model,
        train_dataset,
        val_dataset,
        config,
        device,
        restore_best=True,
        epoch_callback=epoch_callback,
    )


@torch.no_grad()
def evaluate_model_loss(
    model: TemporalJEPA,
    dataset: TensorDataset,
    config: ExperimentConfig,
    device: torch.device,
) -> EpochMetrics:
    loader = make_loader(
        dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        seed=config.train.seed,
        num_workers=config.train.num_workers,
    )
    return _run_epoch(model, loader, device, config, optimizer=None)


@torch.no_grad()
def collect_embeddings(
    model: TemporalJEPA,
    dataset: TensorDataset,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    online_embeddings: list[np.ndarray] = []
    target_embeddings: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    model.eval()
    for context, target, batch_labels in loader:
        context = context.to(device)
        target = target.to(device)
        online_embeddings.append(model.online_encoder(context).cpu().numpy())
        target_embeddings.append(model.target_encoder(target).cpu().numpy())
        labels.append(batch_labels.numpy())

    return (
        np.concatenate(online_embeddings, axis=0),
        np.concatenate(target_embeddings, axis=0),
        np.concatenate(labels, axis=0),
    )


@torch.no_grad()
def collect_paired_embeddings(
    model: TemporalJEPA,
    dataset: TensorDataset,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Encode current/future windows online and future windows with the EMA target."""

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    current_online: list[np.ndarray] = []
    future_online: list[np.ndarray] = []
    future_target: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    model.eval()
    for current, future, batch_labels in loader:
        current = current.to(device)
        future = future.to(device)
        current_online.append(model.online_encoder(current).cpu().numpy())
        future_online.append(model.online_encoder(future).cpu().numpy())
        future_target.append(model.target_encoder(future).cpu().numpy())
        labels.append(batch_labels.numpy())
    return (
        np.concatenate(current_online, axis=0),
        np.concatenate(future_online, axis=0),
        np.concatenate(future_target, axis=0),
        np.concatenate(labels, axis=0),
    )

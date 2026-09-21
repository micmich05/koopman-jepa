from __future__ import annotations

import random
from dataclasses import dataclass

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
    totals = np.zeros(6, dtype=np.float64)

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

            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(model.online_encoder.parameters())
                    + list(model.predictor.parameters()),
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
            ]
        ) * np.array([1.0, batch_size, batch_size, batch_size, batch_size, batch_size])

    count = max(totals[0], 1.0)
    return EpochMetrics(
        loss=float(totals[1] / count),
        prediction_loss=float(totals[2] / count),
        mean_loss=float(totals[3] / count),
        variance_loss=float(totals[4] / count),
        covariance_loss=float(totals[5] / count),
    )


def train_model(
    model: TemporalJEPA,
    train_dataset: TensorDataset,
    val_dataset: TensorDataset,
    config: ExperimentConfig,
    device: torch.device,
) -> list[dict[str, float]]:
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
    optimizer = torch.optim.AdamW(
        list(model.online_encoder.parameters()) + list(model.predictor.parameters()),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )

    history: list[dict[str, float]] = []
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
            "val_loss": val_metrics.loss,
            "val_prediction_loss": val_metrics.prediction_loss,
            "val_mean_loss": val_metrics.mean_loss,
            "val_variance_loss": val_metrics.variance_loss,
            "val_covariance_loss": val_metrics.covariance_loss,
        }
        history.append(row)

        if epoch == 1 or epoch % max(config.train.epochs // 5, 1) == 0:
            print(
                f"epoch={epoch:03d} "
                f"train={train_metrics.loss:.6f} "
                f"val={val_metrics.loss:.6f}"
            )

    return history


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

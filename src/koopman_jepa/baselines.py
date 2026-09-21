from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .model import TemporalEncoder


@dataclass(frozen=True, slots=True)
class RidgeOperatorSelection:
    """A feature-space operator selected without phase labels."""

    matrix: np.ndarray
    feature_mean: np.ndarray
    regularization: float
    validation_mse: float
    validation_mse_by_regularization: dict[float, float]

    def center(self, features: np.ndarray) -> np.ndarray:
        features = _feature_matrix(features, "features")
        if features.shape[1] != self.matrix.shape[0]:
            raise ValueError("features must match the fitted operator dimension")
        return features - self.feature_mean


@dataclass(frozen=True, slots=True)
class PCAFeatureMap:
    """A PCA map fitted without labels on pooled train observations."""

    observation_mean: np.ndarray
    components: np.ndarray
    explained_variance: np.ndarray

    def transform(self, windows: np.ndarray) -> np.ndarray:
        observations = flatten_windows(windows)
        if observations.shape[1] != self.components.shape[1]:
            raise ValueError("windows must match the fitted observation dimension")
        return (observations - self.observation_mean) @ self.components.T


@dataclass(frozen=True, slots=True)
class PCAOperatorSelection:
    """A PCA representation and its validation-selected linear operator."""

    feature_map: PCAFeatureMap
    operator: RidgeOperatorSelection

    def transform(self, windows: np.ndarray) -> np.ndarray:
        return self.operator.center(self.feature_map.transform(windows))


@dataclass(frozen=True, slots=True)
class CNNOperatorSelection:
    """A fixed CNN encoder and its validation-selected linear operator."""

    encoder: TemporalEncoder
    operator: RidgeOperatorSelection
    batch_size: int

    def transform(self, windows: np.ndarray | torch.Tensor) -> np.ndarray:
        features = collect_encoder_features(self.encoder, windows, self.batch_size)
        return self.operator.center(features)


class PhaseClassifier(nn.Module):
    """The shared temporal encoder with a supervised phase head."""

    def __init__(
        self,
        latent_dim: int,
        channels: list[int],
        pooling: str,
        input_length: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least two")
        self.encoder = TemporalEncoder(
            latent_dim=latent_dim,
            channels=channels,
            pooling=pooling,  # type: ignore[arg-type]
            input_length=input_length,
        )
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(self, windows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.encoder(windows)
        return embeddings, self.classifier(embeddings)


@dataclass(frozen=True, slots=True)
class SupervisedEncoderTraining:
    """A phase-supervised encoder frozen at the final epoch."""

    model: PhaseClassifier
    history: tuple[dict[str, float], ...]

    @property
    def encoder(self) -> TemporalEncoder:
        return self.model.encoder


def _feature_matrix(values: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional feature matrix")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must have non-empty sample and feature dimensions")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")
    return matrix


def flatten_windows(windows: np.ndarray) -> np.ndarray:
    """Flatten all non-batch observation axes into a feature matrix."""

    observations = np.asarray(windows, dtype=np.float64)
    if observations.ndim < 2:
        raise ValueError("windows must have a batch axis and observation axes")
    if observations.shape[0] == 0 or any(size == 0 for size in observations.shape[1:]):
        raise ValueError("windows must have non-empty batch and observation axes")
    if not np.isfinite(observations).all():
        raise ValueError("windows must contain only finite values")
    return observations.reshape(observations.shape[0], -1)


def make_random_cnn_encoder(
    seed: int,
    latent_dim: int,
    channels: list[int],
    pooling: str,
    input_length: int,
    device: torch.device | str = "cpu",
) -> TemporalEncoder:
    """Create the exact online-encoder initialization used by a seeded JEPA."""

    if not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        encoder = TemporalEncoder(
            latent_dim=latent_dim,
            channels=channels,
            pooling=pooling,  # type: ignore[arg-type]
            input_length=input_length,
        )
    encoder.requires_grad_(False)
    encoder.eval()
    return encoder.to(device)


def _window_tensor(windows: np.ndarray | torch.Tensor) -> torch.Tensor:
    tensor = torch.as_tensor(windows, dtype=torch.float32)
    if tensor.ndim != 3 or tensor.shape[1] != 1:
        raise ValueError("windows must have shape (samples, 1, length)")
    if tensor.shape[0] == 0 or tensor.shape[2] == 0:
        raise ValueError("windows must have non-empty sample and length dimensions")
    if not torch.isfinite(tensor).all():
        raise ValueError("windows must contain only finite values")
    return tensor


@torch.no_grad()
def collect_encoder_features(
    encoder: nn.Module,
    windows: np.ndarray | torch.Tensor,
    batch_size: int,
) -> np.ndarray:
    """Encode windows deterministically without updating the encoder."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    tensor = _window_tensor(windows)

    try:
        device = next(encoder.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    encoder.eval()
    batches = [
        encoder(tensor[start : start + batch_size].to(device)).detach().cpu().numpy()
        for start in range(0, tensor.shape[0], batch_size)
    ]
    return np.concatenate(batches, axis=0).astype(np.float64, copy=False)


def _phase_labels(
    labels: np.ndarray | torch.Tensor,
    sample_count: int,
    num_classes: int,
    name: str,
) -> torch.Tensor:
    tensor = torch.as_tensor(labels, dtype=torch.long)
    if tensor.shape != (sample_count,):
        raise ValueError(f"{name} must have one label per window")
    if torch.any(tensor < 0) or torch.any(tensor >= num_classes):
        raise ValueError(f"{name} must lie in [0, num_classes)")
    return tensor


def train_supervised_phase_encoder(
    current_windows: np.ndarray | torch.Tensor,
    future_windows: np.ndarray | torch.Tensor,
    current_phases: np.ndarray | torch.Tensor,
    future_phases: np.ndarray | torch.Tensor,
    *,
    seed: int,
    latent_dim: int,
    channels: list[int],
    pooling: str,
    input_length: int,
    num_classes: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device | str = "cpu",
) -> SupervisedEncoderTraining:
    """Train the practical phase-supervised ceiling and freeze its encoder."""

    current = _window_tensor(current_windows)
    future = _window_tensor(future_windows)
    if current.shape != future.shape:
        raise ValueError("current and future windows must have equal shape")
    current_labels = _phase_labels(
        current_phases,
        current.shape[0],
        num_classes,
        "current_phases",
    )
    future_labels = _phase_labels(
        future_phases,
        future.shape[0],
        num_classes,
        "future_phases",
    )
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    if learning_rate <= 0.0 or weight_decay < 0.0:
        raise ValueError("learning_rate must be positive and weight_decay non-negative")

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = PhaseClassifier(
            latent_dim=latent_dim,
            channels=channels,
            pooling=pooling,
            input_length=input_length,
            num_classes=num_classes,
        ).to(device)

    dataset = TensorDataset(
        torch.cat([current, future], dim=0),
        torch.cat([current_labels, future_labels], dim=0),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_count = 0
        for batch_windows, batch_labels in loader:
            batch_windows = batch_windows.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            _, logits = model(batch_windows)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * batch_windows.shape[0]
            total_correct += int((logits.argmax(dim=1) == batch_labels).sum())
            total_count += batch_windows.shape[0]
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": total_loss / total_count,
                "train_accuracy": total_correct / total_count,
            }
        )

    model.eval()
    model.requires_grad_(False)
    return SupervisedEncoderTraining(model=model, history=tuple(history))


def fit_pca_feature_map(
    train_current: np.ndarray,
    train_future: np.ndarray,
    n_components: int,
) -> PCAFeatureMap:
    """Fit PCA on the pooled current and future train marginals."""

    current = flatten_windows(train_current)
    future = flatten_windows(train_future)
    if current.shape != future.shape:
        raise ValueError("train current and future windows must have equal shape")
    if not isinstance(n_components, int) or not 1 <= n_components <= min(
        current.shape[1],
        2 * current.shape[0],
    ):
        raise ValueError("n_components must fit the pooled observation matrix")

    pooled = np.concatenate([current, future], axis=0)
    observation_mean = pooled.mean(axis=0)
    _, singular_values, right_vectors = np.linalg.svd(
        pooled - observation_mean,
        full_matrices=False,
    )
    explained_variance = np.square(singular_values[:n_components]) / (pooled.shape[0] - 1)
    return PCAFeatureMap(
        observation_mean=observation_mean,
        components=right_vectors[:n_components],
        explained_variance=explained_variance,
    )


def _paired_features(
    current: np.ndarray,
    future: np.ndarray,
    split_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    current_matrix = _feature_matrix(current, f"{split_name}_current")
    future_matrix = _feature_matrix(future, f"{split_name}_future")
    if current_matrix.shape != future_matrix.shape:
        raise ValueError(f"{split_name} current and future features must have equal shape")
    return current_matrix, future_matrix


def fit_ridge_operator(
    current: np.ndarray,
    future: np.ndarray,
    regularization: float,
) -> np.ndarray:
    """Fit ``future ~= current @ M.T`` with a normalized ridge objective."""

    current_matrix, future_matrix = _paired_features(current, future, "train")
    regularization = float(regularization)
    if not np.isfinite(regularization) or regularization < 0.0:
        raise ValueError("regularization must be finite and non-negative")

    if regularization == 0.0:
        row_operator, *_ = np.linalg.lstsq(current_matrix, future_matrix, rcond=None)
    else:
        sample_count = current_matrix.shape[0]
        gram = current_matrix.T @ current_matrix / sample_count
        cross_covariance = current_matrix.T @ future_matrix / sample_count
        penalty = regularization * np.eye(current_matrix.shape[1])
        row_operator = np.linalg.solve(gram + penalty, cross_covariance)
    return row_operator.T


def select_ridge_operator(
    train_current: np.ndarray,
    train_future: np.ndarray,
    validation_current: np.ndarray,
    validation_future: np.ndarray,
    regularizations: Sequence[float],
) -> RidgeOperatorSelection:
    """Select ridge strength by validation feature MSE, without phase labels."""

    train_current_matrix, train_future_matrix = _paired_features(
        train_current,
        train_future,
        "train",
    )
    validation_current_matrix, validation_future_matrix = _paired_features(
        validation_current,
        validation_future,
        "validation",
    )
    if train_current_matrix.shape[1] != validation_current_matrix.shape[1]:
        raise ValueError("train and validation feature dimensions must match")

    candidates = tuple(sorted({float(value) for value in regularizations}))
    if not candidates:
        raise ValueError("regularizations must be non-empty")
    if any(not np.isfinite(value) or value < 0.0 for value in candidates):
        raise ValueError("regularizations must be finite and non-negative")

    feature_mean = np.concatenate(
        [train_current_matrix, train_future_matrix],
        axis=0,
    ).mean(axis=0)
    centered_train_current = train_current_matrix - feature_mean
    centered_train_future = train_future_matrix - feature_mean
    centered_validation_current = validation_current_matrix - feature_mean
    centered_validation_future = validation_future_matrix - feature_mean

    matrices: dict[float, np.ndarray] = {}
    validation_mses: dict[float, float] = {}
    for regularization in candidates:
        matrix = fit_ridge_operator(
            centered_train_current,
            centered_train_future,
            regularization,
        )
        prediction = centered_validation_current @ matrix.T
        matrices[regularization] = matrix
        validation_mses[regularization] = float(
            np.mean(np.square(prediction - centered_validation_future))
        )

    selected = min(candidates, key=lambda value: (validation_mses[value], value))
    return RidgeOperatorSelection(
        matrix=matrices[selected],
        feature_mean=feature_mean,
        regularization=selected,
        validation_mse=validation_mses[selected],
        validation_mse_by_regularization=validation_mses,
    )


def fit_raw_window_dmd(
    train_current: np.ndarray,
    train_future: np.ndarray,
    validation_current: np.ndarray,
    validation_future: np.ndarray,
    regularizations: Sequence[float],
) -> RidgeOperatorSelection:
    """Fit validation-selected DMD directly on flattened windows."""

    return select_ridge_operator(
        flatten_windows(train_current),
        flatten_windows(train_future),
        flatten_windows(validation_current),
        flatten_windows(validation_future),
        regularizations,
    )


def fit_pca_dmd(
    train_current: np.ndarray,
    train_future: np.ndarray,
    validation_current: np.ndarray,
    validation_future: np.ndarray,
    n_components: int,
    regularizations: Sequence[float],
) -> PCAOperatorSelection:
    """Fit an unsupervised PCA bottleneck followed by ridge-DMD."""

    feature_map = fit_pca_feature_map(train_current, train_future, n_components)
    operator = select_ridge_operator(
        feature_map.transform(train_current),
        feature_map.transform(train_future),
        feature_map.transform(validation_current),
        feature_map.transform(validation_future),
        regularizations,
    )
    return PCAOperatorSelection(feature_map=feature_map, operator=operator)


def fit_random_cnn_dmd(
    train_current: np.ndarray | torch.Tensor,
    train_future: np.ndarray | torch.Tensor,
    validation_current: np.ndarray | torch.Tensor,
    validation_future: np.ndarray | torch.Tensor,
    *,
    seed: int,
    latent_dim: int,
    channels: list[int],
    pooling: str,
    input_length: int,
    batch_size: int,
    regularizations: Sequence[float],
    device: torch.device | str = "cpu",
) -> CNNOperatorSelection:
    """Fit ridge-DMD on a seeded CNN that is never trained."""

    encoder = make_random_cnn_encoder(
        seed=seed,
        latent_dim=latent_dim,
        channels=channels,
        pooling=pooling,
        input_length=input_length,
        device=device,
    )
    train_current_features = collect_encoder_features(encoder, train_current, batch_size)
    train_future_features = collect_encoder_features(encoder, train_future, batch_size)
    validation_current_features = collect_encoder_features(
        encoder,
        validation_current,
        batch_size,
    )
    validation_future_features = collect_encoder_features(
        encoder,
        validation_future,
        batch_size,
    )
    operator = select_ridge_operator(
        train_current_features,
        train_future_features,
        validation_current_features,
        validation_future_features,
        regularizations,
    )
    return CNNOperatorSelection(
        encoder=encoder,
        operator=operator,
        batch_size=batch_size,
    )


def fit_fixed_cnn_dmd(
    encoder: TemporalEncoder,
    train_current: np.ndarray | torch.Tensor,
    train_future: np.ndarray | torch.Tensor,
    validation_current: np.ndarray | torch.Tensor,
    validation_future: np.ndarray | torch.Tensor,
    *,
    batch_size: int,
    regularizations: Sequence[float],
) -> CNNOperatorSelection:
    """Fit ridge-DMD on an already trained encoder without changing it."""

    encoder.requires_grad_(False)
    encoder.eval()
    operator = select_ridge_operator(
        collect_encoder_features(encoder, train_current, batch_size),
        collect_encoder_features(encoder, train_future, batch_size),
        collect_encoder_features(encoder, validation_current, batch_size),
        collect_encoder_features(encoder, validation_future, batch_size),
        regularizations,
    )
    return CNNOperatorSelection(
        encoder=encoder,
        operator=operator,
        batch_size=batch_size,
    )

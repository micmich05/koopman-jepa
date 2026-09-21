from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


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

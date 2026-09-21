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


def _feature_matrix(values: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional feature matrix")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must have non-empty sample and feature dimensions")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")
    return matrix


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

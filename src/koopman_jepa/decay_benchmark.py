from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from torch.utils.data import TensorDataset

from .baselines import (
    fit_fixed_cnn_dmd,
    fit_pca_feature_map,
    fit_raw_window_dmd,
    fit_ridge_operator,
    flatten_windows,
    make_random_cnn_encoder,
    select_ridge_operator,
    train_supervised_phase_encoder,
)
from .config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig, validate_config
from .koopman import centered_phase_indicators
from .model import TemporalJEPA
from .phase_analysis import (
    evaluate_continuous_decay_operator,
    evaluate_decay_operator_candidates,
)
from .phase_data import DecayPhaseTensorDatasetSplits
from .training import collect_paired_embeddings, select_device, set_seed, train_model

BASELINE_METHODS = (
    "phase_oracle_ols",
    "raw_window_dmd",
    "pca3_dmd",
    "random_cnn3_dmd",
    "supervised_phase_cnn3_dmd",
    "jepa_learned_predictor",
    "jepa_posthoc_dmd",
)


@dataclass(frozen=True, slots=True)
class DecayDevelopmentBenchmark:
    """Validation-only benchmark output; no held-out data is accessed."""

    rows: tuple[dict[str, Any], ...]
    supervised_history: tuple[dict[str, float], ...]
    jepa_histories: dict[float, tuple[dict[str, float], ...]]


def _dataset_arrays(
    dataset: TensorDataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(dataset.tensors) != 3:
        raise ValueError("phase datasets must contain current, future, and phase pairs")
    current, future, phase_pairs = dataset.tensors
    if phase_pairs.ndim != 2 or phase_pairs.shape[1] != 2:
        raise ValueError("phase pairs must have shape (samples, 2)")
    return (
        current.detach().cpu().numpy(),
        future.detach().cpu().numpy(),
        phase_pairs[:, 0].detach().cpu().numpy(),
        phase_pairs[:, 1].detach().cpu().numpy(),
    )


def _evaluate_operator(
    *,
    method: str,
    seed: int,
    rho: float,
    current_features: np.ndarray,
    future_features: np.ndarray,
    current_phases: np.ndarray,
    predictor_matrix: np.ndarray,
    candidate_rhos: tuple[float, ...],
    rollout_horizons: tuple[int, ...],
    selected_regularization: float | None,
) -> dict[str, Any]:
    continuous = evaluate_continuous_decay_operator(
        current_features,
        current_phases,
        predictor_matrix,
        true_rho=rho,
        rollout_horizons=rollout_horizons,
    )
    discrete = evaluate_decay_operator_candidates(
        current_features,
        current_phases,
        predictor_matrix,
        candidate_rhos,
        rollout_horizons=rollout_horizons,
    )
    prediction = current_features @ predictor_matrix.T
    return {
        "method": method,
        "seed": seed,
        "rho": rho,
        "evaluation_split": "validation",
        "selected_regularization": selected_regularization,
        "feature_prediction_mse": float(np.mean(np.square(prediction - future_features))),
        "discrete_action_rho": discrete["predicted_action_rho"],
        "discrete_spectral_rho": discrete["predicted_spectral_rho"],
        **continuous,
    }


def _jepa_config(raw: dict[str, Any], seed: int) -> ExperimentConfig:
    cnn = raw["cnn"]
    train = raw["jepa"]
    config = ExperimentConfig(
        data=DataConfig(context_length=raw["emission"]["window_length"]),
        model=ModelConfig(
            latent_dim=cnn["latent_dim"],
            channels=list(cnn["channels"]),
            predictor_init=train["predictor_init"],
        ),
        train=TrainConfig(
            seed=seed,
            epochs=train["epochs"],
            batch_size=train["batch_size"],
            learning_rate=train["learning_rate"],
            predictor_learning_rate_multiplier=train[
                "predictor_learning_rate_multiplier"
            ],
            freeze_encoder_after_epoch=train["freeze_encoder_after_epoch"],
            weight_decay=train["weight_decay"],
            ema_momentum=train["ema_momentum"],
            mean_weight=train["mean_weight"],
            variance_weight=train["variance_weight"],
            covariance_weight=train["covariance_weight"],
            device=train["device"],
            num_workers=train["num_workers"],
        ),
    )
    validate_config(config)
    return config


def _validate_benchmark_inputs(
    raw: dict[str, Any],
    splits: DecayPhaseTensorDatasetSplits,
    seed: int,
) -> tuple[float, ...]:
    rhos = tuple(float(rho) for rho in raw["rho_values"])
    if tuple(raw["methods"]) != BASELINE_METHODS:
        raise ValueError("configured methods must match the frozen baseline methods")
    if seed not in raw["seeds"]:
        raise ValueError("seed must be declared in the frozen benchmark config")
    if set(splits.train) != set(rhos) or set(splits.validation) != set(rhos):
        raise ValueError("train and validation splits must cover every configured rho")
    if raw["splits"]["heldout_access"] != "after_methods_tests_and_config_are_frozen":
        raise ValueError("held-out access policy does not match the frozen protocol")
    return rhos


def run_decay_baseline_development(
    raw: dict[str, Any],
    splits: DecayPhaseTensorDatasetSplits,
    seed: int,
) -> DecayDevelopmentBenchmark:
    """Run every frozen method on validation while leaving held-out untouched."""

    rhos = _validate_benchmark_inputs(raw, splits, seed)
    canonical_rho = rhos[0]
    ridge_lambdas = tuple(float(value) for value in raw["feature_operator"]["ridge_lambdas"])
    rollout_horizons = tuple(int(value) for value in raw["evaluation"]["rollout_horizons"])
    cnn = raw["cnn"]
    input_length = int(raw["emission"]["window_length"])
    batch_size = int(raw["jepa"]["batch_size"])
    device = select_device(raw["jepa"]["device"])

    canonical_train = _dataset_arrays(splits.train[canonical_rho])
    pca_map = fit_pca_feature_map(
        canonical_train[0],
        canonical_train[1],
        n_components=raw["pca"]["n_components"],
    )
    random_encoder = make_random_cnn_encoder(
        seed=seed,
        latent_dim=cnn["latent_dim"],
        channels=list(cnn["channels"]),
        pooling=cnn["pooling"],
        input_length=input_length,
        device=device,
    )
    supervised = raw["supervised_encoder"]
    supervised_training = train_supervised_phase_encoder(
        canonical_train[0],
        canonical_train[1],
        canonical_train[2],
        canonical_train[3],
        seed=seed,
        latent_dim=cnn["latent_dim"],
        channels=list(cnn["channels"]),
        pooling=cnn["pooling"],
        input_length=input_length,
        num_classes=supervised["num_classes"],
        epochs=supervised["epochs"],
        batch_size=supervised["batch_size"],
        learning_rate=supervised["learning_rate"],
        weight_decay=supervised["weight_decay"],
        device=device,
    )

    rows: list[dict[str, Any]] = []
    jepa_histories: dict[float, tuple[dict[str, float], ...]] = {}
    for rho in rhos:
        train_current, train_future, train_phases, train_future_phases = _dataset_arrays(
            splits.train[rho]
        )
        val_current, val_future, val_phases, val_future_phases = _dataset_arrays(
            splits.validation[rho]
        )

        oracle_train_current = centered_phase_indicators(
            train_phases,
            supervised["num_classes"],
        )
        oracle_train_future = centered_phase_indicators(
            train_future_phases,
            supervised["num_classes"],
        )
        oracle_val_current = centered_phase_indicators(
            val_phases,
            supervised["num_classes"],
        )
        oracle_val_future = centered_phase_indicators(
            val_future_phases,
            supervised["num_classes"],
        )
        oracle_matrix = fit_ridge_operator(
            oracle_train_current,
            oracle_train_future,
            regularization=0.0,
        )
        rows.append(
            _evaluate_operator(
                method="phase_oracle_ols",
                seed=seed,
                rho=rho,
                current_features=oracle_val_current,
                future_features=oracle_val_future,
                current_phases=val_phases,
                predictor_matrix=oracle_matrix,
                candidate_rhos=rhos,
                rollout_horizons=rollout_horizons,
                selected_regularization=0.0,
            )
        )

        raw_fit = fit_raw_window_dmd(
            train_current,
            train_future,
            val_current,
            val_future,
            ridge_lambdas,
        )
        rows.append(
            _evaluate_operator(
                method="raw_window_dmd",
                seed=seed,
                rho=rho,
                current_features=raw_fit.center(flatten_windows(val_current)),
                future_features=raw_fit.center(flatten_windows(val_future)),
                current_phases=val_phases,
                predictor_matrix=raw_fit.matrix,
                candidate_rhos=rhos,
                rollout_horizons=rollout_horizons,
                selected_regularization=raw_fit.regularization,
            )
        )

        pca_fit = select_ridge_operator(
            pca_map.transform(train_current),
            pca_map.transform(train_future),
            pca_map.transform(val_current),
            pca_map.transform(val_future),
            ridge_lambdas,
        )
        rows.append(
            _evaluate_operator(
                method="pca3_dmd",
                seed=seed,
                rho=rho,
                current_features=pca_fit.center(pca_map.transform(val_current)),
                future_features=pca_fit.center(pca_map.transform(val_future)),
                current_phases=val_phases,
                predictor_matrix=pca_fit.matrix,
                candidate_rhos=rhos,
                rollout_horizons=rollout_horizons,
                selected_regularization=pca_fit.regularization,
            )
        )

        random_fit = fit_fixed_cnn_dmd(
            random_encoder,
            train_current,
            train_future,
            val_current,
            val_future,
            batch_size=batch_size,
            regularizations=ridge_lambdas,
        )
        rows.append(
            _evaluate_operator(
                method="random_cnn3_dmd",
                seed=seed,
                rho=rho,
                current_features=random_fit.transform(val_current),
                future_features=random_fit.transform(val_future),
                current_phases=val_phases,
                predictor_matrix=random_fit.operator.matrix,
                candidate_rhos=rhos,
                rollout_horizons=rollout_horizons,
                selected_regularization=random_fit.operator.regularization,
            )
        )

        supervised_fit = fit_fixed_cnn_dmd(
            supervised_training.encoder,
            train_current,
            train_future,
            val_current,
            val_future,
            batch_size=batch_size,
            regularizations=ridge_lambdas,
        )
        rows.append(
            _evaluate_operator(
                method="supervised_phase_cnn3_dmd",
                seed=seed,
                rho=rho,
                current_features=supervised_fit.transform(val_current),
                future_features=supervised_fit.transform(val_future),
                current_phases=val_phases,
                predictor_matrix=supervised_fit.operator.matrix,
                candidate_rhos=rhos,
                rollout_horizons=rollout_horizons,
                selected_regularization=supervised_fit.operator.regularization,
            )
        )

        experiment_config = _jepa_config(raw, seed)
        set_seed(seed)
        model = TemporalJEPA(
            latent_dim=cnn["latent_dim"],
            channels=list(cnn["channels"]),
            predictor_init=raw["jepa"]["predictor_init"],
            pooling=cnn["pooling"],
            input_length=input_length,
        ).to(device)
        history = train_model(
            model,
            splits.train[rho],
            splits.validation[rho],
            experiment_config,
            device,
        )
        jepa_histories[rho] = tuple(history)
        train_online, train_future_online, _, _ = collect_paired_embeddings(
            model,
            splits.train[rho],
            batch_size,
            device,
        )
        val_online, val_future_online, _, _ = collect_paired_embeddings(
            model,
            splits.validation[rho],
            batch_size,
            device,
        )
        learned_predictor = model.predictor.matrix.detach().cpu().numpy()
        rows.append(
            _evaluate_operator(
                method="jepa_learned_predictor",
                seed=seed,
                rho=rho,
                current_features=val_online,
                future_features=val_future_online,
                current_phases=val_phases,
                predictor_matrix=learned_predictor,
                candidate_rhos=rhos,
                rollout_horizons=rollout_horizons,
                selected_regularization=None,
            )
        )

        posthoc = select_ridge_operator(
            train_online,
            train_future_online,
            val_online,
            val_future_online,
            ridge_lambdas,
        )
        rows.append(
            _evaluate_operator(
                method="jepa_posthoc_dmd",
                seed=seed,
                rho=rho,
                current_features=posthoc.center(val_online),
                future_features=posthoc.center(val_future_online),
                current_phases=val_phases,
                predictor_matrix=posthoc.matrix,
                candidate_rhos=rhos,
                rollout_horizons=rollout_horizons,
                selected_regularization=posthoc.regularization,
            )
        )

    return DecayDevelopmentBenchmark(
        rows=tuple(rows),
        supervised_history=supervised_training.history,
        jepa_histories=jepa_histories,
    )

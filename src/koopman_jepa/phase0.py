from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .analysis import evaluate_phase0
from .config import ExperimentConfig, load_config, validate_config
from .data import make_phase0_datasets
from .model import TemporalJEPA
from .plotting import plot_embeddings, plot_history, plot_predictor_spectrum
from .training import collect_embeddings, select_device, set_seed, train_model


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the reduced Phase 0 JEPA reproduction")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--predictor-init", choices=("identity", "random"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device", type=str)
    parser.add_argument("--run-name", type=str)
    return parser.parse_args()


def _apply_overrides(config: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    if args.predictor_init is not None:
        config.model.predictor_init = args.predictor_init
    if args.seed is not None:
        config.train.seed = args.seed
    if args.epochs is not None:
        config.train.epochs = args.epochs
    if args.device is not None:
        config.train.device = args.device
    return config


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _gate_checks(metrics: dict[str, Any], predictor_init: str) -> dict[str, bool]:
    active_identity_error = metrics["active_identity_error"]
    checks = {
        "noncollapsed": metrics["effective_rank"] >= 2.0,
        "regimes_separable": metrics["linear_probe_accuracy"] >= 0.80,
        "active_rank_at_least_three": metrics["active_rank"] >= 3,
        "online_target_aligned": metrics["online_target_centroid_error"] <= 0.25,
    }
    if predictor_init == "identity":
        checks["identity_on_active_subspace"] = (
            active_identity_error is not None and active_identity_error <= 0.25
        )
    return checks


def run(config: ExperimentConfig, run_name: str | None = None) -> Path:
    validate_config(config)
    set_seed(config.train.seed)
    device = select_device(config.train.device)
    datasets = make_phase0_datasets(config.data, config.train.seed)

    model = TemporalJEPA(
        latent_dim=config.model.latent_dim,
        channels=config.model.channels,
        predictor_init=config.model.predictor_init,
    ).to(device)

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    resolved_name = run_name or (
        f"seed{config.train.seed:03d}_{config.model.predictor_init}_{timestamp}"
    )
    output_dir = Path(config.output_dir) / resolved_name
    output_dir.mkdir(parents=True, exist_ok=False)

    print(f"device={device}")
    print(f"run_dir={output_dir}")
    history = train_model(model, datasets.train, datasets.val, config, device)

    train_online, _, train_labels = collect_embeddings(
        model,
        datasets.train,
        config.train.batch_size,
        device,
    )
    test_online, test_target, test_labels = collect_embeddings(
        model,
        datasets.test,
        config.train.batch_size,
        device,
    )
    predictor_matrix = model.predictor.matrix.detach().cpu().numpy()

    metrics = evaluate_phase0(
        train_embeddings=train_online,
        train_labels=train_labels,
        test_embeddings=test_online,
        test_target_embeddings=test_target,
        test_labels=test_labels,
        predictor_matrix=predictor_matrix,
        num_regimes=len(datasets.regime_names),
        seed=config.train.seed,
    )
    metrics["device"] = str(device)
    metrics["predictor_init"] = config.model.predictor_init
    metrics["gate_checks"] = _gate_checks(metrics, config.model.predictor_init)
    metrics["all_run_checks_passed"] = all(metrics["gate_checks"].values())

    _write_json(output_dir / "config.json", config.to_dict())
    _write_json(output_dir / "history.json", history)
    _write_json(output_dir / "metrics.json", metrics)
    np.savez_compressed(
        output_dir / "embeddings.npz",
        train_online=train_online,
        train_labels=train_labels,
        test_online=test_online,
        test_target=test_target,
        test_labels=test_labels,
        predictor_matrix=predictor_matrix,
    )
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config.to_dict(),
            "metrics": metrics,
        },
        output_dir / "checkpoint.pt",
    )
    plot_history(history, output_dir / "loss.png")
    plot_embeddings(
        test_online,
        test_labels,
        datasets.regime_names,
        output_dir / "embeddings_pca.png",
    )
    active_eigenvalues = np.array(
        [
            complex(value["real"], value["imag"])
            for value in metrics["active_eigenvalues"]
        ],
        dtype=np.complex128,
    )
    plot_predictor_spectrum(
        predictor_matrix,
        output_dir / "predictor_spectrum.png",
        active_eigenvalues=active_eigenvalues,
    )

    summary_keys = (
        "effective_rank",
        "linear_probe_accuracy",
        "kmeans_purity",
        "active_rank",
        "active_identity_error",
        "active_eigenvalue_one_error",
        "online_target_centroid_error",
        "all_run_checks_passed",
    )
    print(json.dumps({key: metrics[key] for key in summary_keys}, indent=2))
    return output_dir


def main() -> None:
    args = _parse_args()
    config = _apply_overrides(load_config(args.config), args)
    run(config, run_name=args.run_name)


if __name__ == "__main__":
    main()

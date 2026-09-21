from copy import deepcopy
from pathlib import Path

import yaml

from koopman_jepa.decay_benchmark import BASELINE_METHODS, run_decay_baseline_development
from koopman_jepa.phase_data import PhaseWindowConfig, make_decay_phase_tensor_dataset_splits

ROOT = Path(__file__).resolve().parents[1]


def _smoke_config() -> dict:
    raw = yaml.safe_load(
        (ROOT / "configs" / "koopman_decay_baselines.yaml").read_text(encoding="utf-8")
    )
    raw = deepcopy(raw)
    raw["rho_values"] = [0.0, 1.0]
    raw["seeds"] = [5]
    raw["emission"]["window_length"] = 16
    raw["cnn"]["channels"] = [2]
    raw["supervised_encoder"].update(
        {"epochs": 1, "batch_size": 16, "learning_rate": 0.01}
    )
    raw["jepa"].update(
        {
            "epochs": 2,
            "batch_size": 16,
            "freeze_encoder_after_epoch": None,
            "mean_weight": 0.0,
            "variance_weight": 0.0,
            "covariance_weight": 0.0,
        }
    )
    return raw


def test_development_runner_emits_every_method_without_materializing_heldout() -> None:
    raw = _smoke_config()
    splits = make_decay_phase_tensor_dataset_splits(
        PhaseWindowConfig(window_length=16, max_shift=1),
        raw["rho_values"],
        train_repeats_per_transition=2,
        validation_repeats_per_transition=1,
        seed=5,
    )

    result = run_decay_baseline_development(raw, splits, seed=5)

    assert splits.heldout is None
    assert len(result.rows) == len(BASELINE_METHODS) * len(raw["rho_values"])
    assert {row["method"] for row in result.rows} == set(BASELINE_METHODS)
    assert {row["rho"] for row in result.rows} == {0.0, 1.0}
    assert all(row["evaluation_split"] == "validation" for row in result.rows)
    assert all(row["seed"] == 5 for row in result.rows)
    assert len(result.supervised_history) == 1
    assert set(result.jepa_histories) == {0.0, 1.0}
    assert all(len(history) == 2 for history in result.jepa_histories.values())


def test_development_runner_rejects_seed_outside_frozen_config() -> None:
    raw = _smoke_config()
    splits = make_decay_phase_tensor_dataset_splits(
        PhaseWindowConfig(window_length=16, max_shift=1),
        raw["rho_values"],
        train_repeats_per_transition=1,
        validation_repeats_per_transition=1,
        seed=6,
    )

    try:
        run_decay_baseline_development(raw, splits, seed=6)
    except ValueError as error:
        assert "declared" in str(error)
    else:
        raise AssertionError("undeclared seed should be rejected")


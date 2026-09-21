from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "koopman_decay_baselines.yaml"
PROTOCOL_PATH = ROOT / "docs" / "EXPERIMENT.md"


def test_decay_baseline_protocol_records_validation_only_execution() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    assert config["status"] == "completed_validation_only"
    assert config["executed_seeds"] == [101]
    assert config["heldout_executed"] is False
    assert config["rho_values"] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert config["seeds"] == list(range(101, 111))
    assert config["splits"]["heldout_access"] == (
        "after_methods_tests_and_config_are_frozen"
    )
    assert config["primary_baseline"] == "random_cnn3_dmd"
    assert config["target_method"] == "jepa_learned_predictor"
    assert config["jepa"]["predictor_init"] == "random"
    assert config["evaluation"]["sign_flip_assignments"] == 2 ** len(config["seeds"])
    assert config["evaluation"]["invalid_active_span_absolute_error"] == 1.0


def test_baseline_operator_selection_cannot_use_phase_labels() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    operator = config["feature_operator"]
    assert operator["selection_split"] == "validation"
    assert operator["selection_metric"] == "feature_prediction_mse"
    assert operator["labels_used_for_selection"] is False
    assert operator["ridge_lambdas"] == [0.0, 1e-6, 1e-4, 1e-2, 1.0]
    assert config["evaluation"]["labels_used_only_for_final_metrics"] is True


def test_protocol_declares_the_limited_baseline_scope() -> None:
    protocol = PROTOCOL_PATH.read_text(encoding="utf-8")
    normalized = " ".join(protocol.split())

    assert "posibilidad, no necesidad" in normalized
    assert "DMD crudo y PCA+DMD" in normalized
    assert "No se abrió el held-out" in normalized

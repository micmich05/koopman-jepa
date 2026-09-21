from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "koopman_decay_generalization.yaml"
PROTOCOL_PATH = ROOT / "docs" / "EXPERIMENT.md"


def test_decay_generalization_protocol_is_frozen_before_training() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    assert config["status"] == "frozen_before_training"
    assert config["rho_values"] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert config["seeds"] == list(range(1, 11))
    assert config["operator_identification"]["candidates"] == config["rho_values"]
    assert config["operator_identification"]["minimum_correct_seeds_per_rho"] == 8
    assert config["operator_identification"]["loss_used_for_decision"] is False
    assert config["splits"]["test_constructed"] is False
    assert config["diagnostics"]["rollout_horizons"] == [1, 2, 3, 4, 8]


def test_protocol_states_the_scientific_scope() -> None:
    protocol = PROTOCOL_PATH.read_text(encoding="utf-8")

    assert "Familia continua" in protocol
    assert "8/10 seeds" in protocol
    assert "La loss no interviene" in protocol
    assert "criterio discreto global no pasa" in protocol

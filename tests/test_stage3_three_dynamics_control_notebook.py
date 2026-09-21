import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "stage3_three_dynamics_control.ipynb"
CONFIG_PATH = ROOT / "configs" / "stage3_three_dynamics_control.yaml"


def _notebook() -> dict:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def test_three_dynamics_control_protocol_is_frozen() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    assert config["dynamics"] == ["static", "cyclic", "independent"]
    assert config["seeds"] == list(range(1, 11))
    assert config["operator_identification"]["candidates"] == config["dynamics"]
    assert config["operator_identification"]["minimum_correct_seeds_per_dynamics"] == 8
    assert config["diagnostics"]["rollout_horizons"] == [1, 2, 3, 4, 8]
    assert "maximum_validation_loss_ratio" not in config


def test_three_dynamics_control_notebook_is_unexecuted_and_operator_centered() -> None:
    notebook = _notebook()
    source = "".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert all(
        cell.get("execution_count") is None and cell.get("outputs", []) == []
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert "evaluate_phase_operator_candidates" in source
    assert "global_operator_result" in source
    assert "loss_used_for_decision\": False" in source
    assert "test_constructed\": False" in source
    assert "minimum_correct" in source
    assert "rollout_errors" in source

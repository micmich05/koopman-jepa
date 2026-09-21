import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "three_dynamics_experiment.ipynb"
CONFIG_PATH = ROOT / "configs" / "three_dynamics.yaml"


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


def test_three_dynamics_control_notebook_is_executed_and_operator_centered() -> None:
    notebook = _notebook()
    source = "".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    code_cells = [
        cell for cell in notebook["cells"] if cell["cell_type"] == "code"
    ]

    assert [cell["execution_count"] for cell in code_cells] == list(range(1, 8))
    assert not [
        output
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    assert "evaluate_phase_operator_candidates" in source
    assert "global_operator_result" in source
    assert "loss_used_for_decision\": False" in source
    assert "test_constructed\": False" in source
    assert "minimum_correct" in source
    assert "rollout_errors" in source


def test_three_dynamics_control_records_the_frozen_result() -> None:
    notebook = _notebook()
    rendered_output = json.dumps(notebook["cells"], ensure_ascii=False)

    assert '"global_operator_result\\": true' in rendered_output
    assert rendered_output.count('"correct_action_seeds\\": 10') == 3
    assert rendered_output.count('"correct_spectrum_seeds\\": 10') == 3
    assert rendered_output.count('"full_rank_seeds\\": 10') == 3
    assert '"loss_used_for_decision\\": false' in rendered_output
    assert "el predictor identifica la dinámica" in rendered_output
    assert sum(
        "image/png" in output.get("data", {})
        for cell in notebook["cells"]
        for output in cell.get("outputs", [])
    ) == 2

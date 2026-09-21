import copy
import json
from pathlib import Path

import yaml


def _load_config(name: str) -> dict[str, object]:
    path = Path(__file__).parents[1] / "configs" / name
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_predictor_freeze_config_changes_only_encoder_schedule() -> None:
    reference = _load_config("stage3_cyclic_predictor_fast_final_smoke.yaml")
    candidate = _load_config("stage3_cyclic_predictor_freeze_smoke.yaml")

    normalized = copy.deepcopy(candidate)
    freeze_epoch = normalized["train"].pop(  # type: ignore[union-attr]
        "freeze_encoder_after_epoch"
    )
    assert freeze_epoch == 3
    assert normalized == reference


def test_stage3_cyclic_predictor_freeze_notebook_is_executed() -> None:
    path = (
        Path(__file__).parents[1]
        / "notebooks"
        / "stage3_cyclic_predictor_freeze_smoke.ipynb"
    )
    with path.open(encoding="utf-8") as handle:
        notebook = json.load(handle)
    code_cells = [
        cell for cell in notebook["cells"] if cell["cell_type"] == "code"
    ]
    source = "\n".join("".join(cell["source"]) for cell in code_cells)

    assert "stage3_cyclic_predictor_freeze_smoke.yaml" in source
    assert "stage3_cyclic_predictor_fast_final_smoke.yaml" in source
    assert "normalized == reference_raw" in source
    assert "freeze_encoder_after_epoch == 3" in source
    assert "train_model(" in source
    assert "epoch_callback=trace_epoch" in source
    assert "evaluate_phase_representation" in source
    assert "evaluate_phase_operator_diagnostics" in source
    assert "test_constructed" in source
    assert "assert encoder_freeze_gate_passed" in source
    assert all(isinstance(cell["execution_count"], int) for cell in code_cells)
    errors = [
        output
        for cell in code_cells
        for output in cell["outputs"]
        if output["output_type"] == "error"
    ]
    assert len(errors) == 1
    assert errors[0]["ename"] == "AssertionError"

    rendered = json.dumps(notebook, ensure_ascii=False)
    rendered_output = "\n".join(
        "".join(output.get("text", []))
        + "".join(output.get("data", {}).get("text/markdown", []))
        for cell in code_cells
        for output in cell["outputs"]
    )
    assert "cambia un solo factor" in rendered
    assert "test no se construye" in rendered
    assert '"selected_epoch": 30' in rendered_output
    assert '"validation_loss_ratio": 0.25611721996968134' in rendered_output
    assert '"intertwining_error": 0.32867923737828275' in rendered_output
    assert '"spectral_max_absolute_error": 0.36591994238595005' in rendered_output
    assert '"online_target_phase_basis_error": 1.4164894038126918e-06' in rendered_output
    assert '"encoder_freeze_gate_passed": false' in rendered_output
    assert '"spectrum": false' in rendered_output
    assert "Gate neuronal cíclico con optimización alternada: **FAIL**" in rendered_output

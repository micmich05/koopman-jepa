import copy
import json
from pathlib import Path

import yaml


def _load_config(name: str) -> dict[str, object]:
    path = Path(__file__).parents[1] / "configs" / name
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_final_checkpoint_config_changes_only_selection_policy() -> None:
    reference = _load_config("stage3_cyclic_predictor_fast_smoke.yaml")
    candidate = _load_config("stage3_cyclic_predictor_fast_final_smoke.yaml")

    assert candidate["selection"] == {"metric": "final_epoch", "mode": "fixed"}
    normalized = copy.deepcopy(candidate)
    normalized["selection"] = reference["selection"]
    assert normalized == reference


def test_stage3_cyclic_predictor_fast_final_notebook_is_executed() -> None:
    path = (
        Path(__file__).parents[1]
        / "notebooks"
        / "stage3_cyclic_predictor_fast_final_smoke.ipynb"
    )
    with path.open(encoding="utf-8") as handle:
        notebook = json.load(handle)
    code_cells = [
        cell for cell in notebook["cells"] if cell["cell_type"] == "code"
    ]
    source = "\n".join("".join(cell["source"]) for cell in code_cells)

    assert "stage3_cyclic_predictor_fast_final_smoke.yaml" in source
    assert "stage3_cyclic_predictor_fast_smoke.yaml" in source
    assert "normalized == reference_raw" in source
    assert 'raw["selection"] == {"metric": "final_epoch", "mode": "fixed"}' in source
    assert "train_model(" in source
    assert "epoch_callback=trace_epoch" in source
    assert "train_model_with_validation_checkpoint" not in source
    assert "evaluate_phase_representation" in source
    assert "evaluate_phase_operator_diagnostics" in source
    assert "test_constructed" in source
    assert "assert final_checkpoint_gate_passed" in source
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
    assert "cambia únicamente la política de selección" in rendered
    assert "test no se construye" in rendered
    assert '"selected_epoch": 30' in rendered_output
    assert '"validation_loss_ratio": 4.591613156588604' in rendered_output
    assert '"covariance": 18.990458965301514' in rendered_output
    assert '"intertwining_error": 0.042975913834554444' in rendered_output
    assert '"spectral_max_absolute_error": 0.045352745305285325' in rendered_output
    assert '"final_checkpoint_gate_passed": false' in rendered_output
    assert '"validation_improves": false' in rendered_output
    assert "Gate neuronal cíclico con checkpoint final: **FAIL**" in rendered_output
    assert "Loss final total / predicción / media / varianza / covarianza" in rendered_output

import copy
import json
from pathlib import Path

import yaml


def _load_config(name: str) -> dict[str, object]:
    path = Path(__file__).parents[1] / "configs" / name
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_predictor_fast_config_changes_only_predictor_learning_rate() -> None:
    reference = _load_config("stage3_cyclic_neural_ema_fast_smoke.yaml")
    candidate = _load_config("stage3_cyclic_predictor_fast_smoke.yaml")

    normalized = copy.deepcopy(candidate)
    multiplier = normalized["train"].pop(  # type: ignore[union-attr]
        "predictor_learning_rate_multiplier"
    )
    assert multiplier == 4.0
    assert normalized == reference


def test_stage3_cyclic_predictor_fast_notebook_is_executed() -> None:
    path = (
        Path(__file__).parents[1]
        / "notebooks"
        / "stage3_cyclic_predictor_fast_smoke.ipynb"
    )
    with path.open(encoding="utf-8") as handle:
        notebook = json.load(handle)
    code_cells = [
        cell for cell in notebook["cells"] if cell["cell_type"] == "code"
    ]
    source = "\n".join("".join(cell["source"]) for cell in code_cells)

    assert "stage3_cyclic_predictor_fast_smoke.yaml" in source
    assert "stage3_cyclic_neural_ema_fast_smoke.yaml" in source
    assert "normalized == reference_raw" in source
    assert "predictor_learning_rate_multiplier == 4.0" in source
    assert "epoch_callback=trace_epoch" in source
    assert "evaluate_phase_representation" in source
    assert "evaluate_phase_operator_diagnostics" in source
    assert "test_constructed" in source
    assert "assert predictor_fast_gate_passed" in source
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
    assert '"selected_epoch": 3' in rendered_output
    assert '"best_trace_spectral_epoch": 30' in rendered_output
    assert '"predictor_ever_passed_spectrum": true' in rendered_output
    assert '"best_trace_spectral_error": 0.04535271957022435' in rendered_output
    assert '"best_trace_predictor_online_error": 0.045100050040543974' in rendered_output
    assert '"predictor_fast_gate_passed": false' in rendered_output
    assert '"val_prediction_loss": {' in rendered_output
    assert '"epoch": 1' in rendered_output
    assert "Gate cíclico con predictor `4×`: **FAIL**" in rendered_output
    assert "sí cruza el gate espectral durante el entrenamiento" in rendered_output
    assert "Mínimo de validation prediction loss: época **1**" in rendered_output

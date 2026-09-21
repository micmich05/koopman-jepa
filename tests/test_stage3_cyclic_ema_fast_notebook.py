import copy
import json
from pathlib import Path

import yaml


def _load_config(name: str) -> dict[str, object]:
    path = Path(__file__).parents[1] / "configs" / name
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_ema_fast_config_changes_only_momentum() -> None:
    reference = _load_config("stage3_cyclic_neural_smoke.yaml")
    candidate = _load_config("stage3_cyclic_neural_ema_fast_smoke.yaml")

    assert reference["train"]["ema_momentum"] == 0.99  # type: ignore[index]
    assert candidate["train"]["ema_momentum"] == 0.90  # type: ignore[index]
    normalized = copy.deepcopy(candidate)
    normalized["train"]["ema_momentum"] = 0.99  # type: ignore[index]
    assert normalized == reference


def test_stage3_cyclic_ema_fast_notebook_is_executed() -> None:
    path = (
        Path(__file__).parents[1]
        / "notebooks"
        / "stage3_cyclic_neural_ema_fast_smoke.ipynb"
    )
    with path.open(encoding="utf-8") as handle:
        notebook = json.load(handle)
    code_cells = [
        cell for cell in notebook["cells"] if cell["cell_type"] == "code"
    ]
    source = "\n".join("".join(cell["source"]) for cell in code_cells)

    assert "stage3_cyclic_neural_ema_fast_smoke.yaml" in source
    assert "comparison == reference_raw" in source
    assert "evaluate_phase_representation" in source
    assert "evaluate_phase_operator_diagnostics" in source
    assert "test_constructed" in source
    assert "assert ema_fast_gate_passed" in source
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
    assert "cambia **un solo factor**" in rendered
    assert "test no se construye" in rendered
    assert "Gate cíclico con EMA `0.90`: **FAIL**" in rendered
    assert "Checkpoint: época **27**; ratio validation/baseline: **0.426**" in rendered
    assert "Rango efectivo / probe de fase: **2.927 / 100.00%**" in rendered
    assert "Error de alineación / entrelazamiento: **0.182 / 0.759**" in rendered
    assert "Error espectral máximo del predictor: **1.059**" in rendered
    assert "Diferencia de base online/EMA: **0.182**" in rendered
    assert "Error espectral máximo post-hoc online: **0.045**" in rendered

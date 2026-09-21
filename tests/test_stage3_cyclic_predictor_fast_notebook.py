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


def test_stage3_cyclic_predictor_fast_notebook_is_prepared() -> None:
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
    assert all(cell["execution_count"] is None for cell in code_cells)
    assert all(not cell["outputs"] for cell in code_cells)

    rendered = json.dumps(notebook, ensure_ascii=False)
    assert "cambia un solo factor" in rendered
    assert "test no se construye" in rendered

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "koopman_decay_generalization.ipynb"


def _notebook() -> dict:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def test_decay_notebook_is_prepared_but_not_executed() -> None:
    notebook = _notebook()
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]

    assert all(cell["execution_count"] is None for cell in code_cells)
    assert all(cell["outputs"] == [] for cell in code_cells)
    for cell in code_cells:
        compile("".join(cell["source"]), NOTEBOOK_PATH.name, "exec")


def test_decay_notebook_implements_the_frozen_decision() -> None:
    notebook = _notebook()
    source = "".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert "koopman_decay_generalization.yaml" in source
    assert "make_decay_phase_tensor_dataset_splits" in source
    assert "evaluate_decay_operator_candidates" in source
    assert "minimum_correct" in source
    assert "global_operator_result" in source
    assert "calibration_mae" in source
    assert "loss_used_for_decision" in source
    assert "test_constructed" in source
    assert "train_model(" in source

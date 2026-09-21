import json
from pathlib import Path


def test_stage3_cyclic_neural_diagnostic_notebook_is_prepared() -> None:
    path = (
        Path(__file__).parents[1]
        / "notebooks"
        / "stage3_cyclic_neural_smoke_diagnostic.ipynb"
    )
    with path.open(encoding="utf-8") as handle:
        notebook = json.load(handle)
    code_cells = [
        cell for cell in notebook["cells"] if cell["cell_type"] == "code"
    ]
    source = "\n".join("".join(cell["source"]) for cell in code_cells)

    assert "stage3_cyclic_neural_smoke.yaml" in source
    assert "collect_paired_embeddings" in source
    assert "evaluate_phase_operator_diagnostics" in source
    assert "posthoc_online_operator" in source
    assert "posthoc_target_operator" in source
    assert "test_constructed" in source
    assert "smoke_gate_passed" not in source
    assert all(cell["execution_count"] is None for cell in code_cells)
    assert all(not cell["outputs"] for cell in code_cells)

    rendered = json.dumps(notebook, ensure_ascii=False)
    assert "no redefine el gate" in rendered
    assert "Test sigue sin construirse" in rendered

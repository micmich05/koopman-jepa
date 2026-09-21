import json
from pathlib import Path


def test_stage3_cyclic_neural_diagnostic_notebook_is_executed() -> None:
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
    assert all(isinstance(cell["execution_count"], int) for cell in code_cells)
    assert all(
        output["output_type"] != "error"
        for cell in code_cells
        for output in cell["outputs"]
    )

    rendered = json.dumps(notebook, ensure_ascii=False)
    assert "no redefine el gate" in rendered
    assert "Test sigue sin construirse" in rendered
    assert "Error entre bases de fase online actual/futura: **0.019**" in rendered
    assert "Error entre las bases online y EMA futuras: **0.812**" in rendered
    assert "Error dinámico del predictor como endomorfismo online: **1.279**" in rendered
    assert "coordenada online→EMA que entrenó: **3.338**" in rendered
    assert "Distancia predictor/post-hoc online: **1.302**" in rendered
    assert "Distancia predictor/post-hoc EMA: **3.747**" in rendered
    assert "Error espectral máximo post-hoc online / EMA: **0.037 / 0.864**" in rendered

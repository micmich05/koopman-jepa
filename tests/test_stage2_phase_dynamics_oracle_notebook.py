import json
from pathlib import Path


def test_stage2_phase_dynamics_oracle_notebook_is_executed() -> None:
    path = (
        Path(__file__).parents[1]
        / "notebooks"
        / "stage2_phase_dynamics_oracle.ipynb"
    )
    with path.open(encoding="utf-8") as handle:
        notebook = json.load(handle)
    code_cells = [
        cell for cell in notebook["cells"] if cell["cell_type"] == "code"
    ]
    source = "\n".join("".join(cell["source"]) for cell in code_cells)

    assert "evaluate_phase_dynamics_oracle" in source
    assert 'DYNAMICS = ("static", "cyclic", "independent")' in source
    assert "TOLERANCE = 1e-10" in source
    assert "identical_marginals" in source
    assert "sample_error_semantics" in source
    assert "eight_step_conditional_rollouts" in source
    assert "assert oracle_gate_passed" in source
    assert all(isinstance(cell["execution_count"], int) for cell in code_cells)
    assert all(
        output["output_type"] != "error"
        for cell in code_cells
        for output in cell["outputs"]
    )

    rendered = json.dumps(notebook, ensure_ascii=False)
    assert "no permite predecir cada realización futura" in rendered
    assert "No prueba todavía que un encoder neuronal" in rendered
    assert "Gate oracle de Etapa 2A: **PASS**" in rendered
    assert "Máximo error espectral: **3.33e-15**" in rendered
    assert "error contra una realización: **1.00**" in rendered
    assert "error contra su media condicional: **3.16e-16**" in rendered
    assert "Máximo error de rollout condicional a ocho pasos: **1.48e-14**" in rendered

import json
from pathlib import Path


def test_koopman_oracle_notebook_is_executed() -> None:
    path = (
        Path(__file__).parents[1]
        / "notebooks"
        / "koopman_oracle.ipynb"
    )
    with path.open(encoding="utf-8") as handle:
        notebook = json.load(handle)
    code_cells = [
        cell for cell in notebook["cells"] if cell["cell_type"] == "code"
    ]
    source = "\n".join("".join(cell["source"]) for cell in code_cells)

    assert "centered_phase_indicators" in source
    assert "fit_linear_operator" in source
    assert "restrict_operator" in source
    assert "left_eigendecomposition" in source
    assert "linear_rollout" in source
    assert "expected_spectrum = np.array([-1.0, 1.0j, -1.0j]" in source
    assert "TOLERANCE = 1e-10" in source
    assert "max_eight_step_rollout_error" in source
    assert all(isinstance(cell["execution_count"], int) for cell in code_cells)
    assert all(
        output["output_type"] != "error"
        for cell in code_cells
        for output in cell["outputs"]
    )
    rendered = json.dumps(notebook, ensure_ascii=False)
    assert "Comprobación matemática: **PASS**" in rendered
    assert "Error espectral medio/máximo: **1.14e-15 / 2.11e-15**" in rendered

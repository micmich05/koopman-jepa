import json
from pathlib import Path


def test_stage3_cyclic_predictor_trace_notebook_is_prepared() -> None:
    path = (
        Path(__file__).parents[1]
        / "notebooks"
        / "stage3_cyclic_predictor_trace.ipynb"
    )
    with path.open(encoding="utf-8") as handle:
        notebook = json.load(handle)
    code_cells = [
        cell for cell in notebook["cells"] if cell["cell_type"] == "code"
    ]
    source = "\n".join("".join(cell["source"]) for cell in code_cells)

    assert "stage3_cyclic_neural_ema_fast_smoke.yaml" in source
    assert "epoch_callback=trace_epoch" in source
    assert "train_predictor_gradient_norm" in source
    assert "trace_predictor_relative_movement" in source
    assert "trace_predictor_spectral_max_error" in source
    assert "trace_predictor_posthoc_distance" in source
    assert "test_constructed" in source
    assert "assert smoke" not in source
    assert all(cell["execution_count"] is None for cell in code_cells)
    assert all(not cell["outputs"] for cell in code_cells)

    rendered = json.dumps(notebook, ensure_ascii=False)
    assert "sin cambiar optimizer, checkpoint ni gates" in rendered
    assert "test no se construye" in rendered

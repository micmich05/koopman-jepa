import json
from pathlib import Path


def _mlp_notebook_code() -> list[dict[str, object]]:
    path = (
        Path(__file__).parents[1]
        / "notebooks"
        / "paper_mlp_clustering_development.ipynb"
    )
    with path.open(encoding="utf-8") as handle:
        notebook = json.load(handle)
    return [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]


def test_mlp_development_notebook_is_validation_only() -> None:
    code_cells = _mlp_notebook_code()
    source = "\n".join("".join(cell["source"]) for cell in code_cells)

    assert "paper_mlp_clustering_development.yaml" in source
    assert 'PaperRegimeDataset(config.data, "train"' in source
    assert 'PaperRegimeDataset(config.data, "val"' in source
    assert 'PaperRegimeDataset(config.data, "test"' not in source
    assert "evaluate_paper_mlp_clustering_gate" in source
    assert "evaluate_scale_invariant_seed_stability_gate" in source
    assert "predictive_prerequisites_passed" in source
    assert "Clustering OMITIDO" in source
    assert "assert replay_passed" not in source


def test_mlp_development_notebook_has_completed_outputs_without_errors() -> None:
    code_cells = _mlp_notebook_code()

    assert all(isinstance(cell["execution_count"], int) for cell in code_cells)
    assert all(cell["outputs"] for cell in code_cells)
    assert all(
        output["output_type"] != "error"
        for cell in code_cells
        for output in cell["outputs"]
    )

import json
from pathlib import Path


def test_two_stage_notebook_is_executed_validation_only() -> None:
    path = (
        Path(__file__).parents[1]
        / "notebooks"
        / "paper_mlp_two_stage_one_hidden_development.ipynb"
    )
    with path.open(encoding="utf-8") as handle:
        notebook = json.load(handle)
    code_cells = [
        cell for cell in notebook["cells"] if cell["cell_type"] == "code"
    ]
    source = "\n".join("".join(cell["source"]) for cell in code_cells)

    assert "paper_mlp_two_stage_one_hidden_development.yaml" in source
    assert 'config.model.encoder_projection == "two_stage"' in source
    assert 'PaperRegimeDataset(config.data, "train"' in source
    assert 'PaperRegimeDataset(config.data, "val"' in source
    assert 'PaperRegimeDataset(config.data, "test"' not in source
    assert all(isinstance(cell["execution_count"], int) for cell in code_cells)
    assert all(
        output["output_type"] != "error"
        for cell in code_cells
        for output in cell["outputs"]
    )


def test_two_stage_clustering_diagnostic_is_executed_validation_only() -> None:
    path = (
        Path(__file__).parents[1]
        / "notebooks"
        / "paper_mlp_two_stage_clustering_diagnostic.ipynb"
    )
    with path.open(encoding="utf-8") as handle:
        notebook = json.load(handle)
    code_cells = [
        cell for cell in notebook["cells"] if cell["cell_type"] == "code"
    ]
    source = "\n".join("".join(cell["source"]) for cell in code_cells)

    assert "paper_mlp_two_stage_one_hidden_development.yaml" in source
    assert 'config.model.encoder_projection == "two_stage"' in source
    assert 'PaperRegimeDataset(config.data, "train"' in source
    assert 'PaperRegimeDataset(config.data, "val"' in source
    assert 'PaperRegimeDataset(config.data, "test"' not in source
    assert "min_validation_effective_rank=1.0" in source
    assert all(isinstance(cell["execution_count"], int) for cell in code_cells)
    assert all(
        output["output_type"] != "error"
        for cell in code_cells
        for output in cell["outputs"]
    )

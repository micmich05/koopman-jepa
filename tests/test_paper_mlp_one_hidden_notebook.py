import json
from pathlib import Path


def test_one_hidden_notebook_is_executed_and_never_constructs_test() -> None:
    path = (
        Path(__file__).parents[1]
        / "notebooks"
        / "paper_mlp_one_hidden_development.ipynb"
    )
    with path.open(encoding="utf-8") as handle:
        notebook = json.load(handle)
    code_cells = [
        cell for cell in notebook["cells"] if cell["cell_type"] == "code"
    ]
    source = "\n".join("".join(cell["source"]) for cell in code_cells)

    assert "paper_mlp_one_hidden_development.yaml" in source
    assert "load_paper_mlp_one_hidden_development_config" in source
    assert 'PaperRegimeDataset(config.data, "train"' in source
    assert 'PaperRegimeDataset(config.data, "val"' in source
    assert 'PaperRegimeDataset(config.data, "test"' not in source
    assert all(isinstance(cell["execution_count"], int) for cell in code_cells)
    assert all(
        output["output_type"] != "error"
        for cell in code_cells
        for output in cell["outputs"]
    )


def test_one_hidden_clustering_diagnostic_is_validation_only_and_executed() -> None:
    path = (
        Path(__file__).parents[1]
        / "notebooks"
        / "paper_mlp_one_hidden_clustering_diagnostic.ipynb"
    )
    with path.open(encoding="utf-8") as handle:
        notebook = json.load(handle)
    code_cells = [
        cell for cell in notebook["cells"] if cell["cell_type"] == "code"
    ]
    source = "\n".join("".join(cell["source"]) for cell in code_cells)

    assert 'PaperRegimeDataset(config.data, "train"' in source
    assert 'PaperRegimeDataset(config.data, "val"' in source
    assert 'PaperRegimeDataset(config.data, "test"' not in source
    assert "min_validation_effective_rank=1.0" in source
    assert "evaluate_paper_mlp_seed_clustering" in source
    assert all(isinstance(cell["execution_count"], int) for cell in code_cells)
    assert all(
        output["output_type"] != "error"
        for cell in code_cells
        for output in cell["outputs"]
    )

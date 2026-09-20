import json
from pathlib import Path


def test_one_hidden_notebook_is_unexecuted_and_never_constructs_test() -> None:
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
    assert all(cell["execution_count"] is None for cell in code_cells)
    assert all(not cell["outputs"] for cell in code_cells)

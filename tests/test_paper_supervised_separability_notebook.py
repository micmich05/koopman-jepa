import json
from pathlib import Path


def test_supervised_separability_notebook_is_prepared_validation_only() -> None:
    path = (
        Path(__file__).parents[1]
        / "notebooks"
        / "paper_supervised_separability_probe.ipynb"
    )
    with path.open(encoding="utf-8") as handle:
        notebook = json.load(handle)
    code_cells = [
        cell for cell in notebook["cells"] if cell["cell_type"] == "code"
    ]
    source = "\n".join("".join(cell["source"]) for cell in code_cells)

    assert "paper_mlp_medium_scale_development.yaml" in source
    assert "SupervisedEncoderClassifier" in source
    assert "PaperTemporalEncoder" in source
    assert "cross_entropy" in source
    assert "MIN_VALIDATION_ACCURACY = 0.85" in source
    assert "MIN_PAIR_AWARE_ACCURACY = 0.90" in source
    assert 'PaperRegimeDataset(config.data, "train"' in source
    assert 'PaperRegimeDataset(config.data, "val"' in source
    assert 'PaperRegimeDataset(config.data, "test"' not in source
    assert "evaluate_paper_mlp_seed_clustering" in source
    assert all(cell["execution_count"] is None for cell in code_cells)
    assert all(not cell["outputs"] for cell in code_cells)

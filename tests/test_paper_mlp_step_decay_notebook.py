import json
from pathlib import Path


def test_step_decay_probe_notebook_is_prepared_validation_only() -> None:
    path = (
        Path(__file__).parents[1]
        / "notebooks"
        / "paper_mlp_step_decay_probe.ipynb"
    )
    with path.open(encoding="utf-8") as handle:
        notebook = json.load(handle)
    code_cells = [
        cell for cell in notebook["cells"] if cell["cell_type"] == "code"
    ]
    source = "\n".join("".join(cell["source"]) for cell in code_cells)

    assert "paper_mlp_step_decay_probe.yaml" in source
    assert 'config.train.learning_rate_schedule == "step"' in source
    assert "config.train.learning_rate_decay_epoch == 2" in source
    assert "config.train.learning_rate_decay_factor == 0.1" in source
    assert "MAX_FINAL_VALIDATION_RATIO = 0.5" in source
    assert "MAX_LATE_VALIDATION_RATIO = 0.5" in source
    assert "MIN_CHECKPOINT_RANK = 4.0" in source
    assert "MIN_MEAN_PURITY = 0.49" in source
    assert 'PaperRegimeDataset(config.data, "train"' in source
    assert 'PaperRegimeDataset(config.data, "val"' in source
    assert 'PaperRegimeDataset(config.data, "test"' not in source
    assert "evaluate_paper_mlp_seed_clustering" in source
    assert all(cell["execution_count"] is None for cell in code_cells)
    assert all(not cell["outputs"] for cell in code_cells)

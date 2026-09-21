import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "koopman_decay_baselines_validation.ipynb"


def _notebook() -> dict:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def test_baseline_validation_notebook_is_prepared_but_not_executed() -> None:
    notebook = _notebook()
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]

    assert all(cell["execution_count"] is None for cell in code_cells)
    assert all(cell["outputs"] == [] for cell in code_cells)
    for cell in code_cells:
        compile("".join(cell["source"]), NOTEBOOK_PATH.name, "exec")


def test_baseline_validation_notebook_keeps_heldout_closed() -> None:
    notebook = _notebook()
    source = "".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert "koopman_decay_baselines.yaml" in source
    assert "run_decay_baseline_development" in source
    assert "assert splits.heldout is None" in source
    assert "heldout_repeats_per_transition" not in source
    assert "exact_one_sided_sign_flip_test" not in source


def test_baseline_validation_notebook_reports_every_frozen_method() -> None:
    source = "".join(
        "".join(cell.get("source", [])) for cell in _notebook()["cells"]
    )

    for method in (
        "phase_oracle_ols",
        "raw_window_dmd",
        "pca3_dmd",
        "random_cnn3_dmd",
        "supervised_phase_cnn3_dmd",
        "jepa_learned_predictor",
        "jepa_posthoc_dmd",
    ):
        assert method in source
    assert "Esto no decide la hipótesis" in source


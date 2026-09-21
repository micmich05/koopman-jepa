import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "koopman_decay_baselines_validation.ipynb"


def _notebook() -> dict:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def test_baseline_validation_notebook_is_executed_without_errors() -> None:
    notebook = _notebook()
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]

    assert [cell["execution_count"] for cell in code_cells] == list(range(1, 7))
    assert not [
        output
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    assert sum(
        "image/png" in output.get("data", {})
        for cell in code_cells
        for output in cell.get("outputs", [])
    ) == 1
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


def test_baseline_validation_notebook_records_development_result() -> None:
    rendered = json.dumps(_notebook()["cells"], ensure_ascii=False)

    assert "Corridas evaluadas: 35; held-out materializado: False" in rendered
    assert "`phase_oracle_ols` | 0.000 | 0.000 | 1.000" in rendered
    assert "`raw_window_dmd` | 0.022 | 0.020 | 0.949" in rendered
    assert "`pca3_dmd` | 0.011 | 0.011 | 0.975" in rendered
    assert "`random_cnn3_dmd` | 0.299 | 0.137 | 0.391" in rendered
    assert "`jepa_learned_predictor` | 0.197 | 0.103 | 0.435" in rendered
    assert "`jepa_posthoc_dmd` | 0.029 | 0.026 | 0.952" in rendered
    assert "El DMD post-hoc mejora al predictor JEPA por 0.169 puntos" in rendered

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "koopman_decay_generalization.ipynb"


def _notebook() -> dict:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def _cell_source(notebook: dict, cell_id: str) -> str:
    cell = next(cell for cell in notebook["cells"] if cell["id"] == cell_id)
    return "".join(cell["source"])


def test_decay_notebook_is_prepared_but_not_executed() -> None:
    notebook = _notebook()
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]

    assert all(cell["execution_count"] is None for cell in code_cells)
    assert all(cell["outputs"] == [] for cell in code_cells)
    for cell in code_cells:
        compile("".join(cell["source"]), NOTEBOOK_PATH.name, "exec")


def test_decay_notebook_implements_the_frozen_decision() -> None:
    notebook = _notebook()
    source = "".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert "koopman_decay_generalization.yaml" in source
    assert "make_decay_phase_tensor_dataset_splits" in source
    assert "evaluate_decay_operator_candidates" in source
    assert "minimum_correct" in source
    assert "global_operator_result" in source
    assert "calibration_mae" in source
    assert "loss_used_for_decision" in source
    assert "test_constructed" in source
    assert "train_model(" in source


def test_decay_notebook_postprocessing_accepts_numpy_scalars(monkeypatch) -> None:
    notebook = _notebook()
    namespace: dict = {}
    exec(_cell_source(notebook, "decay-imports"), namespace)
    exec(_cell_source(notebook, "decay-config"), namespace)
    rhos = namespace["rhos"]
    seeds = namespace["seeds"]
    horizons = namespace["rollout_horizons"]
    expected_spectrum = namespace["expected_decay_active_spectrum"]

    results = []
    for rho in rhos:
        eigenvalues = expected_spectrum(rho)
        for seed in seeds:
            results.append(
                {
                    "seed": seed,
                    "rho": rho,
                    "full_rank": np.bool_(True),
                    "action_correct": np.bool_(True),
                    "spectrum_correct": np.bool_(True),
                    "action_errors": {
                        candidate: float(abs(candidate - rho)) for candidate in rhos
                    },
                    "spectral_errors": {
                        candidate: float(abs(candidate - rho)) for candidate in rhos
                    },
                    "action_margin": 0.25,
                    "spectrum_margin": 0.25,
                    "rollout_errors": {
                        candidate: {horizon: float(abs(candidate - rho)) for horizon in horizons}
                        for candidate in rhos
                    },
                    "active_eigenvalues": [
                        {"real": float(value.real), "imag": float(value.imag)}
                        for value in eigenvalues
                    ],
                    "mean_eigenvalue_modulus": rho,
                    "effective_rank": 3.0,
                    "phase_probe_accuracy": 1.0,
                }
            )

    namespace["results"] = results
    monkeypatch.setattr(namespace["plt"], "show", lambda: None)
    exec(_cell_source(notebook, "decay-summary"), namespace)
    exec(_cell_source(notebook, "decay-plots"), namespace)
    exec(_cell_source(notebook, "decay-analysis"), namespace)

    assert namespace["global_operator_result"] is True
    assert namespace["calibration_mae"] < 1e-12

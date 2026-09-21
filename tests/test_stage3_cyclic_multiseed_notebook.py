import json
from pathlib import Path

import yaml


def _load_config(name: str) -> dict[str, object]:
    path = Path(__file__).parents[1] / "configs" / name
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_multiseed_config_reuses_the_passing_recipe() -> None:
    reference = _load_config("stage3_cyclic_predictor_freeze_long_smoke.yaml")
    candidate = _load_config("stage3_cyclic_multiseed_development.yaml")

    assert candidate["seeds"] == list(range(1, 11))
    for key in ("dynamics", "emission", "splits", "model", "train", "selection", "gates"):
        assert candidate[key] == reference[key]
    assert candidate["aggregate_gates"] == {
        "minimum_successful_seeds": 8,
        "maximum_median_intertwining_error": 0.20,
        "maximum_median_spectral_error": 0.20,
    }


def test_stage3_cyclic_multiseed_notebook_is_executed() -> None:
    path = (
        Path(__file__).parents[1]
        / "notebooks"
        / "stage3_cyclic_multiseed_development.ipynb"
    )
    with path.open(encoding="utf-8") as handle:
        notebook = json.load(handle)
    code_cells = [
        cell for cell in notebook["cells"] if cell["cell_type"] == "code"
    ]
    source = "\n".join("".join(cell["source"]) for cell in code_cells)

    assert "stage3_cyclic_multiseed_development.yaml" in source
    assert "stage3_cyclic_predictor_freeze_long_smoke.yaml" in source
    assert "seeds == list(range(1, 11))" in source
    assert "minimum_successful_seeds" in source
    assert "maximum_median_intertwining_error" in source
    assert "maximum_median_spectral_error" in source
    assert "make_phase_tensor_dataset_splits" in source
    assert "seed=seed" in source
    assert "train_model(" in source
    assert "evaluate_phase_representation" in source
    assert "evaluate_phase_operator_diagnostics" in source
    assert "test_constructed" in source
    assert "assert multi_seed_gate_passed" in source
    assert all(isinstance(cell["execution_count"], int) for cell in code_cells)
    assert all(
        output.get("output_type") != "error"
        for cell in code_cells
        for output in cell["outputs"]
    )

    rendered = json.dumps(notebook, ensure_ascii=False)
    rendered_output = "\n".join(
        "".join(output.get("text", []))
        + "".join(output.get("data", {}).get("text/markdown", []))
        for cell in code_cells
        for output in cell["outputs"]
    )
    assert "10 seeds nuevas" in rendered
    assert "test no se construye" in rendered
    assert '"successful_seeds": 8' in rendered_output
    assert '"median_validation_loss_ratio": 0.301005565613913' in rendered_output
    assert '"median_effective_rank": 2.810831580138041' in rendered_output
    assert '"median_intertwining_error": 0.06822191509801676' in rendered_output
    assert '"median_spectral_error": 0.06702866939299468' in rendered_output
    assert '"worst_intertwining_error": 0.5116221835198809' in rendered_output
    assert '"worst_spectral_error": 0.37886467516894656' in rendered_output
    assert '"multi_seed_gate_passed": true' in rendered_output
    assert "Gate multi-seed: **PASS**" in rendered_output
    assert "seed 6: effective_rank, intertwining" in rendered_output
    assert "seed 7: effective_rank, intertwining, spectrum" in rendered_output

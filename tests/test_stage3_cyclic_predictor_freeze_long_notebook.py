import copy
import json
from pathlib import Path

import yaml


def _load_config(name: str) -> dict[str, object]:
    path = Path(__file__).parents[1] / "configs" / name
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_freeze_long_config_changes_only_training_horizon() -> None:
    reference = _load_config("stage3_cyclic_predictor_freeze_smoke.yaml")
    candidate = _load_config("stage3_cyclic_predictor_freeze_long_smoke.yaml")

    assert reference["train"]["epochs"] == 30  # type: ignore[index]
    assert candidate["train"]["epochs"] == 60  # type: ignore[index]
    normalized = copy.deepcopy(candidate)
    normalized["train"]["epochs"] = 30  # type: ignore[index]
    assert normalized == reference


def test_stage3_cyclic_predictor_freeze_long_notebook_is_executed() -> None:
    path = (
        Path(__file__).parents[1]
        / "notebooks"
        / "stage3_cyclic_predictor_freeze_long_smoke.ipynb"
    )
    with path.open(encoding="utf-8") as handle:
        notebook = json.load(handle)
    code_cells = [
        cell for cell in notebook["cells"] if cell["cell_type"] == "code"
    ]
    source = "\n".join("".join(cell["source"]) for cell in code_cells)

    assert "stage3_cyclic_predictor_freeze_long_smoke.yaml" in source
    assert "stage3_cyclic_predictor_freeze_smoke.yaml" in source
    assert "normalized == reference_raw" in source
    assert "experiment_config.train.epochs == 60" in source
    assert "freeze_encoder_after_epoch == 3" in source
    assert "train_model(" in source
    assert "epoch_callback=trace_epoch" in source
    assert "evaluate_phase_representation" in source
    assert "evaluate_phase_operator_diagnostics" in source
    assert "test_constructed" in source
    assert "assert freeze_long_gate_passed" in source
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
    assert "cambia únicamente el horizonte" in rendered
    assert "test no se construye" in rendered
    assert '"selected_epoch": 60' in rendered_output
    assert '"validation_loss_ratio": 0.2363502413782027' in rendered_output
    assert '"effective_rank": 2.9314755102885246' in rendered_output
    assert '"linear_probe_accuracy": 1.0' in rendered_output
    assert '"intertwining_error": 0.07823126813831092' in rendered_output
    assert '"spectral_max_absolute_error": 0.07927158269898033' in rendered_output
    assert '"trained_online_endomorphism_error": 0.06899964533352572' in rendered_output
    assert '"freeze_long_gate_passed": true' in rendered_output
    assert "Gate neuronal cíclico a 60 épocas: **PASS**" in rendered_output

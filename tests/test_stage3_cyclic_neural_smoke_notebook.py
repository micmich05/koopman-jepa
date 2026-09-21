import json
from pathlib import Path

import yaml

from koopman_jepa.config import TrainConfig
from koopman_jepa.phase_data import PhaseWindowConfig, validate_phase_window_config


def test_stage3_cyclic_neural_smoke_config_is_valid() -> None:
    path = Path(__file__).parents[1] / "configs" / "stage3_cyclic_neural_smoke.yaml"
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    emission = PhaseWindowConfig(**config["emission"], repeats_per_transition=1)
    validate_phase_window_config(emission)
    training = TrainConfig(seed=config["base_seed"], **config["train"])
    assert config["dynamics"] == "cyclic"
    assert config["model"]["latent_dim"] == 3
    assert config["model"]["pooling"] == "flatten"
    assert training.mean_weight > 0.0
    assert training.variance_weight > 0.0
    assert config["selection"] == {
        "metric": "validation_total_loss",
        "mode": "min",
    }
    assert config["gates"]["required_active_rank"] == 3
    assert config["gates"]["maximum_spectral_error"] == 0.35


def test_stage3_cyclic_neural_smoke_notebook_is_executed() -> None:
    path = (
        Path(__file__).parents[1]
        / "notebooks"
        / "stage3_cyclic_neural_smoke.ipynb"
    )
    with path.open(encoding="utf-8") as handle:
        notebook = json.load(handle)
    code_cells = [
        cell for cell in notebook["cells"] if cell["cell_type"] == "code"
    ]
    source = "\n".join("".join(cell["source"]) for cell in code_cells)

    assert "stage3_cyclic_neural_smoke.yaml" in source
    assert "make_phase_tensor_dataset_splits" in source
    assert "train_model_with_validation_checkpoint" in source
    assert "evaluate_phase_representation" in source
    assert "test_constructed" in source
    assert "assert smoke_gate_passed" in source
    assert all(isinstance(cell["execution_count"], int) for cell in code_cells)
    errors = [
        output
        for cell in code_cells
        for output in cell["outputs"]
        if output["output_type"] == "error"
    ]
    assert len(errors) == 1
    assert errors[0]["ename"] == "AssertionError"

    rendered = json.dumps(notebook, ensure_ascii=False)
    assert "no contiene la matriz cíclica ni sus eigenvalues" in rendered
    assert "Test no se construye ni consulta" in rendered
    assert "Gate del smoke neuronal cíclico: **FAIL**" in rendered
    assert "ratio validation/baseline: **0.331**" in rendered
    assert "Escala media / rango efectivo: **0.968 / 2.846**" in rendered
    assert "Accuracy del linear probe de fase: **100.00%**" in rendered
    assert "Error de alineación con las indicadoras centradas: **0.153**" in rendered
    assert "Error de entrelazamiento / invariancia activa: **1.278 / 0.000**" in rendered
    assert "Error espectral medio / máximo: **0.763 / 1.074**" in rendered

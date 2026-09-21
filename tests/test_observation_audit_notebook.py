import json
from pathlib import Path

import yaml

from koopman_jepa.phase_data import PhaseWindowConfig, validate_phase_window_config


def test_phase_observation_config_is_valid() -> None:
    path = (
        Path(__file__).parents[1]
        / "configs"
        / "observation_audit.yaml"
    )
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    emission = PhaseWindowConfig(**config["emission"])
    validate_phase_window_config(emission)
    assert config["seed"] == 20260921
    assert config["gates"]["minimum_phase_accuracy"] == 0.99
    assert config["gates"]["maximum_decoded_transition_error"] == 0.01
    assert config["gates"]["require_exact_shared_marginals"] is True


def test_phase_observation_notebook_is_executed() -> None:
    path = (
        Path(__file__).parents[1]
        / "notebooks"
        / "observation_audit.ipynb"
    )
    with path.open(encoding="utf-8") as handle:
        notebook = json.load(handle)
    code_cells = [
        cell for cell in notebook["cells"] if cell["cell_type"] == "code"
    ]
    source = "\n".join("".join(cell["source"]) for cell in code_cells)

    assert "observation_audit.yaml" in source
    assert "make_shared_phase_observation_bundle" in source
    assert "nearest_template_phase_predictions" in source
    assert "exact_shared_marginals" in source
    assert "decoded_temporal_laws" in source
    assert "assert observation_audit_passed" in source
    assert all(isinstance(cell["execution_count"], int) for cell in code_cells)
    assert all(
        output["output_type"] != "error"
        for cell in code_cells
        for output in cell["outputs"]
    )

    rendered = json.dumps(notebook, ensure_ascii=False)
    assert "sólo cambia el acoplamiento temporal" in rendered
    assert "no como encoder aprendido" in rendered
    assert "Auditoría de datos: **PASS**" in rendered
    assert "Máxima diferencia entre marginales observables pareados: **0.00e+00**" in rendered
    assert "Peor accuracy del decoder de fase oracle: **100.00%**" in rendered
    assert "Máximo error de las transiciones decodificadas: **0.00e+00**" in rendered
    assert "Diferencia absoluta media entre los bancos source y target: **0.341**" in rendered

import copy
import json
from pathlib import Path

import yaml


def _load_config(name: str) -> dict[str, object]:
    path = Path(__file__).parents[1] / "configs" / name
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_heldout_config_freezes_development_recipe_and_replay() -> None:
    reference = _load_config("stage3_cyclic_multiseed_development.yaml")
    candidate = _load_config("stage3_cyclic_heldout.yaml")

    normalized = copy.deepcopy(candidate)
    assert normalized["splits"].pop("test_repeats_per_transition") == 64  # type: ignore[union-attr]
    replay = normalized.pop("validation_replay")
    assert replay["absolute_tolerance"] == 1e-8  # type: ignore[index]
    assert replay["successful_seeds"] == 8  # type: ignore[index]
    assert normalized == reference


def test_stage3_cyclic_heldout_notebook_is_executed_with_replay_barrier() -> None:
    path = (
        Path(__file__).parents[1]
        / "notebooks"
        / "stage3_cyclic_heldout.ipynb"
    )
    with path.open(encoding="utf-8") as handle:
        notebook = json.load(handle)
    code_cells = [
        cell for cell in notebook["cells"] if cell["cell_type"] == "code"
    ]
    source = "\n".join("".join(cell["source"]) for cell in code_cells)

    assert "stage3_cyclic_heldout.yaml" in source
    assert "stage3_cyclic_multiseed_development.yaml" in source
    assert "validation_replay_passed" in source
    assert "assert validation_replay_passed" in source
    assert "test_repeats_per_transition=test_repeats_per_transition" in source
    assert source.index("assert validation_replay_passed") < source.index(
        "test_repeats_per_transition=test_repeats_per_transition"
    )
    assert "test_seed == seed + 37" in source
    assert "heldout_gate_passed" in source
    assert "test_consumed" in source
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
    assert "Primera evaluación held-out" in rendered
    assert "sólo después del replay" in rendered
    assert '"validation_replay_passed": true' in rendered_output
    assert '"successful_seeds": 0' in rendered_output
    assert '"median_test_loss_ratio": 0.5815977917787114' in rendered_output
    assert '"median_effective_rank": 2.8191772553190275' in rendered_output
    assert '"median_intertwining_error": 0.06822191509801676' in rendered_output
    assert '"median_spectral_error": 0.06702866939299468' in rendered_output
    assert '"heldout_gate_passed": false' in rendered_output
    assert '"test_consumed": true' in rendered_output
    assert "Gate held-out: **FAIL**" in rendered_output
    assert "seed 10: validation_improves" in rendered_output

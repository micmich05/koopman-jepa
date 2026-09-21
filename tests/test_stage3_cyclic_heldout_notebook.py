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


def test_stage3_cyclic_heldout_notebook_is_prepared_with_replay_barrier() -> None:
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
    assert all(cell["execution_count"] is None for cell in code_cells)
    assert all(not cell["outputs"] for cell in code_cells)

    rendered = json.dumps(notebook, ensure_ascii=False)
    assert "Primera evaluación held-out" in rendered
    assert "sólo después del replay" in rendered

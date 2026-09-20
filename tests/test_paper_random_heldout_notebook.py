import json
from pathlib import Path


def _notebook_cells() -> list[dict[str, object]]:
    path = (
        Path(__file__).parents[1]
        / "notebooks"
        / "paper_linear_random_heldout_smoke.ipynb"
    )
    with path.open(encoding="utf-8") as handle:
        notebook = json.load(handle)
    return notebook["cells"]


def test_random_heldout_notebook_guards_the_only_test_constructor() -> None:
    cells = _notebook_cells()
    code_cells = [cell for cell in cells if cell["cell_type"] == "code"]
    sources_by_id = {
        cell["id"]: "".join(cell["source"])
        for cell in code_cells
    }
    constructor_cells = [
        cell_id
        for cell_id, source in sources_by_id.items()
        if "test_dataset = PaperRegimeDataset" in source
    ]
    cell_ids = [cell["id"] for cell in cells]

    assert constructor_cells == ["build-test-data"]
    assert "replay_gate_open" in sources_by_id["build-test-data"]
    assert cell_ids.index("replay-both-conditions") < cell_ids.index(
        "build-test-data"
    )


def test_random_heldout_notebook_uses_frozen_config_and_paired_gate() -> None:
    cells = _notebook_cells()
    source = "\n".join(
        "".join(cell["source"])
        for cell in cells
        if cell["cell_type"] == "code"
    )

    assert "paper_linear_random_heldout_smoke.yaml" in source
    assert "config.identity_replay" in source
    assert "config.random_replay" in source
    assert "evaluate_paper_linear_random_heldout_gate" in source


def test_random_heldout_diagnostic_is_exploratory_and_guarded() -> None:
    path = (
        Path(__file__).parents[1]
        / "notebooks"
        / "paper_linear_random_heldout_diagnostic.ipynb"
    )
    with path.open(encoding="utf-8") as handle:
        notebook = json.load(handle)
    cells = notebook["cells"]
    code_cells = [cell for cell in cells if cell["cell_type"] == "code"]
    sources_by_id = {
        cell["id"]: "".join(cell["source"])
        for cell in code_cells
    }
    constructor_cells = [
        cell_id
        for cell_id, source in sources_by_id.items()
        if "test_dataset = PaperRegimeDataset" in source
    ]
    all_source = "\n".join(sources_by_id.values())

    assert constructor_cells == ["build-consumed-test"]
    assert "diagnostic_replay_open" in sources_by_id["build-consumed-test"]
    assert "clustering_diagnostics" in all_source
    assert "evaluate_paper_linear_random_heldout_gate" not in all_source
    assert "expected_counts" in all_source

from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "projects"
    / "2026_selectiveRIP"
    / "noteforTG"
    / "readout_optimization.ipynb"
)


def test_readout_optimization_notebook_bootstraps_project_imports() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    first_code_cell = next(
        cell for cell in notebook["cells"] if cell.get("cell_type") == "code"
    )
    source = "".join(first_code_cell.get("source", []))

    assert "%run ../scripts/bootstrap_notebook.py" in source
    assert "from laboneq.serializers import load" in source
    assert "repo_root = PROJECT_ROOT" in source
    assert 'repo_root / "configs" / "descriptors" / "1port.yaml"' in source
    assert "sys.path.insert" not in source

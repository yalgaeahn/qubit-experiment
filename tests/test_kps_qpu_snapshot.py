"""Exercise the KPS notebook save cell from a notebook working directory."""

import json
from datetime import datetime
from pathlib import Path

from laboneq.serializers import load

from qubit_experiment.qpu_types.fixed_transmon.demo_qpus import demo_platform


def test_save_cell_creates_project_snapshot_from_notebook_directory(tmp_path, monkeypatch):
    notebook_path = (
        Path(__file__).resolve().parents[1]
        / "projects/2026_alltoall_kps/notebooks/readout_optimization.ipynb"
    )
    notebook = json.loads(notebook_path.read_text())
    source = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and "save(qpu" in "".join(cell["source"])
    )
    project = tmp_path / "kps"
    notebook_directory = project / "notebooks"
    notebook_directory.mkdir(parents=True)
    monkeypatch.chdir(notebook_directory)
    qpu = demo_platform(1).qpu
    namespace = {"PROJECT_ROOT": project, "qpu": qpu}

    exec(compile(source, str(notebook_path), "exec"), namespace)

    snapshots = list((project / "qpu_parameters").glob("*.json"))
    assert len(snapshots) == 1
    snapshot = snapshots[0]
    datetime.strptime(snapshot.name.split("_", 1)[0], "%Y%m%d-%H%M%S")
    assert not (notebook_directory / "qpu_parameters").exists()
    restored = load(snapshot)
    assert restored.quantum_element_uids == qpu.quantum_element_uids

"""Verify that offline compilation also persists the compiled task output."""

import json

import numpy as np
from laboneq.dsl.session import Session
from laboneq.serializers import load, to_json
from laboneq.workflow.logbook import FolderStore

from qubit_experiment.experiments import resonator_spectroscopy
from qubit_experiment.qpu_types.fixed_transmon.demo_qpus import demo_platform


def test_resonator_compilation_saves_to_folder_store(tmp_path):
    platform = demo_platform(1)
    qubit = platform.qpu.quantum_elements[0]
    options = resonator_spectroscopy.experiment_workflow.options()
    options.logstore(FolderStore(tmp_path, save_mode="raise"))
    options.use_cw(True)
    options.count(4)

    result = resonator_spectroscopy.experiment_workflow(
        session=Session(platform.setup, configure_logging=False),
        qpu=platform.qpu,
        qubit=qubit.uid,
        frequencies=qubit.parameters.readout_resonator_frequency
        + np.array([-1e6, 0.0, 1e6]),
        options=options,
    ).run(until="compile_experiment")

    compiled = result.tasks["compile_experiment"].output
    expected = json.loads(to_json(compiled))
    saved = [
        path
        for path in tmp_path.rglob("*.json")
        if json.loads(path.read_text()) == expected
    ]
    assert saved, "FolderStore must persist the compiled experiment output"
    assert isinstance(load(saved[0]), type(compiled))

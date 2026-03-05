from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

pytest.importorskip("laboneq")
pytest.importorskip("laboneq_applications")
pytest.importorskip("uncertainties")

from analysis import lifetime_measurement as lifetime_analysis
from experiments import lifetime_measurement as lifetime


@dataclass
class _QubitStub:
    uid: str
    parameters: object


def test_workflow_options_expose_tuneup_fields() -> None:
    options = lifetime.experiment_workflow.options()

    assert hasattr(options, "count")
    assert hasattr(options, "transition")
    assert hasattr(options, "use_cal_traces")
    assert hasattr(options, "do_analysis")
    assert hasattr(options, "update")


def test_extract_qubit_parameters_emits_ge_t1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        lifetime_analysis,
        "validate_and_convert_qubits_sweeps",
        lambda qubits: list(qubits),
    )
    qubit = _QubitStub(
        uid="q0",
        parameters=SimpleNamespace(
            ge_T1=12.0e-6,
            ef_T1=7.0e-6,
        ),
    )
    fit_results = {
        "q0": SimpleNamespace(
            params={
                "decay_rate": SimpleNamespace(value=1.0 / 20.0e-6, stderr=50.0),
            }
        )
    }

    out = lifetime_analysis.extract_qubit_parameters([qubit], fit_results)

    assert out["old_parameter_values"]["q0"]["ge_T1"] == pytest.approx(12.0e-6)
    assert out["new_parameter_values"]["q0"]["ge_T1"].nominal_value == pytest.approx(
        20.0e-6
    )


def test_create_experiment_rejects_sequential_averaging_with_cal_traces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        lifetime.validation,
        "validate_and_convert_qubits_sweeps",
        lambda qubits, delays: (list(qubits), [np.asarray(delays, dtype=float)]),
    )
    options = lifetime.TuneupExperimentOptions(
        use_cal_traces=True,
        averaging_mode="sequential",
    )

    with pytest.raises(ValueError, match="SEQUENTIAL"):
        lifetime.create_experiment(
            qpu=SimpleNamespace(),
            qubits=[SimpleNamespace(uid="q0")],
            delays=[1.0e-6, 2.0e-6],
            options=options,
        )

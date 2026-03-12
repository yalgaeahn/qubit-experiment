from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import matplotlib
import numpy as np
import pytest
from laboneq.workflow.reference import Reference

matplotlib.use("Agg")

pytest.importorskip("laboneq")
pytest.importorskip("laboneq_applications")
pytest.importorskip("scipy")

from qubit_experiment.analysis import (
    three_qubit_virtual_z_validation as vz_analysis,
)
from qubit_experiment.experiments import (
    three_qubit_virtual_z_validation as vz,
)


def _assert_no_reference(value) -> None:
    if isinstance(value, Reference):
        pytest.fail(f"Unexpected Reference payload: {value!r}")
    if isinstance(value, dict):
        for item in value.values():
            _assert_no_reference(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_no_reference(item)


def _output_to_dict(output) -> dict[str, object]:
    if isinstance(output, dict):
        return dict(output)
    return dict(vars(output))


def test_workflow_options_expose_validation_fields() -> None:
    options = vz.experiment_workflow.options()
    workflow_defaults = vz.ThreeQVZValidationWorkflowOptions()
    analysis_defaults = vz_analysis.ThreeQVZValidationAnalysisOptions()

    assert hasattr(options, "do_readout_calibration")
    assert hasattr(options, "phase_targets")
    assert hasattr(options, "product_initial_state")
    assert hasattr(options, "repeats_per_phase")
    assert hasattr(options, "count")

    assert workflow_defaults.phase_targets == ("q0", "q1", "q2")
    assert workflow_defaults.product_initial_state == "+++"
    assert workflow_defaults.repeats_per_phase == 1
    assert analysis_defaults.do_plotting is True
    assert analysis_defaults.max_mle_iterations == 2000


def test_resolve_phase_tuple_for_target_maps_one_hot() -> None:
    assert vz._resolve_phase_tuple_for_target_impl("q0", 0.3) == pytest.approx(
        (0.3, 0.0, 0.0)
    )
    assert vz._resolve_phase_tuple_for_target_impl("q1", -0.4) == pytest.approx(
        (0.0, -0.4, 0.0)
    )
    assert vz._resolve_phase_tuple_for_target_impl("q2", 0.5) == pytest.approx(
        (0.0, 0.0, 0.5)
    )


def test_resolve_phase_values_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="at least one value"):
        vz.resolve_phase_values.func([])

    with pytest.raises(ValueError, match="must be numeric"):
        vz.resolve_phase_values.func([0.0, "bad"])


def test_create_experiment_rejects_invalid_acquisition_type() -> None:
    options = SimpleNamespace(
        count=1024,
        acquisition_type="raw",
        averaging_mode=vz.AveragingMode.SINGLE_SHOT,
        repetition_mode=None,
        repetition_time=None,
        reset_oscillator_phase=False,
        active_reset=False,
        active_reset_repetitions=1,
        active_reset_states="ge",
    )

    with pytest.raises(ValueError, match="AcquisitionType.INTEGRATION"):
        vz._create_experiment_impl(
            qpu=SimpleNamespace(),
            qubits=[
                SimpleNamespace(uid="q0"),
                SimpleNamespace(uid="q1"),
                SimpleNamespace(uid="q2"),
            ],
            bus=[
                SimpleNamespace(uid="b0", signals={"drive": "b0/drive", "drive_p": "b0/drive_p"}),
                SimpleNamespace(uid="b1", signals={"drive": "b1/drive", "drive_p": "b1/drive_p"}),
                SimpleNamespace(uid="b2", signals={"drive": "b2/drive", "drive_p": "b2/drive_p"}),
            ],
            phase_tuple=(0.1, 0.0, 0.0),
            stage="product",
            options=options,
        )


def test_create_experiment_rejects_invalid_averaging_mode() -> None:
    options = SimpleNamespace(
        count=1024,
        acquisition_type=vz.AcquisitionType.INTEGRATION,
        averaging_mode="cyclic",
        repetition_mode=None,
        repetition_time=None,
        reset_oscillator_phase=False,
        active_reset=False,
        active_reset_repetitions=1,
        active_reset_states="ge",
    )

    with pytest.raises(ValueError, match="AveragingMode.SINGLE_SHOT"):
        vz._create_experiment_impl(
            qpu=SimpleNamespace(),
            qubits=[
                SimpleNamespace(uid="q0"),
                SimpleNamespace(uid="q1"),
                SimpleNamespace(uid="q2"),
            ],
            bus=[
                SimpleNamespace(uid="b0", signals={"drive": "b0/drive", "drive_p": "b0/drive_p"}),
                SimpleNamespace(uid="b1", signals={"drive": "b1/drive", "drive_p": "b1/drive_p"}),
                SimpleNamespace(uid="b2", signals={"drive": "b2/drive", "drive_p": "b2/drive_p"}),
            ],
            phase_tuple=(0.1, 0.0, 0.0),
            stage="product",
            options=options,
        )


def test_create_experiment_rejects_non_three_qubit_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        vz.validation,
        "validate_and_convert_qubits_sweeps",
        lambda elements: list(elements),
    )

    with pytest.raises(ValueError, match="expects exactly 3 qubits"):
        vz._create_experiment_impl(
            qpu=SimpleNamespace(),
            qubits=[SimpleNamespace(uid="q0"), SimpleNamespace(uid="q1")],
            bus=[
                SimpleNamespace(uid="b0", signals={"drive": "b0/drive", "drive_p": "b0/drive_p"}),
                SimpleNamespace(uid="b1", signals={"drive": "b1/drive", "drive_p": "b1/drive_p"}),
                SimpleNamespace(uid="b2", signals={"drive": "b2/drive", "drive_p": "b2/drive_p"}),
            ],
            phase_tuple=(0.1, 0.0, 0.0),
            stage="product",
        )


def test_create_experiment_rejects_non_three_bus_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        vz.validation,
        "validate_and_convert_single_qubit_sweeps",
        lambda element: element,
    )

    with pytest.raises(ValueError, match="expects exactly 3 bus elements"):
        vz._normalize_three_buses(
            [SimpleNamespace(uid="b0", signals={"drive": "b0/drive", "drive_p": "b0/drive_p"})]
        )


def _run_fake_product_sequence(
    monkeypatch: pytest.MonkeyPatch,
    *,
    phase_tuple=(0.2, 0.0, 0.0),
) -> list[tuple]:
    calls: list[tuple] = []

    class _FakeMeasureSection:
        def __init__(self, uid: str):
            self.uid = uid
            self.length = None

    class _FakeSection:
        def __init__(self, uid: str):
            self.uid = uid

        def __enter__(self):
            return SimpleNamespace(uid=self.uid)

        def __exit__(self, exc_type, exc, tb):
            return False

    class _FakeQop:
        def measure_section_length(self, qubits):
            return 1.0

        def active_reset(self, qubits, **kwargs):
            calls.append(("active_reset", tuple(q.uid for q in qubits)))
            return SimpleNamespace(uid="active_reset")

        def prepare_tomography_state(self, qubit, token):
            calls.append(("prepare", qubit.uid, token))
            return SimpleNamespace(uid=f"prepare_{qubit.uid}_{len(calls)}")

        def rz(self, qubit, angle, **kwargs):
            calls.append(("rz", qubit.uid, float(angle)))
            return SimpleNamespace(uid=f"rz_{qubit.uid}_{len(calls)}")

        def apply_tomography_prerotation(self, qubit, axis):
            calls.append(("basis", qubit.uid, axis))
            return SimpleNamespace(uid=f"basis_{qubit.uid}_{len(calls)}")

        def measure(self, qubit, handle):
            calls.append(("measure", qubit.uid, handle))
            return _FakeMeasureSection(f"measure_{qubit.uid}")

        def passive_reset(self, qubit):
            calls.append(("passive_reset", qubit.uid))

    monkeypatch.setattr(
        vz.validation,
        "validate_and_convert_qubits_sweeps",
        lambda elements: list(elements),
    )
    monkeypatch.setattr(
        vz.validation,
        "validate_and_convert_single_qubit_sweeps",
        lambda element: element,
    )
    monkeypatch.setattr(vz.dsl, "acquire_loop_rt", lambda **kwargs: nullcontext())
    monkeypatch.setattr(
        vz.dsl,
        "section",
        lambda name=None, **kwargs: _FakeSection(name or "section"),
    )
    monkeypatch.setattr(vz, "TOMOGRAPHY_SETTINGS", (("XYZ", ("X", "Y", "Z")),))

    qpu = SimpleNamespace(quantum_operations=_FakeQop())
    qubits = [
        SimpleNamespace(uid="q0"),
        SimpleNamespace(uid="q1"),
        SimpleNamespace(uid="q2"),
    ]
    bus = [
        SimpleNamespace(uid="b0", signals={"drive": "b0/drive", "drive_p": "b0/drive_p"}),
        SimpleNamespace(uid="b1", signals={"drive": "b1/drive", "drive_p": "b1/drive_p"}),
        SimpleNamespace(uid="b2", signals={"drive": "b2/drive", "drive_p": "b2/drive_p"}),
    ]
    options = SimpleNamespace(
        count=8,
        acquisition_type=vz.AcquisitionType.INTEGRATION,
        averaging_mode=vz.AveragingMode.SINGLE_SHOT,
        repetition_mode=None,
        repetition_time=None,
        reset_oscillator_phase=False,
        active_reset=False,
        active_reset_repetitions=1,
        active_reset_states="ge",
    )

    vz._create_experiment_impl(
        qpu=qpu,
        qubits=qubits,
        bus=bus,
        phase_tuple=phase_tuple,
        stage="product",
        product_initial_state="+++",
        options=options,
    )
    return calls


def test_product_stage_inserts_rz_between_prep_and_basis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _run_fake_product_sequence(monkeypatch, phase_tuple=(0.25, 0.0, 0.0))

    assert calls[:3] == [
        ("prepare", "q0", "+"),
        ("prepare", "q1", "+"),
        ("prepare", "q2", "+"),
    ]
    assert [call for call in calls if call[0] == "rz"] == [
        ("rz", "q0", pytest.approx(0.25))
    ]
    first_rz_index = next(index for index, call in enumerate(calls) if call[0] == "rz")
    first_basis_index = next(
        index for index, call in enumerate(calls) if call[0] == "basis"
    )
    assert first_rz_index < first_basis_index


def test_ghz_stage_delegates_final_virtual_z_phases() -> None:
    captured: dict[str, object] = {}

    def _delegate(**kwargs):
        captured.update(kwargs)
        return "ghz-exp"

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(vz.ghz_experiment, "_create_experiment_impl", _delegate)
    try:
        result = vz._create_experiment_impl(
            qpu=SimpleNamespace(),
            qubits=[SimpleNamespace(uid="q0"), SimpleNamespace(uid="q1"), SimpleNamespace(uid="q2")],
            bus=[SimpleNamespace(uid="b0"), SimpleNamespace(uid="b1"), SimpleNamespace(uid="b2")],
            phase_tuple=(0.1, -0.2, 0.3),
            stage="ghz",
        )
    finally:
        monkeypatch.undo()

    assert result == "ghz-exp"
    assert captured["final_virtual_z_phases"] == pytest.approx((0.1, -0.2, 0.3))
    assert captured["ghz_prep"] is True


def test_experiment_workflow_reuses_single_readout_calibration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create_experiment_calls: list[dict[str, object]] = []
    run_calls: list[str] = []

    @vz.workflow.task(save=False)
    def _compile(session, exp):  # noqa: ARG001
        return exp

    @vz.workflow.task(save=False)
    def _run(session, compiled):  # noqa: ARG001
        run_calls.append(compiled)
        if compiled == "readout-cal-exp":
            return {"kind": "calibration"}
        return {"kind": compiled}

    @vz.workflow.task(save=False)
    def _collect_record(**kwargs):  # noqa: ARG001
        return {
            "record": {
                "stage": kwargs["stage"],
                "phase_target": kwargs["phase_target"],
                "phase_value": float(kwargs["phase_value"]),
                "expected_phase": float(kwargs["phase_value"]),
                "repeat": int(kwargs["repeat"]),
                "measured_phase": float(kwargs["phase_value"]),
                "wrapped_phase_error": 0.0,
                "fidelity": 0.99,
                "coherence_magnitude": 0.5,
                "optimizer_success": True,
                "nll": 1.0,
                "min_eig": 0.0,
            },
            "failure": None,
        }

    @vz.workflow.task(save=False)
    def _summarize(stage, run_records):  # noqa: ARG001
        return {
            "stage": stage,
            "aggregate": {"num_total_runs": len(run_records)},
            "per_target": {},
        }

    @vz.workflow.task(save=False)
    def _resolve_max_iterations(analysis_options=None):  # noqa: ARG001
        return 123

    @vz.workflow.task(save=False)
    def _resolve_do_plotting(analysis_options=None):  # noqa: ARG001
        return False

    monkeypatch.setattr(vz, "temporary_qpu", lambda qpu, temporary_parameters: qpu)
    monkeypatch.setattr(
        vz, "temporary_quantum_elements_from_qpu", lambda qpu, elements: elements
    )
    monkeypatch.setattr(
        vz,
        "create_readout_calibration_experiment",
        lambda temp_qpu, qubits: "readout-cal-exp",
    )
    monkeypatch.setattr(
        vz,
        "create_experiment",
        lambda *args, **kwargs: create_experiment_calls.append(kwargs) or f"{kwargs['stage']}-exp",
    )
    monkeypatch.setattr(vz, "compile_experiment", _compile)
    monkeypatch.setattr(vz, "run_experiment", _run)
    monkeypatch.setattr(vz, "collect_virtual_z_run_record", _collect_record)
    monkeypatch.setattr(vz, "summarize_virtual_z_validation", _summarize)
    monkeypatch.setattr(vz, "resolve_analysis_max_mle_iterations", _resolve_max_iterations)
    monkeypatch.setattr(vz, "resolve_do_plotting", _resolve_do_plotting)
    monkeypatch.setattr(
        vz,
        "_select_qubit_for_analysis",
        lambda qubits, index, expected_len, caller: qubits[index],
    )

    result = vz.experiment_workflow(
        session=SimpleNamespace(),
        qpu=SimpleNamespace(),
        qubits=[
            SimpleNamespace(uid="q0"),
            SimpleNamespace(uid="q1"),
            SimpleNamespace(uid="q2"),
        ],
        bus=[
            SimpleNamespace(uid="b0"),
            SimpleNamespace(uid="b1"),
            SimpleNamespace(uid="b2"),
        ],
        phase_values=(0.0, np.pi / 2),
    ).run()

    output = _output_to_dict(result.output)
    assert run_calls.count("readout-cal-exp") == 1
    assert len(create_experiment_calls) == 2
    assert {call["stage"] for call in create_experiment_calls} == {"product", "ghz"}
    assert output["phase_targets"] == ("q0", "q1", "q2")
    assert output["phase_values"] == pytest.approx((0.0, np.pi / 2))
    assert output["repeats_per_phase"] == 1
    assert len(output["product_raw_run_records"]) == 6
    assert len(output["ghz_raw_run_records"]) == 6
    assert output["product_summary"]["aggregate"]["num_total_runs"] == 6
    assert output["ghz_summary"]["aggregate"]["num_total_runs"] == 6
    _assert_no_reference(output)


def test_run_bundle_defaults_dense_phase_grid_and_returns_plain_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Runner:
        def __init__(self, output):
            self._output = output

        def run(self):
            return SimpleNamespace(output=self._output)

    def _workflow_stub(**kwargs):  # noqa: ARG001
        phase_values = kwargs["phase_values"]
        return _Runner(
            {
                "readout_calibration_result": {"kind": "calibration"},
                "product_raw_run_records": [],
                "product_failed_runs": [],
                "product_summary": {"aggregate": {"num_total_runs": 0}, "per_target": {}},
                "ghz_raw_run_records": [],
                "ghz_failed_runs": [],
                "ghz_summary": {"aggregate": {"num_total_runs": 0}, "per_target": {}},
                "phase_values": phase_values,
                "phase_targets": ("q0", "q1", "q2"),
                "repeats_per_phase": 1,
            }
        )

    monkeypatch.setattr(vz, "experiment_workflow", _workflow_stub)

    output = vz.run_bundle(
        session=SimpleNamespace(),
        qpu=SimpleNamespace(),
        qubits=[
            SimpleNamespace(uid="q0"),
            SimpleNamespace(uid="q1"),
            SimpleNamespace(uid="q2"),
        ],
        bus=[
            SimpleNamespace(uid="b0"),
            SimpleNamespace(uid="b1"),
            SimpleNamespace(uid="b2"),
        ],
    )

    assert len(output["phase_values"]) == 13
    assert output["phase_values"][0] == pytest.approx(-np.pi)
    assert output["phase_values"][-1] == pytest.approx(np.pi)
    _assert_no_reference(output)


def test_run_bundle_requires_readout_calibration_when_disabled() -> None:
    options = vz.ThreeQVZValidationWorkflowOptions()
    options.do_readout_calibration = False

    with pytest.raises(ValueError, match="readout calibration"):
        vz.run_bundle(
            session=SimpleNamespace(),
            qpu=SimpleNamespace(),
            qubits=[],
            bus=[],
            options=options,
            phase_values=(0.0,),
        )


def test_extract_product_phase_impl_tracks_positive_phase() -> None:
    phase_value = 0.7
    psi = vz_analysis._product_target_statevector("q1", phase_value, "+++")
    rho = np.outer(psi, psi.conj())

    measured_phase, coherence = vz_analysis._extract_product_phase_impl(rho, "q1")

    assert measured_phase == pytest.approx(phase_value)
    assert coherence == pytest.approx(0.5)


def test_extract_ghz_phase_impl_tracks_positive_phase() -> None:
    phase_value = -0.6
    psi = vz_analysis._ghz_target_statevector(phase_value)
    rho = np.outer(psi, psi.conj())

    measured_phase, coherence = vz_analysis._extract_ghz_phase_impl(rho)

    assert measured_phase == pytest.approx(phase_value)
    assert coherence == pytest.approx(0.5)


def test_summarize_virtual_z_validation_impl_unwraps_phase_and_fits_line() -> None:
    phase_values = np.linspace(-np.pi, np.pi, 9)
    records = []
    for phase_value in phase_values:
        measured_phase = float(np.angle(np.exp(1j * (phase_value + 0.2))))
        records.append(
            {
                "stage": "ghz",
                "phase_target": "q0",
                "phase_value": float(phase_value),
                "expected_phase": float(phase_value),
                "repeat": 1,
                "measured_phase": measured_phase,
                "wrapped_phase_error": 0.2,
                "fidelity": 0.98,
                "coherence_magnitude": 0.5,
                "optimizer_success": True,
                "nll": 1.0,
                "min_eig": 0.0,
            }
        )

    summary = vz_analysis._summarize_virtual_z_validation_impl(
        stage="ghz",
        run_records=records,
    )
    target_summary = summary["per_target"]["q0"]

    assert target_summary["slope"] == pytest.approx(1.0, abs=1e-9)
    assert target_summary["intercept"] == pytest.approx(0.2, abs=1e-9)
    assert target_summary["phase_rmse"] == pytest.approx(0.2, abs=1e-9)
    assert target_summary["n_valid_points"] == len(phase_values)


def test_plot_helpers_save_artifacts(monkeypatch: pytest.MonkeyPatch) -> None:
    artifact_names: list[str] = []

    monkeypatch.setattr(
        vz_analysis.workflow,
        "save_artifact",
        lambda name, fig: artifact_names.append(name),
    )

    sample_records = [
        {
            "stage": "product",
            "phase_target": "q0",
            "phase_value": -0.2,
            "measured_phase": -0.2,
            "fidelity": 0.99,
            "coherence_magnitude": 0.5,
            "repeat": 1,
        },
        {
            "stage": "product",
            "phase_target": "q0",
            "phase_value": 0.2,
            "measured_phase": 0.2,
            "fidelity": 0.98,
            "coherence_magnitude": 0.49,
            "repeat": 1,
        },
    ]
    vz_analysis.plot_phase_tracking.func(
        product_run_records=sample_records,
        ghz_run_records=sample_records,
    )
    vz_analysis.plot_quality_summary.func(
        product_run_records=sample_records,
        ghz_run_records=sample_records,
    )

    assert "three_qubit_virtual_z_validation_phase_tracking" in artifact_names
    assert "three_qubit_virtual_z_validation_quality_summary" in artifact_names

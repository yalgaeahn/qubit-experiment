from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import matplotlib
import pytest

matplotlib.use("Agg")

pytest.importorskip("laboneq")
pytest.importorskip("laboneq_applications")
pytest.importorskip("scipy")

from laboneq.workflow.reference import Reference

from qubit_experiment.analysis import three_qubit_ghz as ghz_analysis
from qubit_experiment.experiments import three_qubit_ghz as ghz


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


def test_workflow_options_expose_ghz_fields_and_hide_product_state_fields() -> None:
    options = ghz.experiment_workflow.options()
    analysis_options = ghz_analysis.analysis_workflow.options()
    workflow_defaults = ghz.ThreeQGhzWorkflowOptions()

    assert hasattr(options, "do_analysis")
    assert hasattr(options, "do_readout_calibration")
    assert hasattr(options, "ghz_prep")
    assert hasattr(options, "final_virtual_z_phases")
    assert hasattr(options, "do_convergence_validation")
    assert hasattr(options, "convergence_repeats")
    assert hasattr(options, "count")
    assert hasattr(analysis_options, "do_plotting")
    assert hasattr(analysis_options, "max_mle_iterations")
    assert workflow_defaults.final_virtual_z_phases == (0.0, 0.0, 0.0)

    assert not hasattr(options, "initial_state")
    assert not hasattr(options, "custom_prep")
    assert not hasattr(options, "shot_sweep_log2_values")
    assert not hasattr(options, "shot_sweep_suite_states")


def test_create_experiment_rejects_disabled_ghz_prep() -> None:
    with pytest.raises(ValueError, match="ghz_prep=True"):
        ghz._create_experiment_impl(
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
            ghz_prep=False,
        )


def test_create_experiment_rejects_invalid_acquisition_type() -> None:
    options = SimpleNamespace(
        count=4096,
        acquisition_type="raw",
        averaging_mode=ghz.AveragingMode.SINGLE_SHOT,
        repetition_mode=None,
        repetition_time=None,
        reset_oscillator_phase=False,
        active_reset=False,
        active_reset_repetitions=1,
        active_reset_states="ge",
    )

    with pytest.raises(ValueError, match="AcquisitionType.INTEGRATION"):
        ghz._create_experiment_impl(
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
            options=options,
        )


def test_create_experiment_rejects_invalid_averaging_mode() -> None:
    options = SimpleNamespace(
        count=4096,
        acquisition_type=ghz.AcquisitionType.INTEGRATION,
        averaging_mode="cyclic",
        repetition_mode=None,
        repetition_time=None,
        reset_oscillator_phase=False,
        active_reset=False,
        active_reset_repetitions=1,
        active_reset_states="ge",
    )

    with pytest.raises(ValueError, match="AveragingMode.SINGLE_SHOT"):
        ghz._create_experiment_impl(
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
            options=options,
        )


def test_create_experiment_rejects_non_three_qubit_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ghz.validation,
        "validate_and_convert_qubits_sweeps",
        lambda qubits: list(qubits),
    )

    with pytest.raises(ValueError, match="expects exactly 3 qubits"):
        ghz._create_experiment_impl(
            qpu=SimpleNamespace(),
            qubits=[SimpleNamespace(uid="q0"), SimpleNamespace(uid="q1")],
            bus=[
                SimpleNamespace(uid="b0", signals={"drive": "b0/drive", "drive_p": "b0/drive_p"}),
                SimpleNamespace(uid="b1", signals={"drive": "b1/drive", "drive_p": "b1/drive_p"}),
                SimpleNamespace(uid="b2", signals={"drive": "b2/drive", "drive_p": "b2/drive_p"}),
            ],
        )


def test_create_experiment_rejects_non_three_bus_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ghz.validation,
        "validate_and_convert_qubits_sweeps",
        lambda qubits: list(qubits),
    )
    monkeypatch.setattr(
        ghz.validation,
        "validate_and_convert_single_qubit_sweeps",
        lambda element: element,
    )

    with pytest.raises(ValueError, match="expects exactly 3 bus elements"):
        ghz._normalize_three_buses(
            [SimpleNamespace(uid="b0", signals={"drive": "b0/drive", "drive_p": "b0/drive_p"})]
        )


def test_create_experiment_rejects_missing_drive_p_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ghz.validation,
        "validate_and_convert_single_qubit_sweeps",
        lambda element: element,
    )

    with pytest.raises(ValueError, match="drive_p"):
        ghz._normalize_three_buses(
            [
                SimpleNamespace(uid="b0", signals={"drive": "b0/drive", "drive_p": "b0/drive_p"}),
                SimpleNamespace(uid="b1", signals={"drive": "b1/drive"}),
                SimpleNamespace(uid="b2", signals={"drive": "b2/drive", "drive_p": "b2/drive_p"}),
            ]
        )


def test_normalize_final_virtual_z_phases_rejects_wrong_length() -> None:
    with pytest.raises(ValueError, match="exactly 3 values"):
        ghz._normalize_final_virtual_z_phases((0.1, 0.2))


def test_normalize_final_virtual_z_phases_rejects_non_numeric() -> None:
    with pytest.raises(ValueError, match="must be numeric"):
        ghz._normalize_final_virtual_z_phases((0.1, "bad", 0.3))


def _run_fake_ghz_sequence(
    monkeypatch: pytest.MonkeyPatch,
    *,
    final_virtual_z_phases=(0.0, 0.0, 0.0),
) -> tuple[list[tuple], list[dict[str, object]]]:
    calls: list[tuple] = []
    section_calls: list[dict[str, object]] = []

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

        def set_bus_frequency(self, bus_elem, frequency, *, line="drive", **kwargs):
            calls.append(("set_bus_frequency", bus_elem.uid, line, frequency))

        def ry(self, qubit, angle, **kwargs):
            calls.append(("ry", qubit.uid, float(angle)))
            return SimpleNamespace(uid=f"ry_{qubit.uid}_{len(calls)}")

        def rz(self, qubit, angle, **kwargs):
            calls.append(("rz", qubit.uid, float(angle)))
            return SimpleNamespace(uid=f"rz_{qubit.uid}_{len(calls)}")

        def rip(self, bus_elem, *, line="drive", **kwargs):
            calls.append(("rip", bus_elem.uid, line))

        def apply_tomography_prerotation(self, qubit, axis):
            calls.append(("basis", qubit.uid, axis))

        def measure(self, qubit, handle):
            calls.append(("measure", qubit.uid, handle))
            return _FakeMeasureSection(f"measure_{qubit.uid}")

        def passive_reset(self, qubit):
            calls.append(("passive_reset", qubit.uid))

    monkeypatch.setattr(
        ghz.validation,
        "validate_and_convert_qubits_sweeps",
        lambda elements: list(elements),
    )
    monkeypatch.setattr(
        ghz.validation,
        "validate_and_convert_single_qubit_sweeps",
        lambda element: element,
    )
    monkeypatch.setattr(
        ghz.dsl,
        "acquire_loop_rt",
        lambda **kwargs: nullcontext(),
    )

    def _section_factory(name=None, **kwargs):
        uid = name or "section"
        section_calls.append(
            {
                "name": uid,
                "play_after": kwargs.get("play_after"),
                "alignment": kwargs.get("alignment"),
            }
        )
        return _FakeSection(uid)

    monkeypatch.setattr(ghz.dsl, "section", _section_factory)
    monkeypatch.setattr(ghz, "TOMOGRAPHY_SETTINGS", (("XYZ", ("X", "Y", "Z")),))

    qpu = SimpleNamespace(quantum_operations=_FakeQop())
    qubits = [
        SimpleNamespace(uid="q0"),
        SimpleNamespace(uid="q1"),
        SimpleNamespace(uid="q2"),
    ]
    bus = [
        SimpleNamespace(
            uid="b0",
            signals={"drive": "b0/drive", "drive_p": "b0/drive_p"},
            parameters=SimpleNamespace(
                resonance_frequency_bus=5.5e9,
                rip_detuning=-10e6,
                resonance_frequency_bus_p=6.0e9,
                rip_p_detuning=-20e6,
            ),
        ),
        SimpleNamespace(
            uid="b1",
            signals={"drive": "b1/drive", "drive_p": "b1/drive_p"},
            parameters=SimpleNamespace(
                resonance_frequency_bus=5.6e9,
                rip_detuning=-11e6,
                resonance_frequency_bus_p=6.1e9,
                rip_p_detuning=-21e6,
            ),
        ),
        SimpleNamespace(
            uid="b2",
            signals={"drive": "b2/drive", "drive_p": "b2/drive_p"},
            parameters=SimpleNamespace(
                resonance_frequency_bus=5.7e9,
                rip_detuning=-12e6,
                resonance_frequency_bus_p=6.2e9,
                rip_p_detuning=-22e6,
            ),
        ),
    ]
    options = SimpleNamespace(
        count=16,
        acquisition_type=ghz.AcquisitionType.INTEGRATION,
        averaging_mode=ghz.AveragingMode.SINGLE_SHOT,
        repetition_mode=None,
        repetition_time=None,
        reset_oscillator_phase=False,
        active_reset=False,
        active_reset_repetitions=1,
        active_reset_states="ge",
    )

    ghz._create_experiment_impl(
        qpu=qpu,
        qubits=qubits,
        bus=bus,
        ghz_prep=True,
        final_virtual_z_phases=final_virtual_z_phases,
        options=options,
    )
    return calls, section_calls


def test_create_experiment_uses_expected_ghz_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls, section_calls = _run_fake_ghz_sequence(
        monkeypatch,
        final_virtual_z_phases=(0.1, -0.2, 0.3),
    )

    assert calls[:6] == [
        ("set_bus_frequency", "b0", "drive", pytest.approx(5.49e9)),
        ("set_bus_frequency", "b1", "drive", pytest.approx(5.589e9)),
        ("set_bus_frequency", "b2", "drive", pytest.approx(5.688e9)),
        ("set_bus_frequency", "b0", "drive_p", pytest.approx(5.98e9)),
        ("set_bus_frequency", "b1", "drive_p", pytest.approx(6.079e9)),
        ("set_bus_frequency", "b2", "drive_p", pytest.approx(6.178e9)),
    ]
    assert calls[6:15] == [
        ("ry", "q0", pytest.approx(float(ghz.np.pi / 2))),
        ("ry", "q1", pytest.approx(float(-0 + ghz.np.pi / 2))),
        ("rip", "b0", "drive"),
        ("rip", "b1", "drive"),
        ("rip", "b2", "drive"),
        ("rz", "q1", pytest.approx(-0.2)),
        ("ry", "q1", pytest.approx(float(-ghz.np.pi / 2))),
        ("rz", "q1", pytest.approx(0.2)),
        ("ry", "q2", pytest.approx(float(ghz.np.pi / 2))),
    ]
    assert [call for call in calls if call[0] == "rz"] == [
        ("rz", "q1", pytest.approx(-0.2)),
        ("rz", "q1", pytest.approx(0.2)),
        ("rz", "q2", pytest.approx(0.3)),
        ("rz", "q2", pytest.approx(-0.3)),
    ]
    assert ("rz", "q0", pytest.approx(0.1)) not in calls

    q2_window_start = calls.index(("ry", "q2", pytest.approx(float(ghz.np.pi / 2))))
    assert calls[q2_window_start : q2_window_start + 7] == [
        ("ry", "q2", pytest.approx(float(ghz.np.pi / 2))),
        ("rip", "b0", "drive_p"),
        ("rip", "b1", "drive_p"),
        ("rip", "b2", "drive_p"),
        ("rz", "q2", pytest.approx(0.3)),
        ("ry", "q2", pytest.approx(float(-ghz.np.pi / 2))),
        ("rz", "q2", pytest.approx(-0.3)),
    ]

    q1_restore_index = max(
        index
        for index, call in enumerate(calls)
        if call == ("rz", "q1", pytest.approx(0.2))
    )
    q2_first_ry_index = next(
        index
        for index, call in enumerate(calls)
        if call == ("ry", "q2", pytest.approx(float(ghz.np.pi / 2)))
    )
    assert q1_restore_index < q2_first_ry_index

    q2_restore_index = max(
        index
        for index, call in enumerate(calls)
        if call == ("rz", "q2", pytest.approx(-0.3))
    )
    first_basis_index = next(
        index for index, call in enumerate(calls) if call[0] == "basis"
    )
    assert q2_restore_index < first_basis_index

    basis_section = next(
        section for section in section_calls if section["name"] == "basis_XYZ"
    )
    assert basis_section["play_after"] == "ghz_prep_XYZ"

    q1_rz_forward_section = next(
        section
        for section in section_calls
        if section["name"] == "ghz_q1_rz_forward_XYZ"
    )
    assert q1_rz_forward_section["play_after"] == "ghz_cz1_XYZ"

    q1_rz_restore_section = next(
        section
        for section in section_calls
        if section["name"] == "ghz_q1_rz_restore_XYZ"
    )
    assert q1_rz_restore_section["play_after"] == "ghz_q1_ry_minus_90_XYZ"

    q2_rz_forward_section = next(
        section
        for section in section_calls
        if section["name"] == "ghz_q2_rz_forward_XYZ"
    )
    assert q2_rz_forward_section["play_after"] == "ghz_cz2_XYZ"

    q2_rz_restore_section = next(
        section
        for section in section_calls
        if section["name"] == "ghz_q2_rz_restore_XYZ"
    )
    assert q2_rz_restore_section["play_after"] == "ghz_q2_ry_minus_90_XYZ"


def test_create_experiment_skips_local_virtual_z_when_q1_q2_phases_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls, section_calls = _run_fake_ghz_sequence(
        monkeypatch,
        final_virtual_z_phases=(0.6, 0.0, 0.0),
    )

    assert all(call[0] != "rz" for call in calls)
    assert all(
        not section["name"].startswith("ghz_q1_rz_")
        and not section["name"].startswith("ghz_q2_rz_")
        for section in section_calls
    )


def test_basis_section_depends_on_outer_prep_section(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, section_calls = _run_fake_ghz_sequence(
        monkeypatch,
        final_virtual_z_phases=(0.3, 0.0, -0.1),
    )

    basis_section = next(
        section for section in section_calls if section["name"] == "basis_XYZ"
    )
    assert basis_section["play_after"] == "ghz_prep_XYZ"
    assert basis_section["play_after"] != "ghz_q2_ry_minus_90_XYZ"
    assert basis_section["play_after"] != "ghz_q2_rz_restore_XYZ"


def test_analysis_workflow_returns_plain_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @ghz_analysis.workflow.task(save=False)
    def _analysis_payload(**kwargs):  # noqa: ARG001
        return {
            "metrics": {"fidelity_to_target": 0.99},
            "optimization_convergence": {"nll_finite": True},
            "tomography_counts": {"XXX": [1, 2, 3, 4]},
            "predicted_counts": {"XXX": [1, 2, 3, 4]},
            "setting_labels": ["XXX"],
            "rho_hat_real": [[1.0, 0.0], [0.0, 0.0]],
            "rho_hat_imag": [[0.0, 0.0], [0.0, 0.0]],
        }

    monkeypatch.setattr(ghz_analysis, "analyze_tomography_run", _analysis_payload)

    options = ghz_analysis.analysis_workflow.options()
    options.do_plotting(False)

    result = ghz_analysis.analysis_workflow(
        tomography_result={"kind": "tomography"},
        q0=SimpleNamespace(uid="q0"),
        q1=SimpleNamespace(uid="q1"),
        q2=SimpleNamespace(uid="q2"),
        readout_calibration_result={"kind": "calibration"},
        options=options,
    ).run()

    assert result.output["metrics"]["fidelity_to_target"] == pytest.approx(0.99)
    _assert_no_reference(result.output)


def test_experiment_workflow_returns_top_level_raw_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    create_experiment_kwargs: dict[str, object] = {}
    final_virtual_z_phases = (0.4, -0.5, 0.6)

    @ghz.workflow.task(save=False)
    def _compile(session, exp):  # noqa: ARG001
        return exp

    @ghz.workflow.task(save=False)
    def _run(session, compiled):  # noqa: ARG001
        calls.append(compiled)
        if compiled == "readout-cal-exp":
            return {"kind": "calibration"}
        return {"kind": "tomography"}

    monkeypatch.setattr(ghz, "temporary_qpu", lambda qpu, temporary_parameters: qpu)
    monkeypatch.setattr(
        ghz, "temporary_quantum_elements_from_qpu", lambda qpu, elements: elements
    )
    monkeypatch.setattr(
        ghz,
        "create_readout_calibration_experiment",
        lambda temp_qpu, qubits: "readout-cal-exp",
    )
    monkeypatch.setattr(ghz, "create_experiment", lambda *args, **kwargs: create_experiment_kwargs.update(kwargs) or "tomography-exp")
    monkeypatch.setattr(ghz, "compile_experiment", _compile)
    monkeypatch.setattr(ghz, "run_experiment", _run)

    options = ghz.experiment_workflow.options()
    options.do_readout_calibration(True)
    options.ghz_prep(True)
    options.final_virtual_z_phases(final_virtual_z_phases)

    result = ghz.experiment_workflow(
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
        options=options,
    ).run()

    output = _output_to_dict(result.output)
    assert calls == ["readout-cal-exp", "tomography-exp"]
    assert set(output) == {
        "tomography_result",
        "readout_calibration_result",
        "ghz_prep",
        "target_state_effective",
    }
    assert output["readout_calibration_result"]["kind"] == "calibration"
    assert output["target_state_effective"] == "ghz"
    assert create_experiment_kwargs["final_virtual_z_phases"] == final_virtual_z_phases
    _assert_no_reference(output)


def test_convergence_validation_workflow_returns_plain_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create_experiment_calls: list[dict[str, object]] = []
    final_virtual_z_phases = (0.7, 0.8, -0.9)

    @ghz.workflow.task(save=False)
    def _compile(session, exp):  # noqa: ARG001
        return exp

    @ghz.workflow.task(save=False)
    def _run(session, compiled):  # noqa: ARG001
        if compiled == "readout-cal-exp":
            return {"kind": "calibration"}
        return {"kind": compiled}

    @ghz.workflow.task(save=False)
    def _ghz_analysis_result(**kwargs):  # noqa: ARG001
        assert kwargs["max_iterations"] == 1234
        return {
            "metrics": {"fidelity_to_target": 0.98, "min_eigenvalue": 0.0},
            "optimization_convergence": {
                "nll_finite": True,
                "negative_log_likelihood": 1.2,
                "optimizer_success": True,
            },
            "negative_log_likelihood": 1.2,
            "optimizer_success": True,
        }

    monkeypatch.setattr(ghz, "temporary_qpu", lambda qpu, temporary_parameters: qpu)
    monkeypatch.setattr(
        ghz, "temporary_quantum_elements_from_qpu", lambda qpu, elements: elements
    )
    monkeypatch.setattr(
        ghz,
        "create_readout_calibration_experiment",
        lambda temp_qpu, qubits: "readout-cal-exp",
    )
    monkeypatch.setattr(
        ghz,
        "create_experiment",
        lambda *args, **kwargs: create_experiment_calls.append(kwargs) or "tomography-exp",
    )
    monkeypatch.setattr(ghz, "compile_experiment", _compile)
    monkeypatch.setattr(ghz, "run_experiment", _run)
    monkeypatch.setattr(
        ghz,
        "_select_qubit_for_analysis",
        lambda qubits, index, expected_len, caller: qubits[index],
    )
    monkeypatch.setattr(ghz, "analyze_tomography_run", _ghz_analysis_result)

    options = ghz.convergence_validation_workflow.options()
    options.do_readout_calibration(True)
    options.convergence_repeats(1)
    options.convergence_do_plotting(False)
    options.final_virtual_z_phases(final_virtual_z_phases)

    analysis_options = ghz_analysis.analysis_workflow.options()
    analysis_options.do_plotting(False)
    analysis_options.max_mle_iterations(1234)

    result = ghz.convergence_validation_workflow(
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
        main_run_optimization_convergence={"nll_finite": True},
        options=options,
        analysis_options=analysis_options,
    ).run()

    output = _output_to_dict(result.output)
    assert output["main_run_optimization_convergence"] == {"nll_finite": True}
    assert output["raw_run_records"][0]["state_label"] == "ghz"
    assert output["statistical_convergence"]["aggregate"]["num_total_runs"] == 1
    assert create_experiment_calls[0]["final_virtual_z_phases"] == final_virtual_z_phases
    _assert_no_reference(output)


def test_run_bundle_reconstructs_ghz_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_virtual_z_phases = (0.2, -0.1, 0.05)
    options = ghz.experiment_workflow.options()
    options.do_analysis(True)
    options.do_readout_calibration(True)
    options.do_convergence_validation(True)
    options.final_virtual_z_phases(final_virtual_z_phases)

    analysis_options = ghz_analysis.analysis_workflow.options()
    analysis_options.do_plotting(False)

    class _Runner:
        def __init__(self, output):
            self._output = output

        def run(self):
            return SimpleNamespace(output=self._output)

    def _experiment_stub(**kwargs):  # noqa: ARG001
        assert kwargs["options"].final_virtual_z_phases == final_virtual_z_phases
        return _Runner(
            SimpleNamespace(
                tomography_result={"kind": "tomography"},
                readout_calibration_result={"kind": "calibration"},
                ghz_prep=True,
                target_state_effective="ghz",
            )
        )

    def _analysis_stub(**kwargs):  # noqa: ARG001
        return _Runner(
            {
                "metrics": {"fidelity_to_target": 0.99},
                "optimization_convergence": {"nll_finite": True},
            }
        )

    def _convergence_stub(**kwargs):  # noqa: ARG001
        assert kwargs["options"].final_virtual_z_phases == final_virtual_z_phases
        return _Runner(
            SimpleNamespace(
                repeats=1,
                raw_run_records=[{"state_label": "ghz"}],
                statistical_convergence={"aggregate": {"num_total_runs": 1}},
                main_run_optimization_convergence={"nll_finite": True},
            )
        )

    monkeypatch.setattr(ghz, "experiment_workflow", _experiment_stub)
    monkeypatch.setattr(ghz, "analysis_workflow", _analysis_stub)
    monkeypatch.setattr(ghz, "convergence_validation_workflow", _convergence_stub)
    monkeypatch.setattr(ghz, "temporary_qpu", lambda qpu, temporary_parameters: qpu)
    monkeypatch.setattr(
        ghz, "temporary_quantum_elements_from_qpu", lambda qpu, elements: elements
    )
    monkeypatch.setattr(
        ghz,
        "_normalize_three_qubits",
        lambda qubits: (qubits[0], qubits[1], qubits[2]),
    )

    output = ghz.run_bundle(
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
        options=options,
        analysis_options=analysis_options,
    )

    assert set(output) == {
        "tomography_result",
        "readout_calibration_result",
        "analysis_result",
        "convergence_report",
        "ghz_prep",
        "target_state_effective",
    }
    assert output["analysis_result"]["metrics"]["fidelity_to_target"] == pytest.approx(0.99)
    assert output["convergence_report"]["main_run_optimization_convergence"] == {
        "nll_finite": True
    }
    assert output["target_state_effective"] == "ghz"
    _assert_no_reference(output)

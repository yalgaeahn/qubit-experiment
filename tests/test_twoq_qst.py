from __future__ import annotations

from types import SimpleNamespace

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

pytest.importorskip("laboneq")
pytest.importorskip("laboneq_applications")
pytest.importorskip("scipy")

from laboneq.workflow.reference import Reference

from qubit_experiment.analysis import twoq_qst as qst_analysis
from qubit_experiment.experiments import twoq_qst as qst


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


def test_analysis_workflow_returns_plain_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @qst_analysis.workflow.task(save=False)
    def _analysis_payload(**kwargs):  # noqa: ARG001
        return {
            "metrics": {"fidelity_to_target": 0.99},
            "optimization_convergence": {"nll_finite": True},
            "tomography_counts": {"XX": [1, 2, 3, 4]},
            "predicted_counts": {"XX": [1, 2, 3, 4]},
            "setting_labels": ["XX"],
            "rho_hat_real": [[1.0, 0.0], [0.0, 0.0]],
            "rho_hat_imag": [[0.0, 0.0], [0.0, 0.0]],
        }

    monkeypatch.setattr(qst_analysis, "analyze_tomography_run", _analysis_payload)

    options = qst_analysis.analysis_workflow.options()
    options.do_plotting(False)

    result = qst_analysis.analysis_workflow(
        tomography_result={"kind": "tomography"},
        ctrl=SimpleNamespace(uid="q0"),
        targ=SimpleNamespace(uid="q1"),
        readout_calibration_result={"kind": "calibration"},
        target_state="00",
        options=options,
    ).run()

    assert result.output["metrics"]["fidelity_to_target"] == pytest.approx(0.99)
    _assert_no_reference(result.output)


def test_experiment_workflow_returns_top_level_raw_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    @qst.workflow.task(save=False)
    def _compile(session, exp):  # noqa: ARG001
        return exp

    @qst.workflow.task(save=False)
    def _run(session, compiled):  # noqa: ARG001
        calls.append(compiled)
        if compiled == "readout-cal-exp":
            return {"kind": "calibration"}
        return {"kind": "tomography"}

    monkeypatch.setattr(qst, "temporary_qpu", lambda qpu, temporary_parameters: qpu)
    monkeypatch.setattr(
        qst, "temporary_quantum_elements_from_qpu", lambda qpu, elements: elements
    )
    monkeypatch.setattr(
        qst,
        "create_readout_calibration_experiment",
        lambda temp_qpu, qubits: "readout-cal-exp",
    )
    monkeypatch.setattr(qst, "create_experiment", lambda *args, **kwargs: "tomography-exp")
    monkeypatch.setattr(qst, "compile_experiment", _compile)
    monkeypatch.setattr(qst, "run_experiment", _run)

    options = qst.experiment_workflow.options()
    options.do_readout_calibration(True)
    options.initial_state("++")
    options.custom_prep(False)

    result = qst.experiment_workflow(
        session=SimpleNamespace(),
        qpu=SimpleNamespace(),
        qubits=[SimpleNamespace(uid="q0"), SimpleNamespace(uid="q1")],
        bus=SimpleNamespace(uid="b0"),
        options=options,
    ).run()

    output = _output_to_dict(result.output)
    assert calls == ["readout-cal-exp", "tomography-exp"]
    assert set(output) == {
        "tomography_result",
        "readout_calibration_result",
        "initial_state",
        "target_state_effective",
        "custom_prep",
    }
    assert output["readout_calibration_result"]["kind"] == "calibration"
    _assert_no_reference(output)


def test_convergence_validation_workflow_returns_plain_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @qst.workflow.task(save=False)
    def _compile(session, exp):  # noqa: ARG001
        return exp

    @qst.workflow.task(save=False)
    def _run(session, compiled):  # noqa: ARG001
        if compiled == "readout-cal-exp":
            return {"kind": "calibration"}
        return {"kind": compiled}

    @qst.workflow.task(save=False)
    def _suite_analysis_result(**kwargs):  # noqa: ARG001
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

    monkeypatch.setattr(qst, "temporary_qpu", lambda qpu, temporary_parameters: qpu)
    monkeypatch.setattr(
        qst, "temporary_quantum_elements_from_qpu", lambda qpu, elements: elements
    )
    monkeypatch.setattr(
        qst,
        "create_readout_calibration_experiment",
        lambda temp_qpu, qubits: "readout-cal-exp",
    )
    monkeypatch.setattr(qst, "create_experiment", lambda *args, **kwargs: "tomography-exp")
    monkeypatch.setattr(qst, "compile_experiment", _compile)
    monkeypatch.setattr(qst, "run_experiment", _run)
    monkeypatch.setattr(
        qst,
        "_select_qubit_for_analysis",
        lambda qubits, index, expected_len, caller: qubits[index],
    )
    monkeypatch.setattr(qst, "analyze_tomography_run", _suite_analysis_result)

    options = qst.convergence_validation_workflow.options()
    options.do_readout_calibration(True)
    options.convergence_repeats_per_state(1)
    options.convergence_suite_states(("00",))
    options.convergence_do_plotting(False)

    analysis_options = qst_analysis.analysis_workflow.options()
    analysis_options.do_plotting(False)
    analysis_options.max_mle_iterations(1234)

    result = qst.convergence_validation_workflow(
        session=SimpleNamespace(),
        qpu=SimpleNamespace(),
        qubits=[SimpleNamespace(uid="q0"), SimpleNamespace(uid="q1")],
        bus=SimpleNamespace(uid="b0"),
        main_run_optimization_convergence={"nll_finite": True},
        options=options,
        analysis_options=analysis_options,
    ).run()

    output = _output_to_dict(result.output)
    assert output["main_run_optimization_convergence"] == {"nll_finite": True}
    assert output["raw_run_records"][0]["state_label"] == "00"
    assert output["statistical_convergence"]["aggregate"]["num_total_runs"] == 1
    _assert_no_reference(output)


def test_shot_sweep_workflow_returns_plain_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @qst.workflow.task(save=False)
    def _compile(session, exp):  # noqa: ARG001
        return exp

    @qst.workflow.task(save=False)
    def _run(session, compiled):  # noqa: ARG001
        if compiled == "readout-cal-exp":
            return {"kind": "calibration"}
        return {"kind": compiled}

    @qst.workflow.task(save=False)
    def _collect_shot_record(**kwargs):  # noqa: ARG001
        assert kwargs["max_iterations"] == 1234
        return {
            "record": {
                "state": "00",
                "log2_shots": 3,
                "shots": 8,
                "repeat": 1,
                "fidelity": 0.97,
                "infidelity": 0.03,
                "log10_infidelity": np.log10(0.03),
                "nll": 1.0,
                "min_eig": 0.0,
            },
            "failure": None,
        }

    @qst.workflow.task(save=False)
    def _validate_records(**kwargs):  # noqa: ARG001
        return {"ok": True}

    @qst.workflow.task(save=False)
    def _aggregate_stats(run_records):  # noqa: ARG001
        return [
            {
                "state": "00",
                "log2_shots": 3,
                "shots": 8,
                "n_total": len(run_records),
                "n_valid_infidelity": len(run_records),
                "infid_mean": 0.03,
                "infid_ci95": 0.0,
                "log10_infid_mean": np.log10(0.03),
                "log10_infid_ci95": 0.0,
            }
        ]

    @qst.workflow.task(save=False)
    def _final_summary(**kwargs):  # noqa: ARG001
        return [
            {
                "state": "00",
                "n_total": 1,
                "n_valid_infidelity": 1,
                "infid_mean": 0.03,
                "infid_ci95": 0.0,
                "log10_infid_mean": np.log10(0.03),
                "log10_infid_ci95": 0.0,
            }
        ]

    monkeypatch.setattr(qst, "temporary_qpu", lambda qpu, temporary_parameters: qpu)
    monkeypatch.setattr(
        qst, "temporary_quantum_elements_from_qpu", lambda qpu, elements: elements
    )
    monkeypatch.setattr(
        qst,
        "create_readout_calibration_experiment",
        lambda temp_qpu, qubits: "readout-cal-exp",
    )
    monkeypatch.setattr(qst, "create_experiment", lambda *args, **kwargs: "tomography-exp")
    monkeypatch.setattr(qst, "compile_experiment", _compile)
    monkeypatch.setattr(qst, "run_experiment", _run)
    monkeypatch.setattr(
        qst,
        "_select_qubit_for_analysis",
        lambda qubits, index, expected_len, caller: qubits[index],
    )
    monkeypatch.setattr(qst, "collect_shot_sweep_run_record", _collect_shot_record)
    monkeypatch.setattr(qst, "validate_shot_sweep_run_records", _validate_records)
    monkeypatch.setattr(qst, "aggregate_shot_sweep_statistics", _aggregate_stats)
    monkeypatch.setattr(qst, "summarize_final_shot_sweep", _final_summary)

    options = qst.shot_sweep_workflow.options()
    options.do_readout_calibration(True)
    options.shot_sweep_log2_values((3,))
    options.shot_sweep_suite_states(("00",))
    options.shot_sweep_repeats_per_point(1)
    options.shot_sweep_do_plotting(False)

    analysis_options = qst_analysis.analysis_workflow.options()
    analysis_options.do_plotting(False)
    analysis_options.max_mle_iterations(1234)

    result = qst.shot_sweep_workflow(
        session=SimpleNamespace(),
        qpu=SimpleNamespace(),
        qubits=[SimpleNamespace(uid="q0"), SimpleNamespace(uid="q1")],
        bus=SimpleNamespace(uid="b0"),
        options=options,
        analysis_options=analysis_options,
    ).run()

    output = _output_to_dict(result.output)
    assert output["validation_checks"] == {"ok": True}
    assert output["aggregated_stats"][0]["shots"] == 8
    assert output["final_summary"][0]["state"] == "00"
    _assert_no_reference(output)


def test_run_bundle_reconstructs_notebook_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = qst.experiment_workflow.options()
    options.do_analysis(True)
    options.do_readout_calibration(True)
    options.do_convergence_validation(True)
    options.do_shot_sweep_convergence(True)

    analysis_options = qst_analysis.analysis_workflow.options()
    analysis_options.do_plotting(False)

    class _Runner:
        def __init__(self, output):
            self._output = output

        def run(self):
            return SimpleNamespace(output=self._output)

    def _experiment_stub(**kwargs):  # noqa: ARG001
        return _Runner(
            SimpleNamespace(
                tomography_result={"kind": "tomography"},
                readout_calibration_result={"kind": "calibration"},
                initial_state="++",
                target_state_effective="++",
                custom_prep=False,
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
        return _Runner(
            SimpleNamespace(
                suite_states=("00",),
                repeats_per_state=1,
                raw_run_records=[{"state_label": "00"}],
                statistical_convergence={"aggregate": {"num_total_runs": 1}},
                main_run_optimization_convergence={"nll_finite": True},
            )
        )

    def _shot_sweep_stub(**kwargs):  # noqa: ARG001
        return _Runner(
            SimpleNamespace(
                suite_states=("00",),
                shot_log2_values=(3,),
                shot_counts=(8,),
                repeats_per_point=1,
                raw_run_records=[{"state": "00"}],
                failed_runs=[],
                validation_checks={"ok": True},
                aggregated_stats=[{"shots": 8}],
                final_summary=[{"state": "00"}],
            )
        )

    monkeypatch.setattr(qst, "experiment_workflow", _experiment_stub)
    monkeypatch.setattr(qst, "analysis_workflow", _analysis_stub)
    monkeypatch.setattr(qst, "convergence_validation_workflow", _convergence_stub)
    monkeypatch.setattr(qst, "shot_sweep_workflow", _shot_sweep_stub)
    monkeypatch.setattr(qst, "temporary_qpu", lambda qpu, temporary_parameters: qpu)
    monkeypatch.setattr(
        qst, "temporary_quantum_elements_from_qpu", lambda qpu, elements: elements
    )
    monkeypatch.setattr(
        qst,
        "_normalize_two_qubits",
        lambda qubits: (qubits[0], qubits[1]),
    )

    output = qst.run_bundle(
        session=SimpleNamespace(),
        qpu=SimpleNamespace(),
        qubits=[SimpleNamespace(uid="q0"), SimpleNamespace(uid="q1")],
        bus=SimpleNamespace(uid="b0"),
        options=options,
        analysis_options=analysis_options,
    )

    assert set(output) == {
        "tomography_result",
        "readout_calibration_result",
        "analysis_result",
        "convergence_report",
        "shot_sweep_report",
        "initial_state",
        "target_state_effective",
        "custom_prep",
    }
    assert output["analysis_result"]["metrics"]["fidelity_to_target"] == pytest.approx(0.99)
    assert output["convergence_report"]["main_run_optimization_convergence"] == {
        "nll_finite": True
    }
    assert output["shot_sweep_report"]["final_summary"][0]["state"] == "00"
    _assert_no_reference(output)


def test_run_bundle_accepts_supplied_readout_calibration_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = qst.experiment_workflow.options()
    options.do_analysis(True)
    options.do_readout_calibration(False)

    class _Runner:
        def __init__(self, output):
            self._output = output

        def run(self):
            return SimpleNamespace(output=self._output)

    monkeypatch.setattr(
        qst,
        "experiment_workflow",
        lambda **kwargs: _Runner(
            SimpleNamespace(
                tomography_result={"kind": "tomography"},
                readout_calibration_result=kwargs["readout_calibration_result"],
                initial_state="++",
                target_state_effective="++",
                custom_prep=False,
            )
        ),
    )
    monkeypatch.setattr(
        qst,
        "analysis_workflow",
        lambda **kwargs: _Runner({"metrics": {"fidelity_to_target": 0.99}}),
    )
    monkeypatch.setattr(qst, "temporary_qpu", lambda qpu, temporary_parameters: qpu)
    monkeypatch.setattr(
        qst, "temporary_quantum_elements_from_qpu", lambda qpu, elements: elements
    )
    monkeypatch.setattr(
        qst,
        "_normalize_two_qubits",
        lambda qubits: (qubits[0], qubits[1]),
    )

    output = qst.run_bundle(
        session=SimpleNamespace(),
        qpu=SimpleNamespace(),
        qubits=[SimpleNamespace(uid="q0"), SimpleNamespace(uid="q1")],
        bus=SimpleNamespace(uid="b0"),
        readout_calibration_result={"kind": "provided"},
        options=options,
    )

    assert output["readout_calibration_result"] == {"kind": "provided"}
    assert output["analysis_result"]["metrics"]["fidelity_to_target"] == pytest.approx(
        0.99
    )

from __future__ import annotations

import json
from pathlib import Path
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

NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "projects"
    / "2026_selectiveRIP"
    / "notebooks"
    / "2Q_QST_TEST_NEW.ipynb"
)
TG_NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "projects"
    / "2026_selectiveRIP"
    / "noteforTG"
    / "ToTG.ipynb"
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


def test_workflow_options_expose_product_state_fields_and_hide_removed_rip_fields() -> None:
    options = qst.experiment_workflow.options()
    analysis_options = qst_analysis.analysis_workflow.options()
    workflow_defaults = qst.TwoQQstWorkflowOptions()

    assert hasattr(options, "do_analysis")
    assert hasattr(options, "do_readout_calibration")
    assert hasattr(options, "initial_state")
    assert hasattr(options, "custom_prep")
    assert hasattr(options, "do_convergence_validation")
    assert hasattr(options, "convergence_repeats_per_state")
    assert hasattr(options, "convergence_suite_states")
    assert hasattr(options, "do_shot_sweep_convergence")
    assert hasattr(options, "shot_sweep_log2_values")
    assert hasattr(options, "shot_sweep_suite_states")
    assert hasattr(options, "shot_sweep_repeats_per_point")
    assert hasattr(options, "count")
    assert hasattr(analysis_options, "do_plotting")
    assert hasattr(analysis_options, "max_mle_iterations")

    assert workflow_defaults.convergence_suite_states == (
        "00",
        "01",
        "10",
        "11",
        "++",
        "+-",
        "-+",
        "--",
    )
    assert workflow_defaults.shot_sweep_suite_states == (
        "00",
        "01",
        "10",
        "11",
        "++",
        "+-",
        "-+",
        "--",
    )
    assert workflow_defaults.shot_sweep_log2_values == tuple(range(3, 13))

    assert not hasattr(options, "validation_mode")
    assert not hasattr(options, "use_rip")
    assert not hasattr(options, "enforce_target_match")


def test_resolve_target_configuration_canonicalizes_matching_target() -> None:
    resolved = qst._resolve_target_configuration_impl(
        custom_prep=False,
        initial_state="ge",
        target_state=None,
    )

    assert resolved["initial_state"] == "01"
    assert resolved["target_state_effective"] == "01"
    assert resolved["custom_prep"] is False


def test_resolve_target_configuration_rejects_mismatched_target() -> None:
    with pytest.raises(ValueError, match="target_state must match initial_state"):
        qst._resolve_target_configuration_impl(
            custom_prep=False,
            initial_state="++",
            target_state="00",
        )


def test_resolve_target_configuration_rejects_custom_prep_for_now() -> None:
    with pytest.raises(
        NotImplementedError,
        match="independently of initial_state",
    ):
        qst._resolve_target_configuration_impl(
            custom_prep=True,
            initial_state="++",
            target_state=None,
        )


def test_create_experiment_rejects_custom_prep_until_implemented() -> None:
    with pytest.raises(
        NotImplementedError,
        match="independently of initial_state",
    ):
        qst._create_experiment_impl(
            qpu=SimpleNamespace(),
            qubits=[SimpleNamespace(uid="q0"), SimpleNamespace(uid="q1")],
            bus=SimpleNamespace(uid="b0"),
            custom_prep=True,
        )


def test_create_experiment_rejects_invalid_acquisition_type() -> None:
    options = SimpleNamespace(
        count=4096,
        acquisition_type="raw",
        averaging_mode=qst.AveragingMode.SINGLE_SHOT,
        repetition_mode=None,
        repetition_time=None,
        reset_oscillator_phase=False,
        active_reset=False,
        active_reset_repetitions=1,
        active_reset_states="ge",
    )

    with pytest.raises(ValueError, match="AcquisitionType.INTEGRATION"):
        qst._create_experiment_impl(
            qpu=SimpleNamespace(),
            qubits=[SimpleNamespace(uid="q0"), SimpleNamespace(uid="q1")],
            bus=SimpleNamespace(uid="b0"),
            options=options,
        )


def test_create_experiment_rejects_invalid_averaging_mode() -> None:
    options = SimpleNamespace(
        count=4096,
        acquisition_type=qst.AcquisitionType.INTEGRATION,
        averaging_mode="cyclic",
        repetition_mode=None,
        repetition_time=None,
        reset_oscillator_phase=False,
        active_reset=False,
        active_reset_repetitions=1,
        active_reset_states="ge",
    )

    with pytest.raises(ValueError, match="AveragingMode.SINGLE_SHOT"):
        qst._create_experiment_impl(
            qpu=SimpleNamespace(),
            qubits=[SimpleNamespace(uid="q0"), SimpleNamespace(uid="q1")],
            bus=SimpleNamespace(uid="b0"),
            options=options,
        )


def test_create_experiment_rejects_non_two_qubit_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        qst.validation,
        "validate_and_convert_qubits_sweeps",
        lambda qubits: list(qubits),
    )

    with pytest.raises(ValueError, match="expects exactly 2 qubits"):
        qst._create_experiment_impl(
            qpu=SimpleNamespace(),
            qubits=[SimpleNamespace(uid="q0")],
            bus=SimpleNamespace(uid="b0"),
        )


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


def test_summarize_statistical_convergence_matches_expected_keys() -> None:
    report = qst_analysis._summarize_statistical_convergence_impl(
        [
            {
                "state_label": "00",
                "repeat_index": 0,
                "fidelity_to_target": 0.95,
                "optimizer_success": True,
                "negative_log_likelihood": 10.0,
                "rho_min_eigenvalue": 0.0,
            },
            {
                "state_label": "00",
                "repeat_index": 1,
                "fidelity_to_target": 0.85,
                "optimizer_success": False,
                "negative_log_likelihood": 12.0,
                "rho_min_eigenvalue": -0.01,
            },
        ]
    )

    assert set(report.keys()) == {"per_state", "aggregate"}
    assert report["per_state"]["00"]["num_runs"] == 2
    assert report["per_state"]["00"]["optimizer_success_rate"] == pytest.approx(0.5)
    assert report["aggregate"]["num_total_runs"] == 2
    assert report["aggregate"]["pooled_fidelity_mean"] == pytest.approx(0.9)


def test_aggregate_shot_sweep_statistics_computes_notebook_schema() -> None:
    rows = [
        {
            "state": "00",
            "log2_shots": 3,
            "shots": 8,
            "repeat": 1,
            "fidelity": 0.9,
            "infidelity": 0.1,
            "log10_infidelity": np.log10(0.1),
            "nll": 10.0,
            "min_eig": 0.0,
        },
        {
            "state": "00",
            "log2_shots": 3,
            "shots": 8,
            "repeat": 2,
            "fidelity": 0.8,
            "infidelity": 0.2,
            "log10_infidelity": np.log10(0.2),
            "nll": 11.0,
            "min_eig": -0.01,
        },
    ]

    aggregated = qst_analysis._aggregate_shot_sweep_statistics_impl(rows)

    assert len(aggregated) == 1
    row = aggregated[0]
    assert row["state"] == "00"
    assert row["log2_shots"] == 3
    assert row["shots"] == 8
    assert row["n_total"] == 2
    assert row["n_valid_infidelity"] == 2
    assert row["infid_mean"] == pytest.approx(0.15)
    assert row["log10_infid_mean"] == pytest.approx(
        np.mean([np.log10(0.1), np.log10(0.2)])
    )


def test_validate_shot_sweep_run_records_reports_missing_bad_and_range_issues() -> None:
    rows = [
        {
            "state": "00",
            "log2_shots": 3,
            "shots": 8,
            "repeat": 1,
            "fidelity": 0.95,
            "infidelity": 0.05,
            "log10_infidelity": np.log10(0.05),
            "nll": 10.0,
            "min_eig": 0.0,
        },
        {
            "state": "00",
            "log2_shots": 3,
            "shots": 8,
            "repeat": 2,
            "fidelity": 0.95,
            "infidelity": 1.1,
            "log10_infidelity": np.log10(1.1),
            "nll": 11.0,
            "min_eig": 0.0,
        },
    ]

    checks = qst_analysis._validate_shot_sweep_run_records_impl(
        run_records=rows,
        suite_states=("00", "01"),
        shot_log2_values=(3, 4),
        repeats_per_point=2,
    )

    assert checks["expected_group_count"] == 4
    assert checks["observed_group_count"] == 1
    assert {"state": "01", "log2_shots": 3} in checks["missing_groups"]
    assert {"state": "00", "log2_shots": 4} in checks["missing_groups"]
    assert checks["bad_repeat_groups"] == []
    assert len(checks["infidelity_range_violations"]) == 1


def test_validate_shot_sweep_run_records_reports_bad_repeat_groups() -> None:
    rows = [
        {
            "state": "00",
            "log2_shots": 3,
            "shots": 8,
            "repeat": 1,
            "fidelity": 0.95,
            "infidelity": 0.05,
            "log10_infidelity": np.log10(0.05),
            "nll": 10.0,
            "min_eig": 0.0,
        }
    ]

    checks = qst_analysis._validate_shot_sweep_run_records_impl(
        run_records=rows,
        suite_states=("00",),
        shot_log2_values=(3,),
        repeats_per_point=2,
    )

    assert checks["missing_groups"] == []
    assert checks["bad_repeat_groups"] == [
        {
            "state": "00",
            "log2_shots": 3,
            "observed_repeats": 1,
            "expected_repeats": 2,
        }
    ]


def test_summarize_final_shot_sweep_selects_largest_log2() -> None:
    final_summary = qst_analysis._summarize_final_shot_sweep_impl(
        aggregated_stats=[
            {
                "state": "00",
                "log2_shots": 3,
                "shots": 8,
                "n_total": 2,
                "n_valid_infidelity": 2,
                "infid_mean": 0.1,
                "infid_ci95": 0.01,
                "log10_infid_mean": -1.0,
                "log10_infid_ci95": 0.1,
            },
            {
                "state": "00",
                "log2_shots": 4,
                "shots": 16,
                "n_total": 2,
                "n_valid_infidelity": 2,
                "infid_mean": 0.05,
                "infid_ci95": 0.005,
                "log10_infid_mean": -1.3,
                "log10_infid_ci95": 0.08,
            },
        ],
        shot_log2_values=(3, 4),
    )

    assert final_summary == [
        {
            "state": "00",
            "n_total": 2,
            "n_valid_infidelity": 2,
            "infid_mean": 0.05,
            "infid_ci95": 0.005,
            "log10_infid_mean": -1.3,
            "log10_infid_ci95": 0.08,
        }
    ]


def test_twoq_qst_source_is_self_contained() -> None:
    experiment_source = (
        Path(__file__).resolve().parents[1]
        / "qubit_experiment"
        / "experiments"
        / "twoq_qst.py"
    ).read_text(encoding="utf-8")
    analysis_source = (
        Path(__file__).resolve().parents[1]
        / "qubit_experiment"
        / "analysis"
        / "twoq_qst.py"
    ).read_text(encoding="utf-8")

    assert "from .two_qubit_qst import" not in experiment_source
    assert "analysis_two_qubit_qst" not in analysis_source
    assert "`experiments.two_qubit_qst`" not in analysis_source


def test_twoq_qst_notebooks_use_canonical_modules() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    notebook_source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    tg_notebook = json.loads(TG_NOTEBOOK.read_text(encoding="utf-8"))
    tg_source = "\n".join(
        "".join(cell.get("source", []))
        for cell in tg_notebook["cells"]
        if cell.get("cell_type") == "code"
    )

    assert "from experiments import twoq_qst,two_qubit_readout_calibration" in notebook_source
    assert "from experiments import two_qubit_state_tomography" not in notebook_source
    assert "twoq_qst.experiment_workflow(" in notebook_source

    assert "from qubit_experiment.experiments import twoq_qst" in tg_source
    assert "from qubit_experiment.experiments import two_qubit_state_tomography" not in tg_source

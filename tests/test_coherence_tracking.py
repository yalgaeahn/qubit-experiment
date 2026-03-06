from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import pytest

matplotlib.use("Agg")

pytest.importorskip("laboneq")
pytest.importorskip("laboneq_applications")
pytest.importorskip("uncertainties")

from laboneq import workflow

from analysis import coherence_tracking as tracking_analysis
from experiments import coherence_tracking as tracking


@dataclass
class _QubitStub:
    uid: str


def _metric_outputs() -> dict[str, dict[str, object]]:
    return {
        "t1": {
            "old_parameter_values": {"q0": {"ge_T1": 10.0e-6}},
            "new_parameter_values": {"q0": {"ge_T1": 11.0e-6}},
        },
        "t2": {
            "old_parameter_values": {"q0": {"ge_T2": 15.0e-6}},
            "new_parameter_values": {"q0": {"ge_T2": 16.0e-6}},
        },
        "t2_star": {
            "old_parameter_values": {
                "q0": {
                    "ge_T2_star": 8.0e-6,
                    "resonance_frequency_ge": 5.0e9,
                }
            },
            "new_parameter_values": {
                "q0": {
                    "ge_T2_star": 9.0e-6,
                    "resonance_frequency_ge": 5.01e9,
                }
            },
        },
    }


def test_workflow_options_default_tracking_fields() -> None:
    options = tracking.experiment_workflow.options()

    assert hasattr(options, "do_plotting")
    assert hasattr(options, "count")
    assert hasattr(options, "use_cal_traces")
    assert hasattr(options, "do_raw_data_plotting")
    assert hasattr(options, "do_qubit_population_plotting")
    assert hasattr(options, "tracked_parameters")
    assert hasattr(options, "refocus_qop")
    assert hasattr(options, "history_path")
    assert hasattr(options, "total_duration_s")
    assert hasattr(options, "interval_s")
    assert hasattr(options, "continue_on_iteration_error")
    assert hasattr(options, "transition")
    assert hasattr(options, "do_analysis")
    assert hasattr(options, "update")


def test_runtime_workflow_options_expose_do_plotting() -> None:
    options = tracking.CoherenceTrackingWorkflowOptions()

    assert hasattr(options, "do_plotting")
    assert options.do_plotting is True


def test_normalize_tracked_parameters_expands_alias_and_deduplicates() -> None:
    out = tracking._normalize_tracked_parameters_value(
        ("coherence", "ge_t1", "RESONANCE_FREQUENCY_GE")
    )

    assert out == ("ge_T1", "ge_T2_star", "ge_T2", "resonance_frequency_ge")


def test_normalize_tracked_parameters_rejects_unknown_key() -> None:
    with pytest.raises(ValueError, match="Unsupported tracked parameter"):
        tracking._normalize_tracked_parameters_value(("ge_T1", "not_a_key"))


def test_resolve_tracking_plan_requires_branch_inputs() -> None:
    with pytest.raises(ValueError, match="t2_star_delays"):
        tracking._resolve_tracking_plan_value(
            ("resonance_frequency_ge",),
            t1_delays=None,
            t2_star_delays=None,
            t2_delays=None,
        )


def test_resolve_iteration_schedule_single_and_long_running_modes() -> None:
    assert tracking._resolve_iteration_schedule_value(None, 300.0) == [
        {"iteration_index": 0, "scheduled_offset_s": 0.0}
    ]
    assert tracking._resolve_iteration_schedule_value(650.0, 300.0) == [
        {"iteration_index": 0, "scheduled_offset_s": 0.0},
        {"iteration_index": 1, "scheduled_offset_s": 300.0},
        {"iteration_index": 2, "scheduled_offset_s": 600.0},
    ]


def test_resolve_iteration_schedule_rejects_non_positive_values() -> None:
    with pytest.raises(ValueError, match="interval_s"):
        tracking._resolve_iteration_schedule_value(None, 0.0)
    with pytest.raises(ValueError, match="total_duration_s"):
        tracking._resolve_iteration_schedule_value(0.0, 300.0)


def test_merge_metric_outputs_preserves_all_parameter_keys() -> None:
    merged = tracking_analysis._merge_metric_outputs(
        _metric_outputs()["t1"],
        _metric_outputs()["t2_star"],
        _metric_outputs()["t2"],
    )

    assert merged["new_parameter_values"]["q0"] == {
        "ge_T1": pytest.approx(11.0e-6),
        "ge_T2_star": pytest.approx(9.0e-6),
        "resonance_frequency_ge": pytest.approx(5.01e9),
        "ge_T2": pytest.approx(16.0e-6),
    }


def test_filter_merged_output_keeps_only_selected_parameters() -> None:
    merged = tracking_analysis._merge_metric_outputs(
        _metric_outputs()["t1"],
        _metric_outputs()["t2_star"],
        _metric_outputs()["t2"],
    )

    filtered = tracking_analysis._filter_merged_output_value(
        merged,
        ("ge_T2_star",),
    )

    assert filtered["tracked_parameters"] == ["ge_T2_star"]
    assert filtered["new_parameter_values"]["q0"] == {
        "ge_T2_star": pytest.approx(9.0e-6)
    }
    assert "resonance_frequency_ge" not in filtered["new_parameter_values"]["q0"]


def test_build_history_rows_emits_selected_parameter_rows() -> None:
    rows = tracking_analysis._build_history_rows(
        _metric_outputs(),
        ("ge_T2_star", "resonance_frequency_ge"),
        timestamp_utc="2026-03-06T00:00:00Z",
    )

    assert [row["parameter_key"] for row in rows] == [
        "ge_T2_star",
        "resonance_frequency_ge",
    ]
    assert rows[0]["resonance_frequency_ge_hz"] == pytest.approx(5.01e9)
    assert rows[1]["value_hz"] == pytest.approx(5.01e9)


def test_load_history_rows_skips_malformed_lines(tmp_path: Path) -> None:
    path = tmp_path / "tracking.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "timestamp_utc": "2026-03-06T00:00:00Z",
                        "parameter_key": "ge_T1",
                    }
                ),
                "{bad json",
                json.dumps(["not", "a", "dict"]),
            ]
        ),
        encoding="utf-8",
    )

    rows = tracking_analysis._load_history_rows(path)

    assert rows == [
        {"timestamp_utc": "2026-03-06T00:00:00Z", "parameter_key": "ge_T1"}
    ]


def test_resolve_history_path_uses_active_folder_store_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    folder_root = tmp_path / "experiment_store"
    folder_root.mkdir()
    fake_store = type("FakeFolderStore", (), {"_folder": folder_root})()
    monkeypatch.setattr(
        tracking_analysis.workflow.logbook,
        "active_logbook_stores",
        lambda: [fake_store],
    )

    out = tracking_analysis._resolve_history_path_value("tracking/coherence_tracking.jsonl")

    assert out == (folder_root / "tracking/coherence_tracking.jsonl").resolve()


def test_materialize_tracking_history_appends_rows(tmp_path: Path) -> None:
    path = tmp_path / "coherence_tracking.jsonl"

    out = tracking_analysis.materialize_tracking_history(
        _metric_outputs(),
        ("ge_T1",),
        path,
        "2026-03-06T00:00:00Z",
    )

    assert out["history_path"] == str(path.resolve())
    assert len(out["history_entries"]) == 1
    assert out["history_entries"][0]["parameter_key"] == "ge_T1"
    assert path.exists()


def test_plot_tracking_history_saves_one_artifact_per_qubit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    saved: list[str] = []
    monkeypatch.setattr(
        tracking_analysis,
        "validate_and_convert_qubits_sweeps",
        lambda qubits: list(qubits),
    )
    monkeypatch.setattr(
        tracking_analysis.workflow,
        "save_artifact",
        lambda name, fig: saved.append(name),
    )

    figures = tracking_analysis.plot_tracking_history(
        qubits=[_QubitStub("q0")],
        history_rows=[
            {
                "timestamp_utc": "2026-03-06T00:00:00Z",
                "qubit_uid": "q0",
                "parameter_key": "ge_T1",
                "value": 10.0e-6,
                "std_dev": None,
                "unit": "s",
            },
            {
                "timestamp_utc": "2026-03-06T00:01:00Z",
                "qubit_uid": "q0",
                "parameter_key": "resonance_frequency_ge",
                "value": 5.01e9,
                "std_dev": None,
                "unit": "Hz",
            },
        ],
        tracked_parameters=("ge_T1", "resonance_frequency_ge"),
    )

    assert "q0" in figures
    assert len(figures["q0"].axes) == 2
    assert saved == ["Coherence_tracking_q0"]


def test_execute_experiment_safe_handles_compile_and_run_failures() -> None:
    class _CompileFailSession:
        def compile(self, experiment):  # noqa: ARG002
            raise RuntimeError("compile boom")

    class _RunFailSession:
        def compile(self, experiment):  # noqa: ARG002
            return "compiled"

        def run(self, compiled):  # noqa: ARG002
            raise RuntimeError("run boom")

    compile_out = tracking._execute_experiment_safe(
        _CompileFailSession(),
        experiment=object(),
        metric_id="t1",
    )
    run_out = tracking._execute_experiment_safe(
        _RunFailSession(),
        experiment=object(),
        metric_id="t2",
    )

    assert compile_out["ok"] is False
    assert compile_out["error_stage"] == "compile"
    assert "compile boom" in str(compile_out["error_message"])
    assert run_out["ok"] is False
    assert run_out["error_stage"] == "run"
    assert "run boom" in str(run_out["error_message"])


def test_materialized_execution_bundle_preserves_child_lookup_after_skipped_branch() -> None:
    @workflow.task(save=False)
    def _default_bundle() -> dict[str, object]:
        return {"result": None}

    @workflow.task(save=False)
    def _selected_bundle() -> dict[str, object]:
        return {"result": "ok"}

    @workflow.task(save=False)
    def _return_value(value: object) -> object:
        return value

    @workflow.workflow(name="materialized_bundle_regression")
    def _bundle_workflow(run_selected: bool) -> None:
        bundle = _default_bundle()
        with workflow.if_(run_selected):
            bundle = _selected_bundle()
        stable_bundle = tracking._materialize_execution_bundle(bundle)
        workflow.return_(_return_value(stable_bundle["result"]))

    assert _bundle_workflow(False).run().output is None
    assert _bundle_workflow(True).run().output == "ok"


def test_materialized_analysis_payload_preserves_workflow_output_after_skipped_branch() -> None:
    @workflow.task(save=False)
    def _empty_payload() -> dict[str, object]:
        return {"result": None}

    @workflow.task(save=False)
    def _return_payload(value: object) -> dict[str, object]:
        return {"result": value}

    @workflow.workflow(name="analysis_payload_child")
    def _child_workflow() -> None:
        workflow.return_(_return_payload("ok"))

    @workflow.task(save=False)
    def _return_value(value: object) -> object:
        return value

    @workflow.workflow(name="analysis_payload_regression")
    def _parent_workflow(run_child: bool) -> None:
        payload = _empty_payload()
        with workflow.if_(run_child):
            child = _child_workflow()
            payload = child.output
        stable_payload = tracking_analysis._materialize_analysis_payload(payload)
        workflow.return_(_return_value(stable_payload["result"]))

    assert _parent_workflow(False).run().output is None
    assert _parent_workflow(True).run().output == "ok"


def test_select_metric_analysis_options_avoids_nested_branch_selection() -> None:
    @workflow.task(save=False)
    def _return_payload(value: object, config: object) -> dict[str, object]:
        return {"result": value, "config": config}

    @workflow.workflow(name="nested_child")
    def _child_workflow(result: object, config: object | None = None) -> None:
        workflow.return_(_return_payload(result, config))

    @workflow.task(save=False)
    def _return_value(value: object) -> object:
        return value

    @workflow.task(save=False)
    def _select_value(flag: bool, when_true: object, when_false: object) -> object:
        return when_true if flag else when_false

    @workflow.workflow(name="nested_parent")
    def _parent_workflow(plotting_enabled: bool) -> None:
        selected_options = tracking_analysis._select_metric_analysis_options(
            plotting_enabled,
            {"do_plotting": False},
        )
        selected_label = _select_value(plotting_enabled, "plot", "no_plot")
        payload = tracking_analysis._materialize_analysis_payload(
            _child_workflow(
                selected_label,
                config=selected_options,
            ).output
        )
        workflow.return_(_return_value(payload["result"]))

    assert _parent_workflow(True).run().output == "plot"
    assert _parent_workflow(False).run().output == "no_plot"


def test_plot_suppressed_analysis_options_returns_concrete_workflow_options() -> None:
    opts = tracking_analysis._plot_suppressed_analysis_options_value(
        tracking_analysis.lifetime_analysis_workflow
    )

    assert hasattr(opts, "_task_options")
    assert not hasattr(opts, "base")
    assert opts.do_plotting is False


def test_analysis_workflow_materializes_parent_result_before_child_analysis(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    @workflow.task(save=False)
    def _build_metric_output(qubit_uid: str) -> dict[str, object]:
        return {
            "old_parameter_values": {qubit_uid: {"ge_T1": 10.0e-6}},
            "new_parameter_values": {qubit_uid: {"ge_T1": 11.0e-6}},
        }

    @workflow.workflow(name="metric_stub")
    def _metric_stub(
        result,  # noqa: ARG001
        qubits,
        delays,  # noqa: ARG001
        options: tracking_analysis.TuneUpAnalysisWorkflowOptions | None = None,  # noqa: ARG001
    ) -> None:
        workflow.return_(_build_metric_output(qubits.uid))

    @workflow.task(save=False)
    def _empty_bundle() -> dict[str, object]:
        return {"result": None}

    @workflow.task(save=False)
    def _selected_bundle() -> dict[str, object]:
        return {"result": "run-result"}

    @workflow.task(save=False)
    def _materialize_bundle(bundle: dict[str, object]) -> dict[str, object]:
        return dict(bundle)

    @workflow.workflow(name="analysis_parent")
    def _parent_workflow(run_child: bool) -> None:
        bundle = _empty_bundle()
        with workflow.if_(run_child):
            bundle = _selected_bundle()
        stable_bundle = _materialize_bundle(bundle)
        stable_result = tracking._extract_execution_result(stable_bundle)
        analysis = tracking_analysis.analysis_workflow(
            qubits=_QubitStub("q0"),
            tracked_parameters=("ge_T1",),
            t1_result=stable_result,
            t1_delays=[0.0, 1.0e-6],
            need_t1=run_child,
            render_metric_plots=False,
            render_history_plot=False,
            history_path=str(tmp_path / "tracking.jsonl"),
        )
        workflow.return_(analysis.output)

    monkeypatch.setattr(tracking_analysis, "lifetime_analysis_workflow", _metric_stub)

    skipped = _parent_workflow(False).run().output
    executed = _parent_workflow(True).run().output

    assert skipped["new_parameter_values"] == {}
    assert executed["new_parameter_values"]["q0"]["ge_T1"] == pytest.approx(11.0e-6)


def test_experiment_workflow_runs_long_running_with_stubbed_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    executed_metrics: list[tuple[str, int]] = []
    analyzed_iterations: list[int] = []

    def _analysis_stub(
        *,
        qubits,
        tracked_parameters,
        history_path,
        timestamp_utc,
        **kwargs,  # noqa: ARG001
    ) -> dict[str, object]:
        iteration_index = int(str(timestamp_utc).rsplit("-", maxsplit=1)[-1])
        analyzed_iterations.append(iteration_index)
        return {
            "metric_outputs": {},
            "old_parameter_values": {qubits.uid: {"ge_T1": 10.0e-6}},
            "new_parameter_values": {
                qubits.uid: {"ge_T1": (11.0 + iteration_index) * 1.0e-6}
            },
            "history_path": str(history_path),
            "history_entries": [{"parameter_key": "ge_T1", "iteration_index": iteration_index}],
            "history_rows": [],
            "tracked_parameters": list(tracked_parameters),
        }

    class _SessionStub:
        def compile(self, experiment):
            return experiment

        def run(self, compiled):
            return {"compiled": compiled}

    def _t1_create(qpu, qubits, delays, options):  # noqa: ARG001
        iteration_index = len(executed_metrics)
        executed_metrics.append(("t1", iteration_index))
        return {"metric": "t1", "qubits": qubits, "delays": delays}

    qubit = _QubitStub("q0")
    monkeypatch.setattr(
        tracking.lifetime_measurement,
        "create_experiment",
        SimpleNamespace(func=_t1_create),
    )
    monkeypatch.setattr(
        tracking.ramsey,
        "create_experiment",
        SimpleNamespace(func=lambda qpu, qubits, delays, detunings, options, echo, refocus_qop: {  # noqa: ARG005
            "metric": "t2" if echo else "t2_star",
            "qubits": qubits,
            "delays": delays,
            "detunings": detunings,
            "refocus_qop": refocus_qop,
        }),
    )
    monkeypatch.setattr(
        tracking,
        "temporary_qpu",
        SimpleNamespace(func=lambda qpu, temporary_parameters: qpu),
    )
    monkeypatch.setattr(
        tracking,
        "temporary_quantum_elements_from_qpu",
        SimpleNamespace(func=lambda qpu, qubits: qubits),
    )
    monkeypatch.setattr(tracking, "run_iteration_analysis_value", _analysis_stub)
    monkeypatch.setattr(
        tracking,
        "_wait_for_iteration_slot_value",
        lambda runtime, iteration_plan: {  # noqa: ARG005
            "run_iteration": True,
            "skip_reason": None,
            "waited_s": 0.0,
            "scheduled_start_utc": f"2026-03-06T00:00:00-{iteration_plan['iteration_index']}",
            "started_utc": f"2026-03-06T00:00:00-{iteration_plan['iteration_index']}",
        },
    )

    options = tracking.experiment_workflow.options()
    options.tracked_parameters(("ge_T1",))
    options.total_duration_s(65.0)
    options.interval_s(30.0)
    options.do_plotting(False)
    options.history_path(str(tmp_path / "tracking.jsonl"))

    result = tracking.experiment_workflow(
        session=_SessionStub(),
        qpu=object(),
        qubits=qubit,
        t1_delays=[0.0, 1.0e-6],
        options=options,
    ).run()

    assert result.output["analysis"]["successful_iterations"] == 3
    assert result.output["analysis"]["failed_iterations"] == 0
    assert len(result.output["analysis"]["iteration_summaries"]) == 3
    assert executed_metrics == [("t1", 0), ("t1", 1), ("t1", 2)]
    assert analyzed_iterations == [0, 1, 2]
    assert result.output["analysis"]["new_parameter_values"]["q0"]["ge_T1"] == pytest.approx(
        13.0e-6
    )


def test_build_final_analysis_output_counts_successes_and_failures() -> None:
    out = tracking._build_final_analysis_output(
        last_successful_analysis={
            "metric_outputs": {"t1": {}},
            "old_parameter_values": {"q0": {"ge_T1": 10.0e-6}},
            "new_parameter_values": {"q0": {"ge_T1": 11.0e-6}},
            "history_path": "/tmp/tracking.jsonl",
            "history_entries": [{"parameter_key": "ge_T1"}],
            "history_rows": [{"parameter_key": "ge_T1"}],
            "tracked_parameters": ["ge_T1"],
        },
        iteration_summaries=[
            {"iteration_index": 0, "status": "completed"},
            {"iteration_index": 1, "status": "failed_execution"},
            {"iteration_index": 2, "status": "skipped_deadline"},
        ],
        history_entries=[{"parameter_key": "ge_T1"}],
        tracked_parameters=("ge_T1",),
        history_path="/tmp/tracking.jsonl",
    )

    assert out["successful_iterations"] == 1
    assert out["failed_iterations"] == 1
    assert out["last_successful_iteration_index"] == 0
    assert out["tracked_parameters"] == ["ge_T1"]
    assert "history_rows" not in out

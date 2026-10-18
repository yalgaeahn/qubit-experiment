"""Hardware-free overnight monitoring regressions using real QPUs and Results."""

from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
import uncertainties as unc
from laboneq import serializers
from laboneq.dsl.result import AcquiredResult, Results
from laboneq.workflow.reference import Reference

from qubit_experiment.analysis import coherence_tracking as analysis
from qubit_experiment.analysis.lifetime_measurement import FitDataT1Options
from qubit_experiment.experiments import coherence_tracking as tracking
from qubit_experiment.qpu_types.fixed_transmon.demo_qpus import demo_platform
from qubit_experiment.workflow import coherence_monitoring as store

DELAYS = np.linspace(0, 120e-6, 121)


def synthetic_result(metric, q, *, cal=True):
    noise = np.random.default_rng(7).normal(0, 0.0002, len(DELAYS))
    if metric == "t1":
        population = np.exp(-DELAYS / 30e-6)
    elif metric == "t2_star":
        population = 0.5 + 0.5 * np.exp(-DELAYS / 40e-6) * np.cos(2 * np.pi * 99e3 * DELAYS)
    else:
        pulse = q.parameters.ge_drive_length_pi or q.parameters.ge_drive_length
        population = 0.5 + 0.5 * np.exp(-(DELAYS + pulse) / 50e-6)
    results = {
        f"{q.uid}/result": AcquiredResult(
            data=(population + noise).astype(complex), axis_name=["delay"], axis=[DELAYS]
        )
    }
    if cal:
        results.update(
            {
                f"{q.uid}/cal_trace/g": AcquiredResult(data=0j, axis_name=[], axis=[]),
                f"{q.uid}/cal_trace/e": AcquiredResult(data=1 + 0j, axis_name=[], axis=[]),
            }
        )
    return Results(acquired_results=results)


@pytest.fixture
def environment(monkeypatch, tmp_path):
    qpu = demo_platform(1).qpu
    clock = SimpleNamespace(now=1000.0)
    monkeypatch.setattr(tracking.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(
        tracking.time, "sleep", lambda seconds: setattr(clock, "now", clock.now + seconds)
    )
    created = []

    def create(qpu, qubits, delays, options, *, echo=False, detunings=None, refocus_qop=None):
        metric = (
            "t1"
            if detunings is None and not echo and refocus_qop is None
            else "t2"
            if echo
            else "t2_star"
        )
        created.append((metric, qubits[0].parameters.resonance_frequency_ge))
        return {"metric": metric, "qubit": qubits[0], "cal": options.use_cal_traces}

    monkeypatch.setattr(
        tracking.lifetime_measurement, "create_experiment", SimpleNamespace(func=create)
    )
    monkeypatch.setattr(tracking.ramsey, "create_experiment", SimpleNamespace(func=create))

    class Session:
        def compile(self, experiment):
            return experiment

        def run(self, experiment):
            return synthetic_result(
                experiment["metric"], experiment["qubit"], cal=experiment["cal"]
            )

    return SimpleNamespace(qpu=qpu, clock=clock, session=Session(), created=created, root=tmp_path)


def run(env, *, duration=61.0, track=True, plotting=False, temporary_parameters=None, **overrides):
    options = tracking.experiment_workflow.options()
    options.tracked_parameters("coherence")
    options.total_duration_s(duration)
    options.interval_s(30.0)
    options.track_frequency(track)
    options.do_plotting(plotting)
    options.history_path(str(env.root / "history.jsonl"))
    options.monitoring_directory(str(env.root / "run"))
    for key, value in overrides.items():
        getattr(options, key)(value)
    return (
        tracking.experiment_workflow(
            env.session,
            env.qpu,
            "q0",
            t1_delays=DELAYS,
            t2_star_delays=DELAYS,
            t2_delays=DELAYS,
            t2_star_detunings=100e3,
            options=options,
            temporary_parameters=temporary_parameters,
        )
        .run()
        .output
    )


def events(env):
    return [json.loads(line) for line in (env.root / "run/events.jsonl").read_text().splitlines()]


def assert_concrete(value):
    assert not isinstance(value, Reference)
    if isinstance(value, dict):
        for child in value.values():
            assert_concrete(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            assert_concrete(child)


def test_overnight_frequency_tracking_and_raw_reanalysis(environment):
    env = environment
    original = deepcopy(env.qpu["q0"].parameters)
    output = run(env)
    assert env.qpu["q0"].parameters == original
    assert output["analysis"]["successful_iterations"] == 3
    for metric in ("t1", "t2_star", "t2"):
        assert output["metric_counts"][metric]["success"] == 3
    for iteration in range(3):
        frequencies = [frequency for _, frequency in env.created[3 * iteration : 3 * iteration + 3]]
        assert frequencies == pytest.approx(
            [original.resonance_frequency_ge + 1000 * iteration] * 3, abs=3
        )
    assert output["final_frequencies"]["q0"] == pytest.approx(
        original.resonance_frequency_ge + 3000, abs=8
    )
    assert all(r["accepted"] for r in output["frequency_history"])
    assert_concrete(output)
    path = env.root / "run/iteration_000000/t1"
    result = serializers.load(path / "raw.json")
    context = json.loads((path / "context.json").read_text())
    assert context["drive_frequencies"]["q0"] == original.resonance_frequency_ge
    repeated = analysis.analyze_monitoring_metric_value(
        "t1",
        result,
        [env.qpu["q0"]],
        DELAYS,
        None,
        tracking.TuneupExperimentOptions(),
        FitDataT1Options(),
    )
    value = repeated["new_parameter_values"]["q0"]["ge_T1"].nominal_value
    assert value == pytest.approx(30e-6, rel=0.002)
    saved = json.loads((path / "analysis.json").read_text())
    assert saved["new_parameter_values"]["q0"]["ge_T1"]["value"] == value
    assert "residual" in saved["diagnostics"]["q0"]
    status = json.loads((env.root / "run/status.json").read_text())
    assert status["status"] == "completed"
    log = events(env)
    assert next(i for i, e in enumerate(log) if e["event"] == "raw_saved") < next(
        i for i, e in enumerate(log) if e["event"] == "metric_finished"
    )


@pytest.mark.parametrize("metric", ["t1", "t2_star", "t2"])
@pytest.mark.parametrize("stage", ["create_experiment", "compile", "run", "analysis"])
def test_metric_failures_do_not_stop_other_metrics_or_later_iterations(
    environment, monkeypatch, metric, stage
):
    env = environment
    if stage == "create_experiment":
        module = tracking.lifetime_measurement if metric == "t1" else tracking.ramsey
        original = module.create_experiment.func

        def broken(*args, **kwargs):
            which = (
                "t1"
                if module is tracking.lifetime_measurement
                else "t2"
                if kwargs["echo"]
                else "t2_star"
            )
            if which == metric:
                raise ValueError("injected create failure")
            return original(*args, **kwargs)

        monkeypatch.setattr(module, "create_experiment", SimpleNamespace(func=broken))
    elif stage == "analysis":
        original = tracking.analyze_monitoring_metric_value

        def broken(which, *args, **kwargs):
            if which == metric:
                raise ValueError("injected analysis failure")
            return original(which, *args, **kwargs)

        monkeypatch.setattr(tracking, "analyze_monitoring_metric_value", broken)
    else:
        original = getattr(env.session, stage)

        def broken(experiment):
            if experiment["metric"] == metric:
                raise ValueError("injected execution failure")
            return original(experiment)

        monkeypatch.setattr(env.session, stage, broken)
    output = run(env)
    assert output["metric_counts"][metric]["failure"] == 3
    for other in {"t1", "t2_star", "t2"} - {metric}:
        assert output["metric_counts"][other]["success"] == 3
    assert len(output["iteration_summaries"]) == 3
    errors = [e for e in events(env) if e["event"] == "metric_finished" and e["metric"] == metric]
    assert all(e["error_stage"] == stage for e in errors)
    if stage == "analysis":
        assert (env.root / f"run/iteration_000000/{metric}/raw.json").exists()


@pytest.mark.parametrize("exception", [KeyboardInterrupt, SystemExit])
def test_operator_interrupt_is_persisted_and_propagated(environment, monkeypatch, exception):
    def stop(_experiment):
        raise exception()

    monkeypatch.setattr(environment.session, "run", stop)
    with pytest.raises(exception):
        run(environment)
    assert len(environment.created) == 1
    assert json.loads((environment.root / "run/status.json").read_text())["status"] == "interrupted"
    assert events(environment)[-1]["event"] == "interrupted"


def test_raw_write_failure_is_fatal(environment, monkeypatch):
    original = store.atomic_write

    def broken(path, payload, **kwargs):
        if kwargs.get("laboneq"):
            raise store.MonitoringStorageError("disk full")
        return original(path, payload, **kwargs)

    monkeypatch.setattr(store, "atomic_write", broken)
    with pytest.raises(store.MonitoringStorageError):
        run(environment)
    assert len(environment.created) == 1
    assert (
        json.loads((environment.root / "run/status.json").read_text())["status"] == "storage_failed"
    )


def test_plot_failure_is_nonfatal(environment, monkeypatch):
    monkeypatch.setattr(
        store, "render_trends", lambda *args: (_ for _ in ()).throw(ValueError("plot error"))
    )
    output = run(environment, plotting=True)
    assert output["analysis"]["successful_iterations"] == 3
    assert any(e["event"] == "plot_failed" for e in events(environment))


def test_deadline_between_metrics_prevents_new_measurements(environment, monkeypatch):
    env = environment
    original = env.session.run

    def slow(experiment):
        result = original(experiment)
        env.clock.now += 15
        return result

    monkeypatch.setattr(env.session, "run", slow)
    output = run(env, duration=10.0)
    assert [metric for metric, _ in env.created] == ["t1"]
    assert (
        output["iteration_summaries"][0]["per_metric_status"]["t2"]["status"] == "skipped_deadline"
    )


def test_overdue_slots_are_skipped(environment, monkeypatch):
    env = environment
    original = env.session.run

    def slow(experiment):
        env.clock.now += 15
        return original(experiment)

    monkeypatch.setattr(env.session, "run", slow)
    output = run(env, duration=121.0)
    assert [r["status"] for r in output["iteration_summaries"]] == [
        "completed",
        "skipped_missed_slot",
        "completed",
        "skipped_missed_slot",
        "partial_success",
    ]
    assert len(env.created) == 7  # only T1 starts at the final slot, before deadline


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"update": True}, "update=False"),
        ({"do_analysis": False}, "Ramsey analysis"),
    ],
)
def test_invalid_tracking_settings_fail_before_hardware(environment, changes, message):
    with pytest.raises(ValueError, match=message):
        run(environment, **changes)
    assert not environment.created


def test_analysis_respects_no_calibration_and_pca(environment):
    q = environment.qpu["q0"]
    result = synthetic_result("t1", q, cal=False)
    payload = analysis.analyze_monitoring_metric_value(
        "t1",
        result,
        [q],
        DELAYS,
        None,
        tracking.TuneupExperimentOptions(use_cal_traces=False),
        FitDataT1Options(),
    )
    assert payload["diagnostics"]["q0"]["processed_data"]["num_cal_traces"] == 0
    assert payload["new_parameter_values"]["q0"]["ge_T1"].nominal_value == pytest.approx(
        30e-6, rel=0.01
    )


@pytest.mark.parametrize(
    "stderr,rate,reason",
    [
        (None, 1 / 30e-6, "uncertainty_missing"),
        (1e5, 1 / 30e-6, "large_relative_uncertainty"),
        (1.0, 0.0, "invalid_coherence_time"),
        (1.0, -2.0, "invalid_coherence_time"),
        (1.0, np.nan, "invalid_coherence_time"),
    ],
)
def test_fit_quality_preserves_estimate_without_crashing(
    environment, monkeypatch, stderr, rate, reason
):
    q = environment.qpu["q0"]
    fit = SimpleNamespace(
        success=True,
        message="fit",
        params={"decay_rate": SimpleNamespace(value=rate, stderr=stderr, vary=True)},
        residual=np.zeros(len(DELAYS)),
        best_fit=np.ones(len(DELAYS)),
        chisqr=0.0,
        redchi=0.0,
    )
    monkeypatch.setattr(
        analysis.lifetime_analysis_module,
        "fit_data",
        SimpleNamespace(func=lambda *a, **k: {"q0": fit}),
    )
    payload = analysis.analyze_monitoring_metric_value(
        "t1",
        synthetic_result("t1", q),
        [q],
        DELAYS,
        None,
        tracking.TuneupExperimentOptions(),
        FitDataT1Options(),
    )
    assert reason in payload["quality"]["q0"]["reasons"]
    if rate > 0:
        assert payload["new_parameter_values"]["q0"]["ge_T1"].nominal_value == pytest.approx(30e-6)
    else:
        assert not payload["new_parameter_values"]["q0"]


@pytest.mark.parametrize(
    "estimate,error,reason",
    [
        (1000, 10, "accepted"),
        (60000, 10, "frequency_step_exceeded"),
        (1000, 30000, "frequency_uncertainty_exceeded"),
        (1000, np.nan, "frequency_uncertainty_missing"),
    ],
)
def test_frequency_guards(environment, estimate, error, reason):
    qpu = environment.qpu
    initial = qpu["q0"].parameters.resonance_frequency_ge
    state = tracking._initialize_tracking_state.func()
    runtime_qpu = tracking.temporary_qpu.func(qpu, None)
    state.update(
        track_frequency=True,
        initial_frequencies={"q0": initial},
        frequencies={"q0": initial},
        detunings={"q0": 100e3},
        runtime_qpu=runtime_qpu,
    )
    tracking._frequency_updates(
        state,
        {
            "t2_star": {
                "new_parameter_values": {
                    "q0": {"resonance_frequency_ge": unc.ufloat(initial + estimate, error)}
                },
                "quality": {"q0": {"fit_success": True}},
            }
        },
        0,
    )
    assert state["frequency_history"][0]["reason"] == reason
    assert qpu["q0"].parameters.resonance_frequency_ge == initial
    assert runtime_qpu["q0"].parameters.resonance_frequency_ge == initial + (
        estimate if reason == "accepted" else 0
    )


def test_twelve_hour_schedule_with_virtual_clock(environment, monkeypatch):
    env = environment
    # Exercise all 144 slots cheaply; real synthetic fitting is covered separately.
    original_run = env.session.run

    def timed_run(experiment):
        result = original_run(experiment)
        env.clock.now += 0.1
        return result

    monkeypatch.setattr(env.session, "run", timed_run)

    def quick_analysis(metric, result, qubits, *args):
        key = {"t1": "ge_T1", "t2_star": "ge_T2_star", "t2": "ge_T2"}[metric]
        return {
            "old_parameter_values": {"q0": {key: 1e-5}},
            "new_parameter_values": {"q0": {key: unc.ufloat(2e-5, 1e-7)}},
            "quality": {"q0": {"fit_success": True, "valid": True, "failed": False, "reasons": []}},
            "diagnostics": {},
        }

    monkeypatch.setattr(tracking, "analyze_monitoring_metric_value", quick_analysis)
    output = run(env, duration=12 * 3600, track=False, interval_s=300.0)
    assert output["analysis"]["successful_iterations"] == 144
    assert len(env.created) == 432
    assert env.clock.now == pytest.approx(1000 + 143 * 300 + 0.3)
    assert output["iteration_summaries"][-1]["scheduled_offset_s"] == 42900.0


def test_failed_history_write_stops_and_preserves_raw(environment, monkeypatch):
    original = store.append_records

    def fail_history(path, rows):
        if str(path).endswith("history.jsonl"):
            raise store.MonitoringStorageError("history disk error")
        return original(path, rows)

    monkeypatch.setattr(store, "append_records", fail_history)
    with pytest.raises(store.MonitoringStorageError):
        run(environment)
    assert len(environment.created) == 1
    assert (environment.root / "run/iteration_000000/t1/raw.json").exists()
    assert events(environment)[-1]["event"] == "storage_failed"


def test_later_interrupt_preserves_completed_results(environment, monkeypatch):
    original = environment.session.run
    calls = []

    def stop_later(experiment):
        calls.append(experiment)
        if len(calls) == 4:
            raise KeyboardInterrupt()
        return original(experiment)

    monkeypatch.setattr(environment.session, "run", stop_later)
    with pytest.raises(KeyboardInterrupt):
        run(environment)
    rows = analysis._load_history_rows(environment.root / "history.jsonl")
    assert len(rows) == 4  # T1, T2*, echo, frequency
    assert all(row["iteration_index"] == 0 for row in rows)
    raw = serializers.load(environment.root / "run/iteration_000000/t1/raw.json")
    np.testing.assert_allclose(
        raw["q0/result"].data, synthetic_result("t1", environment.qpu["q0"])["q0/result"].data
    )
    assert not any(e["event"] == "run_finished" for e in events(environment))


def test_no_analysis_preserves_raw_and_never_updates_frequency(environment):
    output = run(environment, track=False, do_analysis=False)
    assert output["analysis"] is None
    assert not output["frequency_history"]
    assert (environment.root / "run/iteration_000000/t1/raw.json").exists()
    assert not (environment.root / "run/iteration_000000/t1/analysis.json").exists()


def test_stop_after_failure_option_is_preserved(environment, monkeypatch):
    monkeypatch.setattr(
        environment.session, "run", lambda exp: (_ for _ in ()).throw(ValueError("run failure"))
    )
    output = run(environment, continue_on_iteration_error=False)
    assert len(environment.created) == 3
    assert [r["status"] for r in output["iteration_summaries"]] == [
        "failed_execution",
        "skipped_after_failure",
        "skipped_after_failure",
    ]
    assert (
        json.loads((environment.root / "run/status.json").read_text())["status"]
        == "stopped_after_failure"
    )


def test_local_trend_files_are_rendered(environment):
    output = run(environment, duration=1.0, plotting=True)
    path = environment.root / "run/trend_q0.png"
    assert path.read_bytes().startswith(b"\x89PNG")
    assert output["analysis"]["successful_iterations"] == 1


def test_negative_detuning_uses_nearby_frequency_branch(environment):
    q = environment.qpu["q0"]
    output = analysis.analyze_monitoring_metric_value(
        "t2_star",
        synthetic_result("t2_star", q),
        q,
        DELAYS,
        -100e3,
        tracking.TuneupExperimentOptions(),
        FitDataT1Options(),
    )
    assert output["new_parameter_values"]["q0"][
        "resonance_frequency_ge"
    ].nominal_value == pytest.approx(q.parameters.resonance_frequency_ge - 1000, abs=3)


def test_covariance_missing_is_not_reported_as_zero_error(environment, monkeypatch):
    original = analysis.lifetime_analysis_module.fit_data.func

    def no_covariance(*args, **kwargs):
        fits = original(*args, **kwargs)
        fits["q0"].covar = None
        fits["q0"].params["decay_rate"].stderr = 0.0
        return fits

    monkeypatch.setattr(
        analysis.lifetime_analysis_module, "fit_data", SimpleNamespace(func=no_covariance)
    )
    q = environment.qpu["q0"]
    output = analysis.analyze_monitoring_metric_value(
        "t1",
        synthetic_result("t1", q),
        q,
        DELAYS,
        None,
        tracking.TuneupExperimentOptions(),
        FitDataT1Options(),
    )
    assert "uncertainty_missing" in output["quality"]["q0"]["reasons"]
    assert np.isnan(output["new_parameter_values"]["q0"]["ge_T1"].std_dev)


def test_qubit_analysis_failure_is_isolated():
    qpu = demo_platform(2).qpu
    result = synthetic_result("t1", qpu["q0"])
    output = analysis.analyze_monitoring_metric_value(
        "t1",
        result,
        qpu.quantum_elements,
        [DELAYS, DELAYS],
        None,
        tracking.TuneupExperimentOptions(),
        FitDataT1Options(),
    )
    assert output["new_parameter_values"]["q0"]["ge_T1"].nominal_value > 0
    assert output["quality"]["q1"]["failed"]


def test_real_experiments_compile_with_run_local_frequency(tmp_path):
    from laboneq.dsl.session import Session

    from qubit_experiment.experiments import lifetime_measurement, ramsey

    platform = demo_platform(1)
    session = Session(device_setup=platform.setup, configure_logging=False)
    runtime = tracking.temporary_qpu.func(
        platform.qpu, {"q0": {"resonance_frequency_ge": 6.500001e9}}
    )
    q = runtime["q0"]
    for metric in ("t1", "t2_star", "t2"):
        opts = tracking.TuneupExperimentOptions(count=2)
        if metric == "t1":
            experiment = lifetime_measurement.create_experiment.func(
                runtime, q, [0.0, 32e-9, 64e-9], options=opts
            )
        else:
            experiment = ramsey.create_experiment.func(
                runtime,
                q,
                [0.0, 32e-9, 64e-9],
                detunings=100e3 if metric == "t2_star" else None,
                echo=metric == "t2",
                options=opts,
            )
        assert session.compile(experiment=experiment) is not None
    assert platform.qpu["q0"].parameters.resonance_frequency_ge == 6.5e9


def test_compile_finishing_after_deadline_does_not_start_acquisition(environment, monkeypatch):
    original = environment.session.compile

    def slow_compile(experiment):
        environment.clock.now += 20
        return original(experiment)

    monkeypatch.setattr(environment.session, "compile", slow_compile)
    monkeypatch.setattr(
        environment.session, "run", lambda exp: pytest.fail("run started after deadline")
    )
    output = run(environment, duration=10.0)
    assert len(environment.created) == 1
    assert output["iteration_summaries"][0]["status"] == "skipped_deadline"
    assert output["metric_counts"]["t1"]["failure"] == 0


def test_workflow_no_calibration_options_are_forwarded(environment):
    output = run(environment, duration=1.0, track=False, use_cal_traces=False)
    assert output["analysis"]["successful_iterations"] == 1
    path = environment.root / "run/iteration_000000/t1/analysis.json"
    payload = json.loads(path.read_text())
    assert payload["diagnostics"]["q0"]["processed_data"]["num_cal_traces"] == 0


def test_monitoring_does_not_reread_history_each_iteration(environment, monkeypatch):
    monkeypatch.setattr(analysis, "_load_history_rows", lambda path: pytest.fail("history reread"))
    output = run(environment, plotting=False)
    assert output["analysis"]["successful_iterations"] == 3


def test_nonfinite_detuning_is_rejected_before_hardware(environment):
    options = tracking.experiment_workflow.options()
    options.track_frequency(True)
    options.tracked_parameters("ge_T2_star")
    for detuning in (0.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="nonzero detunings"):
            tracking.experiment_workflow(
                environment.session,
                environment.qpu,
                "q0",
                t2_star_delays=DELAYS,
                t2_star_detunings=detuning,
                options=options,
            ).run()
    assert not environment.created


def test_temporary_parameters_only_affect_runtime_qpu(environment):
    initial = environment.qpu["q0"].parameters.resonance_frequency_ge
    override = initial + 200e3
    output = run(
        environment,
        duration=1.0,
        track=False,
        temporary_parameters={"q0": {"resonance_frequency_ge": override}},
    )
    assert environment.qpu["q0"].parameters.resonance_frequency_ge == initial
    assert all(frequency == override for _, frequency in environment.created)
    manifest = json.loads((environment.root / "run/manifest.json").read_text())
    assert manifest["input_qpu"]["q0"]["parameters"]["resonance_frequency_ge"] == initial
    assert manifest["runtime_qpu"]["q0"]["parameters"]["resonance_frequency_ge"] == override
    assert output["final_frequencies"]["q0"] == override


def test_core_event_write_failure_is_fatal(environment, monkeypatch):
    original = store.append_records

    def fail_event(path, rows):
        if any(row.get("event") == "raw_saved" for row in rows):
            raise store.MonitoringStorageError("event storage failure")
        return original(path, rows)

    monkeypatch.setattr(store, "append_records", fail_event)
    with pytest.raises(store.MonitoringStorageError):
        run(environment)
    assert len(environment.created) == 1
    assert serializers.load(environment.root / "run/iteration_000000/t1/raw.json") is not None
    assert (
        json.loads((environment.root / "run/status.json").read_text())["status"] == "storage_failed"
    )

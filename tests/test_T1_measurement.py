"""Single-shot T1 contracts, statistics, plotting, and hardware-free workflows."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from laboneq import serializers
from laboneq.dsl.result import AcquiredResult, Results
from laboneq.dsl.session import Session
from laboneq.simple import AcquisitionType, AveragingMode
from laboneq.workflow.reference import Reference
from scipy.stats import t

from qubit_experiment.analysis import T1_measurement as analysis
from qubit_experiment.analysis import lifetime_measurement as legacy
from qubit_experiment.experiments import T1_measurement as experiment
from qubit_experiment.qpu_types.fixed_transmon.demo_qpus import demo_platform

DELAYS = np.linspace(0, 120e-6, 41)


def synthetic_result(qubits, delays=DELAYS, count=64, *, states="ge", transpose=False):
    rng = np.random.default_rng(17)
    grids = [delays] * len(qubits) if isinstance(delays, np.ndarray) else delays
    shot_grid = np.arange(count, dtype=float)
    acquired = {}
    for q, grid in zip(qubits, grids):
        separation = 1.2 + 0.6j
        low = 0.2 + 0.1j
        noise = rng.normal(0, 0.04, (count, len(grid)))
        population = np.exp(-grid / 30e-6)[None, :] + noise
        signal = (
            low + separation * population + 1j * separation * rng.normal(0, 0.03, population.shape)
        )
        names = ["unnamed_0", q.uid]
        axes = [shot_grid, grid]
        if transpose:
            signal = signal.T
            names, axes = names[::-1], axes[::-1]
        acquired[f"{q.uid}/result"] = AcquiredResult(data=signal, axis_name=names, axis=axes)
        for index, state in enumerate(states):
            samples = rng.normal(0, 0.05 * (index + 1), count) + 1j * rng.normal(0, 0.02, count)
            samples -= samples.mean()
            reference = low + separation * (index + samples)
            acquired[f"{q.uid}/cal_trace/{state}"] = AcquiredResult(
                data=reference, axis_name=["unnamed_0"], axis=[shot_grid]
            )
    return Results(acquired_results=acquired)


@pytest.fixture
def qpu():
    return demo_platform(2).qpu


def analysis_run(qpu, *, options=None, result=None, qubits=None):
    qubits = qpu.quantum_elements if qubits is None else qubits
    if options is None:
        options = analysis.analysis_workflow.options()
        options.logstore([])
        options.do_plotting(False)
    return analysis.analysis_workflow(
        result if result is not None else synthetic_result(qubits),
        qubits,
        [DELAYS] * len(qubits),
        options=options,
    ).run()


def assert_concrete(value):
    assert not isinstance(value, Reference)
    if isinstance(value, dict):
        for child in value.values():
            assert_concrete(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            assert_concrete(child)


@pytest.mark.parametrize("transpose", [False, True])
def test_projection_statistics_and_square_axis(qpu, transpose):
    q = qpu["q0"]
    grid = np.array([0, 10e-6, 20e-6, 30e-6])
    result = synthetic_result([q], grid, 4, transpose=transpose)
    out = analysis.calculate_qubit_population.func(q, result, grid)[q.uid]
    signal = result["q0/result"].data
    if transpose:
        signal = signal.T
    g, e = [result[f"q0/cal_trace/{state}"].data for state in "ge"]
    difference = e.mean() - g.mean()

    def project(x):
        return np.real((x - g.mean()) * difference.conjugate()) / abs(difference) ** 2

    expected = project(signal)
    np.testing.assert_allclose(out["population"], expected.mean(axis=0))
    np.testing.assert_allclose(out["population_std"], expected.std(axis=0, ddof=1))
    np.testing.assert_allclose(out["population_sem"], expected.std(axis=0, ddof=1) / 2)
    np.testing.assert_allclose(out["population_cal_traces"], [0, 1], atol=1e-14)
    np.testing.assert_allclose(
        out["population_cal_traces_std"], [project(g).std(ddof=1), project(e).std(ddof=1)]
    )
    for component, values in (("i", signal.real), ("q", signal.imag)):
        np.testing.assert_allclose(out[f"raw_{component}"], values.mean(axis=0))
        np.testing.assert_allclose(out[f"raw_{component}_std"], values.std(axis=0, ddof=1))
        np.testing.assert_allclose(out[f"raw_{component}_sem"], values.std(axis=0, ddof=1) / 2)
        refs = np.array([g, e]).real if component == "i" else np.array([g, e]).imag
        np.testing.assert_allclose(out[f"reference_{component}_std"], refs.std(axis=1, ddof=1))
    assert out["num_shots"] == 4


def test_population_is_not_clipped(qpu):
    q = qpu["q0"]
    grid = np.array([0.0, 1e-6])
    result = synthetic_result([q], grid, 4)
    result["q0/result"].data[:] = np.array([-1.0, 2.0])[None, :] * (1.2 + 0.6j) + (0.2 + 0.1j)
    out = analysis.calculate_qubit_population.func(q, result, grid)[q.uid]
    np.testing.assert_allclose(out["population"], [-1, 2])


def test_grouped_parallel_axes_with_distinct_qubit_grids(qpu):
    grids = [DELAYS, DELAYS * 2]
    qubits = qpu.quantum_elements
    result = synthetic_result(qubits, grids)
    for q in qubits:
        result[f"{q.uid}/result"].axis_name[1] = [other.uid for other in qubits]
        result[f"{q.uid}/result"].axis[1] = grids
    out = analysis.calculate_qubit_population.func(qubits, result, grids)
    for q, grid in zip(qubits, grids):
        np.testing.assert_array_equal(out[q.uid]["sweep_points"], grid)


@pytest.mark.parametrize(
    "problem",
    [
        "averaged",
        "missing_axes",
        "wrong_delay_name",
        "wrong_delay_grid",
        "wrong_axis_length",
        "missing_reference",
        "scalar_reference",
        "wrong_shot_grid",
        "wrong_shot_name",
        "zero_separation",
        "nonfinite_signal",
        "nonfinite_reference",
        "one_shot",
    ],
)
def test_rejects_unusable_results(qpu, problem):
    q = qpu["q0"]
    result = synthetic_result([q], count=1 if problem == "one_shot" else 64)
    signal = result["q0/result"]
    reference = result["q0/cal_trace/g"]
    if problem == "averaged":
        signal.data = signal.data.mean(axis=0)
    elif problem == "missing_axes":
        signal.axis_name = []
    elif problem == "wrong_delay_name":
        signal.axis_name[1] = "unknown"
    elif problem == "wrong_delay_grid":
        signal.axis[1] = DELAYS + 1e-6
    elif problem == "wrong_axis_length":
        signal.axis[0] = np.arange(63)
    elif problem == "missing_reference":
        del result.acquired_results["q0/cal_trace/g"]
    elif problem == "scalar_reference":
        reference.data = 0j
    elif problem == "wrong_shot_grid":
        reference.axis = [np.arange(64) + 1]
    elif problem == "wrong_shot_name":
        reference.axis_name = ["other"]
    elif problem == "zero_separation":
        reference.data[:] = 0
        result["q0/cal_trace/e"].data[:] = 0
    elif problem == "nonfinite_signal":
        signal.data[0, 0] = np.nan
    elif problem == "nonfinite_reference":
        reference.data[0] = np.inf
    with pytest.raises(ValueError):
        analysis.calculate_qubit_population.func(q, result, DELAYS)


def test_unweighted_fit_matches_lifetime_and_ignores_sem(qpu):
    qubits = qpu.quantum_elements
    processed = analysis.calculate_qubit_population.func(
        qubits, synthetic_result(qubits), [DELAYS] * 2
    )
    original = legacy.fit_data.func(qubits, processed)
    current = analysis.fit_data.func(qubits, processed)
    modified = deepcopy(processed)
    for data in modified.values():
        data["population_sem"] = np.linspace(1e-9, 100, len(DELAYS))
    other = analysis.fit_data.func(qubits, modified)
    for q in qubits:
        assert current[q.uid].best_values == pytest.approx(original[q.uid].best_values)
        assert other[q.uid].best_values == pytest.approx(current[q.uid].best_values)
        assert current[q.uid].params["offset"].value == 0
        assert not current[q.uid].params["offset"].vary


def test_fit_hints_allow_free_offset(qpu):
    q = qpu["q0"]
    processed = analysis.calculate_qubit_population.func(q, synthetic_result([q]), DELAYS)
    processed[q.uid]["population"] += 0.15
    options = analysis.T1AnalysisOptions(
        fit_parameters_hints={"offset": {"value": 0.1, "vary": True}}
    )
    fit = analysis.fit_data.func(q, processed, options=options)[q.uid]
    assert fit.params["offset"].vary
    assert fit.params["offset"].value == pytest.approx(0.15, abs=0.01)


@pytest.mark.parametrize(
    "problem", ["failure", "negative", "zero", "infinite", "missing_error", "missing_covar"]
)
def test_invalid_fit_or_unknown_uncertainty_never_updates(qpu, problem):
    q = qpu["q0"]
    fit = SimpleNamespace(
        success=problem != "failure",
        covar=None if problem == "missing_covar" else np.eye(2),
        params={
            "decay_rate": SimpleNamespace(
                value={"negative": -1.0, "zero": 0.0, "infinite": np.inf}.get(problem, 1 / 30e-6),
                stderr=None if problem == "missing_error" else 100.0,
            )
        },
    )
    out = analysis.extract_qubit_parameters.func(q, {q.uid: fit})
    assert out["new_parameter_values"][q.uid] == {}


def test_fit_restores_unknown_error_from_upstream_zero(qpu, monkeypatch):
    fit = SimpleNamespace(
        covar=None,
        params={
            "decay_rate": SimpleNamespace(vary=True, expr=None, stderr=0),
            "offset": SimpleNamespace(vary=False, expr=None, stderr=0),
        },
    )
    monkeypatch.setattr(
        analysis, "lifetime_fit_data", SimpleNamespace(func=lambda *a, **k: {"q0": fit})
    )
    out = analysis.fit_data.func(qpu["q0"], {})
    assert out["q0"].params["decay_rate"].stderr is None


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"count": 1}, "count"),
        ({"count": True}, "count"),
        ({"averaging_mode": "cyclic"}, "SINGLE_SHOT"),
        ({"averaging_mode": "sequential"}, "SINGLE_SHOT"),
        ({"acquisition_type": "discrimination"}, "INTEGRATION"),
        ({"acquisition_type": "spectroscopy"}, "INTEGRATION"),
        ({"use_cal_traces": False}, "use_cal_traces"),
        ({"do_pca": True}, "do_pca"),
        ({"do_rotation": False}, "do_rotation"),
        ({"transition": "ef"}, "cal_states"),
        ({"cal_states": "eg"}, "cal_states"),
        ({"transition": "gf", "cal_states": "gf"}, "transition"),
    ],
)
def test_invalid_experiment_options(qpu, overrides, match):
    with pytest.raises(ValueError, match=match):
        experiment.create_experiment.func(
            qpu, qpu["q0"], DELAYS, options=experiment.T1ExperimentOptions(**overrides)
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"do_pca": True},
        {"do_rotation": False},
        {"use_cal_traces": False},
        {"transition": "ef"},
        {"cal_states": "gf"},
    ],
)
def test_invalid_analysis_options(qpu, overrides):
    q = qpu["q0"]
    with pytest.raises(ValueError):
        analysis.calculate_qubit_population.func(
            q, synthetic_result([q]), DELAYS, options=analysis.T1AnalysisOptions(**overrides)
        )


@pytest.mark.parametrize(
    "grid", [np.array([-1e-6, 0]), np.array([0.0, np.nan]), np.array([[0.0, 1e-6]])]
)
def test_invalid_delay_grid(qpu, grid):
    with pytest.raises(ValueError):
        experiment.create_experiment.func(qpu, qpu["q0"], grid)


@pytest.mark.parametrize("transition", ["ge", "ef"])
def test_analysis_workflow_outputs_and_transition(qpu, transition):
    qubits = qpu.quantum_elements
    options = analysis.analysis_workflow.options()
    options.logstore([])
    options.do_plotting(False)
    options.transition(transition)
    options.cal_states(transition)
    run = analysis_run(qpu, options=options, result=synthetic_result(qubits, states=transition))
    out = run.output
    assert_concrete(out)
    assert set(out) == {"old_parameter_values", "new_parameter_values", "processed_data"}
    for q in qubits:
        estimate = out["new_parameter_values"][q.uid][f"{transition}_T1"]
        assert estimate.nominal_value == pytest.approx(30e-6, rel=0.02)
        assert estimate.std_dev > 0


@pytest.mark.parametrize(
    "raw,population,fitting",
    [
        (True, True, True),
        (False, True, True),
        (True, False, True),
        (True, True, False),
    ],
)
def test_plot_workflow_branches_and_layout(qpu, raw, population, fitting, monkeypatch):
    timestamp = "2026-10-18 12:34:56"
    monkeypatch.setattr(analysis, "timestamped_title", lambda title: f"{title} ({timestamp})")
    options = analysis.analysis_workflow.options()
    options.logstore([])
    options.save_figures(False)
    options.close_figures(False)
    options.do_raw_data_plotting(raw)
    options.do_qubit_population_plotting(population)
    options.do_fitting(fitting)
    run = analysis_run(qpu, options=options)
    assert_concrete(run.output)
    if population:
        figures = run.tasks["plot_population"].output
        for uid, figure in figures.items():
            ax = figure.axes[0]
            assert len(figure.axes) == 1
            assert len(ax.containers) == 0  # point means; uncertainty uses fit bands
            points = next(line for line in ax.lines if line.get_label() == "data")
            np.testing.assert_allclose(
                points.get_ydata(),
                run.output["processed_data"][uid]["population"],
            )
            labels = ax.get_legend_handles_labels()[1]
            assert "data" in labels
            assert labels.count("g/e reference ± SEM") == 1
            assert timestamp in ax.get_title(loc="left")
            assert ("fit" in labels) == fitting
            assert ("95% confidence band" in labels) == fitting
            assert ("95% prediction band" in labels) == fitting
            assert len(ax.patches) == 2  # reference SEM spans
            assert len(ax.texts) == int(fitting)
            assert ax.get_legend().get_bbox_to_anchor().x0 >= ax.get_window_extent().x1
            figure.canvas.draw()
            plt.close(figure)
    if raw:
        for figure in run.tasks["plot_raw_complex_data_1d"].output.values():
            assert len(figure.axes) == 2
            assert all(len(ax.containers) == 1 and len(ax.patches) == 2 for ax in figure.axes)
            figure.canvas.draw()
            plt.close(figure)
    if not fitting:
        assert all(not entry for entry in run.output["new_parameter_values"].values())


def test_reference_sem_bands_and_raw_sem_errorbar_coordinates(qpu):
    run = analysis_run(qpu)
    q = qpu["q0"]
    data = run.output["processed_data"][q.uid]
    figures = analysis.plot_population.func(
        q,
        run.output["processed_data"],
        {},
        run.output,
        options=analysis.T1PlotOptions(do_fitting=False, save_figures=False),
    )
    ax = figures[q.uid].axes[0]
    for patch, mean, sem in zip(
        ax.patches, data["population_cal_traces"], data["population_cal_traces_sem"]
    ):
        assert patch.get_y() == pytest.approx(mean - sem)
        assert patch.get_height() == pytest.approx(2 * sem)
    plt.close(figures[q.uid])

    raw_figure = analysis.plot_raw_complex_data_1d.func(
        q,
        run.output["processed_data"],
        options=analysis.T1PlotOptions(save_figures=False),
    )[q.uid]
    for ax, component in zip(raw_figure.axes, ("i", "q")):
        segments = ax.containers[0].lines[2][0].get_segments()
        for segment, mean, sem in zip(
            segments, data[f"raw_{component}"], data[f"raw_{component}_sem"]
        ):
            np.testing.assert_allclose(segment[:, 1], [mean - sem, mean + sem])
        assert ax.get_legend_handles_labels()[1].count("g/e reference ± SEM") == 1
        for patch, mean, sem in zip(
            ax.patches, data[f"reference_{component}"], data[f"reference_{component}_sem"]
        ):
            assert patch.get_y() == pytest.approx(mean - sem)
            assert patch.get_height() == pytest.approx(2 * sem)
    plt.close(raw_figure)


def test_fit_bands_match_analytic_covariance_and_mean_residual_variance(qpu):
    q = qpu["q0"]
    processed = analysis.calculate_qubit_population.func(q, synthetic_result([q]), DELAYS)
    fit = analysis.fit_data.func(q, processed)[q.uid]
    fine = np.linspace(DELAYS.min(), DELAYS.max(), 51)
    confidence, prediction = analysis._fit_uncertainty_bands(fit, fine)
    amplitude = fit.best_values["amplitude"]
    exponential = np.exp(-fit.best_values["decay_rate"] * fine)
    derivatives = {
        "amplitude": exponential,
        "decay_rate": -amplitude * fine * exponential,
        "offset": np.ones_like(fine),
    }
    jacobian = np.column_stack([derivatives[name] for name in fit.var_names])
    variance = np.einsum("ij,jk,ik->i", jacobian, fit.covar, jacobian)
    scale = t.ppf(0.975, fit.nfree)
    np.testing.assert_allclose(confidence, scale * np.sqrt(variance), rtol=1e-5)
    residual_variance = np.sum(fit.residual**2) / fit.nfree
    np.testing.assert_allclose(prediction, scale * np.sqrt(variance + residual_variance), rtol=1e-5)
    assert np.all(prediction > confidence)


@pytest.mark.parametrize("problem", ["covariance", "stderr", "dof"])
def test_fit_bands_omit_unavailable_uncertainty(qpu, problem):
    q = qpu["q0"]
    processed = analysis.calculate_qubit_population.func(q, synthetic_result([q]), DELAYS)
    fit = analysis.fit_data.func(q, processed)[q.uid]
    if problem == "covariance":
        fit.covar = None
    elif problem == "stderr":
        fit.params[fit.var_names[0]].stderr = None
    else:
        fit.nfree = 0
    assert analysis._fit_uncertainty_bands(fit, DELAYS) is None


def test_unknown_fit_error_is_annotated(qpu):
    q = qpu["q0"]
    processed = analysis.calculate_qubit_population.func(q, synthetic_result([q]), DELAYS)
    fits = analysis.fit_data.func(q, processed)
    fits[q.uid].covar = None
    parameters = analysis.extract_qubit_parameters.func(q, fits)
    fig = analysis.plot_population.func(
        q, processed, fits, parameters, options=analysis.T1PlotOptions(save_figures=False)
    )[q.uid]
    assert "uncertainty unavailable" in fig.axes[0].texts[0].get_text()
    labels = fig.axes[0].get_legend_handles_labels()[1]
    assert "fit" in labels
    assert "95% confidence band" not in labels
    assert "95% prediction band" not in labels
    plt.close(fig)


@pytest.mark.parametrize(
    "do_analysis,update", [(True, False), (True, True), (False, False), (False, True)]
)
def test_experiment_workflow_raw_output_temporary_parameters_and_update(qpu, do_analysis, update):
    result = synthetic_result(qpu.quantum_elements)
    captured = []

    class SyntheticSession:
        def compile(self, experiment, **kwargs):
            captured.append(experiment)
            return experiment

        def run(self, compiled_experiment, **kwargs):
            return result

    old = qpu["q0"].parameters.ge_T1
    options = experiment.experiment_workflow.options()
    options.logstore([])
    options.count(64)
    options.do_analysis(do_analysis)
    options.update(update)
    options.do_plotting(False)
    run = experiment.experiment_workflow(
        SyntheticSession(),
        qpu,
        ["q0", "q1"],
        [DELAYS] * 2,
        temporary_parameters={"q0": {"ge_T1": 15e-6}},
        options=options,
    ).run()
    assert run.output is result
    assert captured
    if do_analysis:
        out = run.tasks["analysis_workflow"].output
        assert_concrete(out)
        assert out["old_parameter_values"]["q0"]["ge_T1"] == 15e-6
    if do_analysis and update:
        assert qpu["q0"].parameters.ge_T1 == pytest.approx(30e-6, rel=0.02)
    else:
        assert qpu["q0"].parameters.ge_T1 == old


@pytest.mark.parametrize("transition,active_reset", [("ge", False), ("ge", True), ("ef", False)])
def test_new_experiment_compiles_offline(qpu, transition, active_reset):
    platform = demo_platform(2)
    session = Session(device_setup=platform.setup, configure_logging=False)
    opts = experiment.T1ExperimentOptions(
        count=4, transition=transition, cal_states=transition, active_reset=active_reset
    )
    built = experiment.create_experiment.func(
        platform.qpu, platform.qpu.quantum_elements, [DELAYS[:3]] * 2, options=opts
    )
    session.compile(built)


def test_emulated_result_axes_are_accepted(qpu):
    platform = demo_platform(2)
    session = Session(device_setup=platform.setup, configure_logging=False)
    session.connect(do_emulation=True)
    grids = [DELAYS[:3], DELAYS[:3] * 2]
    try:
        built = experiment.create_experiment.func(
            platform.qpu,
            platform.qpu.quantum_elements,
            grids,
            options=experiment.T1ExperimentOptions(count=4),
        )
        result = session.run(session.compile(built))
    finally:
        session.disconnect()
    # Emulation establishes shape/metadata, not physical reference separation.
    for q in platform.qpu.quantum_elements:
        for state, center in zip("ge", (0, 1)):
            result[f"{q.uid}/cal_trace/{state}"].data = np.full(4, complex(center))
    out = analysis.calculate_qubit_population.func(platform.qpu.quantum_elements, result, grids)
    assert set(out) == {"q0", "q1"}
    assert all(item["num_shots"] == 4 for item in out.values())


def test_saved_results_reanalysis(qpu, tmp_path):
    result = synthetic_result(qpu.quantum_elements)
    path = tmp_path / "raw.json"
    serializers.save(result, path)
    restored = serializers.load(path)
    snapshots = []
    for q in qpu.quantum_elements:
        path = tmp_path / f"{q.uid}.json"
        serializers.save(q, path)
        snapshots.append(serializers.load(path))
    out = analysis_run(qpu, result=restored, qubits=snapshots).output
    assert_concrete(out)
    assert out["processed_data"]["q0"]["num_shots"] == 64


def test_fit_failure_preserves_other_qubits(qpu, monkeypatch):
    original = legacy.exponential_decay_fit
    failures = iter([True, False])

    def fit(*args, **kwargs):
        if next(failures):
            raise ValueError("Synthetic failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(legacy, "exponential_decay_fit", fit)
    out = analysis_run(qpu).output
    assert out["new_parameter_values"]["q0"] == {}
    assert "ge_T1" in out["new_parameter_values"]["q1"]
    assert set(out["processed_data"]) == {"q0", "q1"}


def test_saved_figure_artifacts(qpu, monkeypatch):
    saved = []
    monkeypatch.setattr(analysis.workflow, "save_artifact", lambda name, figure: saved.append(name))
    options = analysis.analysis_workflow.options()
    options.logstore([])
    run = analysis_run(qpu, options=options)
    assert set(saved) == {"T1_q0", "T1_q1", "Raw_data_q0", "Raw_data_q1"}
    for task_name in ("plot_population", "plot_raw_complex_data_1d"):
        assert all(
            not plt.fignum_exists(fig.number) for fig in run.tasks[task_name].output.values()
        )


def test_default_options_and_option_builder_boundary(qpu):
    options = experiment.experiment_workflow.options()
    opts = options._base._task_options["create_experiment"]
    assert opts.count == 1024
    assert opts.averaging_mode == AveragingMode.SINGLE_SHOT
    assert opts.acquisition_type == AcquisitionType.INTEGRATION
    assert options._base.do_analysis and not options._base.update
    assert analysis._options(options, experiment.T1WorkflowOptions) is options._base


@pytest.mark.parametrize("transition", ["ge", "ef"])
def test_single_uid_workflow_forwards_transition(qpu, transition):
    q = qpu["q0"]
    result = synthetic_result([q], states=transition)

    class SyntheticSession:
        def compile(self, experiment, **kwargs):
            return experiment

        def run(self, compiled_experiment, **kwargs):
            return result

    options = experiment.experiment_workflow.options()
    options.logstore([])
    options.count(64)
    options.transition(transition)
    options.cal_states(transition)
    options.do_plotting(False)
    run = experiment.experiment_workflow(
        SyntheticSession(),
        qpu,
        "q0",
        DELAYS,
        options=options,
    ).run()
    assert run.output is result
    assert f"{transition}_T1" in run.tasks["analysis_workflow"].output["new_parameter_values"]["q0"]

# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Analyze single-shot T1 IQ data with calibration-based population statistics.

Plot population means with pointwise fit/prediction bands and reference SEM bands.
Keep raw I/Q SEM error bars for acquisition diagnostics.
Fit the mean decay without weights; calibration means are treated as fixed.
Return update-compatible parameters and processed statistics, without modifying a QPU.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import uncertainties as unc
from laboneq import workflow
from laboneq.simple import dsl
from laboneq_applications.analysis.options import (
    BasePlottingOptions,
    FitDataOptions,
    TuneUpAnalysisWorkflowOptions,
)
from laboneq_applications.core.validation import (
    validate_and_convert_qubits_sweeps,
    validate_result,
)

from .lifetime_measurement import FitDataT1Options
from .lifetime_measurement import fit_data as lifetime_fit_data
from .plot_theme import with_plot_theme
from .plotting_helpers import timestamped_title

if TYPE_CHECKING:
    import lmfit
    from laboneq.dsl.result import AcquiredResult, Results
    from laboneq_applications.typing import QuantumElements, QubitSweepPoints


@workflow.task_options(base_class=FitDataOptions)
class T1AnalysisOptions:
    """Common calibration and unweighted-fit options for single-shot T1."""

    transition: str = workflow.option_field("ge", description="Transition: ge or ef.")
    cal_states: str | tuple = workflow.option_field(
        "ge", description="Ordered reference states; must match transition."
    )
    use_cal_traces: bool = workflow.option_field(
        True, description="Calibration traces are required."
    )
    do_pca: bool = workflow.option_field(False, description="PCA is not supported.")


@workflow.task_options(base_class=BasePlottingOptions)
class T1PlotOptions:
    """Lifetime-style T1 plotting options."""

    do_fitting: bool = workflow.option_field(True, description="Display the decay fit.")


def _options(options, default_class):
    """Resolve plain options or an OptionBuilder at a concrete Python boundary."""
    return default_class() if options is None else getattr(options, "_base", options)


def validate_calibration_options(opts) -> tuple[str, str]:
    """Validate the common experiment/analysis calibration contract."""
    if not opts.use_cal_traces:
        raise ValueError("T1_measurement requires use_cal_traces=True.")
    if opts.do_pca or not opts.do_rotation:
        raise ValueError("T1_measurement requires do_pca=False and do_rotation=True.")
    if opts.transition not in ("ge", "ef"):
        raise ValueError("T1_measurement transition must be 'ge' or 'ef'.")
    states = tuple(opts.cal_states)
    if states != tuple(opts.transition):
        raise ValueError("cal_states must match transition in order: 'ge' or 'ef'.")
    return states


def validate_delays(qubits, delays):
    """Validate finite, nonnegative, nonempty 1D delay grids."""
    qubits, delays = validate_and_convert_qubits_sweeps(qubits, delays)
    arrays = []
    for q, values in zip(qubits, delays):
        values = np.asarray(values, dtype=float)
        if (
            values.ndim != 1
            or values.size == 0
            or not np.all(np.isfinite(values))
            or np.any(values < 0)
        ):
            raise ValueError(f"{q.uid}: delays must be a nonempty finite nonnegative 1D grid.")
        arrays.append(values)
    return qubits, arrays


def _axis_metadata(acquired: AcquiredResult, handle: str, ndim: int):
    """Validate scalar or parallel-sweep metadata for every data dimension."""
    data = np.asarray(acquired.data, dtype=complex)
    if data.ndim != ndim or not np.all(np.isfinite(data)):
        raise ValueError(f"{handle}: expected finite {ndim}D single-shot IQ data.")
    names, axes = acquired.axis_name, acquired.axis
    if (
        not isinstance(names, (list, tuple))
        or not isinstance(axes, (list, tuple))
        or len(names) != ndim
        or len(axes) != ndim
    ):
        raise ValueError(f"{handle}: missing or unsupported axis metadata.")
    dimensions = []
    for size, name, axis in zip(data.shape, names, axes):
        labels = [name] if isinstance(name, str) else name
        values = [axis] if isinstance(name, str) else axis
        if (
            not isinstance(labels, (list, tuple))
            or not isinstance(values, (list, tuple))
            or not labels
            or len(labels) != len(values)
            or not all(isinstance(label, str) for label in labels)
            or len(set(labels)) != len(labels)
        ):
            raise ValueError(f"{handle}: unsupported parallel axis metadata.")
        dimension = {}
        for label, value in zip(labels, values):
            grid = np.asarray(value, dtype=float)
            if grid.ndim != 1 or len(grid) != size or not np.all(np.isfinite(grid)):
                raise ValueError(f"{handle}: axis grid does not match data shape.")
            dimension[label] = grid
        dimensions.append(dimension)
    return data, dimensions


def _single_shots(result: Results, uid: str, delays: np.ndarray, states):
    """Identify shot/delay axes by their metadata, including square arrays."""
    handle = dsl.handles.result_handle(uid)
    try:
        acquired = result[handle]
    except KeyError as err:
        raise ValueError(f"Missing single-shot signal {handle}.") from err
    data, dimensions = _axis_metadata(acquired, handle, 2)
    # create_experiment names the sweep axis after the qubit UID. Never infer
    # the delay axis from shape: count can equal the number of delay points.
    delay_axes = [i for i, dimension in enumerate(dimensions) if uid in dimension]
    if len(delay_axes) != 1:
        raise ValueError(f"{handle}: expected exactly one delay axis named {uid!r}.")
    delay_axis = delay_axes[0]
    shot_axis = 1 - delay_axis
    if not np.array_equal(dimensions[delay_axis][uid], delays):
        raise ValueError(f"{handle}: delay axis does not match the requested delays.")
    if len(dimensions[shot_axis]) != 1:
        raise ValueError(f"{handle}: expected one shot axis name.")
    shot_name, shot_grid = next(iter(dimensions[shot_axis].items()))
    data = np.moveaxis(data, shot_axis, 0)
    if data.shape[0] < 2:
        raise ValueError(f"{handle}: at least two shots are required for SD and SEM.")
    references = []
    for state in states:
        handle = dsl.handles.calibration_trace_handle(uid, state)
        try:
            reference = result[handle]
        except KeyError as err:
            raise ValueError(f"Missing single-shot reference {handle}.") from err
        shots, ref_dimensions = _axis_metadata(reference, handle, 1)
        if (
            shots.size != data.shape[0]
            or list(ref_dimensions[0]) != [shot_name]
            or not np.array_equal(ref_dimensions[0][shot_name], shot_grid)
        ):
            raise ValueError(f"{handle}: reference shot axis must match the signal shot axis.")
        references.append(shots)
    return data, references


def _statistics(values, axis=0):
    """Sample SD and SEM of real samples along the shot axis."""
    sd = np.std(values, axis=axis, ddof=1)
    return np.mean(values, axis=axis), sd, sd / np.sqrt(values.shape[axis])


@workflow.workflow
def analysis_workflow(
    result: Results,
    qubits: QuantumElements,
    delays: QubitSweepPoints,
    options: TuneUpAnalysisWorkflowOptions | None = None,
) -> None:
    """Analyze saved or newly acquired single-shot T1 results."""
    processed_data = calculate_qubit_population(qubits, result, delays)
    fit_results = fit_data(qubits, processed_data)
    parameters = extract_qubit_parameters(qubits, fit_results)
    with workflow.if_(options.do_plotting):
        with workflow.if_(options.do_raw_data_plotting):
            plot_raw_complex_data_1d(qubits, processed_data)
        with workflow.if_(options.do_qubit_population_plotting):
            plot_population(qubits, processed_data, fit_results, parameters)
    output = assemble_analysis_output(parameters, processed_data)
    workflow.return_(output)


@workflow.task
def calculate_qubit_population(
    qubits: QuantumElements,
    result: Results,
    delays: QubitSweepPoints,
    options: T1AnalysisOptions | None = None,
) -> dict:
    """Project shots using fixed reference means and calculate SD/SEM statistics."""
    opts = _options(options, T1AnalysisOptions)
    states = validate_calibration_options(opts)
    validate_result(result)
    qubits, delays = validate_delays(qubits, delays)
    processed = {}
    for q, grid in zip(qubits, delays):
        shots, references = _single_shots(result, q.uid, grid, states)
        means = np.array([np.mean(ref) for ref in references])
        separation = means[1] - means[0]
        distance = abs(separation)
        if not np.isfinite(distance) or distance == 0:
            raise ValueError(f"{q.uid}: reference means must have finite nonzero separation.")
        # Divide by the norm in two steps to avoid squaring very small/large IQ scales.
        direction = separation.conjugate() / distance

        def project(z):
            return np.real((z - means[0]) * direction) / distance

        population_shots = project(shots)
        ref_population_shots = np.array([project(ref) for ref in references])
        if not np.all(np.isfinite(population_shots)) or not np.all(
            np.isfinite(ref_population_shots)
        ):
            raise ValueError(f"{q.uid}: nonfinite population after reference normalization.")
        population, population_std, population_sem = _statistics(population_shots)
        cal_population, cal_std, cal_sem = _statistics(ref_population_shots, axis=1)
        i, i_std, i_sem = _statistics(shots.real)
        quadrature, q_std, q_sem = _statistics(shots.imag)
        ref_i, ref_i_std, ref_i_sem = _statistics(np.array(references).real, axis=1)
        ref_q, ref_q_std, ref_q_sem = _statistics(np.array(references).imag, axis=1)
        processed[q.uid] = {
            "sweep_points": grid,
            "num_shots": shots.shape[0],
            "transition": opts.transition,
            "cal_states": states,
            "num_cal_traces": 2,
            "population": population,
            "population_std": population_std,
            "population_sem": population_sem,
            "population_cal_traces": cal_population,
            "population_cal_traces_std": cal_std,
            "population_cal_traces_sem": cal_sem,
            "data_raw": i + 1j * quadrature,
            "data_raw_cal_traces": means,
            "raw_i": i,
            "raw_i_std": i_std,
            "raw_i_sem": i_sem,
            "raw_q": quadrature,
            "raw_q_std": q_std,
            "raw_q_sem": q_sem,
            "reference_i": ref_i,
            "reference_i_std": ref_i_std,
            "reference_i_sem": ref_i_sem,
            "reference_q": ref_q,
            "reference_q_std": ref_q_std,
            "reference_q_sem": ref_q_sem,
            "uncertainty_method": "shot SEM; calibration means fixed",
        }
    return processed


@workflow.task
def fit_data(
    qubits: QuantumElements,
    processed_data_dict: dict,
    options: T1AnalysisOptions | None = None,
) -> dict[str, lmfit.model.ModelResult]:
    """Reuse the existing lifetime fit without reading SD or SEM as weights."""
    opts = _options(options, T1AnalysisOptions)
    validate_calibration_options(opts)
    results = lifetime_fit_data.func(
        qubits,
        processed_data_dict,
        options=FitDataT1Options(
            do_fitting=opts.do_fitting,
            do_rotation=True,
            do_pca=False,
            use_cal_traces=True,
            fit_parameters_hints=opts.fit_parameters_hints,
        ),
    )
    # The upstream helper converts unavailable stderr to zero. Restore an
    # explicit unknown when covariance is absent; never report false precision.
    for fit in results.values():
        if fit.covar is None:
            for parameter in fit.params.values():
                if parameter.vary or parameter.expr is not None:
                    parameter.stderr = None
    return results


def _t1_estimate(fit):
    """Return a physical T1 and its error, or None for an invalid fit."""
    if fit is None or not fit.success:
        return None
    rate = fit.params["decay_rate"]
    if not np.isfinite(rate.value) or rate.value <= 0:
        return None
    t1 = 1.0 / rate.value
    if not np.isfinite(t1):
        return None
    error = rate.stderr
    if fit.covar is None or error is None or not np.isfinite(error) or error < 0:
        return t1, None
    t1_error = (error / rate.value) * t1
    return t1, t1_error if np.isfinite(t1_error) else None


@workflow.task
def extract_qubit_parameters(
    qubits: QuantumElements,
    fit_results: dict,
    options: T1AnalysisOptions | None = None,
) -> dict:
    """Publish only successful positive T1 estimates with known uncertainty."""
    opts = _options(options, T1AnalysisOptions)
    validate_calibration_options(opts)
    qubits = validate_and_convert_qubits_sweeps(qubits)
    output = {"old_parameter_values": {}, "new_parameter_values": {}}
    key = f"{opts.transition}_T1"
    for q in qubits:
        output["old_parameter_values"][q.uid] = {key: getattr(q.parameters, key)}
        output["new_parameter_values"][q.uid] = {}
        estimate = _t1_estimate(fit_results.get(q.uid)) if opts.do_fitting else None
        if estimate is not None and estimate[1] is not None:
            output["new_parameter_values"][q.uid][key] = unc.ufloat(*estimate)
        elif opts.do_fitting:
            message = "%s: no valid T1 update with known uncertainty."
            if workflow.execution_info() is not None:
                workflow.log(logging.WARNING, message, q.uid)
            else:
                logging.getLogger(__name__).warning(message, q.uid)
    return output


@workflow.task(save=False)
def assemble_analysis_output(parameters: dict, processed_data: dict) -> dict:
    """Assemble concrete output inside a task, avoiding nested workflow References."""
    return {**parameters, "processed_data": processed_data}


def _reference_bands(ax, states, means, deviations):
    for index, (mean, deviation) in enumerate(zip(means, deviations)):
        ax.axhline(mean, color="black", linestyle="--", linewidth=0.8, zorder=1)
        ax.axhspan(
            mean - deviation,
            mean + deviation,
            color="#808080",
            alpha=0.16,
            linewidth=0,
            label=f"{'/'.join(states)} reference ± SEM" if index == 0 else "_nolegend_",
            zorder=0,
        )


def _fit_uncertainty_bands(fit, grid):
    """Return approximate 95% pointwise bands for the unweighted mean-data fit.

    The prediction target is one new shot-averaged datum with the fitted residual
    variance. Assume independent, normal, constant-variance mean residuals and
    fixed calibration; these are neither simultaneous nor single-shot bands.
    """
    if (
        fit.covar is None
        or fit.nfree <= 0
        or not np.all(np.isfinite(fit.covar))
        or not np.isfinite(fit.redchi)
        or fit.redchi < 0
    ):
        return None
    parameters = fit.params.copy()
    for name, parameter in parameters.items():
        if name in fit.var_names:
            if (
                parameter.stderr is None
                or not np.isfinite(parameter.stderr)
                or parameter.stderr <= 0
            ):
                return None
        else:
            # lmfit checks stderr even for fixed parameters, which do not enter
            # the covariance propagation. A missing fixed stderr is harmless.
            parameter.stderr = 0.0
    confidence = fit.eval_uncertainty(params=parameters, x=grid, sigma=0.95)
    prediction = fit.dely_predicted.copy()
    if (
        not np.all(np.isfinite(confidence))
        or not np.all(np.isfinite(prediction))
        or np.any(confidence < 0)
        or np.any(prediction < confidence)
    ):
        return None
    return confidence, prediction


@workflow.task
@with_plot_theme
def plot_population(
    qubits: QuantumElements,
    processed_data_dict: dict,
    fit_results: dict,
    qubit_parameters: dict,
    options: T1PlotOptions | None = None,
) -> dict:
    """Publication-style means, 95% fit/prediction bands, and reference SEM."""
    opts = _options(options, T1PlotOptions)
    qubits = validate_and_convert_qubits_sweeps(qubits)
    figures = {}
    for q in qubits:
        data = processed_data_dict[q.uid]
        grid = data["sweep_points"]
        fig, ax = plt.subplots(figsize=(6.8, 3.7))
        ax.set_title(
            timestamped_title(f"T1 measurement · {q.uid}"), loc="left", fontsize=10, pad=10
        )
        ax.set_xlabel("Delay, $\\tau$ ($\\mu$s)", fontsize=10)
        ax.set_ylabel(f"$|{data['cal_states'][-1]}\\rangle$ population", fontsize=10)
        ax.plot(
            grid * 1e6,
            data["population"],
            linestyle="none",
            marker="o",
            color="#0000FF",
            markersize=2.6,
            markeredgewidth=0,
            zorder=4,
            label="data",
        )
        _reference_bands(
            ax,
            data["cal_states"],
            data["population_cal_traces"],
            data["population_cal_traces_sem"],
        )
        fit = fit_results.get(q.uid) if opts.do_fitting else None
        estimate = _t1_estimate(fit)
        if estimate is not None:
            fine = np.linspace(np.min(grid), np.max(grid), 501)
            fitted = fit.model.func(fine, **fit.best_values)
            bands = _fit_uncertainty_bands(fit, fine)
            if bands is not None:
                confidence, prediction = bands
                ax.fill_between(
                    fine * 1e6,
                    fitted - prediction,
                    fitted + prediction,
                    color="#90CAF9",
                    alpha=0.30,
                    linewidth=0,
                    zorder=1,
                    label="95% prediction band",
                )
                ax.fill_between(
                    fine * 1e6,
                    fitted - confidence,
                    fitted + confidence,
                    color="#4285D4",
                    alpha=0.40,
                    linewidth=0,
                    zorder=2,
                    label="95% confidence band",
                )
            ax.plot(
                fine * 1e6,
                fitted,
                color="#FF0000",
                linestyle="-",
                linewidth=1.4,
                zorder=3,
                label="fit",
            )
            value, error = estimate
            text = f"$T_1 = {value * 1e6:.2f}"
            text += (
                f" \\pm {error * 1e6:.2f}\\;\\mu\\mathrm{{s}}$"
                if error is not None
                else "\\;\\mu\\mathrm{s}$\n(fit uncertainty unavailable)"
            )
            ax.text(0.97, 0.84, text, ha="right", va="top", fontsize=9, transform=ax.transAxes)
        handles, labels = ax.get_legend_handles_labels()
        entries = dict(zip(labels, handles))
        order = ["data", "fit", "95% confidence band", "95% prediction band"]
        order += [f"{'/'.join(data['cal_states'])} reference ± SEM"]
        order = [label for label in order if label in entries]
        ax.legend(
            [entries[label] for label in order],
            order,
            loc="center left",
            bbox_to_anchor=(1.03, 0.5),
            fontsize=8,
            handlelength=1.8,
            frameon=False,
        )
        ax.grid(False)
        ax.tick_params(direction="out", length=3, width=0.8, labelsize=9)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
        fig.subplots_adjust(left=0.12, right=0.65, bottom=0.17, top=0.88)
        if opts.save_figures:
            workflow.save_artifact(f"T1_{q.uid}", fig)
        if opts.close_figures:
            plt.close(fig)
        figures[q.uid] = fig
    return figures


@workflow.task
@with_plot_theme
def plot_raw_complex_data_1d(
    qubits: QuantumElements,
    processed_data_dict: dict,
    options: T1PlotOptions | None = None,
) -> dict:
    """Keep the existing two-panel raw I/Q layout, adding SEM bars and SEM bands."""
    opts = _options(options, T1PlotOptions)
    qubits = validate_and_convert_qubits_sweeps(qubits)
    figures = {}
    for q in qubits:
        data = processed_data_dict[q.uid]
        fig, axes = plt.subplots(2, 1, figsize=(8.2, 5.6), sharex=True)
        for ax, component in zip(axes, ("i", "q")):
            ax.errorbar(
                data["sweep_points"] * 1e6,
                data[f"raw_{component}"],
                yerr=data[f"raw_{component}_sem"],
                fmt="o-",
                color="#0000FF",
                markersize=2.5,
                markeredgewidth=0.4,
                linewidth=0.7,
                elinewidth=0.8,
                capsize=2,
                capthick=0.8,
                barsabove=True,
                zorder=3,
                label=f"{component.upper()} ± SEM",
            )
            _reference_bands(
                ax,
                data["cal_states"],
                data[f"reference_{component}"],
                data[f"reference_{component}_sem"],
            )
            ax.set_ylabel(f"{component.upper()} (a.u.)")
            ax.grid(alpha=0.22)
            ax.legend(loc="best", frameon=False)
        axes[0].set_title(timestamped_title(f"Raw data {q.uid}"))
        axes[1].set_xlabel("Pulse Delay, $\\tau$ ($\\mu$s)")
        fig.tight_layout()
        if opts.save_figures:
            workflow.save_artifact(f"Raw_data_{q.uid}", fig)
        if opts.close_figures:
            plt.close(fig)
        figures[q.uid] = fig
    return figures

# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Local lifetime-measurement analysis.

Fits an exponential decay to extract `ge_T1`.
Uses repo-local raw-data plotting helpers and plot theme handling.
Returns `old_parameter_values` and `new_parameter_values` for update workflows.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import uncertainties as unc
from analysis.plot_theme import with_plot_theme
from analysis.plotting_helpers import (
    plot_raw_complex_data_1d,
    timestamped_title,
)
from laboneq import workflow
from laboneq_applications.analysis.calibration_traces_rotation import (
    calculate_qubit_population,
)
from laboneq_applications.analysis.fitting_helpers import exponential_decay_fit
from laboneq_applications.analysis.options import (
    ExtractQubitParametersTransitionOptions,
    FitDataOptions,
    PlotPopulationOptions,
    TuneUpAnalysisWorkflowOptions,
)
from laboneq_applications.core.validation import validate_and_convert_qubits_sweeps

if TYPE_CHECKING:
    import lmfit
    import matplotlib as mpl
    from laboneq.workflow.tasks.run_experiment import RunExperimentResults
    from laboneq_applications.typing import QuantumElements, QubitSweepPoints
    from numpy.typing import ArrayLike


@workflow.task_options(base_class=FitDataOptions)
class FitDataT1Options:
    """Options for fitting lifetime-measurement traces."""

    do_pca: bool = workflow.option_field(
        False,
        description="Whether to force PCA projection for lifetime fitting.",
    )
    use_cal_traces: bool = workflow.option_field(
        True,
        description="Whether calibration traces are expected in the experiment.",
    )


@workflow.workflow
def analysis_workflow(
    result: RunExperimentResults,
    qubits: QuantumElements,
    delays: QubitSweepPoints,
    options: TuneUpAnalysisWorkflowOptions | None = None,
) -> None:
    """Run the local lifetime-measurement analysis workflow."""
    processed_data_dict = calculate_qubit_population(qubits, result, delays)
    fit_results = fit_data(qubits, processed_data_dict)
    qubit_parameters = extract_qubit_parameters(qubits, fit_results)
    with workflow.if_(options.do_plotting):
        with workflow.if_(options.do_raw_data_plotting):
            plot_raw_complex_data_1d(
                qubits,
                result,
                delays,
                xlabel="Pulse Delay, $\\tau$ ($\\mu$s)",
                xscaling=1e6,
            )
        with workflow.if_(options.do_qubit_population_plotting):
            plot_population(qubits, processed_data_dict, fit_results, qubit_parameters)
    workflow.return_(qubit_parameters)


@workflow.task
def fit_data(
    qubits: QuantumElements,
    processed_data_dict: dict[str, dict[str, ArrayLike]],
    options: FitDataT1Options | None = None,
) -> dict[str, lmfit.model.ModelResult]:
    """Fit an exponential decay model to lifetime-measurement traces."""
    opts = FitDataT1Options() if options is None else options
    qubits = validate_and_convert_qubits_sweeps(qubits)
    fit_results = {}
    if not opts.do_fitting:
        return fit_results

    for q in qubits:
        swpts_fit = processed_data_dict[q.uid]["sweep_points"]
        data_to_fit = processed_data_dict[q.uid][
            "population" if opts.do_rotation else "data_raw"
        ]
        param_hints = {
            "offset": {"value": 0, "vary": opts.do_pca or not opts.use_cal_traces},
        }
        param_hints_user = opts.fit_parameters_hints
        if param_hints_user is None:
            param_hints_user = {}
        param_hints.update(param_hints_user)
        try:
            fit_res = exponential_decay_fit(
                swpts_fit,
                data_to_fit,
                param_hints=param_hints,
            )
            fit_results[q.uid] = fit_res
        except ValueError as err:
            workflow.log(logging.ERROR, "Fit failed for %s: %s.", q.uid, err)

    return fit_results


@workflow.task
def extract_qubit_parameters(
    qubits: QuantumElements,
    fit_results: dict[str, lmfit.model.ModelResult],
    options: ExtractQubitParametersTransitionOptions | None = None,
) -> dict[str, dict[str, dict[str, int | float | unc.core.Variable | None]]]:
    """Extract old/new `T1` values from the fit results."""
    opts = ExtractQubitParametersTransitionOptions() if options is None else options
    qubits = validate_and_convert_qubits_sweeps(qubits)
    qubit_parameters = {
        "old_parameter_values": {q.uid: {} for q in qubits},
        "new_parameter_values": {q.uid: {} for q in qubits},
    }

    for q in qubits:
        old_t1 = q.parameters.ef_T1 if "f" in opts.transition else q.parameters.ge_T1
        qubit_parameters["old_parameter_values"][q.uid] = {
            f"{opts.transition}_T1": old_t1,
        }

        if opts.do_fitting and q.uid in fit_results:
            fit_res = fit_results[q.uid]
            dec_rt = unc.ufloat(
                fit_res.params["decay_rate"].value,
                fit_res.params["decay_rate"].stderr,
            )
            qubit_parameters["new_parameter_values"][q.uid] = {
                f"{opts.transition}_T1": 1 / dec_rt,
            }

    return qubit_parameters


@workflow.task
@with_plot_theme
def plot_population(
    qubits: QuantumElements,
    processed_data_dict: dict[str, dict[str, ArrayLike]],
    fit_results: dict[str, lmfit.model.ModelResult] | None,
    qubit_parameters: dict[
        str,
        dict[str, dict[str, int | float | unc.core.Variable | None]],
    ]
    | None,
    options: PlotPopulationOptions | None = None,
) -> dict[str, mpl.figure.Figure]:
    """Create local lifetime-measurement population plots."""
    opts = PlotPopulationOptions() if options is None else options
    qubits = validate_and_convert_qubits_sweeps(qubits)
    figures = {}
    for q in qubits:
        sweep_points = processed_data_dict[q.uid]["sweep_points"]
        data = processed_data_dict[q.uid][
            "population" if opts.do_rotation else "data_raw"
        ]
        num_cal_traces = processed_data_dict[q.uid]["num_cal_traces"]

        fig, ax = plt.subplots()
        ax.set_title(timestamped_title(f"Lifetime Measurement {q.uid}"))
        ax.set_xlabel("Pulse Delay, $\\tau$ ($\\mu$s)")
        ax.set_ylabel(
            "Principal Component (a.u)"
            if (num_cal_traces == 0 or opts.do_pca)
            else f"$|{opts.cal_states[-1]}\\rangle$-State Population",
        )
        ax.plot(sweep_points * 1e6, data, "o", zorder=2, label="data")
        if processed_data_dict[q.uid]["num_cal_traces"] > 0:
            xlims = ax.get_xlim()
            ax.hlines(
                processed_data_dict[q.uid]["population_cal_traces"],
                *xlims,
                linestyles="--",
                colors="gray",
                zorder=0,
                label="calib.\ntraces",
            )
            ax.set_xlim(xlims)

        if opts.do_fitting and q.uid in fit_results:
            fit_res_qb = fit_results[q.uid]
            swpts_fine = np.linspace(sweep_points[0], sweep_points[-1], 501)
            ax.plot(
                swpts_fine * 1e6,
                fit_res_qb.model.func(swpts_fine, **fit_res_qb.best_values),
                "r-",
                zorder=1,
                label="fit",
            )
            if (
                qubit_parameters is not None
                and len(qubit_parameters["new_parameter_values"][q.uid]) > 0
            ):
                old_t1 = qubit_parameters["old_parameter_values"][q.uid][
                    f"{opts.transition}_T1"
                ]
                new_t1 = qubit_parameters["new_parameter_values"][q.uid][
                    f"{opts.transition}_T1"
                ]
                textstr = (
                    "$T_1$: "
                    f"{new_t1.nominal_value * 1e6:.4f} $\\mu$s $\\pm$ "
                    f"{new_t1.std_dev * 1e6:.4f} $\\mu$s"
                )
                textstr += "\nPrevious value: " + f"{old_t1 * 1e6:.4f} $\\mu$s"
                ax.text(0, -0.15, textstr, ha="left", va="top", transform=ax.transAxes)

        ax.legend(
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            handlelength=1.5,
            frameon=False,
        )

        if opts.save_figures:
            workflow.save_artifact(f"T1_{q.uid}", fig)

        if opts.close_figures:
            plt.close(fig)

        figures[q.uid] = fig

    return figures

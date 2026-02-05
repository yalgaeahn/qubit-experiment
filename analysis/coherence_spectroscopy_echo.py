# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np
import uncertainties as unc
from laboneq import workflow

from laboneq_applications.analysis.calibration_traces_rotation import (
    calculate_qubit_population_2d,
)
from analysis.fitting_helpers import (
    coherent_spec_fit,
    coherent_spec_photon_numbers,
    exponential_decay_fit,
)
from laboneq_applications.analysis.options import (
    ExtractQubitParametersTransitionOptions,
    FitDataOptions,
    PlotPopulationOptions,
    TuneUpAnalysisWorkflowOptions,
)
from laboneq_applications.analysis.plotting_helpers import (
    plot_raw_complex_data_1d,
    timestamped_title,
)
from laboneq_applications.core.validation import validate_and_convert_qubits_sweeps

if TYPE_CHECKING:
    import lmfit
    import matplotlib as mpl
    from laboneq.workflow.tasks.run_experiment import RunExperimentResults
    from numpy.typing import ArrayLike

    from laboneq_applications.typing import QuantumElements, QubitSweepPoints


@workflow.task_options(base_class=FitDataOptions)
class FitDataRamseyOptions:
    """Options for the `fit_data` task of the Ramsey analysis.

    See [FitDataOptions] for additional accepted options.

    Attributes:
        do_pca:
            Whether to perform principal component analysis on the raw data independent
            of whether there were calibration traces in the experiment.
            Default: `False`.
        transition:
            Transition to perform the experiment on. May be any
            transition supported by the quantum operations.
            Default: `"ge"` (i.e. ground to first excited state).
    """

    do_pca: bool = workflow.option_field(
        False,
        description="Whether to perform principal component analysis on the raw data"
        " independent of whether there were calibration traces in the experiment.",
    )
    transition: Literal["ge", "ef"] = workflow.option_field(
        "ge",
        description="Transition to perform the experiment on. May be any"
        " transition supported by the quantum operations.",
    )


def validate_and_convert_detunings(
    qubits: QuantumElements,
    detunings: float | Sequence[float] | None = None,
) -> Sequence[float]:
    """Validate the detunings used in a Ramsey experiment, and convert them to iterable.

    Check for the following conditions:
        - qubits must be a sequence.
        - detunings must be a sequence
        - detunings must have the same length as qubits
    If detunings is None, it is instantiated to a list of zeros with the same length
    as qubits.

    Args:
        qubits: the qubits used in the experiment/analysis
    frequencies:
        The CW drive frequencies swept in the outer loop.
    detunings:
        The detuning in Hz introduced in order to generate oscillations of the qubit
        state vector around the Bloch sphere. This detuning and the frequency of the
        fitted oscillations is used to calculate the true qubit resonance frequency.
        `detunings` is a list of float values for each qubit following the order
        in qubits.

    Returns:
        a list containing the validated detunings
    """
    if not isinstance(qubits, Sequence):
        qubits = [qubits]

    if detunings is None:
        detunings = [0] * len(qubits)

    if not isinstance(detunings, Sequence):
        detunings = [detunings]

    if len(detunings) != len(qubits):
        raise ValueError(
            f"Length of qubits and detunings must be the same but "
            f"currently len(qubits) = {len(qubits)} and "
            f"len(detunings) = {len(detunings)}."
        )

    return detunings


@workflow.workflow
def analysis_workflow(
    result: RunExperimentResults,
    qubits: QuantumElements,
    delays: QubitSweepPoints,
    frequencies: QubitSweepPoints,
    options: TuneUpAnalysisWorkflowOptions | None = None,
) -> None:
    """The Echo analysis Workflow.

    The workflow consists of the following steps:

    - [calculate_qubit_population]()
    - [fit_data]()
    - [extract_qubit_parameters]()
    - [plot_raw_complex_data_1d]()
    - [plot_raw_iq_heatmap_2d]()
    - [plot_population]()

    Arguments:
        result:
            The experiment results returned by the run_experiment task.
        qubits:
            The qubits on which to run the analysis. May be either a single qubit or
            a list of qubits. The UIDs of these qubits must exist in the result.
        delays:
            The delays that were swept over in the Echo experiment for
            each qubit. If `qubits` is a single qubit, `delays` must be an array of
            numbers. Otherwise, it must be a list of arrays of numbers.
        options:
            The options for building the workflow, passed as an instance of
            [TuneUpAnalysisWorkflowOptions]. See the docstring of this class for
            more details.

    Returns:
        WorkflowBuilder:
            The builder for the analysis workflow.

    Example:
        ```python
        result = analysis_workflow(
            results=results
            qubits=[q0, q1],
            delays=[
                np.linspace(0, 20e-6, 51),
                np.linspace(0, 30e-6, 52),
            ],
            options=analysis_workflow.options()
        ).run()
        ```
    """
    processed_data_dict = calculate_qubit_population_2d(
        qubits=qubits,
        result=result,
        sweep_points_1d=delays,
        sweep_points_2d=frequencies,
    )
    fit_results = fit_data(qubits, processed_data_dict)
    qubit_parameters = extract_qubit_parameters(qubits, fit_results)
    mid_fit_results = fit_mid_rate_data(qubits, qubit_parameters)
    with workflow.if_(options.do_plotting):
        with workflow.if_(options.do_raw_data_plotting):
            plot_raw_iq_heatmap_2d(qubits, processed_data_dict)
            plot_population_heatmap_2d(qubits, processed_data_dict)
        with workflow.if_(options.do_qubit_population_plotting):
            plot_population(
                qubits, processed_data_dict, fit_results, qubit_parameters
            )
            plot_t2_star_vs_frequency(qubits, qubit_parameters)
            plot_MID_rate_vs_frequency(qubits, qubit_parameters, mid_fit_results)
            plot_photon_numbers_vs_frequency(qubits, qubit_parameters)
    workflow.return_(qubit_parameters)


@workflow.task
def fit_data(
    qubits: QuantumElements,
    processed_data_dict: dict[str, dict[str, ArrayLike]],
    options: FitDataRamseyOptions | None = None,
) -> dict[str, dict[str, ArrayLike]]:
    """Perform a fit of an exponential-decay model to the data.

    Arguments:
        qubits:
            The qubits on which to run the analysis. May be either a single qubit or
            a list of qubits. The UIDs of these qubits must exist in the
            processed_data_dict.
        processed_data_dict: the processed data dictionary returned by process_raw_data
        options:
            The options class for this task as an instance of [FitDataRamseyOptions].
            See the docstring of this class for accepted options.

    Returns:
        dict with qubit UIDs as keys and the fit results for each qubit as values.
    """
    opts = FitDataRamseyOptions() if options is None else options
    qubits = validate_and_convert_qubits_sweeps(qubits)
    fit_results = {}
    if not opts.do_fitting:
        return fit_results

    for q in qubits:
        delays = processed_data_dict[q.uid]["sweep_points_1d"]
        cw_freqs = processed_data_dict[q.uid]["sweep_points_2d"]
        data_to_fit = processed_data_dict[q.uid][
            "population" if opts.do_rotation else "data_raw"
        ]
        num_cal_traces = processed_data_dict[q.uid].get("num_cal_traces", 0)
        echo_pulse_length = (
            q.parameters.ef_drive_length
            if "f" in opts.transition
            else q.parameters.ge_drive_length
        )
        swpts_fit = np.asarray(delays) + echo_pulse_length

        param_hints = {
            "amplitude": {"value": 0.5},
            "offset": {"value": 0.5, "vary": opts.do_pca or num_cal_traces == 0},
        }
        param_hints_user = opts.fit_parameters_hints
        if param_hints_user is None:
            param_hints_user = {}
        param_hints.update(param_hints_user)

        freq_fits: list[lmfit.model.ModelResult] = []
        for i, freq in enumerate(cw_freqs):
            try:
                fit_res = exponential_decay_fit(
                    swpts_fit,
                    data_to_fit[i, :],
                    param_hints=param_hints,
                )
                freq_fits.append(fit_res)
            except ValueError as err:
                workflow.log(
                    logging.ERROR,
                    "Fit failed for %s at CW freq %s Hz: %s.",
                    q.uid,
                    freq,
                    err,
                )
                freq_fits.append(None)

        fit_results[q.uid] = {
            "frequencies": cw_freqs,
            "fits": freq_fits,
        }

    return fit_results


@workflow.task
def extract_qubit_parameters(
    qubits: QuantumElements,
    fit_results: dict[str, dict[str, ArrayLike]],
    options: ExtractQubitParametersTransitionOptions | None = None,
) -> dict[str, dict[str, dict[str, int | float | unc.core.Variable | None]]]:
    """Extract the qubit parameters from the fit results.

    Arguments:
        qubits:
            The qubits on which to run the analysis. May be either a single qubit or
            a list of qubits.
        fit_results: the fit-results dictionary returned by fit_data
        options:
            The options for extracting the qubit parameters.
            See [ExtractQubitParametersTransitionOptions] for accepted options.

    Returns:
        dict with extracted qubit parameters and the previous values for those qubit
        parameters. The dictionary has the following form:
        ```python
        {
            "new_parameter_values": {
                q.uid: {
                    qb_param_name: qb_param_value
                },
            }
            "old_parameter_values": {
                q.uid: {
                    qb_param_name: qb_param_value
                },
            },
            "per_frequency": {
                q.uid: {
                    "frequencies": np.array([...]),
                    "t2_star": [unc.ufloat, ...],
                    "best_index": int,
                }
            }
        }
        ```
        If the do_fitting option is False, the new_parameter_values are not extracted
        and the function only returns the old_parameter_values.
        If a qubit uid is not found in fit_results, the new_parameter_values entry for
        that qubit is left empty.
    """
    opts = ExtractQubitParametersTransitionOptions() if options is None else options
    qubits = validate_and_convert_qubits_sweeps(qubits)
    qubit_parameters = {
        "old_parameter_values": {q.uid: {} for q in qubits},
        "new_parameter_values": {q.uid: {} for q in qubits},
        "per_frequency": {q.uid: {} for q in qubits},
    }

    for i, q in enumerate(qubits):
        # Store the qubit frequency and T2_star values
        old_qb_freq = (
            q.parameters.resonance_frequency_ef
            if "f" in opts.transition
            else q.parameters.resonance_frequency_ge
        )
        old_t2_star = (
            q.parameters.ef_T2_star
            if "f" in opts.transition
            else q.parameters.ge_T2_star
        )
        qubit_parameters["old_parameter_values"][q.uid] = {
            f"resonance_frequency_{opts.transition}": old_qb_freq,
            f"{opts.transition}_T2_star": old_t2_star,
        }

        if not (opts.do_fitting and q.uid in fit_results):
            continue

        freq_data = fit_results[q.uid]
        cw_freqs = np.array(freq_data["frequencies"])
        fits = freq_data["fits"]

        t2_list = []
        fit_success_list = []
        for fit_res in fits:
            min_t2_star = 0.1e-6  # 0.1 us in seconds
            max_rel_t2_err = 0.5
            if fit_res is None:
                fit_success_list.append(False)
                t2_list.append(unc.ufloat(np.nan, np.nan))
                continue
            decay_rate = fit_res.params["decay_rate"].value
            decay_rate_stderr = fit_res.params["decay_rate"].stderr
            if decay_rate is None or not np.isfinite(decay_rate) or decay_rate <= 0:
                fit_success_list.append(False)
                t2_list.append(unc.ufloat(np.nan, np.nan))
                continue
            t2_value = 1 / decay_rate
            t2_stderr = None
            if decay_rate_stderr is not None and np.isfinite(decay_rate_stderr):
                t2_stderr = decay_rate_stderr / (decay_rate**2)
            rel_t2_err = np.inf
            if t2_stderr is not None and np.isfinite(t2_stderr) and t2_value != 0:
                rel_t2_err = abs(t2_stderr / t2_value)
            is_success = (
                fit_res is not None
                and getattr(fit_res, "success", True)
                and np.isfinite(t2_value)
                and t2_value > min_t2_star
                and rel_t2_err <= max_rel_t2_err
            )
            fit_success_list.append(bool(is_success))
            if not is_success:
                t2_list.append(unc.ufloat(np.nan, np.nan))
                continue
            t2_star = unc.ufloat(t2_value, t2_stderr or 0.0)
            t2_list.append(t2_star)

        # pick the frequency with the largest T2* (ignoring NaNs) as the update target
        t2_values = np.array([t.n for t in t2_list], dtype=float)
        finite_mask = np.isfinite(t2_values)
        best_index = int(np.nanargmax(t2_values)) if finite_mask.any() else 0

        # find local minima (T2* dips). Use only successful fits and ignore edges.
        minima_indices: list[int] = []
        if len(t2_values) >= 3:
            safe_t2 = np.array(t2_values, dtype=float)
            safe_t2[~np.isfinite(safe_t2)] = np.inf
            safe_t2[~np.asarray(fit_success_list, dtype=bool)] = np.inf
            for idx in range(1, len(safe_t2) - 1):
                if (
                    np.isfinite(safe_t2[idx])
                    and safe_t2[idx] <= safe_t2[idx - 1]
                    and safe_t2[idx] <= safe_t2[idx + 1]
                ):
                    minima_indices.append(idx)
        # Keep at most two deepest minima (smallest T2*)
        if minima_indices:
            minima_indices = sorted(minima_indices, key=lambda i: t2_values[i])[:2]

        # Estimate measurement-induced dephasing rate Γm by subtracting a baseline
        # dephasing rate computed from the median of the top 10% largest T2* values.
        valid_mask = finite_mask & (t2_values > 0)
        if valid_mask.any():
            valid_indices = np.where(valid_mask)[0]
            valid_t2 = t2_values[valid_mask]
            top_thresh = np.nanpercentile(valid_t2, 90.0)
            top_mask = valid_t2 >= top_thresh
            if np.count_nonzero(top_mask) == 0:
                top_mask = np.ones_like(valid_t2, dtype=bool)
            top_t2 = valid_t2[top_mask]
            baseline_target = float(np.nanmedian(top_t2))
            candidate_indices = valid_indices[top_mask]
            closest = np.nanargmin(np.abs(t2_values[candidate_indices] - baseline_target))
            baseline_actual_index = int(candidate_indices[int(closest)])
            baseline_t2 = t2_list[baseline_actual_index]
        else:
            baseline_actual_index = 0
            baseline_t2 = unc.ufloat(np.nan, np.nan)
        baseline_gamma = (1 / baseline_t2) if valid_mask.any() else unc.ufloat(np.nan, np.nan)

        mid_rate_list: list[unc.core.Variable] = []
        for t2_star in t2_list:
            if not np.isfinite(t2_star.n) or t2_star.n <= 0 or not np.isfinite(baseline_gamma.n):
                mid_rate_list.append(unc.ufloat(np.nan, np.nan))
                continue
            gamma_total = 1 / t2_star
            gamma_mid = gamma_total - baseline_gamma
            if gamma_mid.n < 0:
                gamma_mid = unc.ufloat(0.0, gamma_mid.s)
            mid_rate_list.append(gamma_mid)

        qubit_parameters["per_frequency"][q.uid] = {
            "frequencies": cw_freqs,
            "t2_star": t2_list,
            "best_index": best_index,
            "mid_rate": mid_rate_list,
            "mid_baseline_index": baseline_actual_index,
            "fit_success": fit_success_list,
            "minima_indices": minima_indices,
            "minima_frequencies": [cw_freqs[i] for i in minima_indices],
            "minima_t2_star": [t2_list[i] for i in minima_indices],
        }

        if finite_mask.any():
            qubit_parameters["new_parameter_values"][q.uid] = {
                f"{opts.transition}_T2_star": t2_list[best_index],
                "selected_cw_frequency": cw_freqs[best_index],
            }

    return qubit_parameters


@workflow.task
def fit_mid_rate_data(
    qubits: QuantumElements,
    qubit_parameters: dict[str, dict[str, dict[str, int | float | unc.core.Variable]]],
    param_hints: dict[str, dict[str, float | bool | str]] | None = None,
) -> dict[str, object]:
    """Fit the extracted MID rate Γm as a function of CW frequency."""
    qubits = validate_and_convert_qubits_sweeps(qubits)
    fit_results: dict[str, object] = {}

    for q in qubits:
        per_freq = qubit_parameters["per_frequency"].get(q.uid, {})
        if not per_freq:
            continue
        freqs = np.asarray(per_freq.get("frequencies", []), dtype=float)
        mid_rates = per_freq.get("mid_rate", [])
        if len(freqs) == 0 or len(mid_rates) == 0:
            continue

        mid_nom = np.array([rate.n for rate in mid_rates], dtype=float)
        fit_success = np.asarray(per_freq.get("fit_success", []), dtype=bool)
        if fit_success.size != mid_nom.size:
            fit_success = np.ones_like(mid_nom, dtype=bool)
        mask = np.isfinite(freqs) & np.isfinite(mid_nom) & (mid_nom >= 0) & fit_success
        if np.count_nonzero(mask) < 4:  # noqa: PLR2004
            logging.warning("Not enough finite MID-rate points to fit for %s.", q.uid)
            fit_results[q.uid] = None
            continue

        try:
            fit_res = coherent_spec_fit(freqs[mask], mid_nom[mask], param_hints=param_hints)
            fit_results[q.uid] = fit_res
            kappa = fit_res.best_values.get("kappa", np.nan)
            chi = fit_res.best_values.get("chi", np.nan)
            eps = fit_res.best_values.get("epsilon_rf", np.nan)
            bare = fit_res.best_values.get("bare_freq", np.nan)
            eta = fit_res.best_values.get("eta", 0.0)
            if np.all(np.isfinite([kappa, chi, eps, bare])):
                n_plus, n_minus = coherent_spec_photon_numbers(
                    freqs, kappa, chi, eps, bare, eta
                )
            else:
                n_plus = np.full_like(freqs, np.nan, dtype=float)
                n_minus = np.full_like(freqs, np.nan, dtype=float)
            per_freq["n_plus"] = n_plus
            per_freq["n_minus"] = n_minus
        except ValueError as err:
            logging.error("MID fit failed for %s: %s.", q.uid, err)
            fit_results[q.uid] = None
            per_freq["n_plus"] = np.full_like(freqs, np.nan, dtype=float)
            per_freq["n_minus"] = np.full_like(freqs, np.nan, dtype=float)

    return fit_results


@workflow.task
def plot_photon_numbers_vs_frequency(
    qubits: QuantumElements,
    qubit_parameters: dict[str, dict[str, dict[str, int | float | unc.core.Variable]]],
) -> dict[str, mpl.figure.Figure]:
    """Plot extracted photon numbers n_plus/n_minus as a function of CW frequency."""
    qubits = validate_and_convert_qubits_sweeps(qubits)
    figures: dict[str, mpl.figure.Figure] = {}

    for q in qubits:
        per_freq = qubit_parameters["per_frequency"].get(q.uid, {})
        if not per_freq:
            continue
        freqs = np.asarray(per_freq.get("frequencies", []), dtype=float)
        n_plus = np.asarray(per_freq.get("n_plus", []), dtype=float)
        n_minus = np.asarray(per_freq.get("n_minus", []), dtype=float)
        if freqs.size == 0 or n_plus.size == 0 or n_minus.size == 0:
            continue

        mask = np.isfinite(freqs) & np.isfinite(n_plus) & np.isfinite(n_minus)
        if np.count_nonzero(mask) == 0:
            continue

        fig, ax = plt.subplots()
        ax.plot(freqs[mask] / 1e9, n_plus[mask], "o-", label="$\\bar{n}_+$")
        ax.plot(freqs[mask] / 1e9, n_minus[mask], "o-", label="$\\bar{n}_-$")
        ax.set_xlabel("CW frequency (GHz)")
        ax.set_ylabel("Photon number")
        ax.set_title(timestamped_title(f"Photon numbers vs CW freq ({q.uid})"))
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        figures[q.uid] = fig

    return figures


@workflow.task
def plot_population(
    qubits: QuantumElements,
    processed_data_dict: dict[str, dict[str, ArrayLike]],
    fit_results: dict[str, dict[str, ArrayLike]] | None,
    qubit_parameters: dict[
        str,
        dict[str, dict[str, int | float | unc.core.Variable | None]],
    ]
    | None,
    options: PlotPopulationOptions | None = None,
) -> dict[str, mpl.figure.Figure]:
    """Create the Echo plots.

    Arguments:
        qubits:
            The qubits on which to run the analysis. May be either a single qubit or
            a list of qubits. The UIDs of these qubits must exist in the
            processed_data_dict.
        processed_data_dict: the processed data dictionary returned by process_raw_data
        fit_results: the fit-results dictionary returned by fit_data
        qubit_parameters: the qubit-parameters dictionary returned by
            extract_qubit_parameters
    options:
        The options class for this task as an instance of [PlotPopulationOptions].
        See the docstring of this class for accepted options.

    Returns:
        dict with qubit UIDs as keys and the figures for each qubit as values.

        If a qubit uid is not found in fit_results, the fit and the textbox with the
        extracted qubit parameters are not plotted.
    """
    opts = PlotPopulationOptions() if options is None else options
    qubits = validate_and_convert_qubits_sweeps(qubits)
    figures = {}
    for i, q in enumerate(qubits):
        sweep_points = processed_data_dict[q.uid]["sweep_points_1d"]
        data_full = processed_data_dict[q.uid][
            "population" if opts.do_rotation else "data_raw"
        ]
        # plot the trace at the selected (or first) CW frequency as a sanity check
        per_freq = qubit_parameters["per_frequency"][q.uid] if qubit_parameters else {}
        best_index = per_freq.get("best_index", 0)
        minima_indices = per_freq.get("minima_indices", [])
        indices_to_plot = [int(best_index)]
        for idx in minima_indices:
            idx = int(idx)
            if idx not in indices_to_plot:
                indices_to_plot.append(idx)

        num_cal_traces = processed_data_dict[q.uid]["num_cal_traces"]

        for cw_index in indices_to_plot:
            data = data_full[cw_index, :] if data_full.ndim > 1 else data_full

            fig, ax = plt.subplots()
            ax.set_title(timestamped_title(f"Echo {q.uid} @ CW idx {cw_index}"))
            ax.set_xlabel("Pulse Separation, $\\tau$ ($\\mu$s)")
            ax.set_ylabel(
                "Principal Component (a.u)"
                if (num_cal_traces == 0 or opts.do_pca)
                else f"$|{opts.cal_states[-1]}\\rangle$-State Population",
            )
            ax.plot(sweep_points * 1e6, data, "o", zorder=2, label="data")
            if processed_data_dict[q.uid]["num_cal_traces"] > 0:
                # plot lines at the calibration traces
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
                freq_data = fit_results[q.uid]
                cw_freqs = freq_data["frequencies"]
                fit_res_qb = freq_data["fits"][cw_index]
                if fit_res_qb is None:
                    if cw_index == best_index:
                        figures[q.uid] = fig
                    else:
                        figures[f"{q.uid}_minima_{cw_index}"] = fig
                    continue
                # plot fit
                swpts_fine = np.linspace(sweep_points[0], sweep_points[-1], 501)
                ax.plot(
                    swpts_fine * 1e6,
                    fit_res_qb.model.func(swpts_fine, **fit_res_qb.best_values),
                    "r-",
                    zorder=1,
                    label="fit",
                )

                if qubit_parameters is not None and len(
                    qubit_parameters["new_parameter_values"][q.uid]
                ) > 0:
                    # textbox (include for minima as well)
                    old_qb_freq = qubit_parameters["old_parameter_values"][q.uid][
                        f"resonance_frequency_{opts.transition}"
                    ]
                    old_t2_star = qubit_parameters["old_parameter_values"][q.uid][
                        f"{opts.transition}_T2_star"
                    ]
                    new_t2_star = qubit_parameters["new_parameter_values"][q.uid][
                        f"{opts.transition}_T2_star"
                    ]
                    decay_rate = fit_res_qb.params["decay_rate"].value
                    decay_rate_err = fit_res_qb.params["decay_rate"].stderr
                    if decay_rate is not None and np.isfinite(decay_rate) and decay_rate > 0:
                        fit_t2 = unc.ufloat(
                            1 / decay_rate,
                            (decay_rate_err / (decay_rate**2))
                            if decay_rate_err is not None
                            and np.isfinite(decay_rate_err)
                            else 0.0,
                        )
                    else:
                        fit_t2 = unc.ufloat(np.nan, np.nan)
                    display_t2 = new_t2_star if cw_index == best_index else fit_t2
                    textstr = (
                        f"Old qubit frequency: {old_qb_freq / 1e9:.6f} GHz"
                    )
                    textstr += (
                        f"\n$T_2^*$: {display_t2.nominal_value * 1e6:.4f} $\\mu$s $\\pm$ "
                        f"{display_t2.std_dev * 1e6:.4f} $\\mu$s"
                    )
                    textstr += f"\nOld $T_2^*$: {old_t2_star * 1e6:.4f} $\\mu$s"
                    textstr += (
                        f"\nCW freq: {cw_freqs[cw_index] / 1e9:.6f} GHz (index {cw_index})"
                    )
                    if cw_index != best_index:
                        textstr = f"[minima]\n{textstr}"
                    ax.text(0, -0.15, textstr, ha="left", va="top", transform=ax.transAxes)
            ax.legend(
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                handlelength=1.5,
                frameon=False,
            )

            if opts.save_figures:
                artifact_name = (
                    f"Echo_{q.uid}"
                    if cw_index == best_index
                    else f"Echo_{q.uid}_minima_{cw_index}"
                )
                workflow.save_artifact(artifact_name, fig)

            if opts.close_figures:
                plt.close(fig)

            if cw_index == best_index:
                figures[q.uid] = fig
            else:
                figures[f"{q.uid}_minima_{cw_index}"] = fig

    return figures


@workflow.task
def plot_t2_star_vs_frequency(
    qubits: QuantumElements,
    qubit_parameters: dict[str, dict[str, dict[str, int | float | unc.core.Variable]]],
) -> dict[str, mpl.figure.Figure]:
    """Plot extracted $T_2^*$ as a function of the CW drive frequency."""

    figures: dict[str, mpl.figure.Figure] = {}
    qubits = validate_and_convert_qubits_sweeps(qubits)
    for q in qubits:
        per_freq = qubit_parameters["per_frequency"].get(q.uid, {})
        if not per_freq:
            continue
        freqs = np.array(per_freq["frequencies"])
        t2_list = per_freq["t2_star"]
        if len(t2_list) == 0:
            continue

        t2_vals = np.array([t.n for t in t2_list]) * 1e6
        t2_errs = np.array([t.s for t in t2_list]) * 1e6

        fig, ax = plt.subplots()
        ax.errorbar(
            freqs / 1e9,
            t2_vals,
            yerr=t2_errs,
            fmt="o-",
            capsize=3,
            label="$T_2^*$",
        )
        # Mark which point was used as the MID baseline (largest T2*)
        baseline_index = per_freq.get("mid_baseline_index", None)
        if baseline_index is not None and 0 <= int(baseline_index) < len(freqs):
            baseline_freq = freqs[int(baseline_index)]
            baseline_t2 = t2_vals[int(baseline_index)]
            baseline_t2_ufloat = t2_list[int(baseline_index)]
            baseline_gamma = (
                (1 / baseline_t2_ufloat) if np.isfinite(baseline_t2_ufloat.n) else None
            )
            ax.axhline(
                baseline_t2,
                color="k",
                linestyle="--",
                linewidth=1.0,
                alpha=0.7,
                label="Baseline (top 10% median $T_2^*$)",
            )
            ax.plot(
                baseline_freq / 1e9,
                baseline_t2,
                "ks",
                markersize=5,
                label="Baseline point",
            )
            if baseline_gamma is not None and np.isfinite(baseline_gamma.n):
                ax.text(
                    0.98,
                    0.02,
                    f"Baseline $\\Gamma_0$: {baseline_gamma.n / 1e6:.3f} MHz",
                    ha="right",
                    va="bottom",
                    transform=ax.transAxes,
                )
        ax.set_xlabel("CW frequency (GHz)")
        ax.set_ylabel("$T_2^*$ ($\\mu$s)")
        ax.set_title(timestamped_title(f"Coherence vs CW freq ({q.uid})"))
        ax.grid(True, alpha=0.3)
        ax.legend()

        figures[q.uid] = fig
    return figures
########################### 
@workflow.task
def plot_MID_rate_vs_frequency(
    qubits: QuantumElements,
    qubit_parameters: dict[str, dict[str, dict[str, int | float | unc.core.Variable]]],
    mid_fit_results: dict[str, object] | None = None,
) -> dict[str, mpl.figure.Figure]:
    r"""Plot extracted MID rate $\Gamma_m$ as a function of the CW drive frequency."""
    qubits = validate_and_convert_qubits_sweeps(qubits)
    figures: dict[str, mpl.figure.Figure] = {}

    for q in qubits:
        per_freq = qubit_parameters["per_frequency"].get(q.uid, {})
        if not per_freq:
            continue
        freqs = np.asarray(per_freq.get("frequencies", []), dtype=float)
        mid_rates = per_freq.get("mid_rate", [])
        if len(freqs) == 0 or len(mid_rates) == 0:
            continue

        mid_nom = np.array([rate.n for rate in mid_rates], dtype=float)
        mid_err = np.array([rate.s for rate in mid_rates], dtype=float)
        fit_success = np.asarray(per_freq.get("fit_success", []), dtype=bool)
        if fit_success.size != mid_nom.size:
            fit_success = np.ones_like(mid_nom, dtype=bool)
        mask = np.isfinite(freqs) & np.isfinite(mid_nom) & fit_success
        if np.count_nonzero(mask) == 0:
            continue

        f_plot = freqs[mask]
        g_plot = mid_nom[mask]
        g_err = mid_err[mask]

        fig, ax = plt.subplots()
        ax.errorbar(
            f_plot / 1e9,
            g_plot / 1e6,
            yerr=g_err / 1e6,
            fmt="o",
            capsize=3,
            label="$\\Gamma_m$ (extracted)",
        )
        ax.set_xlabel("CW frequency (GHz)")
        ax.set_ylabel("$\\Gamma_m$ (MHz)")
        ax.set_title(timestamped_title(f"MID rate vs CW freq ({q.uid})"))
        ax.grid(True, alpha=0.3)

        fit_res = mid_fit_results.get(q.uid) if mid_fit_results else None
        if fit_res is not None:
            swpts_fine = np.linspace(np.min(f_plot), np.max(f_plot), 801)
            fit_curve = fit_res.model.func(swpts_fine, **fit_res.best_values)
            ax.plot(
                swpts_fine / 1e9,
                fit_curve / 1e6,
                "r-",
                label="MID model fit",
            )

            kappa_fit = fit_res.best_values.get("kappa", np.nan)
            chi_fit = fit_res.best_values.get("chi", np.nan)
            eps_fit = fit_res.best_values.get("epsilon_rf", np.nan)
            eta_fit = fit_res.best_values.get("eta", np.nan)
            bare_fit = fit_res.best_values.get("bare_freq", np.nan)
            textstr = (
                f"$\\kappa$: {kappa_fit / 1e6:.3f} MHz\n"
                f"$\\chi$: {chi_fit / 1e6:.3f} MHz\n"
                f"$\\epsilon_{{rf}}$: {eps_fit / 1e6:.3f} MHz\n"
                f"$\\eta$: {eta_fit:.3f}\n"
                f"$f_r$: {bare_fit / 1e9:.6f} GHz"
            )
            ax.text(0.02, 0.98, textstr, ha="left", va="top", transform=ax.transAxes)

        ax.legend(loc="best")
        figures[q.uid] = fig

    return figures



######################
@workflow.task
def plot_population_heatmap_2d(
    qubits: QuantumElements,
    processed_data_dict: dict[str, dict[str, ArrayLike]],
) -> dict[str, mpl.figure.Figure]:
    """Plot the extracted population as a 2D map over delays and CW frequencies."""

    qubits = validate_and_convert_qubits_sweeps(qubits)
    figures: dict[str, mpl.figure.Figure] = {}

    for q in qubits:
        data = processed_data_dict[q.uid]["population"]  # shape: (freq, delay)
        taus = np.array(processed_data_dict[q.uid]["sweep_points_1d"])
        freqs = np.array(processed_data_dict[q.uid]["sweep_points_2d"])

        fig, ax = plt.subplots()
        im = ax.imshow(
            data,
            aspect="auto",
            origin="lower",
            extent=[taus[0] * 1e6, taus[-1] * 1e6, freqs[0] / 1e9, freqs[-1] / 1e9],
            vmin=0,
            vmax=1,
        )
        ax.set_xlabel("Pulse Separation, $\\tau$ ($\\mu$s)")
        ax.set_ylabel("CW frequency (GHz)")
        ax.set_title(timestamped_title(f"Population {q.uid}"))
        fig.colorbar(im, ax=ax, label="Population |e>")
        figures[q.uid] = fig
    return figures


@workflow.task
def plot_raw_iq_heatmap_2d(
    qubits: QuantumElements,
    processed_data_dict: dict[str, dict[str, ArrayLike]],
) -> dict[str, mpl.figure.Figure]:
    """Plot the raw I/Q data as 2D heatmaps over delays and CW frequencies."""

    qubits = validate_and_convert_qubits_sweeps(qubits)
    figures: dict[str, mpl.figure.Figure] = {}

    for q in qubits:
        raw_data = processed_data_dict[q.uid]["data_raw"]  # shape: (freq, delay)
        taus = np.array(processed_data_dict[q.uid]["sweep_points_1d"])
        freqs = np.array(processed_data_dict[q.uid]["sweep_points_2d"])

        real_part = np.real(raw_data)
        imag_part = np.imag(raw_data)
        color_limit = np.nanmax(
            np.abs(np.concatenate([real_part.ravel(), imag_part.ravel()]))
        )
        if not np.isfinite(color_limit) or color_limit == 0:
            color_limit = 1.0

        fig_width, fig_height = plt.rcParams["figure.figsize"]
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(1.3 * fig_width, fig_height),
            sharex=True,
            sharey=True,
        )
        extent = [taus[0] * 1e6, taus[-1] * 1e6, freqs[0] / 1e9, freqs[-1] / 1e9]

        for ax, data, title in zip(
            axes,
            (real_part, imag_part),
            ("I (Real)", "Q (Imag)"),
        ):
            im = ax.imshow(
                data,
                aspect="auto",
                origin="lower",
                extent=extent,
                vmin=-color_limit,
                vmax=color_limit,
                cmap="RdBu_r",
            )
            ax.set_title(title)
            ax.set_xlabel("Pulse Separation, $\\tau$ ($\\mu$s)")
            fig.colorbar(im, ax=ax, label="Amplitude (arb.)")

        axes[0].set_ylabel("CW frequency (GHz)")
        fig.suptitle(timestamped_title(f"Raw IQ {q.uid}"))
        figures[q.uid] = fig
    return figures

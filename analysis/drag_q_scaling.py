# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""This module defines the analysis for a DRAG calibration experiment.

The experiment is defined in laboneq_applications.experiments. See the docstring of
this module for more details about the experiment and its parameters.

In this analysis, we first interpret the raw data into qubit populations using
principle-component analysis or rotation and projection on the measured calibration
states. Then we fit the measured qubit population as a function of the beta parameter
and determine the optimal beta parameter. Finally, we plot the data and the fit.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np
import uncertainties as unc
from laboneq import workflow
from laboneq.simple import dsl

from laboneq_applications.analysis.calibration_traces_rotation import (
    CalculateQubitPopulationOptions,
    calculate_population_1d,
    extract_raw_data_dict,
)
from laboneq_applications.analysis.fitting_helpers import (
    fit_data_lmfit,
    linear,
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
from laboneq_applications.core.validation import (
    validate_and_convert_qubits_sweeps,
    validate_result,
)

if TYPE_CHECKING:
    import lmfit
    import matplotlib as mpl
    from laboneq.workflow.tasks.run_experiment import RunExperimentResults
    from numpy.typing import ArrayLike

    from laboneq_applications.typing import QuantumElements, QubitSweepPoints


ALLXY_SEQUENCE_IDS: tuple[str, ...] = (
    "allxy_00_II",
    "allxy_01_XX",
    "allxy_02_YY",
    "allxy_03_XY",
    "allxy_04_YX",
    "allxy_05_xI",
    "allxy_06_yI",
    "allxy_07_xy",
    "allxy_08_yx",
    "allxy_09_xY",
    "allxy_10_yX",
    "allxy_11_Xy",
    "allxy_12_Yx",
    "allxy_13_xX",
    "allxy_14_Xx",
    "allxy_15_yY",
    "allxy_16_Yy",
    "allxy_17_XI",
    "allxy_18_YI",
    "allxy_19_xx",
    "allxy_20_yy",
)
ALLXY_TARGETS: np.ndarray = np.array(
    [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        1.0,
        1.0,
        1.0,
        1.0,
    ],
    dtype=float,
)


def _safe_ufloat(value: float, stderr: float | None) -> unc.core.Variable:
    std = 0.0 if stderr is None or not np.isfinite(stderr) else float(abs(stderr))
    return unc.ufloat(float(value), std)


def _get_old_beta_for_transition(q: object, transition: str) -> float:
    return (
        q.parameters.ef_drive_pulse["beta"]
        if "f" in transition
        else q.parameters.ge_drive_pulse["beta"]
    )


def _normalize_single_qubit_sweeps(
    qubits: QuantumElements,
    q_scalings: QubitSweepPoints,
) -> QubitSweepPoints:
    single_qubit_input = not isinstance(qubits, Sequence) or hasattr(qubits, "uid")
    if not single_qubit_input:
        return q_scalings
    if isinstance(q_scalings, np.ndarray):
        return q_scalings
    if isinstance(q_scalings, Sequence) and not isinstance(q_scalings, (str, bytes)):
        if len(q_scalings) == 1:
            first = q_scalings[0]
            if isinstance(first, np.ndarray):
                return first
            if isinstance(first, Sequence) and not isinstance(first, (str, bytes)):
                return np.asarray(first, dtype=float)
    return q_scalings


@workflow.workflow
def analysis_workflow(
    result: RunExperimentResults,
    qubits: QuantumElements,
    q_scalings: QubitSweepPoints,
    sequence_set: Literal["xy3", "allxy21"] = "xy3",
    options: TuneUpAnalysisWorkflowOptions | None = None,
) -> None:
    """The analysis Workflow for the DRAG quadrature-scaling calibration.

    The workflow consists of the following steps:

    - [calculate_qubit_population_for_pulse_ids]()
    - [fit_data]()
    - [extract_qubit_parameters]()
    - [plot_raw_complex_data_1d]()
    - [plot_population]()

    Arguments:
        result:
            The experiment results returned by the run_experiment task.
        qubits:
            The qubits on which to run the analysis. May be either a single qubit or
            a list of qubits. The UIDs of these qubits must exist in the result.
        q_scalings:
            The quadrature scaling factors that were swept over in the experiment for
            each qubit. If `qubits` is a single qubit, `q_scalings` must be a list of
            numbers or an array. Otherwise, it must be a list of lists of numbers or
            arrays.
        options:
            The options for building the workflow, passed as an instance of
            [TuneUpAnalysisWorkflowOptions]. See the docstring of this class for
            more details.

    Returns:
        WorkflowBuilder:
            The builder for the analysis workflow.

    Example:
        ```python
        options = analysis_workflow.options()
        options.close_figures(False)
        result = analysis_workflow(
            results=results
            qubits=[q0, q1],
            q_scalings=[
                np.linspace(-0.05, 0.05, 11),
                np.linspace(-0.05, 0.05, 11),
            ],
            options=options,
        ).run()
        ```
    """
    q_scalings = _normalize_single_qubit_sweeps(qubits, q_scalings)
    processed_data_dict = calculate_qubit_population_for_pulse_ids(
        qubits, result, q_scalings
    )
    fit_results = {}
    allxy_beta_results = {}
    if sequence_set == "xy3":
        fit_results = fit_data(qubits, processed_data_dict)
        qubit_parameters = extract_qubit_parameters(qubits, fit_results)
    elif sequence_set == "allxy21":
        allxy_beta_results = estimate_beta_from_allxy_scores(qubits, processed_data_dict)
        qubit_parameters = extract_qubit_parameters_from_allxy(
            qubits, allxy_beta_results
        )
    else:
        raise ValueError(
            f"Unsupported sequence_set: {sequence_set}. Choose xy3 or allxy21."
        )
    with workflow.if_(options.do_plotting):
        with workflow.if_(options.do_raw_data_plotting):
            plot_raw_complex_data_1d(
                qubits,
                result,
                q_scalings,
                xlabel="DRAG Quadrature Scaling Factor, $\\beta$",
            )
        with workflow.if_(options.do_qubit_population_plotting):
            plot_population(
                qubits,
                processed_data_dict,
                fit_results,
                qubit_parameters,
            )
            if sequence_set == "allxy21":
                plot_allxy_scores(qubits, allxy_beta_results, qubit_parameters)
    workflow.return_(qubit_parameters)


@workflow.task
def calculate_qubit_population_for_pulse_ids(
    qubits: QuantumElements,
    result: RunExperimentResults,
    q_scalings: QubitSweepPoints,
    options: CalculateQubitPopulationOptions | None = None,
) -> dict[str, dict[str, dict[str, ArrayLike]]]:
    """Processes the raw data from the experiment result.

     The data is processed in the following way:

     - If calibration traces were used in the experiment, the raw data is rotated based
     on the calibration traces.
     See [calibration_traces_rotation.py/rotate_data_to_cal_trace_results] for more
     details.
     - If no calibration traces were used in the experiment, or do_pca = True is passed
     in options, principal-component analysis is performed on the data.
     See [calibration_traces_rotation.py/principal_component_analysis] for more details.

    Arguments:
        qubits:
            The qubits on which the experiments was run. May be either
            a single qubit or a list of qubits.
        result: the result of the experiment, returned by the run_experiment task.
        q_scalings:
            The quadrature scaling factors that were swept over in the experiment for
            each qubit. If `qubits` is a single qubit, `q_scalings` must be a list of
            numbers or an array. Otherwise, it must be a list of lists of numbers or
            arrays.
        options:
            The options for processing the raw data.
            See [CalculateQubitPopulationOptions] for accepted options.

    Returns:
        dict with qubit UIDs as keys. The dictionary of processed data for each qubit
        further has "y180" and "my180" as keys.
        See [calibration_traces_rotation.py/calculate_population_1d] for what this
        dictionary looks like.

    Raises:
        TypeError:
            If result is not an instance of RunExperimentResults.
    """
    validate_result(result)
    opts = CalculateQubitPopulationOptions() if options is None else options
    qubits, q_scalings = validate_and_convert_qubits_sweeps(qubits, q_scalings)
    processed_data_dict = {}
    for q, qscales in zip(qubits, q_scalings):
        processed_data_dict[q.uid] = {}
        if opts.use_cal_traces:
            calibration_traces = [
                result[dsl.handles.calibration_trace_handle(q.uid, cs)].data
                for cs in opts.cal_states
            ]
            do_pca = opts.do_pca
        else:
            calibration_traces = []
            do_pca = True

        for pulse_id in result[dsl.handles.result_handle(q.uid)]:
            raw_data = result[dsl.handles.result_handle(q.uid, suffix=pulse_id)].data
            if opts.do_rotation:
                data_dict = calculate_population_1d(
                    raw_data,
                    qscales,
                    calibration_traces,
                    do_pca=do_pca,
                )
            else:
                data_dict = extract_raw_data_dict(raw_data, qscales, calibration_traces)
            processed_data_dict[q.uid][pulse_id] = data_dict
    return processed_data_dict


@workflow.task
def fit_data(
    qubits: QuantumElements,
    processed_data_dict: dict[str, dict[str, dict[str, ArrayLike]]],
    options: FitDataOptions | None = None,
) -> dict[str, dict[str, lmfit.model.ModelResult]]:
    """Perform a fit of a linear model to the data.

    Arguments:
        qubits:
            The qubits on which to run the analysis. May be either a single qubit or
            a list of qubits. The UIDs of these qubits must exist in the
            processed_data_dict.
        processed_data_dict: the processed data dictionary returned by process_raw_data
        options:
            The options class for this task as an instance of [FitDataOptions]. See
            the docstring of this class for accepted options.

    Returns:
        dict with qubit UIDs as keys, "y180"/"my180" as subkeys and the fit results
        for each qubit as values.
    """
    opts = FitDataOptions() if options is None else options
    qubits = validate_and_convert_qubits_sweeps(qubits)
    fit_results = {}
    if not opts.do_fitting:
        return fit_results

    for q in qubits:
        fit_results[q.uid] = {}
        for pulse_id in processed_data_dict[q.uid]:
            swpts_fit = processed_data_dict[q.uid][pulse_id]["sweep_points"]
            data_to_fit = processed_data_dict[q.uid][pulse_id][
                "population" if opts.do_rotation else "data_raw"
            ]
            if pulse_id == "xx":
                param_hints = {
                    "gradient": {"value": 0, "vary": False},
                    "intercept": {"value": np.mean(data_to_fit)},
                }
            else:
                gradient = (data_to_fit[-1] - data_to_fit[0]) / (
                    swpts_fit[-1] - swpts_fit[0]
                )
                param_hints = {
                    "gradient": {"value": gradient},
                    "intercept": {"value": data_to_fit[-1] - gradient * swpts_fit[-1]},
                }
            param_hints_user = opts.fit_parameters_hints
            if param_hints_user is None:
                param_hints_user = {}
            param_hints.update(param_hints_user)
            try:
                fit_res = fit_data_lmfit(
                    model=linear,
                    x=swpts_fit,
                    y=data_to_fit,
                    param_hints=param_hints,
                )
                fit_results[q.uid][pulse_id] = fit_res
            except ValueError as err:
                workflow.log(logging.ERROR, "Fit failed for %s: %s.", q.uid, err)

    return fit_results


@workflow.task
def estimate_beta_from_allxy_scores(
    qubits: QuantumElements,
    processed_data_dict: dict[str, dict[str, dict[str, ArrayLike]]],
    options: FitDataOptions | None = None,
    min_score_drop: float = 0.05,
) -> dict[str, dict[str, bool | str | float | unc.core.Variable | ArrayLike | None]]:
    """Estimate beta by minimizing an ALLXY score over a fine beta sweep."""
    opts = FitDataOptions() if options is None else options
    qubits = validate_and_convert_qubits_sweeps(qubits)
    beta_results: dict[
        str,
        dict[str, bool | str | float | unc.core.Variable | ArrayLike | None],
    ] = {}
    data_key = "population" if opts.do_rotation else "data_raw"
    for q in qubits:
        qid = q.uid
        beta_results[qid] = {
            "accepted": False,
            "reason": "No ALLXY data available.",
            "beta_opt": None,
            "beta_nominal": np.nan,
            "beta_std": np.nan,
            "score_drop": np.nan,
            "score_min": np.nan,
            "score_reference": np.nan,
            "curvature": np.nan,
            "sweep_points": np.array([]),
            "scores": np.array([]),
        }
        if qid not in processed_data_dict:
            workflow.log(logging.ERROR, "No processed data for qubit %s.", qid)
            continue
        missing_ids = [
            sequence_id
            for sequence_id in ALLXY_SEQUENCE_IDS
            if sequence_id not in processed_data_dict[qid]
        ]
        if missing_ids:
            beta_results[qid]["reason"] = (
                "Missing ALLXY sequences: " + ",".join(sorted(missing_ids))
            )
            workflow.log(
                logging.ERROR,
                "ALLXY beta estimate failed for %s due to missing sequences.",
                qid,
            )
            continue

        sweep_points = np.asarray(
            processed_data_dict[qid][ALLXY_SEQUENCE_IDS[0]]["sweep_points"], dtype=float
        )
        if sweep_points.size < 3:
            beta_results[qid]["reason"] = "Need at least 3 beta points."
            continue

        data_matrix = []
        invalid_shape = False
        for sequence_id in ALLXY_SEQUENCE_IDS:
            sequence_data = np.asarray(
                processed_data_dict[qid][sequence_id][data_key],
            )
            if sequence_data.shape != sweep_points.shape:
                beta_results[qid]["reason"] = (
                    "Inconsistent ALLXY data shape across sequences."
                )
                invalid_shape = True
                break
            if np.iscomplexobj(sequence_data):
                sequence_data = np.real(sequence_data)
            data_matrix.append(sequence_data.astype(float))
        if invalid_shape:
            continue

        data_matrix_arr = np.vstack(data_matrix)
        scores = np.mean((data_matrix_arr - ALLXY_TARGETS[:, None]) ** 2, axis=0)
        min_idx = int(np.argmin(scores))
        score_min = float(scores[min_idx])
        score_reference = float(np.median(scores))
        score_drop = (score_reference - score_min) / max(abs(score_reference), 1e-12)

        beta_discrete = float(sweep_points[min_idx])
        beta_nominal = beta_discrete
        curvature = np.nan
        beta_std = (
            float(abs(sweep_points[1] - sweep_points[0]))
            if sweep_points.size > 1
            else 0.0
        )

        lo = max(0, min_idx - 2)
        hi = min(len(sweep_points), min_idx + 3)
        if hi - lo >= 3:
            coeffs = np.polyfit(sweep_points[lo:hi], scores[lo:hi], 2)
            curvature = float(coeffs[0])
            if curvature > 0:
                beta_candidate = float(-coeffs[1] / (2 * coeffs[0]))
                if float(np.min(sweep_points)) <= beta_candidate <= float(
                    np.max(sweep_points)
                ):
                    beta_nominal = beta_candidate
                residual = scores[lo:hi] - np.polyval(coeffs, sweep_points[lo:hi])
                residual_std = float(np.std(residual)) if residual.size > 1 else 0.0
                beta_std = float(
                    np.sqrt(max(residual_std, 1e-12) / max(curvature, 1e-12))
                )

        reasons = []
        accepted = True
        if min_idx in (0, len(sweep_points) - 1):
            reasons.append("Score minimum lies on sweep boundary.")
            accepted = False
        if score_drop < min_score_drop:
            reasons.append(
                f"Score drop {score_drop:.3f} below threshold {min_score_drop:.3f}."
            )
            accepted = False
        if not (float(np.min(sweep_points)) <= beta_nominal <= float(np.max(sweep_points))):
            reasons.append("Estimated beta is out of the swept range.")
            accepted = False
        if not np.isfinite(beta_nominal):
            reasons.append("Estimated beta is not finite.")
            accepted = False

        beta_results[qid] = {
            "accepted": accepted,
            "reason": "OK" if accepted else " ".join(reasons),
            "beta_opt": _safe_ufloat(beta_nominal, beta_std) if accepted else None,
            "beta_nominal": beta_nominal,
            "beta_std": beta_std,
            "score_drop": score_drop,
            "score_min": score_min,
            "score_reference": score_reference,
            "curvature": curvature,
            "sweep_points": sweep_points,
            "scores": scores,
        }
    return beta_results


@workflow.task
def extract_qubit_parameters_from_allxy(
    qubits: QuantumElements,
    allxy_beta_results: dict[
        str, dict[str, bool | str | float | unc.core.Variable | ArrayLike | None]
    ],
    options: ExtractQubitParametersTransitionOptions | None = None,
) -> dict[str, dict]:
    """Extract qubit parameters from ALLXY score minima."""
    opts = ExtractQubitParametersTransitionOptions() if options is None else options
    qubits = validate_and_convert_qubits_sweeps(qubits)
    qubit_parameters = {
        "old_parameter_values": {q.uid: {} for q in qubits},
        "new_parameter_values": {q.uid: {} for q in qubits},
        "diagnostics": {q.uid: {} for q in qubits},
    }
    for q in qubits:
        qid = q.uid
        old_beta = _get_old_beta_for_transition(q, opts.transition)
        beta_param_name = f"{opts.transition}_drive_pulse.beta"
        qubit_parameters["old_parameter_values"][qid] = {beta_param_name: old_beta}
        if not opts.do_fitting:
            qubit_parameters["diagnostics"][qid] = {
                "accepted": False,
                "reason": "Fitting disabled.",
                "method": "allxy21",
            }
            continue
        result = allxy_beta_results.get(qid, {})
        accepted = bool(result.get("accepted", False))
        reason = str(result.get("reason", "No ALLXY result."))
        qubit_parameters["diagnostics"][qid] = {
            "accepted": accepted,
            "reason": reason,
            "method": "allxy21",
            "score_drop": result.get("score_drop", np.nan),
            "curvature": result.get("curvature", np.nan),
        }
        if accepted and result.get("beta_opt") is not None:
            qubit_parameters["new_parameter_values"][qid] = {
                beta_param_name: result["beta_opt"]
            }
    return qubit_parameters


@workflow.task
def extract_qubit_parameters(
    qubits: QuantumElements,
    fit_results: dict[str, dict[str, lmfit.model.ModelResult]],
    options: ExtractQubitParametersTransitionOptions | None = None,
) -> dict[str, dict]:
    """Extract the qubit parameters from the fit results.

    Arguments:
        qubits:
            The qubits on which to run the analysis. May be either a single qubit or
            a list of qubits. The UIDs of these qubits must exist in the
            processed_data_dict.
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
        "diagnostics": {q.uid: {} for q in qubits},
    }

    for q in qubits:
        qid = q.uid
        beta_param_name = f"{opts.transition}_drive_pulse.beta"
        old_beta = _get_old_beta_for_transition(q, opts.transition)
        qubit_parameters["old_parameter_values"][qid] = {beta_param_name: old_beta}

        if not opts.do_fitting:
            qubit_parameters["diagnostics"][qid] = {
                "accepted": False,
                "reason": "Fitting disabled.",
                "method": "xy3",
            }
            continue
        if qid not in fit_results:
            qubit_parameters["diagnostics"][qid] = {
                "accepted": False,
                "reason": "No fit results for qubit.",
                "method": "xy3",
            }
            continue
        if not {"xy", "xmy"}.issubset(fit_results[qid]):
            qubit_parameters["diagnostics"][qid] = {
                "accepted": False,
                "reason": "Missing xy/xmy fit results.",
                "method": "xy3",
            }
            continue

        gradient = {}
        intercept = {}
        for i, pulse_id in enumerate(["xy", "xmy"]):
            gradient[i] = _safe_ufloat(
                fit_results[qid][pulse_id].params["gradient"].value,
                fit_results[qid][pulse_id].params["gradient"].stderr,
            )
            intercept[i] = _safe_ufloat(
                fit_results[qid][pulse_id].params["intercept"].value,
                fit_results[qid][pulse_id].params["intercept"].stderr,
            )
        intercept_diff_mean = intercept[0] - intercept[1]
        slope_diff_mean = gradient[1] - gradient[0]
        slope_threshold = 1e-12
        if abs(slope_diff_mean.nominal_value) <= slope_threshold:
            workflow.log(
                logging.ERROR,
                "Could not extract DRAG beta for %s because slope difference was too small.",
                qid,
            )
            qubit_parameters["diagnostics"][qid] = {
                "accepted": False,
                "reason": "Slope difference too small.",
                "method": "xy3",
            }
            continue
        new_beta = intercept_diff_mean / slope_diff_mean
        qubit_parameters["new_parameter_values"][qid] = {beta_param_name: new_beta}
        qubit_parameters["diagnostics"][qid] = {
            "accepted": True,
            "reason": "OK",
            "method": "xy3",
        }
    return qubit_parameters


@workflow.task
def plot_population(
    qubits: QuantumElements,
    processed_data_dict: dict[str, dict[str, dict[str, ArrayLike]]],
    fit_results: dict[str, dict[str, lmfit.model.ModelResult]] | None,
    qubit_parameters: dict[
        str,
        dict[str, dict[str, int | float | unc.core.Variable | None]],
    ]
    | None,
    options: PlotPopulationOptions | None = None,
) -> dict[str, mpl.figure.Figure]:
    """Create the DRAG quadrature-scaling calibration plots.

    Arguments:
        qubits:
            The qubits on which to run the analysis. May be either a single qubit or
            a list of qubits. The UIDs of these qubits must exist in
            processed_data_dict and qubit_parameters.
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
    fit_results = {} if fit_results is None else fit_results
    qubit_parameters = {} if qubit_parameters is None else qubit_parameters
    figures = {}
    for q in qubits:
        pulse_ids = list(processed_data_dict[q.uid])
        fig, ax = plt.subplots()
        ax.set_title(timestamped_title(f"DRAG Q-Scaling {q.uid}"))
        ax.set_xlabel("Quadrature Scaling Factor, $\\beta$")
        num_cal_traces = processed_data_dict[q.uid][pulse_ids[0]]["num_cal_traces"]
        ax.set_ylabel(
            "Principal Component (a.u)"
            if (num_cal_traces == 0 or opts.do_pca)
            else f"$|{opts.cal_states[-1]}\\rangle$-State Population",
        )

        for pulse_id in pulse_ids:
            sweep_points = processed_data_dict[q.uid][pulse_id]["sweep_points"]
            data = processed_data_dict[q.uid][pulse_id][
                "population" if opts.do_rotation else "data_raw"
            ]
            # plot data
            [line] = ax.plot(sweep_points, data, "o", zorder=2, label=pulse_id)
            if (
                opts.do_fitting
                and q.uid in fit_results
                and pulse_id in fit_results[q.uid]
            ):
                fit_res_qb = fit_results[q.uid][pulse_id]
                # plot fit
                sweep_points = processed_data_dict[q.uid][pulse_id]["sweep_points"]
                swpts_fine = np.linspace(sweep_points[0], sweep_points[-1], 501)
                ax.plot(
                    swpts_fine,
                    fit_res_qb.model.func(swpts_fine, **fit_res_qb.best_values),
                    c=line.get_color(),
                    zorder=1,
                    label="fit",
                )

        # the block plotting the lines at the calibration traces needs to come after
        # the xlims have been determined by plotting the data because we want these
        # lines to extend across the entire width of the axis
        if processed_data_dict[q.uid][pulse_ids[0]]["num_cal_traces"] > 0:
            # plot lines at the calibration traces
            xlims = ax.get_xlim()
            ax.hlines(
                processed_data_dict[q.uid][pulse_ids[0]]["population_cal_traces"],
                *xlims,
                linestyles="--",
                colors="gray",
                zorder=0,
                label="calib.\ntraces",
            )
            ax.set_xlim(xlims)

        q_new_params = qubit_parameters.get("new_parameter_values", {}).get(q.uid, {})
        if q_new_params:
            beta_param_name = f"{opts.transition}_drive_pulse.beta"
            new_beta = q_new_params.get(beta_param_name)
            if new_beta is not None:
                beta_nominal = (
                    new_beta.nominal_value
                    if hasattr(new_beta, "nominal_value")
                    else float(new_beta)
                )
                beta_std = (
                    new_beta.std_dev if hasattr(new_beta, "std_dev") else None
                )
                if (
                    q.uid in fit_results
                    and pulse_ids
                    and pulse_ids[0] in fit_results[q.uid]
                ):
                    fit_res_qb = fit_results[q.uid][pulse_ids[0]]
                    ax.plot(
                        beta_nominal,
                        fit_res_qb.model.func(
                            beta_nominal,
                            **fit_res_qb.best_values,
                        ),
                        "sk",
                        zorder=3,
                        markersize=plt.rcParams["lines.markersize"] + 1,
                    )
                else:
                    ax.axvline(
                        beta_nominal,
                        color="k",
                        linestyle="--",
                        zorder=1,
                        label="chosen $\\beta$",
                    )

                old_beta = (
                    qubit_parameters.get("old_parameter_values", {})
                    .get(q.uid, {})
                    .get(beta_param_name)
                )
                textstr = f"$\\beta$: {beta_nominal:.4f}"
                if beta_std is not None:
                    textstr += f" $\\pm$ {beta_std:.4f}"
                if old_beta is not None:
                    textstr += "\nPrevious value: " + f"{old_beta:.4f}"
                ax.text(0, -0.15, textstr, ha="left", va="top", transform=ax.transAxes)

        ax.legend(
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            handlelength=1.5,
            frameon=False,
        )

        if opts.save_figures:
            workflow.save_artifact(f"Drag_q_scaling_{q.uid}", fig)

        if opts.close_figures:
            plt.close(fig)

        figures[q.uid] = fig

    return figures


@workflow.task
def plot_allxy_scores(
    qubits: QuantumElements,
    allxy_beta_results: dict[
        str, dict[str, bool | str | float | unc.core.Variable | ArrayLike | None]
    ],
    qubit_parameters: dict[str, dict] | None = None,
    options: PlotPopulationOptions | None = None,
) -> dict[str, mpl.figure.Figure]:
    """Create score-vs-beta plots for ALLXY beta optimization."""
    opts = PlotPopulationOptions() if options is None else options
    qubits = validate_and_convert_qubits_sweeps(qubits)
    qubit_parameters = {} if qubit_parameters is None else qubit_parameters
    figures = {}
    for q in qubits:
        qid = q.uid
        result = allxy_beta_results.get(qid, {})
        sweep_points = np.asarray(result.get("sweep_points", np.array([])), dtype=float)
        scores = np.asarray(result.get("scores", np.array([])), dtype=float)
        if sweep_points.size == 0 or scores.size == 0:
            continue
        fig, ax = plt.subplots()
        ax.set_title(timestamped_title(f"ALLXY score {qid}"))
        ax.set_xlabel("Quadrature Scaling Factor, $\\beta$")
        ax.set_ylabel("ALLXY score (MSE)")
        ax.plot(sweep_points, scores, "o-", label="score")
        beta_param_name = f"{opts.transition}_drive_pulse.beta"
        new_beta = (
            qubit_parameters.get("new_parameter_values", {})
            .get(qid, {})
            .get(beta_param_name)
        )
        if new_beta is not None:
            beta_nominal = (
                new_beta.nominal_value
                if hasattr(new_beta, "nominal_value")
                else float(new_beta)
            )
            ax.axvline(beta_nominal, color="k", linestyle="--", label="chosen $\\beta$")
        ax.legend(frameon=False)
        if opts.save_figures:
            workflow.save_artifact(f"Drag_q_scaling_allxy_score_{qid}", fig)
        if opts.close_figures:
            plt.close(fig)
        figures[qid] = fig
    return figures

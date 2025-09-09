# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""This module defines the analysis for a resonator-spectroscopy experiment.

The experiment is defined in laboneq_applications.experiments.

In this analysis, we first interpret the raw data into the signal magnitude and phase.
Then we either extract the frequency corresponding to the min or max of the magnitude
data, or we fit a Lorentzian model to the signal magnitude and extract frequency
corresponding to the peak of the Lorentzian from the fit. Finally, we plot the data and
the fit.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import uncertainties as unc
from laboneq import workflow
from laboneq.simple import dsl

from laboneq_applications.analysis.fitting_helpers import (
    lorentzian_fit,
    fit_data_lmfit,
    )
                                                           
from laboneq_applications.analysis.options import BasePlottingOptions
from laboneq_applications.analysis.plotting_helpers import (
    plot_raw_complex_data_1d,
    timestamped_title,
)
from laboneq_applications.core import validation

if TYPE_CHECKING:
    import lmfit
    import matplotlib as mpl
    from laboneq.dsl.quantum.quantum_element import QuantumElement
    from laboneq.workflow.tasks.run_experiment import RunExperimentResults
    from numpy.typing import ArrayLike


@workflow.workflow_options
class ResonatorSpectroscopyAnalysisWorkflowOptions:
    """Option class for spectroscopy analysis workflows.

    Attributes:
        do_plotting:
            Whether to create plots.
            Default: 'True'.
        do_raw_data_plotting:
            Whether to plot the raw data.
            Default: True.
        do_plotting_magnitude_phase:
            Whether to plot the magnitude and phase.
            Default: True.
        do_plotting_real_imaginary:
            Whether to plot the real and imaginary data.
            Default: True.
    """

    do_plotting: bool = workflow.option_field(
        True, description="Whether to create plots."
    )
    do_raw_data_plotting: bool = workflow.option_field(
        True, description="Whether to plot the raw data."
    )
    do_plotting_magnitude_phase: bool = workflow.option_field(
        True, description="Whether to plot the magnitude and phase."
    )
    do_plotting_real_imaginary: bool = workflow.option_field(
        True, description="Whether to plot the real and imaginary data."
    )


@workflow.task_options
class FitDataResSpecOptions:
    """Options for the `fit_data` task of the resonator spectroscopy analysis.

    Attributes:
        fit_lorentzian:
            Whether to fit a Lorentzian model to the data.
            Default: `False`.
        fit_parameters_hints:
            Parameters hints accepted by lmfit
            Default: None.
        fit_complex_resonator:
            Whether to fit a complex resonator model to the data.
            Default: `False`.
        fit_complex_parameters_hints:
            Parameter hints for the complex resonator model
            Default: `None`.
        fit_scresonators:
            Whether to fit using the external scresonators package (DCM method).
            Default: `False`.
        scresonators_method:
            The scresonators method to use (e.g. 'DCM'). Only 'DCM' supported
            by this integration.
            Default: `"DCM"`.
    """

    fit_lorentzian: bool = workflow.option_field(
        False, description="Whether to fit a Lorentzian model to the data."
    )

    fit_complex_resonator: bool = workflow.option_field(
        False,
        description=(
            "Whether to fit a complex resonator model (complex S21) to the data."
        ),
    )

    fit_parameters_hints: dict[str, dict[str, float | bool | str]] | None = (
        workflow.option_field(None, description="Parameters hints accepted by lmfit")
    )

    fit_complex_parameters_hints: dict[str, dict[str, float | bool | str]] | None = (
        workflow.option_field(
            None, description="Parameter hints for the complex resonator model"
        )
    )

    fit_scresonators: bool = workflow.option_field(
        False,
        description=(
            "Whether to fit using the external scresonators package (complex S21 circle fit)."
        ),
    )

    scresonators_method: str = workflow.option_field(
        "DCM", description="Method to use with scresonators (currently only 'DCM')."
    )

   


@workflow.task_options
class ExtractQubitParametersResSpecOptions:
    """Options for the `extract_qubit_parameters` task of the resonator spec. analysis.

    Attributes:
        find_peaks:
            Whether to search for peaks (True) or dips (False) in the spectrum.
            Default: `False`.
    """

    find_peaks: bool = workflow.option_field(
        False,
        description="Whether to search for peaks (True) or dips (False) "
        "in the spectrum.",
    )


@workflow.task_options(base_class=BasePlottingOptions)
class PlotMagnitudePhaseOptions:
    """Options for the `plot_magnitude_phase` task of the resonator spec. analysis.

    Attributes:
        fit_lorentzian:
            Whether to fit a Lorentzian model to the data.
            Default: `False`.
        find_peaks:
            Whether to search for peaks (True) or dips (False) in the spectrum.
            Default: `False`.

    Additional attributes from `BasePlottingOptions`:
        save_figures:
            Whether to save the figures.
            Default: `True`.
        close_figures:
            Whether to close the figures.
            Default: `True`.

    """

    fit_lorentzian: bool = workflow.option_field(
        False, description="Whether to fit a Lorentzian model to the data."
    )
    find_peaks: bool = workflow.option_field(
        False,
        description="Whether to search for peaks (True) or dips (False) "
        "in the spectrum.",
    )


@workflow.workflow
def analysis_workflow(
    result: RunExperimentResults,
    qubit: QuantumElement,
    frequencies: ArrayLike,
    options: ResonatorSpectroscopyAnalysisWorkflowOptions | None = None,
) -> None:
    """The Resonator Spectroscopy analysis Workflow.

    The workflow consists of the following steps:

    - [calculate_signal_magnitude_and_phase]()
    - [fit_data]()
    - [extract_qubit_parameters]()
    - [plot_raw_complex_data_1d]()
    - [plot_magnitude_phase]()
    - [plot_real_imaginary]()

    Arguments:
        result:
            The experiment results returned by the run_experiment task.
        qubit:
            The qubit on which to run the analysis. The UID of this qubit must exist
            in the result.
        frequencies:
            The array of frequencies that were swept over in the experiment.
        options:
            The options for building the workflow, passed as an instance of
                [ResonatorSpectroscopyAnalysisWorkflowOptions]. See the docstring of
                [ResonatorSpectroscopyAnalysisWorkflowOptions] for more details.

    Returns:
        WorkflowBuilder:
            The builder for the analysis workflow.

    Example:
        ```python
        result = analysis_workflow(
            results=results
            qubit=q0,
            frequencies=np.linspace(7.0, 7.1, 101),
            options=analysis_workflow.options(),
        ).run()
        ```
    """
    # Ensure options have sensible defaults if None
    opts = (
        ResonatorSpectroscopyAnalysisWorkflowOptions()
        if options is None
        else options
    )

    processed_data_dict = calculate_signal_magnitude_and_phase(
        qubit, result, frequencies
    )
    fit_result = fit_data(processed_data_dict)
    qubit_parameters = extract_qubit_parameters(qubit, processed_data_dict, fit_result)
    with workflow.if_(opts.do_plotting):
        with workflow.if_(opts.do_raw_data_plotting):
            plot_raw_complex_data_1d(
                qubit,
                result,
                frequencies,
                xlabel="Readout Frequency, $f_{\\mathrm{RO}}$ (GHz)",
                xscaling=1e-9,
            )
        with workflow.if_(opts.do_plotting_magnitude_phase):
            plot_magnitude_phase(
                qubit, processed_data_dict, fit_result, qubit_parameters
            )
        with workflow.if_(opts.do_plotting_real_imaginary):
            #plot_real_imaginary(qubit, result, frequencies, fit_result)
            plot_real_imaginary(qubit, processed_data_dict, fit_result)
    workflow.return_(qubit_parameters)


@workflow.task
def calculate_signal_magnitude_and_phase(
    qubit: QuantumElement,
    result: RunExperimentResults,
    frequencies: ArrayLike,
) -> dict[str, ArrayLike]:
    """Calculates the magnitude and phase of the spectroscopy signal in result.

    Arguments:
        result:
            The experiment results returned by the run_experiment task.
        qubit:
            The qubit on which to run the analysis. The UID of this qubit must exist
            in the result.
        frequencies:
            The array of frequencies that were swept over in the experiment.

    Returns:
        dictionary with the following data:
            sweep_points
            data_raw
            magnitude
            phase
    """
    validation.validate_result(result)
    qubit, frequencies = validation.validate_and_convert_single_qubit_sweeps(
        qubit, frequencies
    )

    raw_data = result[dsl.handles.result_handle(qubit.uid)].data
    return {
        "sweep_points": frequencies,
        "data_raw": raw_data,
        "magnitude": np.abs(raw_data),
        "phase": np.angle(raw_data),
    }


@workflow.task
def fit_data(
    processed_data_dict: dict[str, ArrayLike],
    options: FitDataResSpecOptions | None = None,
) -> lmfit.model.ModelResult | None:
    """Fit resonator data using Lorentzian or complex resonator models.

    Arguments:
        processed_data_dict: the processed data dictionary returned by
            calculate_signal_magnitude_and_phase
        options:
            The options for this task as an instance of [FitDataResSpecOptions].
            See the docstring of this class for more details.

    Returns:
        The lmfit fit result, or None if no fitting requested.
    """
    opts = FitDataResSpecOptions() if options is None else options
    fit_result = None

    if opts.fit_scresonators:
        # Try scresonators-based complex fit (external package)
        swpts_fit = np.asarray(processed_data_dict["sweep_points"], dtype=float).ravel()
        raw_data = np.asarray(processed_data_dict["data_raw"], dtype=np.complex128).ravel()
        try:
            from analysis.scresonators_adapter import fit_with_scresonators

            fit_res = fit_with_scresonators(swpts_fit, raw_data, method=opts.scresonators_method)
            fit_result = fit_res
        except ImportError:
            workflow.log(
                logging.ERROR,
                "scresonators not available. Install via 'pip install scresonators-fit' to enable.",
            )
        except Exception as err:
            workflow.log(logging.ERROR, "scresonators fit failed: %s", err)
    elif opts.fit_complex_resonator:
        # Normalize shapes/dtypes for fitting
        swpts_fit = np.asarray(processed_data_dict["sweep_points"], dtype=float).ravel()
        raw_data = np.asarray(processed_data_dict["data_raw"], dtype=np.complex128).ravel()
        x = np.concatenate([swpts_fit, swpts_fit])
        y = np.concatenate([np.real(raw_data), np.imag(raw_data)])
        def complex_resonator_model(
            x: ArrayLike,
            amp: float,
            alpha: float,
            tau: float,
            phi: float,
            Q: float,
            Q_e_real: float,
            Q_e_imag: float,
            fr : float
        ) -> ArrayLike:
            n = len(x) // 2
            freqs = x[:n]
            Q_e = Q_e_real + 1j*Q_e_imag
            background = np.exp(1j*alpha-2*np.pi*1j*freqs*tau)
            resp = 1 - ((Q/np.abs(Q_e)*np.exp(1j*phi)))/(1+2j*Q*(freqs/fr-1))

            s21 = amp *background * resp

            out_real = np.real(s21) #+ i_offset
            out_imag = np.imag(s21) #+ q_offset
            return np.concatenate([out_real, out_imag])
     
        if opts.fit_complex_parameters_hints is None:
            mag = np.abs(raw_data)
            idx0 = int(np.argmin(mag))
            fr_guess = float(swpts_fit[idx0])
            Q_guess = 1e4
            depth = float(max(1e-6, 1.0 - mag[idx0]))
            Qc_real_guess = max(10.0, Q_guess / depth)
            f_min = float(np.min(swpts_fit))
            f_max = float(np.max(swpts_fit))
            param_hints = {
                "amp": {"value": float(np.max(mag)), "min": 0.0},
                "alpha": {"value": 0.0, "min": -2 * np.pi, "max": 2 * np.pi},
                "tau": {"value": 0.0, "min": -1e-6, "max": 1e-6},
                "phi": {
                    "value": float(np.mean(np.angle(raw_data))),
                    "min": -2 * np.pi,
                    "max": 2 * np.pi,
                },
                "Q": {"value": Q_guess, "min": 1.0},
                "Q_e_real": {"value": Qc_real_guess, "min": 1.0},
                "Q_e_imag": {"value": 0.0},
                "fr": {"value": fr_guess, "min": f_min, "max": f_max},
            }
        else:
            param_hints = opts.fit_complex_parameters_hints

        try:
            fit_res = fit_data_lmfit(
                complex_resonator_model,
                x,
                y,
                param_hints=param_hints,
            )
            fit_result = fit_res
        except ValueError as err:
            workflow.log(logging.ERROR, "Fit failed: %s", err)
    elif opts.fit_lorentzian:
        swpts_fit = processed_data_dict["sweep_points"]
        data_to_fit = processed_data_dict["magnitude"]
        try:
            fit_res = lorentzian_fit(
                swpts_fit,
                data_to_fit,
                param_hints=opts.fit_parameters_hints,
            )
            fit_result = fit_res
        except ValueError as err:
            workflow.log(logging.ERROR, "Fit failed: %s", err)

    return fit_result
        
  
@workflow.task
def extract_qubit_parameters(
    qubit: QuantumElement,
    processed_data_dict: dict[str, ArrayLike],
    fit_result: lmfit.model.ModelResult | None,
    options: ExtractQubitParametersResSpecOptions | None = None,
) -> dict[str, dict[str, dict[str, int | float | unc.core.Variable | None]]]:
    """Extract the qubit parameters from the fit results.

    Arguments:
        qubit:
            The qubit on which to run the analysis.
        processed_data_dict: the processed data dictionary returned by
            calculate_signal_magnitude_and_phase.
        fit_result: the lmfit ModelResults returned by fit_data
        options:
            The options for this task as an instance of
            [ExtractQubitParametersResSpecOptions].
            See the docstring of this class for more details.

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
        If the fit_results is None, the new_parameter_values entry for the qubit is
        left empty.
    """
    opts = ExtractQubitParametersResSpecOptions() if options is None else options
    qubit = validation.validate_and_convert_single_qubit_sweeps(qubit)

    qubit_parameters = {
        "old_parameter_values": {qubit.uid: {}},
        "new_parameter_values": {qubit.uid: {}},
    }

    # Store the readout resonator frequency value
    qubit_parameters["old_parameter_values"][qubit.uid] = {
        "readout_resonator_frequency": qubit.parameters.readout_resonator_frequency,
    }

    # Extract and store the readout resonator frequency value
    if fit_result is not None:
        # Extract central frequency depending on model used
        if "position" in fit_result.params:
            rr_freq = unc.ufloat(
                fit_result.params["position"].value,
                fit_result.params["position"].stderr,
            )
        elif "fr" in fit_result.params:
            rr_freq = unc.ufloat(
                fit_result.params["fr"].value,
                fit_result.params["fr"].stderr,
            )
        elif "f0" in fit_result.params:
            rr_freq = unc.ufloat(
                fit_result.params["f0"].value,
                getattr(fit_result.params["f0"], "stderr", 0) or 0,
            )
        else:
            rr_freq = None
    else:
        # find frequency at min or max of the signal magnitude
        take_extremum = np.argmax if opts.find_peaks else np.argmin
        freqs = processed_data_dict["sweep_points"]
        signal_magnitude = processed_data_dict["magnitude"]
        rr_freq = unc.ufloat(
            freqs[take_extremum(signal_magnitude)],
            0,
        )

    new_vals: dict[str, int | float | unc.core.Variable | None] = {}
    if rr_freq is not None:
        new_vals["readout_resonator_frequency"] = rr_freq

    # If complex fit present, also compute and store Q factors
    if fit_result is not None:
        # Our internal complex model (Qe_real/Qe_imag)
        if all(k in fit_result.params for k in ("Q", "Q_e_real", "Q_e_imag")):
            Q_total = float(fit_result.params["Q"].value)
            Qe_real = float(fit_result.params["Q_e_real"].value)
            Qe_imag = float(fit_result.params["Q_e_imag"].value)
            denom = Qe_real * Qe_real + Qe_imag * Qe_imag
            inv_Qe_real = Qe_real / denom if denom != 0 else 0.0
            inv_Q_total = 1.0 / Q_total if Q_total != 0 else 0.0
            inv_Qi = max(inv_Q_total - inv_Qe_real, 0.0)
            Qi = (1.0 / inv_Qi) if inv_Qi > 0 else float("inf")
            Qe_external = (1.0 / inv_Qe_real) if inv_Qe_real > 0 else float("inf")
            new_vals["loaded_quality_factor"] = Q_total
            new_vals["internal_quality_factor"] = Qi
            new_vals["external_quality_factor"] = Qe_external
        # scresonators: params contain Q (loaded), Qc (external), and they add Qi
        elif all(k in fit_result.params for k in ("Q", "Qc")):
            Q_total = float(fit_result.params["Q"].value)
            Qc = float(fit_result.params["Qc"].value)
            Qi = fit_result.params.get("Qi")
            Qi_val = float(Qi.value) if Qi is not None else None
            new_vals["loaded_quality_factor"] = Q_total
            if Qi_val is not None:
                new_vals["internal_quality_factor"] = Qi_val
            new_vals["external_quality_factor"] = Qc

    qubit_parameters["new_parameter_values"][qubit.uid] = new_vals

    return qubit_parameters


@workflow.task
def plot_magnitude_phase(
    qubit: QuantumElement,
    processed_data_dict: dict[str, ArrayLike],
    fit_result: lmfit.model.ModelResult | None,
    qubit_parameters: dict[
        str, dict[str, dict[str, int | float | unc.core.Variable | None]]
    ],
    options: PlotMagnitudePhaseOptions | None = None,
) -> mpl.figure.Figure | None:
    """Plot the magnitude and phase of the spectroscopy signal.

    Arguments:
        qubit:
            The qubit on which to run the analysis. qubit_parameters.
        processed_data_dict: the processed data dictionary returned by
            calculate_signal_magnitude_and_phase.
        fit_result: the lmfit ModelResults returned by fit_data
        qubit_parameters: the qubit-parameters dictionary returned by
            extract_qubit_parameters
        options:
            The options for this task as an instance of [PlotMagnitudePhaseOptions].
            See the docstring of this class for more details.

    Returns:
        the matplotlib figure

        If there are no new_parameter_values for the qubit, then fit result and the
        textbox with the extracted readout resonator frequency are not plotted.
    """
    opts = PlotMagnitudePhaseOptions() if options is None else options
    qubit = validation.validate_and_convert_single_qubit_sweeps(qubit)

    sweep_points = processed_data_dict["sweep_points"]
    magnitude = processed_data_dict["magnitude"]
    phase = processed_data_dict["phase"]

    fig, axs = plt.subplots(nrows=2, sharex=True)
    axs[0].set_title(timestamped_title(f"Magnitude-Phase {qubit.uid}"))
    axs[0].plot(
        sweep_points / 1e9,
        magnitude,
        "-",
        zorder=2,
        label="data",
    )
    axs[0].set_ylabel("Transmission Signal\nMagnitude, $|S_{21}|$ (a.u.)")
    axs[1].plot(sweep_points / 1e9, phase, "-", zorder=2, label="data")
    axs[1].set_ylabel("Transmission Signal\nPhase, arg($S_{21}$) (rad)")
    axs[1].set_xlabel("Readout Frequency, $f_{\\mathrm{RO}}$ (GHz)")
    fig.align_ylabels()
    fig.subplots_adjust(hspace=0.1)

    if fit_result is not None:
        # Overlay fit curve depending on model type
        if opts.fit_lorentzian:
            # Plot Lorentzian fit of the magnitude
            swpts_fine = np.linspace(sweep_points[0], sweep_points[-1], 501)
            axs[0].plot(
                swpts_fine / 1e9,
                fit_result.model.func(swpts_fine, **fit_result.best_values),
                "r-",
                zorder=1,
                label="fit",
            )
            axs[0].legend(loc="best", frameon=False)
        else:
            # If complex resonator fit, overlay magnitude of complex fit
            try:
                if all(k in fit_result.params for k in ("Q_e_real", "Q_e_imag", "Q")):
                    swpts_fine = np.linspace(sweep_points[0], sweep_points[-1], 501)
                    x_fine = np.concatenate([swpts_fine, swpts_fine])
                    fit_vals = fit_result.model.func(x_fine, **fit_result.best_values)
                    n = len(swpts_fine)
                    mag_fit = np.hypot(fit_vals[:n], fit_vals[n:])
                    axs[0].plot(
                        swpts_fine / 1e9,
                        mag_fit,
                        "r-",
                        zorder=1,
                        label="fit",
                    )
                    axs[0].legend(loc="best", frameon=False)
            except Exception as err:
                workflow.log(logging.WARNING, "Could not overlay complex fit magnitude: %s", err)

    if len(qubit_parameters["new_parameter_values"][qubit.uid]) > 0:
        rr_freq = qubit_parameters["new_parameter_values"][qubit.uid][
            "readout_resonator_frequency"
        ]

        # Point at the extracted readout resonator frequency
        if opts.fit_lorentzian:
            # Point on the magnitude plot from the fit result
            axs[0].plot(
                rr_freq.nominal_value / 1e9,
                fit_result.model.func(
                    rr_freq.nominal_value,
                    **fit_result.best_values,
                ),
                "or",
                zorder=3,
                markersize=plt.rcParams["lines.markersize"] + 1,
            )
            # Legend
            axs[0].legend(
                loc="center left",
                bbox_to_anchor=(1, 0),
                handlelength=1.5,
                frameon=False,
            )
        else:
            take_extremum = np.argmax if opts.find_peaks else np.argmin
            # Point on the magnitude plot at the rr freq
            axs[0].plot(
                rr_freq.nominal_value / 1e9,
                magnitude[take_extremum(magnitude)],
                "or",
                zorder=3,
                markersize=plt.rcParams["lines.markersize"] + 1,
            )

        # Line on the phase plot corresponding to the rr freq
        ylims = axs[1].get_ylim()
        axs[1].vlines(
            rr_freq.nominal_value / 1e9,
            *ylims,
            linestyles="--",
            colors="r",
            zorder=0,
        )
        axs[1].set_ylim(ylims)

        # Textbox
        old_rr_freq = qubit_parameters["old_parameter_values"][qubit.uid][
            "readout_resonator_frequency"
        ]
        textstr = (
            f"Readout-resonator frequency: "
            f"{rr_freq.nominal_value / 1e9:.4f} GHz $\\pm$ "
            f"{rr_freq.std_dev / 1e6:.4f} MHz"
        )
        textstr += f"\nPrevious value: {old_rr_freq / 1e9:.4f} GHz"
        axs[1].text(0, -0.35, textstr, ha="left", va="top", transform=axs[1].transAxes)

    if opts.save_figures:
        workflow.save_artifact(f"Magnitude_Phase_{qubit.uid}", fig)

    if opts.close_figures:
        plt.close(fig)

    return fig


@workflow.task
def plot_real_imaginary(
    qubit: QuantumElement,
    #result: RunExperimentResults,
    processed_data_dict: dict[str, ArrayLike],
    fit_result: lmfit.model.ModelResult | None,
    options: BasePlottingOptions | None = None,
) -> mpl.figure.Figure | None:
    """Plot the real versus imaginary parts of the spectroscopy signal.

    Arguments:
        qubit:
            The qubit on which to run the analysis.
        result:
            The experiment results returned by the run_experiment task.
        processed_data_dict:
            The processed data dictionary returned by
            [calculate_signal_magnitude_and_phase].
        fit_result:
            The lmfit ModelResults returned by [fit_data].
        options:
            The options for this task as an instance of [BasePlottingOptions].
            See the docstring for this class for more information.

    Returns:
        the matplotlib figure
    """
     
    opts = BasePlottingOptions() if options is None else options
    #validation.validate_result(result)
    qubit = validation.validate_and_convert_single_qubit_sweeps(qubit)

    #raw_data = result[dsl.handles.result_handle(qubit.uid)].data
    sweep_points = processed_data_dict["sweep_points"]
    raw_data = processed_data_dict["data_raw"]

    fig, ax = plt.subplots()
    ax.set_title(timestamped_title(f"Real-Imaginary {qubit.uid}"))
    ax.set_xlabel("Real Transmission Signal, Re($S_{21}$) (a.u.)")
    ax.set_ylabel("Imaginary Transmission Signal, Im($S_{21}$) (a.u.)")
    ax.plot(
        np.real(raw_data),
        np.imag(raw_data),
        "o",
        zorder=1,
        label="data",
    )
    if fit_result is not None:
        swpts_fine = np.linspace(sweep_points[0], sweep_points[-1], 501)
        x_fine = np.concatenate([swpts_fine, swpts_fine])
        fit_eval = fit_result.model.func(x_fine, **fit_result.best_values)
        n = len(swpts_fine)
        ax.plot(fit_eval[:n], fit_eval[n:], "r-", zorder=2, label="fit")
        ax.legend(loc="best", frameon=False)

    if opts.save_figures:
        workflow.save_artifact(f"Real_Imaginary_{qubit.uid}", fig)

    if opts.close_figures:
        plt.close(fig)

    return fig
 

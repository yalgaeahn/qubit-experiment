# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""This module defines the analysis for an IQ-blob experiment.

The experiment is defined in laboneq_applications.experiments.

In this analysis, we collect the single shots acquire for each prepared stated and
then use LinearDiscriminantAnalysis from the sklearn library to classify the data into
the prepared states. From this classification, we calculate the correct-state-assignment
matrix and the correct-state-assignment fidelity. Finally, we plot the single shots for
each prepared state and the correct-state-assignment matrix.
"""

from __future__ import annotations

import logging
from itertools import product
from typing import TYPE_CHECKING

import matplotlib.colors as mc
import matplotlib.pyplot as plt
import numpy as np
from laboneq import workflow
from laboneq.simple import dsl
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.inspection import DecisionBoundaryDisplay
# from sklearn.metrics import confusion_matrix

from laboneq_applications.analysis.options import BasePlottingOptions
from laboneq_applications.analysis.plotting_helpers import timestamped_title
from laboneq_applications.core.validation import (
    validate_and_convert_qubits_sweeps,
    validate_result,
)


from laboneq_applications.analysis.options import (
    BasePlottingOptions,
    DoFittingOption,
)


if TYPE_CHECKING:
    from collections.abc import Sequence

    import matplotlib as mpl
    from laboneq.workflow.tasks.run_experiment import RunExperimentResults
    from numpy.typing import ArrayLike

    from laboneq_applications.typing import QuantumElements


@workflow.workflow_options
class IQTrajAnalysisWorkflowOptions:
    """Option class for IQ-blob analysis workflows.

    Attributes:
        do_fitting:
            Whether to perform the fit.
            Default: `True`.
        do_plotting:
            Whether to create plots.
            Default: 'True'.
        do_plotting_iq_blobs:
            Whether to create the IQ-blob plots of the single shots.
            Default: 'True'.
        do_plotting_assignment_matrices:
            Whether to create the assignment matrix plots.
            Default: 'True'.
    """

    do_plotting: bool = workflow.option_field(
        True, description="Whether to create plots."
    )
    do_plotting_iq_traj: bool = workflow.option_field(
        True, description="Whether to create the IQ-blob plots of the single shots."
    )
    chunk_size: int = workflow.option_field(
        8, description="Size of chung 8 -> 4ns"
    )
    sample_dt_ns: float = workflow.option_field(
        0.5,
        description="Sampling period in ns for raw traces.",
    )
    apply_software_demodulation: bool = workflow.option_field(
        False,
        description=(
            "Apply extra software demodulation at IF. "
            "Keep False for standard LabOne Q RAW traces."
        ),
    )
    phase_mask_relative_threshold: float = workflow.option_field(
        0.08,
        description=(
            "Hide phase samples where |IQ| is below this fraction of the "
            "per-trace max amplitude."
        ),
    )
    plot_phase_boundaries: bool = workflow.option_field(
        True,
        description=(
            "Plot readout/integration timing boundaries on time-domain "
            "and phase panels."
        ),
    )

@workflow.workflow
def analysis_workflow(
    result: RunExperimentResults,
    qubits: QuantumElements,
    states: Sequence[str],
    options: IQTrajAnalysisWorkflowOptions | None = None,
) -> None:
    """The IQ Blobs analysis Workflow.

    The workflow consists of the following steps:

    - [demodulate_time_traces]
    - [avg_chunk_time_traces]
    - [plot_iq_traj]


    Arguments:
        result:
            The experiment results returned by the run_experiment task.
        qubits:
            The qubits on which to run the analysis. May be either a single qubit or
            a list of qubits. The UIDs of these qubits must exist in the result.
        states:
            The basis states the qubits should be prepared in. May be either a string,
            e.g. "gef", or a list of letters, e.g. ["g","e","f"].
        options:
            The options for building the workflow as an instance of
            [IQBlobAnalysisWorkflowOptions]. See the docstring of this class for more
            details.

    Returns:
        WorkflowBuilder:
            The builder for the analysis workflow.
        ```
    """
    opts = IQTrajAnalysisWorkflowOptions() if options is None else options

    processed_data_dict = demodulate_time_traces(
        qubits,
        result,
        states,
        sample_dt_ns=opts.sample_dt_ns,
        apply_software_demodulation=opts.apply_software_demodulation,
    )
    processed_data_dict = average_chunk_time_traces(
        qubits,
        processed_data_dict,
        states,
        options=opts,
    )
    with workflow.if_(opts.do_plotting):
        plot_iq_trajectories(
            qubits,
            states,
            processed_data_dict,
            chunk_size=opts.chunk_size,
            sample_dt_ns=opts.sample_dt_ns,
            options2=opts,
        )
    workflow.return_(None)

@workflow.task
def demodulate_time_traces(
    qubits: QuantumElements,
    result: RunExperimentResults,
    states: Sequence[str],
    sample_dt_ns: float = 0.5,
    apply_software_demodulation: bool = False,
) -> dict[str, dict[str, ArrayLike | dict]]:
    
    
    qubits = validate_and_convert_qubits_sweeps(qubits)
    validate_result(result)
    processed_data_dict = {q.uid: {} for q in qubits} 
    for q in qubits:
        time_trace = {}
        for s in states:
            time_trace[s] = np.asarray(
                result[dsl.handles.calibration_trace_handle(q.uid, s)].data,
                dtype=complex,
            )
            if apply_software_demodulation:
                if (
                    q.parameters.readout_resonator_frequency is None
                    or q.parameters.readout_lo_frequency is None
                ):
                    raise ValueError(
                        "readout_resonator_frequency/readout_lo_frequency must be set "
                        "when apply_software_demodulation=True."
                    )
                if_freq = (
                    q.parameters.readout_resonator_frequency
                    - q.parameters.readout_lo_frequency
                )
                i_raw = np.real(time_trace[s])
                q_raw = np.imag(time_trace[s])
                times_s = np.arange(len(i_raw), dtype=float) * sample_dt_ns * 1e-9
                cos = np.cos(2 * np.pi * if_freq * times_s)
                sin = np.sin(2 * np.pi * if_freq * times_s)
                i_demod = i_raw * cos + q_raw * sin
                q_demod = -i_raw * sin + q_raw * cos
                processed_data_dict[q.uid][s] = i_demod + 1j * q_demod
            else:
                processed_data_dict[q.uid][s] = time_trace[s]
    return processed_data_dict


@workflow.task
def average_chunk_time_traces(
    qubits: QuantumElements,
    processed_data_dict: dict[str, dict[str, ArrayLike | dict]],
    states: Sequence[Literal["g", "e", "f"]],
    options: IQTrajAnalysisWorkflowOptions | None = None,
) -> dict[str, list[ArrayLike]]:
    """Truncate the time traces to align on the granularity grid.

    The granularity is passed via the options and is typically 16 samples (default).

    Arguments:
        qubits:
            The qubits on which to run the analysis. May be either a single qubit or
            a list of qubits.
        result:
            The experiment results returned by the run_experiment task.
        states:
            The basis states the qubits should be prepared in. May be either a string,
            e.g. "gef", or a list of letters, e.g. ["g","e","f"].
        options:
            The options for this task as an instance of [TruncateTimeTracesOptions].
            See the docstring of this class for more details.

    Returns:
        dict with qubit UIDs as keys and the list of truncated time-traces for each
        qubit as keys.
    """
    qubits = validate_and_convert_qubits_sweeps(qubits)
    opts = IQTrajAnalysisWorkflowOptions() if options is None else options
    if opts.chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {opts.chunk_size}.")

    for q in qubits:  # type: ignore
        for s in states:
            time_trace = np.asarray(processed_data_dict[q.uid][s])
            n_samples = len(time_trace)
            n_aligned = (n_samples // opts.chunk_size) * opts.chunk_size
            if n_samples == 0:
                averaged_trace = time_trace
            elif n_aligned == 0:
                averaged_trace = np.asarray([time_trace.mean()])
            else:
                averaged_trace = time_trace[:n_aligned].reshape(
                    -1, opts.chunk_size
                ).mean(axis=1)
            processed_data_dict[q.uid][s] = averaged_trace

    return processed_data_dict




@workflow.task
def plot_iq_trajectories(
    qubits: QuantumElements,
    states: Sequence[str],
    processed_data_dict: dict[str, dict[str, ArrayLike | dict]],
    options: BasePlottingOptions | None = None,
    chunk_size: int = 8,
    sample_dt_ns: float = 0.5,
    options2: IQTrajAnalysisWorkflowOptions | None = None,
) -> dict[str, mpl.figure.Figure]:
    """Create the IQ-blobs plots.

    Arguments:
        qubits:
            The qubits on which to run the analysis. May be either a single qubit or
            a list of qubits. The UIDs of these qubits must exist in
            processed_data_dict and fit_results.
        states:
            The basis states the qubits should be prepared in. May be either a string,
            e.g. "gef", or a list of letters, e.g. ["g","e","f"].
        processed_data_dict: the processed data dictionary returned by collect_shots.
        fit_results: the classification fit results returned by fit_data.
        options:
            The options class for this task as an instance of [BasePlottingOptions]. See
            the docstring of this class for accepted options.

    Returns:
        dict with qubit UIDs as keys and the figures for each qubit as values.

        If a qubit uid is not found in fit_results, the fit is not plotted.
    """
    opts = BasePlottingOptions() if options is None else options
    opts2 = IQTrajAnalysisWorkflowOptions() if options2 is None else options2 
    qubits = validate_and_convert_qubits_sweeps(qubits)
    figures = {}
    color_map = {0:'b', 1:'r', 2:'g'}
    plot_chunk_size = opts2.chunk_size if options2 is not None else chunk_size
    plot_sample_dt_ns = opts2.sample_dt_ns if options2 is not None else sample_dt_ns
    phase_mask_relative_threshold = max(float(opts2.phase_mask_relative_threshold), 0.0)
    for q in qubits:
        # shots_per_state = processed_data_dict[q.uid]["shots_per_state"]
        # shots_combined = processed_data_dict[q.uid]["shots_combined"]

        fig = plt.figure(figsize=(24, 12), constrained_layout=True)
        grid = fig.add_gridspec(2, 3, hspace=0.22, wspace=0.22)
        ax_i = fig.add_subplot(grid[0, 0])
        ax_q = fig.add_subplot(grid[0, 1], sharex=ax_i)
        ax_phase = fig.add_subplot(grid[0, 2], sharex=ax_i)
        ax_iq = fig.add_subplot(grid[1, 0:2])
        ax_amp = fig.add_subplot(grid[1, 2], sharex=ax_i)

        fig.suptitle(timestamped_title(f"IQ Trajectories {q.uid}"), fontsize=14)

        ax_i.set_title("I(t)")
        ax_i.set_xlabel("Time (ns)")
        ax_i.set_ylabel("Real Signal Component, $V_{\\mathrm{I}}$ (a.u.)")

        ax_q.set_title("Q(t)")
        ax_q.set_xlabel("Time (ns)")
        ax_q.set_ylabel("Imaginary Signal Component, $V_{\\mathrm{Q}}$ (a.u.)")

        ax_phase.set_title("Phase(t)")
        ax_phase.set_xlabel("Time (ns)")
        ax_phase.set_ylabel("Phase (rad)")
        ax_phase.text(
            0.01,
            0.98,
            f"Masked if |IQ| < {phase_mask_relative_threshold:.0%} of peak",
            transform=ax_phase.transAxes,
            va="top",
            fontsize=9,
        )

        ax_iq.set_title("IQ Plane")
        ax_iq.set_xlabel("Real Signal Component, $V_{\\mathrm{I}}$ (a.u.)")
        ax_iq.set_ylabel("Imaginary Signal Component, $V_{\\mathrm{Q}}$ (a.u.)")

        ax_amp.set_title("|IQ|(t)")
        ax_amp.set_xlabel("Time (ns)")
        ax_amp.set_ylabel("Magnitude, $|IQ|$ (a.u.)")

        if opts2.plot_phase_boundaries:
            # In this workflow, RAW time traces are aligned to the acquisition window
            # start in practice. Use integration-start as t=0 for boundary overlays.
            readout_length_s = getattr(q.parameters, "readout_length", None)
            readout_integration_delay_s = getattr(
                q.parameters, "readout_integration_delay", None
            )
            readout_integration_length_s = getattr(
                q.parameters, "readout_integration_length", None
            )
            boundary_lines = []
            int_delay_ns = (
                float(readout_integration_delay_s) * 1e9
                if readout_integration_delay_s is not None
                else 0.0
            )
            # Integration-start aligned axis: int_start is always at t=0.
            boundary_lines.append(("int_start", 0.0, ":"))
            if readout_integration_length_s is not None:
                boundary_lines.append(
                    ("int_end", float(readout_integration_length_s) * 1e9, "-.")
                )
            if readout_length_s is not None:
                boundary_lines.append(
                    ("ro_end", float(readout_length_s) * 1e9 - int_delay_ns, "--")
                )
            for label, boundary_ns, style in boundary_lines:
                ax_i.axvline(
                    boundary_ns, color="k", linestyle=style, linewidth=1.0, alpha=0.3
                )
                ax_q.axvline(
                    boundary_ns, color="k", linestyle=style, linewidth=1.0, alpha=0.3
                )
                ax_phase.axvline(
                    boundary_ns, color="k", linestyle=style, linewidth=1.0, alpha=0.3
                )
                ax_amp.axvline(
                    boundary_ns, color="k", linestyle=style, linewidth=1.0, alpha=0.3
                )
            ax_phase.text(
                0.01,
                0.90,
                "t=0: int_start  |  --: ro_end  |  -.: int_end",
                transform=ax_phase.transAxes,
                va="top",
                fontsize=8,
                alpha=0.8,
            )

        for i, s in enumerate(states):
            color = color_map.get(i, f"C{i}")
            complex_trace = np.asarray(processed_data_dict[q.uid][s], dtype=complex)
            I_trace = np.real(complex_trace)
            Q_trace = np.imag(complex_trace)
            amplitude_trace = np.abs(complex_trace)
            phase_trace = np.unwrap(np.angle(complex_trace))
            amplitude_max = float(np.max(amplitude_trace)) if amplitude_trace.size else 0.0
            phase_mask_threshold = phase_mask_relative_threshold * amplitude_max
            phase_trace_masked = np.array(phase_trace, copy=True)
            phase_trace_masked[amplitude_trace < phase_mask_threshold] = np.nan
            
            axis = np.arange(len(I_trace)) * plot_chunk_size * plot_sample_dt_ns
            marker_every = max(len(axis) // 120, 1)
            ax_i.plot(axis, I_trace, "-", color=color, linewidth=1.3, label=s)
            ax_i.plot(
                axis,
                I_trace,
                "o",
                color=color,
                markersize=2.0,
                alpha=0.55,
                markevery=marker_every,
            )

            ax_q.plot(axis, Q_trace, "-", color=color, linewidth=1.3, label=s)
            ax_q.plot(
                axis,
                Q_trace,
                "o",
                color=color,
                markersize=2.0,
                alpha=0.55,
                markevery=marker_every,
            )

            ax_iq.plot(I_trace, Q_trace, "-", color=color, linewidth=1.1, label=s)
            ax_iq.scatter(
                I_trace[::marker_every],
                Q_trace[::marker_every],
                c=color,
                s=10,
                alpha=0.6,
            )

            valid_phase = np.isfinite(phase_trace_masked)
            ax_phase.scatter(
                axis[valid_phase],
                phase_trace_masked[valid_phase],
                color=color,
                s=10,
                alpha=0.7,
            )
            ax_phase.plot(axis, phase_trace_masked, "-", color=color, linewidth=1.3, label=s)

            ax_amp.plot(axis, amplitude_trace, "-", color=color, linewidth=1.3, label=s)

            dI = np.gradient(I_trace)
            dQ = np.gradient(Q_trace)

            step = max(len(I_trace) // 90, 2)
            ax_iq.quiver(
                I_trace[::step],
                Q_trace[::step],
                dI[::step],
                dQ[::step],
                angles="xy",
                scale_units="xy",
                scale=1,
                color=color,
                alpha=0.6,
                width=0.002,
            )

        ax_iq.set_aspect("equal", adjustable="box")
        for axis_obj in (ax_i, ax_q, ax_phase, ax_amp):
            axis_obj.grid(True, linestyle=":", alpha=0.35)
        ax_iq.grid(True, linestyle=":", alpha=0.25)

        handles, labels = ax_i.get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper right", frameon=False, title="State")
   
        # if opts.save_figures:
        #     workflow.save_artifact(f"IQ_Trajectories_{q.uid}", fig)

        if opts.close_figures:
            plt.close(fig)
        
        figures[q.uid] = fig

    return figures

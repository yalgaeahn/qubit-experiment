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

    processed_data_dict = demodulate_time_traces(qubits,result,states)
    processed_data_dict =average_chunk_time_traces(qubits, processed_data_dict, states)
    with workflow.if_(options.do_plotting):
        plot_iq_trajectories(qubits,states,processed_data_dict) 
    workflow.return_(None)

@workflow.task
def demodulate_time_traces(
    qubits: QuantumElements,
    result: RunExperimentResults,
    states: Sequence[str],
) -> dict[str, dict[str, ArrayLike | dict]]:
    
    
    qubits = validate_and_convert_qubits_sweeps(qubits)
    validate_result(result)
    states_map = {"g": 0, "e": 1, "f": 2}
    processed_data_dict = {q.uid: {} for q in qubits} 
    for q in qubits:
        _IF_freq = q.parameters.readout_resonator_frequency - q.parameters.readout_lo_frequency
        time_trace = {}
        for s in states:
            time_trace[s] = result[dsl.handles.calibration_trace_handle(q.uid, s)].data
            _I = np.real(time_trace[s])
            _Q = np.imag(time_trace[s])
            cos = np.cos(2*np.pi*_IF_freq*np.arange(len(_I))/2 * 1e-9)
            sin = np.sin(2*np.pi*_IF_freq*np.arange(len(_Q))/2 * 1e-9)
            # demodulate the data
            I_demod = _I * cos + _Q * sin
            # Rotate the quadratures into the demodulated frame
            Q_demod = -_I * sin + _Q * cos
            processed_data_dict[q.uid][s] = I_demod + 1j*Q_demod
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

    for q in qubits: # type: ignore
        #averaged_time_traces = {}
        for s in states:
            time_trace = processed_data_dict[q.uid][s]
            _N= len(time_trace)
            processed_data_dict[q.uid][s]=time_trace[:_N].reshape(-1,opts.chunk_size).mean(axis=1)
    
    return processed_data_dict




@workflow.task
def plot_iq_trajectories(
    qubits: QuantumElements,
    states: Sequence[str],
    processed_data_dict: dict[str, dict[str, ArrayLike | dict]],
    options: BasePlottingOptions | None = None,
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
    for q in qubits:
        # shots_per_state = processed_data_dict[q.uid]["shots_per_state"]
        # shots_combined = processed_data_dict[q.uid]["shots_combined"]

        fig, ax = plt.subplots(1,3, figsize=(28,10))
        
        ax[0].set_title(timestamped_title(f"IQ Trajectories {q.uid}"))
        ax[0].set_xlabel("Time (ns)")
        ax[0].set_ylabel("Real Signal Component, $V_{\\mathrm{I}}$ (a.u.)")

        ax[1].set_title(timestamped_title(f"IQ Trajectories {q.uid}"))
        ax[1].set_xlabel("Time (ns)")
        ax[1].set_ylabel("Imaginary Signal Component, $V_{\\mathrm{Q}}$ (a.u.)")
        
        ax[2].set_title(timestamped_title(f"IQ Trajectories {q.uid}"))
        ax[2].set_xlabel("Real Signal Component, $V_{\\mathrm{I}}$ (a.u.)")
        ax[2].set_ylabel("Imaginary Signal Component, $V_{\\mathrm{Q}}$ (a.u.)")

        for i, s in enumerate(states):
          
            I_trace = np.real(processed_data_dict[q.uid][s])
            Q_trace = np.imag(processed_data_dict[q.uid][s])
            
            axis = np.arange(len(I_trace))*opts2.chunk_size
            ax[0].scatter(axis,I_trace,color=color_map[i],label=s)
            ax[0].plot(axis,I_trace,'-',color=color_map[i],label=s)

            ax[1].scatter(axis,Q_trace,color=color_map[i],label=s)
            ax[1].plot(axis,Q_trace,'-',color=color_map[i],label=s)

            ax[2].scatter(I_trace,Q_trace,c=color_map[i],label=s)
            ax[2].plot(I_trace,Q_trace,'-',c=color_map[i],label=s)

            dI = np.gradient(I_trace)
            dQ = np.gradient(Q_trace)

            step=2
            ax[2].quiver(I_trace[::step], Q_trace[::step], dI[::step], dQ[::step], angles='xy', scale_units='xy', scale=1, color=color_map[i])
            ax[2].set_aspect("equal")
       
            ax[0].legend(frameon=False)
            ax[1].legend(frameon=False)
            ax[2].legend(frameon=False)
   
        # if opts.save_figures:
        #     workflow.save_artifact(f"IQ_Trajectories_{q.uid}", fig)

        if opts.close_figures:
            plt.close(fig)
        
        figures[q.uid] = fig

    return figures


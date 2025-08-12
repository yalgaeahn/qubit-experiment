"""2D plotting analysis workflow for direct CR Hamiltonian Tomography.

Works with experiments in `experiments/direct_cr_hamiltonian_tomography.py`.

This analysis provides:
- Raw I/Q heatmaps versus (CR amplitude, pulse length)
- Magnitude and phase heatmaps versus (CR amplitude, pulse length)

Notes:
- It expects the experiment to use `dsl.handles.result_handle(targ.uid)` for the
    measurement handle of the target qubit, as implemented in the experiment file.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from laboneq import workflow

from laboneq_applications.analysis.plotting_helpers import (
    plot_raw_complex_data_2d,
    plot_signal_magnitude_and_phase_2d,
)

if TYPE_CHECKING:
    from laboneq.workflow.tasks.run_experiment import RunExperimentResults

    from laboneq_applications.typing import QuantumElements, QubitSweepPoints

if TYPE_CHECKING:
    from laboneq.workflow.tasks.run_experiment import RunExperimentResults
    from laboneq_applications.typing import QuantumElements, QubitSweepPoints


@workflow.workflow_options
class HamiltonianTomography2DAnalysisOptions:
    """Options for CR Hamiltonian tomography plotting.

    Attributes:
        do_raw_data_plotting: Whether to plot raw complex I/Q
        do_plotting_magnitude_phase: Whether to plot magnitude/phase
    """

    do_raw_data_plotting = workflow.option_field(
        True, description="Whether to plot the raw complex I/Q heatmaps."
    )
    do_plotting_magnitude_phase = workflow.option_field(
        True, description="Whether to plot the magnitude and phase heatmaps."
    )


@workflow.workflow(name="analysis_workflow")
def analysis_workflow(
    result: RunExperimentResults,
    qubits: QuantumElements,
    sweep_points_1d: QubitSweepPoints,
    sweep_points_2d: QubitSweepPoints,
    label_sweep_points_1d: str,
    label_sweep_points_2d: str,
    scaling_sweep_points_2d: float = 1.0,
    options: Optional[HamiltonianTomography2DAnalysisOptions] = None,
) -> None:
    """Plot direct-CR tomography data as 2D heatmaps.

    Arguments:
        result: Results from run_experiment
        qubits: Target qubit(s) whose readout handles exist in results
        sweep_points_1d: First sweep axis values (e.g., CR amplitude)
        sweep_points_2d: Second sweep axis values (e.g., pulse length)
        label_sweep_points_1d: Label for the first axis
        label_sweep_points_2d: Label for the second axis
        scaling_sweep_points_2d: Optional scaling for display (e.g., 1e9 to ns)
        options: Plotting options
    """

    # Default options if not provided
    opts = HamiltonianTomography2DAnalysisOptions() if options is None else options

    with workflow.if_(opts.do_raw_data_plotting):
        plot_raw_complex_data_2d(
            result=result,
            qubits=qubits,
            sweep_points_1d=sweep_points_1d,
            sweep_points_2d=sweep_points_2d,
            label_sweep_points_1d=label_sweep_points_1d,
            label_sweep_points_2d=label_sweep_points_2d,
            scaling_sweep_points_2d=scaling_sweep_points_2d,
        )

    with workflow.if_(opts.do_plotting_magnitude_phase): 
        plot_signal_magnitude_and_phase_2d(
            result=result,
            qubits=qubits,
            sweep_points_1d=sweep_points_1d,
            sweep_points_2d=sweep_points_2d,
            label_sweep_points_1d=label_sweep_points_1d,
            label_sweep_points_2d=label_sweep_points_2d,
            scaling_sweep_points_2d=scaling_sweep_points_2d,
        )

    
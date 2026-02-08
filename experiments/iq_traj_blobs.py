# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Combined IQ trajectory + integrated IQ blob experiment.

This experiment acquires raw single-shot calibration traces and runs a combined
analysis that produces both trajectory plots and integrated IQ blob plots from
exactly the same acquisition data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from laboneq import workflow
from laboneq.simple import (
    AcquisitionType,
    AveragingMode,
    Experiment,
    SweepParameter,
    dsl,
)
from laboneq.workflow.tasks import compile_experiment, run_experiment

from analysis.iq_traj_blobs import analysis_workflow
from laboneq_applications.core import validation
from laboneq_applications.experiments.options import BaseExperimentOptions
from laboneq_applications.tasks.parameter_updating import (
    temporary_qpu,
    temporary_quantum_elements_from_qpu,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from laboneq.dsl.quantum import QuantumParameters
    from laboneq.dsl.quantum.qpu import QPU
    from laboneq.dsl.session import Session

    from laboneq_applications.typing import QuantumElements


@workflow.task_options(base_class=BaseExperimentOptions)
class IQTrajBlobsExperimentOptions:
    """Options for the combined iq_traj_blobs experiment."""

    acquisition_type: AcquisitionType = workflow.option_field(
        AcquisitionType.RAW,
        description="Acquisition type used for the experiment.",
    )
    averaging_mode: AveragingMode = workflow.option_field(
        AveragingMode.SINGLE_SHOT,
        description="Averaging mode used for the experiment.",
    )
    max_raw_results_per_rt: int = workflow.option_field(
        1024,
        description=(
            "Hardware/controller limit for raw results per real-time execution. "
            "Used to automatically split large RAW single-shot acquisitions "
            "into near-time chunks."
        ),
    )
    max_shots_per_rt: int | None = workflow.option_field(
        None,
        description=(
            "Optional manual cap for shots per real-time RAW execution to avoid "
            "instrument memory overflow for long traces."
        ),
    )


@workflow.workflow_options
class IQTrajBlobsExperimentWorkflowOptions:
    """Workflow options for the combined iq_traj_blobs experiment."""

    do_analysis: bool = workflow.option_field(
        True, description="Whether to run the analysis workflow."
    )


@workflow.workflow(name="iq_traj_blobs")
def experiment_workflow(
    session: Session,
    qpu: QPU,
    qubits: QuantumElements,
    states: Sequence[str],
    temporary_parameters: dict[str | tuple[str, str, str], dict | QuantumParameters]
    | None = None,
    options: IQTrajBlobsExperimentWorkflowOptions | None = None,
) -> None:
    """Run combined IQ trajectory/blob experiment and optional analysis."""
    options = (
        IQTrajBlobsExperimentWorkflowOptions() if options is None else options
    )
    temp_qpu = temporary_qpu(qpu, temporary_parameters)
    qubits = temporary_quantum_elements_from_qpu(temp_qpu, qubits)
    exp = create_experiment(temp_qpu, qubits, states)
    compiled_exp = compile_experiment(session, exp)
    result = run_experiment(session, compiled_exp)
    with workflow.if_(options.do_analysis):
        analysis_workflow(result, qubits, states)
    workflow.return_(result)


@workflow.task
@dsl.qubit_experiment
def create_experiment(
    qpu: QPU,
    qubits: QuantumElements,
    states: Sequence[str],
    options: IQTrajBlobsExperimentOptions | None = None,
) -> Experiment:
    """Create an experiment that captures raw single-shot calibration traces."""
    opts = IQTrajBlobsExperimentOptions() if options is None else options
    qubits = validation.validate_and_convert_qubits_sweeps(qubits)
    states = list(states)
    qop = qpu.quantum_operations
    target_count = int(opts.count)

    if target_count < 1:
        raise ValueError("'count' must be >= 1.")

    def _add_calibration_traces() -> None:
        qop.calibration_traces.omit_section(
            qubits=qubits,
            states=states,
            active_reset=opts.active_reset,
            active_reset_states=opts.active_reset_states,
            active_reset_repetitions=opts.active_reset_repetitions,
        )

    rt_count = target_count
    full_nt_steps = 0
    remainder_count = 0
    if opts.acquisition_type == AcquisitionType.RAW:
        raw_results_per_shot = len(qubits) * len(states)
        max_raw_results = int(opts.max_raw_results_per_rt)
        if max_raw_results < 1:
            raise ValueError("'max_raw_results_per_rt' must be >= 1.")

        max_shots_per_rt = max_raw_results // raw_results_per_shot
        if max_shots_per_rt < 1:
            raise ValueError(
                "Requested qubits/states exceed RAW result capacity per real-time "
                "execution. Reduce number of qubits/states or increase "
                "'max_raw_results_per_rt'."
            )

        rt_count = min(target_count, max_shots_per_rt)
        if opts.max_shots_per_rt is not None:
            manual_cap = int(opts.max_shots_per_rt)
            if manual_cap < 1:
                raise ValueError("'max_shots_per_rt' must be >= 1 when provided.")
            rt_count = min(rt_count, manual_cap)
        full_nt_steps = target_count // rt_count
        remainder_count = target_count % rt_count
    else:
        full_nt_steps = 1

    if full_nt_steps > 1:
        nt_step = SweepParameter(uid="nt_chunk", values=list(range(full_nt_steps)))
        with dsl.sweep(
            uid="nt_chunk_sweep",
            name="nt_chunk_sweep",
            parameter=nt_step,
        ):
            with dsl.acquire_loop_rt(
                count=rt_count,
                averaging_mode=opts.averaging_mode,
                acquisition_type=opts.acquisition_type,
                repetition_mode=opts.repetition_mode,
                repetition_time=opts.repetition_time,
                reset_oscillator_phase=opts.reset_oscillator_phase,
            ):
                _add_calibration_traces()
        if remainder_count > 0:
            with dsl.acquire_loop_rt(
                count=remainder_count,
                averaging_mode=opts.averaging_mode,
                acquisition_type=opts.acquisition_type,
                repetition_mode=opts.repetition_mode,
                repetition_time=opts.repetition_time,
                reset_oscillator_phase=opts.reset_oscillator_phase,
            ):
                _add_calibration_traces()
    elif full_nt_steps == 1:
        with dsl.acquire_loop_rt(
            count=rt_count,
            averaging_mode=opts.averaging_mode,
            acquisition_type=opts.acquisition_type,
            repetition_mode=opts.repetition_mode,
            repetition_time=opts.repetition_time,
            reset_oscillator_phase=opts.reset_oscillator_phase,
        ):
            _add_calibration_traces()
    else:
        with dsl.acquire_loop_rt(
            count=remainder_count,
            averaging_mode=opts.averaging_mode,
            acquisition_type=opts.acquisition_type,
            repetition_mode=opts.repetition_mode,
            repetition_time=opts.repetition_time,
            reset_oscillator_phase=opts.reset_oscillator_phase,
        ):
            _add_calibration_traces()

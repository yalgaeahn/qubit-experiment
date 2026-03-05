"""Experiment workflow for readout amplitude optimization."""

from __future__ import annotations

from typing import TYPE_CHECKING

from laboneq import workflow
from laboneq.dsl.enums import AcquisitionType, AveragingMode
from laboneq.simple import Experiment, SweepParameter, dsl

from analysis.readout_amplitude_sweep import analysis_workflow
from laboneq_applications.core import validation
from laboneq_applications.experiments.options import BaseExperimentOptions
from laboneq_applications.tasks.parameter_updating import (
    temporary_qpu,
    temporary_quantum_elements_from_qpu,
    update_qpu,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from laboneq.dsl.quantum import QuantumParameters
    from laboneq.dsl.quantum.qpu import QPU
    from laboneq.dsl.quantum.quantum_element import QuantumElement
    from laboneq.dsl.session import Session
    from numpy.typing import ArrayLike


@workflow.task_options(base_class=BaseExperimentOptions)
class ReadoutAmplitudeSweepExperimentOptions:
    """Options for readout amplitude sweep acquisition."""

    count: int = workflow.option_field(
        4096,
        description="Number of single shots per sweep point and state.",
    )
    acquisition_type: AcquisitionType = workflow.option_field(
        AcquisitionType.INTEGRATION,
        description="Acquire integrated complex IQ samples.",
    )
    averaging_mode: AveragingMode = workflow.option_field(
        AveragingMode.SINGLE_SHOT,
        description="Single-shot acquisition mode.",
    )
    states: str | tuple[str, ...] = workflow.option_field(
        "ge",
        description="Calibration states. Only g/e is supported in this workflow.",
    )


@workflow.workflow_options
class ReadoutAmplitudeSweepWorkflowOptions:
    """Workflow options for readout amplitude optimization."""

    do_analysis: bool = workflow.option_field(
        True, description="Whether to run analysis."
    )
    update: bool = workflow.option_field(
        False, description="Whether to apply optimized parameters to the input qpu."
    )


def _states_to_tuple(states: str | Sequence[str]) -> tuple[str, ...]:
    state_tuple = tuple(states) if isinstance(states, str) else tuple(states)
    if len(state_tuple) != 2 or set(state_tuple) != {"g", "e"}:
        raise ValueError("Only g/e states are supported. Use states='ge'.")
    return state_tuple


@workflow.task(save=False)
def _compile_experiment_no_log(session: Session, experiment: Experiment):
    return session.compile(experiment=experiment)


@workflow.task(save=False)
def _run_experiment_no_log(session: Session, compiled_experiment):
    return session.run(compiled_experiment)


@workflow.workflow(name="readout_amplitude_sweep")
def experiment_workflow(
    session: Session,
    qpu: QPU,
    qubit: QuantumElement,
    amplitudes: ArrayLike,
    temporary_parameters: dict[str | tuple[str, str, str], dict | QuantumParameters]
    | None = None,
    options: ReadoutAmplitudeSweepWorkflowOptions | None = None,
) -> None:
    """Run readout amplitude sweep and optionally update qpu."""
    options = ReadoutAmplitudeSweepWorkflowOptions() if options is None else options
    temp_qpu = temporary_qpu(qpu, temporary_parameters)
    qubit = temporary_quantum_elements_from_qpu(temp_qpu, qubit)

    exp = create_experiment(
        qpu=temp_qpu,
        qubit=qubit,
        amplitudes=amplitudes,
    )
    compiled = _compile_experiment_no_log(session, exp)
    result = _run_experiment_no_log(session, compiled)

    analysis_result = None
    with workflow.if_(options.do_analysis):
        analysis_result = analysis_workflow(
            result=result,
            qubit=qubit,
            amplitudes=amplitudes,
        )
        with workflow.if_(options.update):
            update_qpu(qpu, analysis_result.output["new_parameter_values"])

    workflow.return_({"status": "completed"})


@workflow.task(save=False)
@dsl.qubit_experiment
def create_experiment(
    qpu: QPU,
    qubit: QuantumElement,
    amplitudes: ArrayLike,
    options: ReadoutAmplitudeSweepExperimentOptions | None = None,
) -> Experiment:
    """Create a calibration-trace experiment with readout amplitude sweep."""
    opts = ReadoutAmplitudeSweepExperimentOptions() if options is None else options
    states = _states_to_tuple(opts.states)
    qubit, amplitude_points = validation.validate_and_convert_single_qubit_sweeps(
        qubit, amplitudes
    )

    if AcquisitionType(opts.acquisition_type) != AcquisitionType.INTEGRATION:
        raise ValueError("readout_amplitude_sweep requires acquisition_type=INTEGRATION.")
    if AveragingMode(opts.averaging_mode) != AveragingMode.SINGLE_SHOT:
        raise ValueError("readout_amplitude_sweep requires averaging_mode=SINGLE_SHOT.")

    qop = qpu.quantum_operations
    measure_section_length = qop.measure_section_length(qubit)
    with dsl.sweep(
        name=f"readout_amplitude_{qubit.uid}",
        parameter=SweepParameter(f"readout_amplitude_{qubit.uid}", amplitude_points),
    ) as amplitude:
        with dsl.acquire_loop_rt(
            count=opts.count,
            averaging_mode=opts.averaging_mode,
            acquisition_type=opts.acquisition_type,
            repetition_mode=opts.repetition_mode,
            repetition_time=opts.repetition_time,
            reset_oscillator_phase=opts.reset_oscillator_phase,
        ):
            qop.set_readout_amplitude(qubit, amplitude=amplitude)
            qop.calibration_traces.omit_section(
                qubits=qubit,
                states=states,
                active_reset=opts.active_reset,
                active_reset_states=opts.active_reset_states,
                active_reset_repetitions=opts.active_reset_repetitions,
                measure_section_length=measure_section_length,
            )

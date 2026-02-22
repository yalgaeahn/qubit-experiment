# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""This module defines the IQ time-trace experiment."""

from __future__ import annotations

from typing import TYPE_CHECKING

from laboneq import workflow
from laboneq.simple import AcquisitionType, AveragingMode, Experiment, dsl
from laboneq.workflow.tasks import compile_experiment, run_experiment

from analysis.iq_time_trace import analysis_workflow
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
    from laboneq.dsl.session import Session

    from laboneq_applications.typing import QuantumElements


_FIND_FLAT_WINDOW_INTEGRATION_LENGTH_S = 2.0e-6


def _validate_states(states: Sequence[str]) -> None:
    invalid_states = sorted({s for s in states if s not in ("g", "e", "f")})
    if invalid_states:
        raise ValueError(
            "Invalid states for iq_time_trace: "
            f"{invalid_states}. Only 'g', 'e', 'f' are supported."
        )


@workflow.task_options(base_class=BaseExperimentOptions)
class IQTimeTraceExperimentOptions:
    """Options for the IQ time-trace experiment."""

    acquisition_type: AcquisitionType = workflow.option_field(
        AcquisitionType.RAW,
        description="Acquisition type to use for the experiment.",
    )
    averaging_mode: AveragingMode = workflow.option_field(
        AveragingMode.CYCLIC,
        description="Averaging mode used for the experiment.",
    )
    reset_oscillator_phase: bool = workflow.option_field(
        True,
        description="Reset oscillator phase at each repetition.",
    )


@workflow.workflow_options
class IQTimeTraceExperimentWorkflowOptions:
    """Options for IQ time-trace workflow."""

    do_analysis: bool = workflow.option_field(
        True, description="Whether to run the analysis workflow."
    )
    update: bool = workflow.option_field(
        False, description="Whether to update qubit parameters from analysis results."
    )
    find_flat_window: bool = workflow.option_field(
        False,
        description=(
            "Force readout_integration_delay=0 and readout_integration_length=2.0e-6 "
            "during acquisition, then detect flat_start/flat_end from IQ traces."
        ),
    )


@workflow.task
def _apply_find_flat_window_capture_policy(qubits: QuantumElements) -> None:
    """Force capture settings used for flat-window detection."""
    validated_qubits = validation.validate_and_convert_qubits_sweeps(qubits)
    for q in validated_qubits:
        q.parameters.readout_integration_delay = 0.0
        q.parameters.readout_integration_length = _FIND_FLAT_WINDOW_INTEGRATION_LENGTH_S


@workflow.workflow(name="iq_time_trace")
def experiment_workflow(
    session: Session,
    qpu: QPU,
    qubits: QuantumElements,
    states: Sequence[str],
    temporary_parameters: dict[str | tuple[str, str, str], dict | QuantumParameters]
    | None = None,
    options: IQTimeTraceExperimentWorkflowOptions | None = None,
) -> None:
    """Run IQ time-trace experiment and optional analysis."""
    opts = IQTimeTraceExperimentWorkflowOptions() if options is None else options
    if opts.find_flat_window and not opts.do_analysis:
        raise ValueError(
            "find_flat_window=True requires do_analysis=True because flat-window "
            "detection is performed in analysis_workflow."
        )

    temp_qpu = temporary_qpu(qpu, temporary_parameters)
    qubits = temporary_quantum_elements_from_qpu(temp_qpu, qubits)
    qubits = validation.validate_and_convert_qubits_sweeps(qubits)
    with workflow.if_(opts.find_flat_window):
        _apply_find_flat_window_capture_policy(qubits)
    exp = create_experiment(
        qpu=temp_qpu,
        qubits=qubits,
        states=states,
    )
    compiled_exp = compile_experiment(session, exp)
    result = run_experiment(session, compiled_exp)
    analysis_result = None
    with workflow.if_(opts.do_analysis):
        analysis_result = analysis_workflow(
            result=result,
            qubits=qubits,
            states=states,
            find_flat_window=opts.find_flat_window,
        )
        with workflow.if_(opts.update):
            with workflow.if_(opts.find_flat_window):
                update_qpu(qpu, analysis_result.output["new_parameter_values"])
    workflow.return_(result)


@workflow.task
@dsl.qubit_experiment
def create_experiment(
    qpu: QPU,
    qubits: QuantumElements,
    states: Sequence[str],
    options: IQTimeTraceExperimentOptions | None = None,
) -> Experiment:
    """Create IQ time-trace experiment using calibration traces."""
    opts = IQTimeTraceExperimentOptions() if options is None else options
    _validate_states(states)
    acquisition_type = AcquisitionType(opts.acquisition_type)
    if acquisition_type != AcquisitionType.RAW:
        raise ValueError(
            "The only allowed acquisition_type for this experiment is "
            "AcquisitionType.RAW because the experiment acquires raw traces."
        )
    qubits = validation.validate_and_convert_qubits_sweeps(qubits)

    qop = qpu.quantum_operations
    max_measure_section_length = qop.measure_section_length(qubits)

    with dsl.acquire_loop_rt(
        count=opts.count,
        averaging_mode=opts.averaging_mode,
        acquisition_type=acquisition_type,
        repetition_mode=opts.repetition_mode,
        repetition_time=opts.repetition_time,
        reset_oscillator_phase=opts.reset_oscillator_phase,
    ):
        qop.calibration_traces.omit_section(
            qubits=qubits,
            states=states,
            active_reset=opts.active_reset,
            active_reset_states=opts.active_reset_states,
            active_reset_repetitions=opts.active_reset_repetitions,
            measure_section_length=max_measure_section_length,
        )

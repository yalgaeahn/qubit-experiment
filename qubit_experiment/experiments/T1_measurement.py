# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Measure T1 using single-shot integrated IQ and per-shot calibration traces.

Repeat the lifetime delay sweep followed by two reference states for each shot.
Only integration/single-shot acquisition with calibration rotation is supported.
Use the dedicated T1 analysis; persistent QPU updates are opt-in and validated.
"""

from __future__ import annotations

from numbers import Integral
from typing import TYPE_CHECKING

from laboneq import workflow
from laboneq.simple import (
    AcquisitionType,
    AveragingMode,
    Experiment,
    SectionAlignment,
    SweepParameter,
    dsl,
)
from laboneq.workflow.tasks import compile_experiment, run_experiment
from laboneq_applications.experiments.options import TuneupExperimentOptions, TuneUpWorkflowOptions
from laboneq_applications.tasks.parameter_updating import (
    temporary_qpu,
    temporary_quantum_elements_from_qpu,
    update_qpu,
)

from qubit_experiment.analysis.T1_measurement import (
    _options,
    analysis_workflow,
    validate_calibration_options,
    validate_delays,
)

if TYPE_CHECKING:
    from laboneq.dsl.quantum import QuantumParameters
    from laboneq.dsl.quantum.qpu import QPU
    from laboneq.dsl.session import Session
    from laboneq_applications.typing import QuantumElements, QubitSweepPoints


@workflow.task_options(base_class=TuneupExperimentOptions)
class T1ExperimentOptions:
    """Single-shot lifetime acquisition; count is shots per delay/reference."""

    count: int = workflow.option_field(
        1024, description="Shots per delay and reference; at least two."
    )
    acquisition_type: str | AcquisitionType = workflow.option_field(
        AcquisitionType.INTEGRATION, description="Only integrated complex IQ is supported."
    )
    averaging_mode: str | AveragingMode = workflow.option_field(
        AveragingMode.SINGLE_SHOT, description="Only single-shot acquisition is supported."
    )
    do_pca: bool = workflow.option_field(False, description="PCA is not supported.")
    do_rotation: bool = workflow.option_field(True, description="Calibration rotation is required.")


@workflow.workflow_options(base_class=TuneUpWorkflowOptions)
class T1WorkflowOptions:
    """Analyze by default and never update parameters unless explicitly requested."""

    do_analysis: bool = workflow.option_field(True, description="Analyze the single-shot results.")
    update: bool = workflow.option_field(False, description="Apply valid T1 parameter estimates.")


@workflow.workflow(name="T1_measurement")
def experiment_workflow(
    session: Session,
    qpu: QPU,
    qubits: list[str] | str,
    delays: QubitSweepPoints,
    temporary_parameters: dict[str | tuple[str, str, str], dict | QuantumParameters] | None = None,
    options: T1WorkflowOptions | None = None,
) -> None:
    """Return original Results; analysis is available in the nested workflow tasks."""
    temp_qpu = temporary_qpu(qpu, temporary_parameters)
    elements = temporary_quantum_elements_from_qpu(temp_qpu, qubits)
    experiment = create_experiment(temp_qpu, elements, delays)
    compiled = compile_experiment(session, experiment)
    result = run_experiment(session, compiled)
    with workflow.if_(options.do_analysis):
        analysis = analysis_workflow(result, elements, delays)
        with workflow.if_(options.update):
            update_qpu(qpu, analysis.output["new_parameter_values"])
    workflow.return_(result)


@workflow.task
@dsl.qubit_experiment
def create_experiment(
    qpu: QPU,
    qubits: QuantumElements,
    delays: QubitSweepPoints,
    options: T1ExperimentOptions | None = None,
) -> Experiment:
    """Build the local lifetime sequence with a validated single-shot contract."""
    opts = _options(options, T1ExperimentOptions)
    validate_calibration_options(opts)
    if isinstance(opts.count, bool) or not isinstance(opts.count, Integral) or opts.count < 2:
        raise ValueError("T1_measurement count must be an integer >= 2.")
    if AveragingMode(opts.averaging_mode) != AveragingMode.SINGLE_SHOT:
        raise ValueError("T1_measurement requires AveragingMode.SINGLE_SHOT.")
    if AcquisitionType(opts.acquisition_type) != AcquisitionType.INTEGRATION:
        raise ValueError("T1_measurement requires AcquisitionType.INTEGRATION.")
    qubits, delays = validate_delays(qubits, delays)
    sweep_parameters = [
        SweepParameter(f"delays_{q.uid}", grid, axis_name=q.uid) for q, grid in zip(qubits, delays)
    ]
    qop = qpu.quantum_operations
    measure_length = qop.measure_section_length(qubits)
    with dsl.acquire_loop_rt(
        count=opts.count,
        averaging_mode=opts.averaging_mode,
        acquisition_type=opts.acquisition_type,
        repetition_mode=opts.repetition_mode,
        repetition_time=opts.repetition_time,
        reset_oscillator_phase=opts.reset_oscillator_phase,
    ):
        with dsl.sweep(name="T1_measurement_sweep", parameter=sweep_parameters):
            if opts.active_reset:
                qop.active_reset(
                    qubits,
                    active_reset_states=opts.active_reset_states,
                    number_resets=opts.active_reset_repetitions,
                    measure_section_length=measure_length,
                )
            with dsl.section(name="main", alignment=SectionAlignment.RIGHT):
                with dsl.section(name="main_drive", alignment=SectionAlignment.RIGHT):
                    for q, delay in zip(qubits, sweep_parameters):
                        qop.prepare_state.omit_section(q, opts.transition[0])
                        section = qop.x180(q, transition=opts.transition)
                        section.alignment = SectionAlignment.RIGHT
                        qop.delay(q, time=delay)
                with dsl.section(name="main_measure", alignment=SectionAlignment.LEFT):
                    for q in qubits:
                        qop.measure(q, dsl.handles.result_handle(q.uid))
                        qop.passive_reset(q)
        qop.calibration_traces.omit_section(
            qubits=qubits,
            states=opts.cal_states,
            active_reset=opts.active_reset,
            active_reset_states=opts.active_reset_states,
            active_reset_repetitions=opts.active_reset_repetitions,
        )

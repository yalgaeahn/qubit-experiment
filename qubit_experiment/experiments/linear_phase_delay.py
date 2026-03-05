# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Linear phase-delay calibration experiment.

This experiment sweeps readout frequency and collects complex S21 in a fixed qubit
state (default: |g>). The paired analysis extracts the linear phase-delay tau.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from laboneq import workflow
from laboneq.dsl.enums import AcquisitionType
from laboneq.simple import Experiment, SweepParameter, dsl
from laboneq.workflow.tasks import (
    compile_experiment,
    run_experiment,
)

from analysis.linear_phase_delay import analysis_workflow
from laboneq_applications.core import validation
from laboneq_applications.experiments.options import (
    BaseExperimentOptions,
    TuneUpWorkflowOptions,
)
from laboneq_applications.tasks.parameter_updating import (
    temporary_qpu,
    temporary_quantum_elements_from_qpu,
    update_qpu,
)

if TYPE_CHECKING:
    from laboneq.dsl.quantum import QuantumParameters
    from laboneq.dsl.quantum.qpu import QPU
    from laboneq.dsl.quantum.quantum_element import QuantumElement
    from laboneq.dsl.session import Session
    from numpy.typing import ArrayLike


@workflow.task_options(base_class=BaseExperimentOptions)
class LinearPhaseDelayExperimentOptions:
    """Options for linear phase-delay experiment."""

    acquisition_type: AcquisitionType = workflow.option_field(
        AcquisitionType.SPECTROSCOPY, description="Acquisition type to use."
    )


@workflow.workflow(name="linear_phase_delay")
def experiment_workflow(
    session: Session,
    qpu: QPU,
    qubit: QuantumElement,
    frequencies: ArrayLike,
    state: str = "g",
    temporary_parameters: dict[str | tuple[str, str, str], dict | QuantumParameters]
    | None = None,
    options: TuneUpWorkflowOptions | None = None,
) -> None:
    """Run linear phase-delay calibration workflow."""
    options = TuneUpWorkflowOptions() if options is None else options
    temp_qpu = temporary_qpu(qpu, temporary_parameters)
    qubit = temporary_quantum_elements_from_qpu(temp_qpu, qubit)

    exp = create_experiment(
        qpu=temp_qpu,
        qubit=qubit,
        frequencies=frequencies,
        state=state,
    )
    compiled_exp = compile_experiment(session, exp)
    result = run_experiment(session, compiled_exp)

    with workflow.if_(options.do_analysis):
        analysis_results = analysis_workflow(
            result=result,
            qubit=qubit,
            frequencies=frequencies,
            state=state,
        )
        qubit_parameters = analysis_results.output
        with workflow.if_(options.update):
            update_qpu(qpu, qubit_parameters["new_parameter_values"])

    workflow.return_(result)


@workflow.task
@dsl.qubit_experiment
def create_experiment(
    qpu: QPU,
    qubit: QuantumElement,
    frequencies: ArrayLike,
    state: str = "g",
    options: LinearPhaseDelayExperimentOptions | None = None,
) -> Experiment:
    """Create linear phase-delay experiment."""
    opts = LinearPhaseDelayExperimentOptions() if options is None else options
    qubit, frequencies = validation.validate_and_convert_single_qubit_sweeps(
        qubit, frequencies
    )
    if AcquisitionType(opts.acquisition_type) != AcquisitionType.SPECTROSCOPY:
        raise ValueError(
            "linear_phase_delay supports only AcquisitionType.SPECTROSCOPY."
        )
    if not isinstance(state, str) or len(state) == 0:
        raise ValueError("state must be a non-empty string.")

    qop = qpu.quantum_operations
    with dsl.acquire_loop_rt(
        count=opts.count,
        averaging_mode=opts.averaging_mode,
        acquisition_type=opts.acquisition_type,
        repetition_mode=opts.repetition_mode,
        repetition_time=opts.repetition_time,
        reset_oscillator_phase=opts.reset_oscillator_phase,
    ):
        with dsl.sweep(
            name=f"frequency_sweep_{qubit.uid}",
            parameter=SweepParameter(f"frequency_{qubit.uid}", frequencies),
        ) as frequency:
            qop.set_frequency(qubit, frequency=frequency, readout=True)
            qop.prepare_state(qubit, state)
            qop.measure(
                qubit,
                dsl.handles.result_handle(qubit.uid, suffix=state),
            )
            qop.passive_reset(qubit)

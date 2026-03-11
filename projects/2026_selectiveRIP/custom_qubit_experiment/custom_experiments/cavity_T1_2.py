# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""This module defines the qubit spectroscopy experiment.

In this experiment, we sweep the frequency of a qubit drive pulse to characterize
the qubit transition frequency.

The qubit spectroscopy experiment has the following pulse sequence:

    qb --- [ prep transition ] --- [ x180_transition (swept frequency)] --- [ measure ]

If multiple qubits are passed to the `run` workflow, the above pulses are applied
in parallel on all the qubits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from laboneq import workflow
from laboneq.simple import (
    Experiment,
    SectionAlignment,
    SweepParameter,
    dsl,
)
from laboneq.workflow.tasks import (
    compile_experiment,
    run_experiment,
)

from laboneq_applications.analysis.qubit_spectroscopy import analysis_workflow
from laboneq_applications.core import validation
from laboneq_applications.core.validation import validate_and_convert_qubits_sweeps
from laboneq_applications.experiments.options import (
    TuneUpWorkflowOptions,
    BaseExperimentOptions
)


from qubit_experiment.experiments.options import QubitSpectroscopyExperimentOptions
from laboneq_applications.tasks.parameter_updating import (
    temporary_qpu,
    temporary_quantum_elements_from_qpu,
    update_qpu
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from laboneq.dsl.quantum import QuantumParameters
    from laboneq.dsl.quantum.qpu import QPU
    from laboneq.dsl.session import Session

    from laboneq_applications.typing import QuantumElements, QubitSweepPoints

from laboneq.simple import *
@workflow.workflow(name="cavity_t1_spectroscopy")
def experiment_workflow(
    session: Session,
    qpu: QPU,
    qubit: QuantumElement,
    bus: QuantumElement,
    delay_time: float,
    CW_amplitude: float,
    CW_frequency: float,
    temporary_parameters: dict[str | tuple[str, str, str], dict | QuantumParameters]
    | None = None,
    options: TuneUpWorkflowOptions | None = None,
) -> None:
    temp_qpu = temporary_qpu(qpu, temporary_parameters)
    qubit = temporary_quantum_elements_from_qpu(temp_qpu, qubit)
    bus = temporary_quantum_elements_from_qpu(temp_qpu, bus)
    
    exp = create_experiment(
        temp_qpu,
        qubit,
        bus,
        delay_time,
        CW_amplitude,
        CW_frequency,
    )
    compiled_exp = compile_experiment(session, exp)
    result = run_experiment(session, compiled_exp)
    workflow.return_(result)


@workflow.task_options(base_class=BaseExperimentOptions)
class CavityT1SpectroscopyOptions:
    bus_pulse_length: float = workflow.option_field(
        10e-6, description="Bus CW drive pulse length"
    )


@workflow.task
@dsl.qubit_experiment
def create_experiment(
    qpu: QPU,
    qubit: QuantumElement,
    bus: QuantumElement,
    delay_time: float,
    CW_amplitude: float,
    CW_frequency: float,
    options: CavityT1SpectroscopyOptions | None = None,
) -> Experiment:
    
    opts = CavityT1SpectroscopyOptions() if options is None else options
    
    qop = qpu.quantum_operations
    with dsl.acquire_loop_rt(
        count=opts.count,
        averaging_mode=opts.averaging_mode,
        acquisition_type=opts.acquisition_type,
        repetition_mode=opts.repetition_mode,
        repetition_time=opts.repetition_time,
        reset_oscillator_phase=opts.reset_oscillator_phase,
    ):
        qop.set_bus_frequency.omit_section(bus, CW_frequency)
        
        # 1. Bus CW drive
        with dsl.section(uid="bus_drive_section", alignment=SectionAlignment.LEFT):
            qop.bus_spectroscopy_drive(
                bus,
                amplitude=CW_amplitude,
                phase=0.0,
                length=opts.bus_pulse_length,
            )
        
        # 2. Fixed delay
        with dsl.section(
            uid="delay_section",
            alignment=SectionAlignment.LEFT,
            play_after="bus_drive_section",
        ):
            qop.delay(qubit, time=delay_time)
        
        # 3. x180 pulse (qubit spectroscopy 대신)
        with dsl.section(
            uid="qubit_drive_section", 
            alignment=SectionAlignment.LEFT, 
            play_after="delay_section",
        ):
            qop.x180(qubit)
        
        # 4. Measure
        qop.measure(qubit, dsl.handles.result_handle(qubit.uid))
        qop.passive_reset(qubit, delay=200e-6)


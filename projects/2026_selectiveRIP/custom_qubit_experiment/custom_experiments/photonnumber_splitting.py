# Copyright 2025 AHNYALGAE
# SPDX-License-Identifier: Apache-2.0

"""This module is modified version of Ramsey experiment.
.- Fixed issues with the increment oscillator phase
 - Uses INTEGRATION + SINGLESHOT 
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from laboneq import workflow
from laboneq.simple import (
    AveragingMode,
    Experiment,
    SectionAlignment,
    SweepParameter,
    dsl,
)
from laboneq.workflow.tasks import (
    compile_experiment,
    run_experiment,
)

from analysis import coherence_spectroscopy as analysis_coherence
from laboneq_applications.analysis.ramsey import (
    validate_and_convert_detunings,
)
from laboneq_applications.core import validation
from laboneq_applications.experiments.options import (
    TuneupExperimentOptions,
    TuneUpWorkflowOptions,
    BaseExperimentOptions
)
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
    from laboneq.dsl.quantum_element import QuantumElement
    from laboneq_applications.typing import QubitSweepPoints

@workflow.workflow(name="photon_number_calibration")
def experiment_workflow(
    session: Session,
    qpu: QPU,
    qubit: QuantumElement,
    bus: QuantumElement,
    qubit_frequencies: QubitSweepPoints,
    CW_frequencies: QubitSweepPoints,
    CW_amplitude: float,
    CW_phase: float = 0.0,
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
        qubit_frequencies,
        CW_frequencies,
        CW_amplitude,
        CW_phase,
    )
    compiled_exp = compile_experiment(session, exp)
    result = run_experiment(session, compiled_exp)
    workflow.return_(result)

@workflow.task_options(base_class=BaseExperimentOptions)
class PhotonNumberCalibrationOptions:
    bus_pulse_length: float = workflow.option_field(
        12e-6, description="Bus CW drive pulse length"
    )
    ring_up: float = workflow.option_field(
        10e-6, description="Wait time for cavity to ring up before qubit spec"
    )
    ring_down: float = workflow.option_field(
        100e-9, description="Wait time after bus drive before measure"
    )
    use_cal_traces: bool = workflow.option_field(
        True, description="Whether to include calibration traces in the experiment."
    )
    cal_states: str | tuple = workflow.option_field(
        "ge", description="The states to prepare in the calibration traces."
    )


@workflow.task
@dsl.qubit_experiment
def create_experiment(
    qpu: QPU,
    qubit: QuantumElement,
    bus: QuantumElement,
    qubit_frequencies: QubitSweepPoints,
    CW_frequencies: QubitSweepPoints,
    CW_amplitude: float,
    CW_phase: float = 0.0,
    options: PhotonNumberCalibrationOptions | None = None,
) -> Experiment:
    
    opts = PhotonNumberCalibrationOptions() if options is None else options
    
    bus, cw_frequencies = validation.validate_and_convert_single_qubit_sweeps(bus, CW_frequencies)
    q, q_frequencies = validation.validate_and_convert_single_qubit_sweeps(qubit, qubit_frequencies)
    
    if (
        opts.use_cal_traces
        and AveragingMode(opts.averaging_mode) == AveragingMode.SEQUENTIAL
    ):
        raise ValueError(
            "'AveragingMode.SEQUENTIAL' cannot be used with calibration traces."
        )
    
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
            name="CW_freq_sweep",
            parameter=SweepParameter("CW_drive_freqs", cw_frequencies),
            auto_chunking=False,
        ) as cw_frequency:
            qop.set_bus_frequency.omit_section(bus, cw_frequency)
        
            with dsl.sweep(
                name="qubit_freq_sweep",
                parameter=SweepParameter(f"qubit_freqs_{q.uid}", q_frequencies),
                auto_chunking=True,
            ) as qubit_frequency:
                qop.set_frequency(q, qubit_frequency)
                
                with dsl.section(name="main", alignment=SectionAlignment.LEFT):
                    # 1. Bus CW drive
                    with dsl.section(
                        name="bus_drive", alignment=SectionAlignment.LEFT
                    ) as bus_sec:
                        qop.bus_spectroscopy_drive(
                            bus,
                            amplitude=CW_amplitude,
                            phase=CW_phase,
                            length=opts.bus_pulse_length,
                        )
                    
                    # 2. Qubit spectroscopy (ring_up 후 시작, bus drive와 동시에)
                    with dsl.section(
                        name="qubit_spec", alignment=SectionAlignment.LEFT
                    ):
                        qop.delay(q, opts.ring_up)
                        qop.qubit_spectroscopy_drive(q)
                    
                    # 3. Measure (bus drive 끝난 후)
                    with dsl.section(
                        name="main_measure", 
                        alignment=SectionAlignment.LEFT, 
                        play_after=bus_sec.uid
                    ):
                        qop.delay(q, opts.ring_down)
                        qop.measure(q, dsl.handles.result_handle(q.uid))
                        qop.passive_reset(q)
        
        if opts.use_cal_traces:
            qop.calibration_traces.omit_section(
                qubits=qubit,
                states=opts.cal_states,
                active_reset=opts.active_reset,
                active_reset_states=opts.active_reset_states,
                active_reset_repetitions=opts.active_reset_repetitions,
            )

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

@workflow.workflow(name="photon_number_spectroscopy")
def experiment_workflow(
    session: Session,
    qpu: QPU,
    qubit: QuantumElement,
    bus: QuantumElement,
    frequencies: QubitSweepPoints,
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
        frequencies,
        CW_amplitude,
        CW_frequency,
    )
    compiled_exp = compile_experiment(session, exp)
    result = run_experiment(session, compiled_exp)
    workflow.return_(result)


@workflow.task_options(base_class=BaseExperimentOptions)
class PhotonNumberSpectroscopyOptions:
    bus_pulse_length: float = workflow.option_field(
        10e-6, description="Bus CW drive pulse length"
    )


@workflow.task
@dsl.qubit_experiment
def create_experiment(
    qpu: QPU,
    qubit: QuantumElement,
    bus: QuantumElement,
    frequencies: QubitSweepPoints,
    CW_amplitude: float,
    CW_frequency: float,
    options: PhotonNumberSpectroscopyOptions | None = None,
) -> Experiment:
    
    opts = PhotonNumberSpectroscopyOptions() if options is None else options
    
    q, q_frequencies = validation.validate_and_convert_single_qubit_sweeps(qubit, frequencies)
    
    qop = qpu.quantum_operations
    with dsl.acquire_loop_rt(
        count=opts.count,
        averaging_mode=opts.averaging_mode,
        acquisition_type=opts.acquisition_type,
        repetition_mode=opts.repetition_mode,
        repetition_time=opts.repetition_time,
        reset_oscillator_phase=opts.reset_oscillator_phase,
    ):
        # Bus frequency 설정 (sweep 밖)
        qop.set_bus_frequency.omit_section(bus, CW_frequency)
        
        with dsl.sweep(
            name=f"freqs_{q.uid}",
            parameter=SweepParameter(f"frequency_{q.uid}", q_frequencies),
        ) as frequency:
            qop.set_frequency(q, frequency)
            
            # 1. Bus CW drive (UID 지정 필수)
            with dsl.section(uid="bus_drive_section", alignment=SectionAlignment.LEFT):
                qop.bus_spectroscopy_drive(
                    bus,
                    amplitude=CW_amplitude,
                    phase=0.0,
                    length=opts.bus_pulse_length,
                )
            
            # 2. Qubit spectroscopy drive 
            # [핵심 수정] play_after를 사용하여 bus_drive_section이 끝난 뒤 실행되도록 강제
            with dsl.section(
                uid="qubit_drive_section", 
                alignment=SectionAlignment.LEFT, 
                play_after="bus_drive_section"  # <--- 이 부분이 있어야 순차 실행됩니다.
            ):
                qop.qubit_spectroscopy_drive(q)
            
            # Measurement
            # 측정은 보통 qubit 라인을 쓰므로 위 섹션(qubit drive) 뒤에 자동으로 붙지만,
            # 더 명확하게 하려면 여기도 play_after="qubit_drive_section"을 걸어줄 수 있습니다.
            # (일단은 dsl이 같은 라인 충돌을 감지해 뒤로 미뤄줄 것입니다)
            qop.measure(q, dsl.handles.result_handle(q.uid))
            qop.passive_reset(q, delay=200e-6)


# def create_experiment(
#     qpu: QPU,
#     qubit: QuantumElement,
#     bus: QuantumElement,
#     frequencies: QubitSweepPoints,
#     CW_amplitude: float,
#     CW_frequency: float,
#     options: PhotonNumberSpectroscopyOptions | None = None,
# ) -> Experiment:
    
#     opts = PhotonNumberSpectroscopyOptions() if options is None else options
    
#     q, q_frequencies = validation.validate_and_convert_single_qubit_sweeps(qubit, frequencies)
    
#     qop = qpu.quantum_operations
#     with dsl.acquire_loop_rt(
#         count=opts.count,
#         averaging_mode=opts.averaging_mode,
#         acquisition_type=opts.acquisition_type,
#         repetition_mode=opts.repetition_mode,
#         repetition_time=opts.repetition_time,
#         reset_oscillator_phase=opts.reset_oscillator_phase,
#     ):
#         # Bus frequency 설정 (sweep 밖에서 한번만)
#         qop.set_bus_frequency.omit_section(bus, CW_frequency)
        
#         with dsl.sweep(
#             name=f"freqs_{q.uid}",
#             parameter=SweepParameter(f"frequency_{q.uid}", q_frequencies),
#         ) as frequency:
#             qop.set_frequency(q, frequency)
            
#             # Bus CW drive + qubit spec drive 동시에
#             qop.bus_spectroscopy_drive(
#                 bus,
#                 amplitude=CW_amplitude,
#                 phase=0.0,
#                 length=opts.bus_pulse_length,
#             )
#             qop.qubit_spectroscopy_drive(q)
            
#             qop.measure(q, dsl.handles.result_handle(q.uid))
#             qop.passive_reset(q, delay=200e-6)

# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""RIP v2 experiment (Ramsey with bus tone).

This variant performs a Ramsey sequence on target qubit(s) while driving a
bus element concurrently for the duration of the Ramsey delay. It supports
detuning the second x90 phase based on a provided detuning and setting the
bus drive frequency and amplitude.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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

from qubit_experiment.analysis.rip import (
    validate_and_convert_detunings,
)
from laboneq_applications.core import validation
from laboneq_applications.experiments.options import (
    TuneupExperimentOptions,
    TuneUpWorkflowOptions,
)
from laboneq_applications.tasks.parameter_updating import (
    temporary_qpu,
    temporary_quantum_elements_from_qpu,
    update_qubits,
    update_qpu
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from laboneq.dsl.quantum import QuantumParameters
    from laboneq.dsl.quantum.qpu import QPU
    from laboneq.dsl.session import Session

    from laboneq_applications.typing import QuantumElements, QubitSweepPoints

@workflow.workflow(name="rip_selective")
def experiment_workflow(
    session: Session,
    qpu: QPU,
    ctrl: QuantumElements,
    targ: QuantumElements,
    bus: QuantumElements,
    bus2: QuantumElements,
    bus3: QuantumElements,
    bus_frequency: float,
    bus_amplitude: float,
    bus2_frequency: float,
    bus2_amplitude: float,
    bus3_frequency: float,
    bus3_amplitude: float,
    delays: QubitSweepPoints,
    c_prep: str = "g",
    detunings: float | Sequence[float] | None = None,
    temporary_parameters: dict[str | tuple[str, str, str], dict | QuantumParameters]
    | None = None,
    options: TuneUpWorkflowOptions | None = None,
) -> None:
    temp_qpu = temporary_qpu(qpu, temporary_parameters)
    targ = temporary_quantum_elements_from_qpu(temp_qpu, targ)
    bus = temporary_quantum_elements_from_qpu(temp_qpu, bus)
    bus2 = temporary_quantum_elements_from_qpu(temp_qpu, bus2)
    bus3 = temporary_quantum_elements_from_qpu(temp_qpu, bus3)
    ctrl = temporary_quantum_elements_from_qpu(temp_qpu, ctrl)

    exp = create_experiment(
        temp_qpu,
        targ,
        ctrl,
        bus,
        bus2,
        bus3,
        bus_frequency,
        bus_amplitude,
        bus2_frequency,
        bus2_amplitude,
        bus3_frequency,
        bus3_amplitude,
        delays=delays,
        detunings=detunings,
        c_prep=c_prep,
    )
    compiled_exp = compile_experiment(session, exp)
    result = run_experiment(session, compiled_exp)
    workflow.return_(result)


@workflow.task
@dsl.qubit_experiment
def create_experiment(
    qpu: QPU,
    targ: QuantumElements,
    ctrl: QuantumElements,
    bus: QuantumElements,
    bus2: QuantumElements,
    bus3: QuantumElements,
    bus_frequency: float,
    bus_amplitude: float,
    bus2_frequency: float,
    bus2_amplitude: float,
    bus3_frequency: float,
    bus3_amplitude: float,
    delays: QubitSweepPoints,
    detunings: float | Sequence[float] | None = None,
    c_prep: str = "g",
    options: TuneupExperimentOptions | None = None,
) -> Experiment:
    opts = TuneupExperimentOptions() if options is None else options
    targ, delays = validation.validate_and_convert_qubits_sweeps(targ, delays)
    bus = validation.validate_and_convert_qubits_sweeps(bus)
    bus2 = validation.validate_and_convert_qubits_sweeps(bus2)
    bus3 = validation.validate_and_convert_qubits_sweeps(bus3)
    ctrl = validation.validate_and_convert_qubits_sweeps(ctrl)
    detunings = validate_and_convert_detunings(targ, detunings)
    if (
        opts.use_cal_traces
        and AveragingMode(opts.averaging_mode) == AveragingMode.SEQUENTIAL
    ):
        raise ValueError(
            "'AveragingMode.SEQUENTIAL' (or {AveragingMode.SEQUENTIAL}) cannot be used "
            "with calibration traces because the calibration traces are added "
            "outside the sweep."
        )

    swp_delays = []
    swp_phases = []
    for i, q in enumerate(targ):
        q_delays = delays[i]
        swp_delays += [
            SweepParameter(
                uid=f"wait_time_{q.uid}",
                values=q_delays,
            ),
        ]
        swp_phases += [
            SweepParameter(
                uid=f"x90_phases_{q.uid}",
                values=np.array(
                    [
                        #((wait_time - q_delays[0]) * detunings[i] * 2 * np.pi)
                        (wait_time * detunings[i] * 2 * np.pi)
                        % (2 * np.pi)
                        for wait_time in q_delays   
                    ]
                ),
            ),
        ]

    max_measure_section_length = qpu.measure_section_length(targ)
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
            name="rip2_sweep",
            parameter=swp_delays + swp_phases,
            auto_chunking=True,
        ):
            if opts.active_reset:
                qop.active_reset(
                    targ,
                    active_reset_states=opts.active_reset_states,
                    number_resets=opts.active_reset_repetitions,
                    measure_section_length=max_measure_section_length,
                )
            with dsl.section(name="main", alignment=SectionAlignment.LEFT):
                for q_c in ctrl:
                    prep = qop.prepare_state(q_c, c_prep)
                with dsl.section(name="main_drive", alignment=SectionAlignment.LEFT, play_after=prep.uid):
                    for q_t, q_c, q_b, q_b2, q_b3, wait_time, phase in zip(
                        targ, ctrl, bus, bus2, bus3, swp_delays, swp_phases
                    ):
                        # Bus 1
                        qop.bus_delay(q_b, time=128e-9)
                        qop.set_bus_frequency(q_b, frequency=bus_frequency)
                        qop.rip(q_b, amplitude=bus_amplitude, length=wait_time)
                        # Bus 2
                        qop.bus_delay(q_b2, time=128e-9)
                        qop.set_bus_frequency(q_b2, frequency=bus2_frequency)
                        qop.rip(q_b2, amplitude=bus2_amplitude, length=wait_time)
                        # Bus 3
                        qop.bus_delay(q_b3, time=128e-9)
                        qop.set_bus_frequency(q_b3, frequency=bus3_frequency)
                        qop.rip(q_b3, amplitude=bus3_amplitude, length=wait_time)
                        # Target Ramsey
                        qop.ramsey(
                            q_t, wait_time, phase, transition=opts.transition
                        )
                with dsl.section(name="main_measure", alignment=SectionAlignment.LEFT):
                    for q_t in targ:
                        sec = qop.measure(q_t, dsl.handles.result_handle(q_t.uid))
                        sec.length = max_measure_section_length
                        qop.passive_reset(q_t)
        if opts.use_cal_traces:
            qop.calibration_traces.omit_section(
                qubits=targ,
                states=opts.cal_states,
                active_reset=opts.active_reset,
                active_reset_states=opts.active_reset_states,
                active_reset_repetitions=opts.active_reset_repetitions,
                measure_section_length=max_measure_section_length,
            )

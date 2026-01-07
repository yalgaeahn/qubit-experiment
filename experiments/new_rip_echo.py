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


from analysis.new_rip_echo import analysis_workflow
from analysis.rip import (
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


@workflow.workflow(name="rip_echo")
def experiment_workflow(
    session: Session,
    qpu: QPU,
    ctrl: QuantumElements,
    targ: QuantumElements,
    bus: QuantumElements,
    delays: QubitSweepPoints,
    rip_detunings: QubitSweepPoints,
    ramsey_detunings: float | None,
    temporary_parameters: dict[str | tuple[str, str, str], dict | QuantumParameters]
    | None = None,
    options: TuneUpWorkflowOptions | None = None,
) -> None:

    temp_qpu = temporary_qpu(qpu, temporary_parameters)
    targ = temporary_quantum_elements_from_qpu(temp_qpu, targ)
    bus = temporary_quantum_elements_from_qpu(temp_qpu, bus)
    ctrl = temporary_quantum_elements_from_qpu(temp_qpu, ctrl)


    exp = create_experiment(
        temp_qpu,
        ctrl,
        targ,
        bus,
        delays=delays,
        rip_detunings=rip_detunings,
        ramsey_detunings=ramsey_detunings
    )
    compiled_exp = compile_experiment(session, exp)
    result = run_experiment(session, compiled_exp)
    with workflow.if_(options.do_analysis):
        analysis_results = analysis_workflow(result, targ, delays, rip_detunings)
        #qubit_parameters = analysis_results.output
        # with workflow.if_(options.update):
        #     update_qpu(qpu, qubit_parameters["new_parameter_values"])
    workflow.return_(result)



@workflow.task
@dsl.qubit_experiment
def create_experiment(
    qpu: QPU,
    ctrl: QuantumElements,
    targ: QuantumElements,
    bus: QuantumElements,
    delays: QubitSweepPoints,
    rip_detunings: QubitSweepPoints,
    ramsey_detunings: float | None = None,
    options: TuneupExperimentOptions | None = None,
) -> Experiment:
    """Creates a Ramsey Experiment where the phase of the second pulse is swept.

    Arguments:
        qpu:
            The qpu consisting of the original qubits and quantum operations.
        qubits:
            The qubits to run the experiments on. May be either a single
            qubit or a list of qubits.
        delays:
            The delays (in seconds) of the second x90 pulse to sweep over for each
            qubit. If `qubits` is a single qubit, `delays` must be a list of numbers
            or an array. Otherwise, it must be a list of lists of numbers or arrays.
        detunings:
            The effective detuning in Hz used to calculate the phase increment
            of the second pulse in the Ramsey sequence.
            For perfectly resonant excitation pulses,
            this simulates oscillations of the qubit state vector
            around the Bloch sphere at the given frequency.
            This parameter and the fitted frequency of the oscillations
            can then be used to calculate the true qubit resonance frequency.
            `detunings` is a list of float values for each qubit following the order
            in `qubits`.
        options:
            The options for building the experiment.
            See [TuneupExperimentOptions] and [BaseExperimentOptions] for
            accepted options.
            Overwrites the options from [TuneupExperimentOptions] and
            [BaseExperimentOptions].

    Returns:
        experiment:
            The generated LabOne Q experiment instance to be compiled and executed.

    Raises:
        ValueError:
            If the lengths of `qubits` and `delays` do not match.

        ValueError:
            If `delays` is not a list of numbers when a single qubit is passed.

        ValueError:
            If `delays` is not a list of lists of numbers when a list of qubits
            is passed.

        ValueError:
            If the experiment uses calibration traces and the averaging mode is
            sequential.

    Example:
        ```python
        options = TuneupExperimentOptions()
        qpu = QPU(
            quantum_elements=[TunableTransmonQubit("q0"), TunableTransmonQubit("q1")],
            quantum_operations=TunableTransmonOperations(),
        )
        temp_qubits = qpu.copy_quantum_elements()
        create_experiment(
            qpu=qpu,
            qubits=temp_qubits,
            delays=[
                np.linspace(0, 20e-6, 51),
                np.linspace(0, 30e-6, 52),
            ],
            detunings = [1e6, 1.346e6],
            options=options,
        )
        ```
    """
    # Define the custom options for the experiment
    opts = TuneupExperimentOptions() if options is None else options
    _targ, delays = validation.validate_and_convert_single_qubit_sweeps(targ, delays)
    _bus, rip_detunings = validation.validate_and_convert_single_qubit_sweeps(bus, rip_detunings)
    
   
    # bus = validation.validate_and_convert_single_qubit_sweeps(bus)
    # ctrl = validation.validate_and_convert_single_qubit_sweeps(ctrl)
    # bus, detunings = validation.validate_and_convert_single_qubit_sweeps(bus, detunings)

    if (
        opts.use_cal_traces
        and AveragingMode(opts.averaging_mode) == AveragingMode.SEQUENTIAL
    ):
        raise ValueError(
            "'AveragingMode.SEQUENTIAL' (or {AveragingMode.SEQUENTIAL}) cannot be used "
            "with calibration traces because the calibration traces are added "
            "outside the sweep."
        )

    swp_delays = SweepParameter(uid=f"wait_time_{targ.uid}",values=delays) 
    swp_phases = SweepParameter(
                uid=f"x90_phases_{targ.uid}",
                values=np.array(
                    [
                        ((wait_time - delays[0]) * ramsey_detunings * 2 * np.pi)
                        % (2 * np.pi)
                        for wait_time in delays
                    ]
                    )
                )
    swp_detunings = SweepParameter(uid=f"rip_detuning_{bus.uid}", values = rip_detunings)

    # len(swp_delays)=len(swp_phases) => multi dimensional sweep 할때 1d 병렬 sweep으로 동작
 
    # We will fix the length of the measure section to the longest section among
    # the qubits to allow the qubits to have different readout and/or
    # integration lengths.
    # measure_section_length expects an iterable of qubits
    max_measure_section_length = qpu.measure_section_length([targ])
    qop = qpu.quantum_operations #combined operation
    with dsl.acquire_loop_rt(
        count=opts.count,
        averaging_mode=opts.averaging_mode,
        acquisition_type=opts.acquisition_type,
        repetition_mode=opts.repetition_mode,
        repetition_time=opts.repetition_time,
        reset_oscillator_phase=opts.reset_oscillator_phase,
    ):
        with dsl.sweep(
            name="rip_detuning_sweep",
            parameter=swp_detunings,
            auto_chunking=True
        ) as rip_detuning: 
            with dsl.sweep(
                name="rip_echo_ramsey_sweep",
                parameter=[swp_delays, swp_phases],
                auto_chunking=False
            ):
                if opts.active_reset:
                    qop.active_reset(
                        targ,
                        active_reset_states=opts.active_reset_states,
                        number_resets=opts.active_reset_repetitions,
                        #measure_section_length=max_measure_section_length,
                    )

                with dsl.section(name="ctrl_prep", alignment=SectionAlignment.LEFT) as ctrl_prep:
                    qop.prepare_state(ctrl, "g")
                with dsl.section(name="main_drive", alignment=SectionAlignment.LEFT, play_after=ctrl_prep.uid):
                    ###############ECHO SEQUENCE##################################    
                    qop.set_bus_frequency(bus, frequency=rip_detuning+bus.parameters.resonance_frequency_bus)
            
                    
                    sec1=qop.x90(targ)
                    qop.delay(targ, time=swp_delays) 
                    with dsl.section(name="rip_drive1",alignment=SectionAlignment.LEFT, play_after=sec1.uid) as rip1:
                        qop.rip(bus, amplitude=bus.parameters.rip_amplitude, length=swp_delays)
                
                    with dsl.section(name="flip",alignment=SectionAlignment.LEFT, play_after=rip1.uid) as flip:
                        qop.x180(ctrl)
                        qop.x180(targ)
                    with dsl.section(name="rip_drive2",alignment=SectionAlignment.LEFT, play_after=flip.uid) as rip2:
                        qop.rip(bus, amplitude=bus.parameters.rip_amplitude, length=swp_delays)
                    
                    qop.delay(targ, time=swp_delays)
                    qop.x90(targ) #increment_oscillator_phase = swp_phases)

                 
                    with dsl.section(name="main_measure", alignment=SectionAlignment.LEFT):
                        sec = qop.measure(targ, dsl.handles.result_handle(targ.uid))
                        # Fix the length of the measure section
                        #sec.length = max_measure_section_length
                        qop.passive_reset(targ)
        if opts.use_cal_traces:
            qop.calibration_traces.omit_section(
                qubits=targ,
                states=opts.cal_states,
                active_reset=opts.active_reset,
                active_reset_states=opts.active_reset_states,
                active_reset_repetitions=opts.active_reset_repetitions,
                #measure_section_length=max_measure_section_length,
            )

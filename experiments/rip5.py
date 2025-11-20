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


from laboneq_applications.analysis.ramsey import analysis_workflow
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
    bus2: QuantumElements,
    bus_frequency: float,
    bus_amplitude: float,
    bus2_frequency: float,
    bus2_amplitude: float,
    delays: QubitSweepPoints,
    c_prep: str = "g",
    detunings: float | None = None,
    temporary_parameters: dict[str | tuple[str, str, str], dict | QuantumParameters]
    | None = None,
    options: TuneUpWorkflowOptions | None = None,
) -> None:
    """The Ramsey Workflow.

    The workflow consists of the following steps:

    - [create_experiment]()
    - [compile_experiment]()
    - [run_experiment]()
    - [analysis_workflow]()
    - [update_qubits]()

    Arguments:
        session:
            The connected session to use for running the experiment.
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
            The detuning in Hz to generate oscillating qubit occupations. `detunings`
            is a list of float values for each qubits following the order in `qubits`.
        temporary_parameters:
            The temporary parameters with which to update the quantum elements and
            topology edges. For quantum elements, the dictionary key is the quantum
            element UID. For topology edges, the dictionary key is the edge tuple
            `(tag, source node UID, target node UID)`.
        options:
            The options for building the workflow.
            In addition to options from [WorkflowOptions], the following
            custom options are supported:
                - create_experiment: The options for creating the experiment.

    Returns:
        result:
            The result of the workflow.

    Example:
        ```python
        options = experiment_workflow.options()
        options.create_experiment.count(10)
        options.create_experiment.transition("ge")
        setup = DeviceSetup("my_device")
        qpu = QPU(
            quantum_elements=[TunableTransmonQubit("q0"), TunableTransmonQubit("q1")],
            quantum_operations=TunableTransmonOperations(),
        )
        temp_qubits = qpu.copy_quantum_elements()
        result = experiment_workflow(
            session=session,
            qpu=qpu,
            qubits=temp_qubits,
            delays=[[0.1, 0.5, 1], [0.1, 0.5, 1]],
            detunings = {'q0':1e6,'q1':1.346e6},
            options=options,
        ).run()
        ```
    """
    temp_qpu = temporary_qpu(qpu, temporary_parameters)
    targ = temporary_quantum_elements_from_qpu(temp_qpu, targ)
    bus = temporary_quantum_elements_from_qpu(temp_qpu, bus)
    bus2 = temporary_quantum_elements_from_qpu(temp_qpu, bus2)
    ctrl = temporary_quantum_elements_from_qpu(temp_qpu, ctrl)


    exp = create_experiment(
        temp_qpu,
        targ,
        ctrl,
        bus,
        bus2,
        bus_frequency,
        bus_amplitude,
        bus2_frequency,
        bus2_amplitude,
        delays=delays,
        detunings=detunings,
        c_prep=c_prep,
    )
    compiled_exp = compile_experiment(session, exp)
    result = run_experiment(session, compiled_exp)
    with workflow.if_(options.do_analysis):
        analysis_results = analysis_workflow(result, targ, delays, detunings)
        #qubit_parameters = analysis_results.output
        # with workflow.if_(options.update):
        #     update_qpu(qpu, qubit_parameters["new_parameter_values"])
    workflow.return_(result)



@workflow.task
@dsl.qubit_experiment
def create_experiment(
    qpu: QPU,
    targ: QuantumElements,
    ctrl: QuantumElements,
    bus: QuantumElements,
    bus2: QuantumElements,
    bus_frequency: float,
    bus_amplitude: float,
    bus2_frequency: float,
    bus2_amplitude: float,
    delays: QubitSweepPoints,
    detunings: float | None = None,
    c_prep: str = "g",
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
    targ, delays = validation.validate_and_convert_single_qubit_sweeps(targ, delays)
    bus = validation.validate_and_convert_single_qubit_sweeps(bus)
    bus2 = validation.validate_and_convert_single_qubit_sweeps(bus2)
    ctrl = validation.validate_and_convert_single_qubit_sweeps(ctrl)
    #detunings = validate_and_convert_detunings(targ, detunings)
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
                        ((wait_time - delays[0]) * detunings * 2 * np.pi)
                        % (2 * np.pi)
                        for wait_time in delays
                    ]
                    )
                )

    # len(swp_delays)=len(swp_phases) => multi dimensional sweep 할때 1d 병렬 sweep으로 동작
 
    # We will fix the length of the measure section to the longest section among
    # the qubits to allow the qubits to have different readout and/or
    # integration lengths.
    # measure_section_length expects an iterable of qubits
    max_measure_section_length = qpu.measure_section_length([targ])
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
            name="rip_echo",
            parameter=[swp_delays, swp_phases],
            auto_chunking=True,
            #chunk_count=1
        ):
            if opts.active_reset:
                qop.active_reset(
                    targ,
                    active_reset_states=opts.active_reset_states,
                    number_resets=opts.active_reset_repetitions,
                    measure_section_length=max_measure_section_length,
                )

            with dsl.section(name="ctrl_prep", alignment=SectionAlignment.LEFT) as ctrl_prep:
                qop.prepare_state(ctrl, c_prep)
            with dsl.section(name="main_drive", alignment=SectionAlignment.LEFT, play_after=ctrl_prep.uid):
                ###############ECHO SEQUENCE##################################    
                qop.set_frequency(bus, frequency=bus_frequency)
                qop.set_frequency(bus2, frequency=bus2_frequency)
                
                
                sec1=qop.x90(targ)
                qop.delay(targ, time=swp_delays)
                with dsl.section(name="rip_drive1",alignment=SectionAlignment.LEFT, play_after=sec1.uid) as rip1:
                    qop.x180(bus, amplitude=bus_amplitude, length=swp_delays)
                    qop.x180(bus2, amplitude=bus2_amplitude, length=swp_delays)
                # with dsl.section(name="flip",alignment=SectionAlignment.LEFT, play_after=rip1.uid) as flip:
                #     qop.x180(ctrl)
                #     qop.x180(targ)
                # with dsl.section(name="rip_drive2",alignment=SectionAlignment.LEFT, play_after=flip.uid) as rip2:
                #     qop.x180(bus, amplitude=bus_amplitude, length=swp_delays)
                #     qop.x180(bus2, amplitude=bus2_amplitude, length=swp_delays)
                #qop.delay(targ, time=swp_delays)
                qop.x90(targ,phase=swp_phases)

                # qop.ramsey(
                #     targ, swp_delays, swp_phases, transition=opts.transition
                # )
                with dsl.section(name="main_measure", alignment=SectionAlignment.LEFT):
                    sec = qop.measure(targ, dsl.handles.result_handle(targ.uid))
                    # Fix the length of the measure section
                    sec.length = max_measure_section_length
                    qop.passive_reset(targ)
        if opts.use_cal_traces:
            qop.calibration_traces.omit_section(
                qubits=targ,
                states=opts.cal_states,
                active_reset=opts.active_reset,
                active_reset_states=opts.active_reset_states,
                active_reset_repetitions=opts.active_reset_repetitions,
                measure_section_length=max_measure_section_length,
            )

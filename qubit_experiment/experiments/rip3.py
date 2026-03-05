# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""RIP v2 experiment (Ramsey with bus tone).

This variant performs a Ramsey sequence on target qubit(s) while driving a
bus element concurrently for the duration of the Ramsey delay. It supports
detuning the second x90 phase based on a provided detuning and setting the
bus drive frequency and amplitude.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

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
    from laboneq.dsl.quantum import QuantumParameters
    from laboneq.dsl.quantum.qpu import QPU
    from laboneq.dsl.quantum.quantum_element import QuantumElement
    from laboneq.dsl.session import Session


@workflow.workflow(name="rip2")
def experiment_workflow(
    session: Session,
    qpu: QPU,
    ctrl: QuantumElement,
    targ: QuantumElement,
    bus: QuantumElement,
    c: QuantumElement,
    bus_frequency: float,
    bus_amplitude: float,
    c_frequency: float,
    c_amplitude: float,
    delays: Sequence[float] | np.ndarray,
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
        ctrl/targ/bus/c:
            Single quantum elements involved in the experiment. `ctrl` prepares the
            control qubit, `targ` is the Ramsey target, `bus` is the bus element, and
            `c` is the secondary drive element.
        delays:
            The Ramsey delays (in seconds) for the target qubit. Provide a 1D array or
            list of floats.
        detunings:
            Optional detuning in Hz used to generate oscillations in the Ramsey signal.
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
        result = experiment_workflow(...).run()
        ```
    """
    temp_qpu = temporary_qpu(qpu, temporary_parameters)
    targ = temporary_quantum_elements_from_qpu(temp_qpu, targ)
    bus = temporary_quantum_elements_from_qpu(temp_qpu, bus)
    ctrl = temporary_quantum_elements_from_qpu(temp_qpu, ctrl)
    c = temporary_quantum_elements_from_qpu(temp_qpu, c)

    exp = create_experiment(
        temp_qpu,
        targ,
        ctrl,
        bus,
        c,
        bus_frequency,
        bus_amplitude,
        c_frequency,
        c_amplitude,
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
    targ: QuantumElement,
    ctrl: QuantumElement,
    bus: QuantumElement,
    c: QuantumElement,
    bus_frequency: float,
    bus_amplitude: float,
    c_frequency: float,
    c_amplitude: float,
    delays: Sequence[float] | np.ndarray,
    detunings: float | None = None,
    c_prep: str = "g",
    options: TuneupExperimentOptions | None = None,
) -> Experiment:
    """Creates a Ramsey Experiment where the phase of the second pulse is swept.

    Arguments:
        qpu:
            The qpu consisting of the original qubits and quantum operations.
        ctrl/targ/bus/c:
            Single quantum elements involved in the experiment. They correspond to the
            control, target, bus, and additional drive elements.
        delays:
            1D Ramsey delay sweep (seconds) applied to the target qubit.
        detunings:
            Optional detuning in Hz used to calculate the Ramsey phase increment for
            the target.
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
            If `delays` cannot be interpreted as a 1D sweep for the target qubit.

        ValueError:
            If the experiment uses calibration traces and the averaging mode is
            sequential.

    Example:
        ```python
        options = TuneupExperimentOptions()
        create_experiment(
            qpu,
            targ=target_qubit,
            ctrl=control_qubit,
            bus=bus_element,
            c=c_element,
            delays=np.linspace(0, 20e-6, 51),
            detunings=1e6,
            options=options,
        )
        ```
    """
    # Define the custom options for the experiment
    opts = TuneupExperimentOptions() if options is None else options
    ctrl = validation.validate_and_convert_single_qubit_sweeps(ctrl)
    targ, q_delays = validation.validate_and_convert_single_qubit_sweeps(targ, delays)
    bus = validation.validate_and_convert_single_qubit_sweeps(bus)
    c = validation.validate_and_convert_single_qubit_sweeps(c)
    q_delays = np.asarray(q_delays, dtype=float).ravel()
    detuning = validate_and_convert_detunings([targ], detunings)[0]
    if (
        opts.use_cal_traces
        and AveragingMode(opts.averaging_mode) == AveragingMode.SEQUENTIAL
    ):
        raise ValueError(
            "'AveragingMode.SEQUENTIAL' (or {AveragingMode.SEQUENTIAL}) cannot be used "
            "with calibration traces because the calibration traces are added "
            "outside the sweep."
        )

    swp_delays = SweepParameter(
        uid=f"wait_time_{targ.uid}",
        values=q_delays.tolist(),
    )
    phase_values = ((q_delays - q_delays[0]) * detuning * 2 * np.pi) % (2 * np.pi)
    swp_phases = SweepParameter(
        uid=f"x90_phases_{targ.uid}",
        values=phase_values.tolist(),
    )

    # We will fix the length of the measure section to the longest section among
    # the qubits to allow the qubits to have different readout and/or
    # integration lengths.
    qop = qpu.quantum_operations
    max_measure_section_length = qop.measure_section_length([targ])
    with dsl.acquire_loop_rt(
        count=opts.count,
        averaging_mode=opts.averaging_mode,
        acquisition_type=opts.acquisition_type,
        repetition_mode=opts.repetition_mode,
        repetition_time=opts.repetition_time,
        reset_oscillator_phase=opts.reset_oscillator_phase,
    ):
        with dsl.sweep(
            name="rip3_sweep",
            parameter=[swp_delays, swp_phases],
            auto_chunking=True,
        ):
            if opts.active_reset:
                qop.active_reset(
                    [targ],
                    active_reset_states=opts.active_reset_states,
                    number_resets=opts.active_reset_repetitions,
                    measure_section_length=max_measure_section_length,
                )
            with dsl.section(name="main", alignment=SectionAlignment.LEFT):
                prep = qop.prepare_state(ctrl, c_prep)
                with dsl.section(
                    name="main_drive",
                    alignment=SectionAlignment.LEFT,
                    play_after=prep.uid,
                ):
                    qop.delay(bus, time=64e-9)
                    qop.delay(c, time=64e-9)
                    qop.set_frequency(bus, frequency=bus_frequency)
                    qop.set_frequency(c, frequency=c_frequency)
                    qop.x180(c, amplitude=c_amplitude, length=swp_delays)
                    qop.x180(bus, amplitude=bus_amplitude, length=swp_delays)
                    qop.ramsey(
                        targ, swp_delays, swp_phases, transition=opts.transition
                    )
                with dsl.section(name="main_measure", alignment=SectionAlignment.LEFT):
                    sec = qop.measure(targ, dsl.handles.result_handle(targ.uid))
                    # Fix the length of the measure section
                    sec.length = max_measure_section_length
                    qop.passive_reset(targ)
        if opts.use_cal_traces:
            qop.calibration_traces.omit_section(
                qubits=[targ],
                states=opts.cal_states,
                active_reset=opts.active_reset,
                active_reset_states=opts.active_reset_states,
                active_reset_repetitions=opts.active_reset_repetitions,
                measure_section_length=max_measure_section_length,
            )

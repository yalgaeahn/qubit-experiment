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




@workflow.task_options(base_class=BaseExperimentOptions)
class CoherenceSpectroscopyExperimentOptions:

    ring_up: float = workflow.option_field(
        200e-9, description="Waiting for cavity to ring up"
    )
 
    use_cal_traces: bool = workflow.option_field(
        True, description="Whether to include calibration traces in the experiment."
    )
    cal_states: str | tuple = workflow.option_field(
        "ge", description="The states to prepare in the calibration traces."
    )
    transition: Literal["ge", "ef"] = workflow.option_field(
        "ge",
        description="Transition to perform the experiment on. May be any"
        " transition supported by the quantum operations.",
    )




@workflow.workflow(name="coherence_spectroscopy")
def experiment_workflow(
    session: Session,
    qpu: QPU,
    qubit: QuantumElement,
    bus: QuantumElement,
    delays: QubitSweepPoints,
    CW_frequencies: QubitSweepPoints,
    CW_amplitude: float,
    CW_phase: float,
    detunings: float | Sequence[float] | None = None,
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
    - [update_qpu]()

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
 
        
    """
    temp_qpu = temporary_qpu(qpu, temporary_parameters)
    qubit = temporary_quantum_elements_from_qpu(temp_qpu, qubit)
    exp = create_experiment(
        temp_qpu,
        qubit,
        bus,
        delays,
        CW_frequencies,
        CW_amplitude,
        CW_phase,
        detunings,
    )
    compiled_exp = compile_experiment(session, exp)
    result = run_experiment(session, compiled_exp)
    with workflow.if_(options.do_analysis):
        analysis_results = analysis_coherence.analysis_workflow(
            result, qubit, delays, CW_frequencies, detunings
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
    bus: QuantumElement,
    delays: QubitSweepPoints,
    CW_frequencies: QubitSweepPoints,
    CW_amplitude : float,
    CW_phase: float,
    detunings: float | Sequence[float] | None = None,
    options: CoherenceSpectroscopyExperimentOptions | None = None,
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

    """
    # Define the custom options for the experiment
    opts = CoherenceSpectroscopyExperimentOptions() if options is None else options
    
    bus, frequencies = validation.validate_and_convert_single_qubit_sweeps(bus, CW_frequencies)
    q, delays = validation.validate_and_convert_single_qubit_sweeps(qubit,delays)
    detunings = validate_and_convert_detunings(qubit, detunings)
    detuning = detunings[0]
    
    if (
        opts.use_cal_traces
        and AveragingMode(opts.averaging_mode) == AveragingMode.SEQUENTIAL
    ):
        raise ValueError(
            "'AveragingMode.SEQUENTIAL' (or {AveragingMode.SEQUENTIAL}) cannot be used "
            "with calibration traces because the calibration traces are added "
            "outside the sweep."
        )
    




    swp_delays = SweepParameter(uid=f"wait_time_{q.uid}",values=delays) 
    swp_phases = SweepParameter(
                uid=f"x90_phases_{q.uid}",
                values=np.array(
                    [
                        ((wait_time - delays[0]) * detuning * 2 * np.pi)
                        % (2 * np.pi)
                        for wait_time in delays
                    ]
                    )
                )

    # len(swp_delays)=len(swp_phases) => multi dimensional sweep 할때 1d 병렬 sweep으로 동작

    # We will fix the length of the measure section to the longest section among
    # the qubits to allow the qubits to have different readout and/or
    # integration lengths.
    #max_measure_section_length = qpu.measure_section_length(q)
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
            parameter=SweepParameter("CW_drive_freqs",frequencies),
            auto_chunking=True,
        ) as frequency:
            qop.set_frequency.omit_section(bus,frequency)
            # qop.qubit_spectroscopy_drive(bus,CW_amplitude,CW_phase)
        
        
            with dsl.sweep(
                name="ramsey_sweep",
                parameter=[swp_delays, swp_phases],
            ):
                
                with dsl.section(name="main", alignment=SectionAlignment.LEFT):
                    with dsl.section(
                        name="spec_drive", alignment=SectionAlignment.LEFT
                    ):
                        qop.qubit_spectroscopy_drive(
                            bus,
                            amplitude=CW_amplitude,
                            phase=CW_phase,
                            #length=drive_length,
                        )
                    with dsl.section(name="main_drive", alignment=SectionAlignment.LEFT):
                        qop.delay(q,opts.ring_up)
                        qop.prepare_state.omit_section(q, opts.transition[0])
                        qop.ramsey.omit_section(
                            q, swp_delays, swp_phases,echo_pulse="x180", transition=opts.transition 
                        )
                    with dsl.section(name="main_measure", alignment=SectionAlignment.LEFT):
                        sec = qop.measure(q, dsl.handles.result_handle(q.uid))
                        # Fix the length of the measure section
                        #sec.length = max_measure_section_length
                        qop.passive_reset(q)
        
        
        
        
        
        
        
        if opts.use_cal_traces:
            qop.calibration_traces.omit_section(
                qubits=qubit,
                states=opts.cal_states,
                active_reset=opts.active_reset,
                active_reset_states=opts.active_reset_states,
                active_reset_repetitions=opts.active_reset_repetitions,
                #measure_section_length=max_measure_section_length,
            )

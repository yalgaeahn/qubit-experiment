# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""This module defines the time-rabi experiment.

In this experiment, we sweep the length of a drive pulse on a given qubit transition
in order to determine the pulse length that induces a rotation of pi.

The time-rabi experiment has the following pulse sequence:

    qb --- [ prep transition ] --- [ x180_transition ] --- [ measure ]

If multiple qubits are passed to the `run` workflow, the above pulses are applied
in parallel on all the qubits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from laboneq.simple import Experiment, SweepParameter, dsl, SectionAlignment
from laboneq.workflow import if_, task, workflow
from laboneq.workflow.tasks import (
    compile_experiment,
    run_experiment,
)

from analysis.time_rabi import analysis_workflow
from laboneq_applications.core import validation
from laboneq_applications.experiments.options import (
    TuneupExperimentOptions,
    TuneUpWorkflowOptions,
)
from laboneq_applications.tasks.parameter_updating import update_qpu

from laboneq_applications.tasks.parameter_updating import (
    temporary_qpu,
    temporary_quantum_elements_from_qpu,
)

if TYPE_CHECKING:
    from laboneq.dsl.quantum import QuantumParameters
    from laboneq.dsl.quantum.qpu import QPU
    from laboneq.dsl.session import Session

    from laboneq_applications.typing import QuantumElements, QubitSweepPoints


@workflow(name="time_rabi")
def experiment_workflow(
    session: Session,
    qpu: QPU,
    qubits: QuantumElements,
    lengths: QubitSweepPoints,
    options: TuneUpWorkflowOptions | None = None,
    temporary_parameters: dict[str | tuple[str, str, str], dict | QuantumParameters]
    | None = None,
) -> None:
    """The Time Rabi Workflow.

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
        lengths:
            The drive-pulse lengths to sweep over for each qubit. If `qubits` is a
            single qubit, `lengths` must be a list of numbers or an array. Otherwise
            it must be a list of lists of numbers or arrays.
        options:
            The options for building the workflow.
            In addition to options from [WorkflowOptions], the following
            custom options are supported:
                - create_experiment: The options for creating the experiment.

    Returns:
        WorkflowBuilder:
            The builder of the experiment workflow.

    Example:
        ```python
        options = experiment_workflow.options()
        options.count(10)
        options.create_experiment.transition = "ge"
        qpu = QPU(
            qubits=[TunableTransmonQubit("q0"), TunableTransmonQubit("q1")],
            quantum_operations=TunableTransmonOperations(),
        )
        temp_qubits = qpu.copy_qubits()
        result = experiment_workflow(
            session=session,
            qpu=qpu,
            qubits=temp_qubits,
            lengths=[
                np.linspace(100e-9, 500e-9, 11),
                np.linspace(100e-9, 500e-9, 11),
            ],
            options=options,
        ).run()
        ```
    """

    temp_qpu = temporary_qpu(qpu, temporary_parameters)
    qubits = temporary_quantum_elements_from_qpu(temp_qpu, qubits)
    
    exp = create_experiment(
        temp_qpu,
        qubits,
        lengths=lengths,
    )
  
    compiled_exp = compile_experiment(session, exp)
    _result = run_experiment(session, compiled_exp)
    with if_(options.do_analysis):
        analysis_results = analysis_workflow(_result, qubits, lengths)
        qubit_parameters = analysis_results.tasks["extract_qubit_parameters"].output
        with if_(options.update):
            update_qpu(qpu, qubit_parameters["new_parameter_values"])


@task
@dsl.qubit_experiment
def create_experiment(
    qpu: QPU,
    qubits: QuantumElements,
    lengths: QubitSweepPoints,
    options: TuneupExperimentOptions | None = None,
) -> Experiment:
    """Creates a length-Rabi experiment Workflow.

    Arguments:
        qpu:
            The qpu consisting of the original qubits and quantum operations.
        qubits:
            The qubits to run the experiments on. May be either a single
            qubit or a list of qubits.
        lengths:
            The drive-pulse lengths to sweep over for each qubit. If `qubits` is a
            single qubit, `lengths` must be a list of numbers or an array. Otherwise
            it must be a list of lists of numbers or arrays.
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
            If the qubits and qubit_lengths are not of the same length.

        ValueError:
            If qubit_lengths is not a list of numbers when a single qubit is passed.

        ValueError:
            If qubit_lengths is not a list of lists of numbers.

    Example:
        ```python
        options = TuneupExperimentOptions()
        options.count = 10
        options.transition = "ge"
        options.cal_traces = True
        qpu = QPU(
            qubits=[TunableTransmonQubit("q0"), TunableTransmonQubit("q1")],
            quantum_operations=TunableTransmonOperations(),
        )
        temp_qubits = qpu.copy_qubits()
        create_experiment(
            qpu=qpu,
            qubits=temp_qubits,
            lengths=[
                np.linspace(100e-9, 500e-9, 11),
                np.linspace(100e-9, 500e-9, 11),
            ],
            options=options,
        )
        ```
    """
    # Define the custom options for the experiment
    opts = TuneupExperimentOptions() if options is None else options
    #qubits, lengths = validation.validate_and_convert_single_qubit_sweeps(qubits, lengths)
    qubits, lengths = validation.validate_and_convert_qubits_sweeps(
        qubits, lengths
    )
    lengths_sweep_pars = [
        SweepParameter(f"length_{q.uid}", q_lengths, axis_name=f"{q.uid}")
        for q, q_lengths in zip(qubits, lengths)
    ]


    qop = qpu.quantum_operations
    max_measure_section_length = qop.measure_section_length(qubits)
    with dsl.acquire_loop_rt(
        count=opts.count,
        averaging_mode=opts.averaging_mode,
        acquisition_type=opts.acquisition_type,
        repetition_mode=opts.repetition_mode,
        repetition_time=opts.repetition_time,
        reset_oscillator_phase=opts.reset_oscillator_phase,
    ):
        with dsl.sweep(
            name="rabi_length_sweep",
            parameter=lengths_sweep_pars,
        ):
            if opts.active_reset:
                qop.active_reset(
                    qubits,
                    active_reset_states=opts.active_reset_states,
                    number_resets=opts.active_reset_repetitions,
                    measure_section_length=max_measure_section_length,
                )
            with dsl.section(name="main", alignment=SectionAlignment.RIGHT):
                with dsl.section(name="main_drive", alignment=SectionAlignment.RIGHT):
                    for q, q_lengths in zip(qubits, lengths_sweep_pars):
                        qop.prepare_state.omit_section(q, state=opts.transition[0])
                        ge_drive_length_pi = (
                            q.parameters.ge_drive_length_pi
                            if q.parameters.ge_drive_length_pi is not None
                            else q.parameters.ge_drive_length
                        )

                        sec = qop.qubit_spectroscopy_drive(
                            q,
                            amplitude=q.parameters.ge_drive_amplitude_pi,
                            phase=0.0,
                            length=q_lengths,
                            pulse={
                                "sigma": 0.25,
                                "risefall_sigma_ratio": None,
                                "width": q_lengths - ge_drive_length_pi,
                            },
                        )

                        # sec = qop.x180(
                        #     q, amplitude=q_lengths, transition=opts.transition
                        # )
                        sec.alignment = SectionAlignment.RIGHT
                with dsl.section(name="main_measure", alignment=SectionAlignment.LEFT):
                    for q in qubits:
                        sec = qop.measure(q, dsl.handles.result_handle(q.uid))
                        # Fix the length of the measure section
                        sec.length = max_measure_section_length
                        qop.passive_reset(q)

        if opts.use_cal_traces:
            qop.calibration_traces.omit_section(
                qubits=qubits,
                states=opts.cal_states,
                active_reset=opts.active_reset,
                active_reset_states=opts.active_reset_states,
                active_reset_repetitions=opts.active_reset_repetitions,
                measure_section_length=max_measure_section_length,
            )

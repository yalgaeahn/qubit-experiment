# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""This module defines the DRAG quadrature-scaling calibration experiment.

In this experiment, we determine the quadrature scaling factor, beta, of a DRAG pulse,
which is optimal for cancelling dynamics phase errors that occur during the application
of the pulse. The DRAG drive pulse has the following form:

v(t) = i(t) + q(t),

where the quadrature component is give by the derivative of the in-phase component,
scaled by a scaling factor beta:

q(t) = beta * d(i(t)) / d(t)

In order to determine the optimal beta for compensating phase errors, we apply a pulse
sequence that is sensitive to phase errors and sweep the value of beta for all the
drive pulses in the sequence. In the experiment workflow defined in this file, we
refer to the beta parameter as a q-scaling.

The DRAG quadrature-scaling calibration experiment has the following pulse sequence:

    qb --- [ prep transition ] --- [ x90_transition ]
    --- [ y180_transition ] --- [ measure ]

    qb --- [ prep transition ] --- [ x90_transition ]
    --- [ my180_transition ] --- [ measure ]

If multiple qubits are passed to the `run` workflow, the above pulses are applied
in parallel on all the qubits.
"""

from __future__ import annotations

from collections.abc import Sequence
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

from analysis.drag_q_scaling import analysis_workflow
from laboneq_applications.core import validation
from laboneq_applications.experiments.options import (
    TuneupExperimentOptions,
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
    from laboneq.dsl.session import Session

    from laboneq_applications.typing import QuantumElements, QubitSweepPoints


ALLXY_SEQUENCE_DEFINITIONS: tuple[tuple[str, tuple[str, str]], ...] = (
    ("allxy_00_II", ("I", "I")),
    ("allxy_01_XX", ("X", "X")),
    ("allxy_02_YY", ("Y", "Y")),
    ("allxy_03_XY", ("X", "Y")),
    ("allxy_04_YX", ("Y", "X")),
    ("allxy_05_xI", ("x", "I")),
    ("allxy_06_yI", ("y", "I")),
    ("allxy_07_xy", ("x", "y")),
    ("allxy_08_yx", ("y", "x")),
    ("allxy_09_xY", ("x", "Y")),
    ("allxy_10_yX", ("y", "X")),
    ("allxy_11_Xy", ("X", "y")),
    ("allxy_12_Yx", ("Y", "x")),
    ("allxy_13_xX", ("x", "X")),
    ("allxy_14_Xx", ("X", "x")),
    ("allxy_15_yY", ("y", "Y")),
    ("allxy_16_Yy", ("Y", "y")),
    ("allxy_17_XI", ("X", "I")),
    ("allxy_18_YI", ("Y", "I")),
    ("allxy_19_xx", ("x", "x")),
    ("allxy_20_yy", ("y", "y")),
)

XY3_SEQUENCE_DEFINITIONS: tuple[tuple[str, str, float], ...] = (
    ("xx", "x180", 0.0),
    ("xy", "y180", np.pi / 2),
    ("xmy", "y180", -np.pi / 2),
)


def _play_allxy_gate(qop: object, q: object, gate: str, beta: SweepParameter, transition: str) -> None:
    if gate == "I":
        return
    gate_to_operation = {
        "X": qop.x180,
        "Y": qop.y180,
        "x": qop.x90,
        "y": qop.y90,
    }
    if gate not in gate_to_operation:
        raise ValueError(f"Unsupported ALLXY gate token: {gate}")
    sec = gate_to_operation[gate](q, pulse={"beta": beta}, transition=transition)
    sec.alignment = SectionAlignment.RIGHT


def _extract_beta_from_parameter_dict(parameter_dict: dict | None) -> float | None:
    if not parameter_dict:
        return None
    for key, value in parameter_dict.items():
        if key.endswith("_drive_pulse.beta"):
            return value.nominal_value if hasattr(value, "nominal_value") else float(value)
    return None


def _normalize_single_qubit_sweeps(
    qubits: QuantumElements,
    q_scalings: QubitSweepPoints,
) -> QubitSweepPoints:
    single_qubit_input = not isinstance(qubits, Sequence) or hasattr(qubits, "uid")
    if not single_qubit_input:
        return q_scalings
    if isinstance(q_scalings, np.ndarray):
        return q_scalings
    if isinstance(q_scalings, Sequence) and not isinstance(q_scalings, (str, bytes)):
        if len(q_scalings) == 1:
            first = q_scalings[0]
            if isinstance(first, np.ndarray):
                return first
            if isinstance(first, Sequence) and not isinstance(first, (str, bytes)):
                return np.asarray(first, dtype=float)
    return q_scalings


@workflow.task
def normalize_q_scalings_for_qubits(
    qubits: QuantumElements, q_scalings: QubitSweepPoints
) -> QubitSweepPoints:
    """Normalize sweep-point shape for single-qubit workflows."""
    return _normalize_single_qubit_sweeps(qubits, q_scalings)


@workflow.task
def normalize_mode(mode: str) -> str:
    """Normalize and validate DRAG calibration execution mode."""
    mode_normalized = mode.lower()
    if mode_normalized not in ("coarse", "allxy", "hybrid"):
        raise ValueError(f"Unsupported mode: {mode}. Choose coarse, allxy, or hybrid.")
    return mode_normalized


@workflow.task
def build_fine_q_scalings(
    qubits: QuantumElements,
    q_scalings: QubitSweepPoints,
    coarse_qubit_parameters: dict | None = None,
    fine_span: float = 0.02,
    fine_points: int = 9,
) -> QubitSweepPoints:
    """Build fine beta sweeps centered on coarse estimates."""
    if fine_points < 3:
        raise ValueError("fine_points must be at least 3.")
    qubits_input = qubits
    qubits, q_scalings = validation.validate_and_convert_qubits_sweeps(qubits, q_scalings)
    new_sweeps = []
    coarse_new_params = (
        {}
        if coarse_qubit_parameters is None
        else coarse_qubit_parameters.get("new_parameter_values", {})
    )
    for q, coarse_sweep in zip(qubits, q_scalings):
        coarse_sweep = np.asarray(coarse_sweep, dtype=float)
        if coarse_sweep.size == 0:
            raise ValueError(f"Empty q_scalings for qubit {q.uid}.")
        sweep_min = float(np.min(coarse_sweep))
        sweep_max = float(np.max(coarse_sweep))
        sweep_range = sweep_max - sweep_min
        beta_center = _extract_beta_from_parameter_dict(coarse_new_params.get(q.uid, {}))
        if beta_center is None:
            beta_center = 0.5 * (sweep_min + sweep_max)
        beta_center = float(np.clip(beta_center, sweep_min, sweep_max))
        effective_span = fine_span if fine_span > 0 else 0.2 * sweep_range
        effective_span = min(effective_span, sweep_range) if sweep_range > 0 else 0.0
        if effective_span <= 0:
            new_sweeps.append(np.full(fine_points, beta_center, dtype=float))
            continue
        beta_low = max(sweep_min, beta_center - effective_span / 2)
        beta_high = min(sweep_max, beta_center + effective_span / 2)
        if beta_high <= beta_low:
            beta_low, beta_high = sweep_min, sweep_max
        new_sweeps.append(np.linspace(beta_low, beta_high, fine_points, dtype=float))
    single_qubit_input = not isinstance(qubits_input, Sequence) or hasattr(
        qubits_input, "uid"
    )
    return new_sweeps[0] if single_qubit_input else new_sweeps


@workflow.task
def select_hybrid_qubit_parameters(
    qubits: QuantumElements,
    coarse_qubit_parameters: dict | None,
    fine_qubit_parameters: dict | None,
) -> dict:
    """Select fine ALLXY result when available; otherwise keep coarse result."""
    qubits = validation.validate_and_convert_qubits_sweeps(qubits)
    coarse_qubit_parameters = coarse_qubit_parameters or {}
    fine_qubit_parameters = fine_qubit_parameters or {}
    selected = {
        "old_parameter_values": {q.uid: {} for q in qubits},
        "new_parameter_values": {q.uid: {} for q in qubits},
        "diagnostics": {q.uid: {} for q in qubits},
    }
    coarse_old = coarse_qubit_parameters.get("old_parameter_values", {})
    coarse_new = coarse_qubit_parameters.get("new_parameter_values", {})
    coarse_diag = coarse_qubit_parameters.get("diagnostics", {})
    fine_old = fine_qubit_parameters.get("old_parameter_values", {})
    fine_new = fine_qubit_parameters.get("new_parameter_values", {})
    fine_diag = fine_qubit_parameters.get("diagnostics", {})
    for q in qubits:
        qid = q.uid
        selected["old_parameter_values"][qid] = coarse_old.get(qid, fine_old.get(qid, {}))
        if fine_new.get(qid):
            selected["new_parameter_values"][qid] = fine_new[qid]
            selected["diagnostics"][qid] = {
                "method": "allxy21",
                **fine_diag.get(qid, {}),
            }
        else:
            selected["new_parameter_values"][qid] = coarse_new.get(qid, {})
            selected["diagnostics"][qid] = {
                "method": "xy3",
                **coarse_diag.get(qid, {}),
            }
    return selected


@workflow.workflow(name="drag_q_scaling")
def experiment_workflow(
    session: Session,
    qpu: QPU,
    qubits: QuantumElements,
    q_scalings: QubitSweepPoints,
    temporary_parameters: dict[str | tuple[str, str, str], dict | QuantumParameters]
    | None = None,
    options: TuneUpWorkflowOptions | None = None,
    mode: Literal["coarse", "allxy", "hybrid"] = "coarse",
    fine_span: float = 0.02,
    fine_points: int = 9,
) -> None:
    """The DRAG quadrature-scaling calibration workflow.

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
        q_scalings:
            The DRAG quadrature scaling factors to sweep over for each qubit
            (see docstring at the top of the module). If `qubits` is a single qubit,
            `q_scalings` must be a list of numbers or an array. Otherwise it must be a
            list of lists of numbers or arrays.
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
        WorkflowBuilder:
            The builder of the experiment workflow.

    Example:
        ```python
        options = experiment_workflow()
        options.count(10)
        options.transition("ge")
        qpu = QPU(
            quantum_elements=[TunableTransmonQubit("q0"), TunableTransmonQubit("q1")],
            quantum_operations=TunableTransmonOperations(),
        )
        temp_qubits = qpu.copy_quantum_elements()
        result = experiment_workflow(
            session=session,
            qpu=qpu,
            qubits=temp_qubits,
            q_scalings=[
                np.linspace(-0.05, 0.05, 11),
                np.linspace(-0.04, 0.04, 11),
            ],
            options=options,
        ).run()
        ```
    """
    mode_normalized = normalize_mode(mode)
    temp_qpu = temporary_qpu(qpu, temporary_parameters)
    qubits = temporary_quantum_elements_from_qpu(temp_qpu, qubits)
    q_scalings = normalize_q_scalings_for_qubits(qubits, q_scalings)

    with workflow.if_(mode_normalized == "coarse"):
        coarse_exp = create_experiment(
            temp_qpu,
            qubits,
            q_scalings=q_scalings,
            sequence_set="xy3",
        )
        coarse_compiled_exp = compile_experiment(session, coarse_exp)
        coarse_result = run_experiment(session, coarse_compiled_exp)
        with workflow.if_(options.do_analysis):
            coarse_analysis_results = analysis_workflow(
                coarse_result, qubits, q_scalings, sequence_set="xy3"
            )
            qubit_parameters = coarse_analysis_results.output
            with workflow.if_(options.update):
                update_qpu(qpu, qubit_parameters["new_parameter_values"])
        workflow.return_(coarse_result)

    with workflow.elif_(mode_normalized == "allxy"):
        allxy_exp = create_experiment(
            temp_qpu,
            qubits,
            q_scalings=q_scalings,
            sequence_set="allxy21",
        )
        allxy_compiled_exp = compile_experiment(session, allxy_exp)
        allxy_result = run_experiment(session, allxy_compiled_exp)
        with workflow.if_(options.do_analysis):
            allxy_analysis_results = analysis_workflow(
                allxy_result, qubits, q_scalings, sequence_set="allxy21"
            )
            qubit_parameters = allxy_analysis_results.output
            with workflow.if_(options.update):
                update_qpu(qpu, qubit_parameters["new_parameter_values"])
        workflow.return_(allxy_result)

    with workflow.else_():
        coarse_exp = create_experiment(
            temp_qpu,
            qubits,
            q_scalings=q_scalings,
            sequence_set="xy3",
        )
        coarse_compiled_exp = compile_experiment(session, coarse_exp)
        coarse_result = run_experiment(session, coarse_compiled_exp)
        coarse_qubit_parameters = None
        with workflow.if_(options.do_analysis):
            coarse_analysis_results = analysis_workflow(
                coarse_result, qubits, q_scalings, sequence_set="xy3"
            )
            coarse_qubit_parameters = coarse_analysis_results.output

        fine_q_scalings = build_fine_q_scalings(
            qubits,
            q_scalings,
            coarse_qubit_parameters=coarse_qubit_parameters,
            fine_span=fine_span,
            fine_points=fine_points,
        )
        fine_exp = create_experiment(
            temp_qpu,
            qubits,
            q_scalings=fine_q_scalings,
            sequence_set="allxy21",
        )
        fine_compiled_exp = compile_experiment(session, fine_exp)
        fine_result = run_experiment(session, fine_compiled_exp)
        with workflow.if_(options.do_analysis):
            fine_analysis_results = analysis_workflow(
                fine_result,
                qubits,
                fine_q_scalings,
                sequence_set="allxy21",
            )
            qubit_parameters = select_hybrid_qubit_parameters(
                qubits,
                coarse_qubit_parameters,
                fine_analysis_results.output,
            )
            with workflow.if_(options.update):
                update_qpu(qpu, qubit_parameters["new_parameter_values"])
        workflow.return_(fine_result)


@workflow.task
@dsl.qubit_experiment
def create_experiment(
    qpu: QPU,
    qubits: QuantumElements,
    q_scalings: QubitSweepPoints,
    options: TuneupExperimentOptions | None = None,
    sequence_set: Literal["xy3", "allxy21"] = "xy3",
) -> Experiment:
    """Creates a DRAG quadrature-scaling calibration Experiment.

    Arguments:
        qpu:
            The qpu consisting of the original qubits and quantum operations.
        qubits:
            The qubits to run the experiments on. May be either a single
            qubit or a list of qubits.
        q_scalings:
            The DRAG quadrature scaling factors to sweep over for each qubit
            (see docstring at the top of the module). If `qubits` is a single qubit,
            `q_scalings` must be a list of numbers or an array. Otherwise it must be a
            list of lists of numbers or arrays.
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
            If the qubits and q_scalings are not of the same length.

        ValueError:
            If q_scalings is not a list of numbers when a single qubit is passed.

        ValueError:
            If q_scalings is not a list of lists of numbers.

        ValueError:
            If the experiment uses calibration traces and the averaging mode is
            sequential.

    Example:
        ```python
        options = TuneupExperimentOptions()
        options.count(10)
        options.transition("ge")
        qpu = QPU(
            quantum_elements=[TunableTransmonQubit("q0"), TunableTransmonQubit("q1")],
            quantum_operations=TunableTransmonOperations(),
        )
        temp_qubits = qpu.copy_quantum_elements()
        create_experiment(
            qpu=qpu,
            qubits=temp_qubits,
            q_scalings=[
                np.linspace(-0.05, 0.05, 11),
                np.linspace(-0.05, 0.05, 11),
            ],
            options=options,
        )
        ```
    """
    # Define the custom options for the experiment
    opts = TuneupExperimentOptions() if options is None else options
    q_scalings = _normalize_single_qubit_sweeps(qubits, q_scalings)
    qubits, q_scalings = validation.validate_and_convert_qubits_sweeps(
        qubits, q_scalings
    )
    if (
        opts.use_cal_traces
        and AveragingMode(opts.averaging_mode) == AveragingMode.SEQUENTIAL
    ):
        raise ValueError(
            "'AveragingMode.SEQUENTIAL' (or {AveragingMode.SEQUENTIAL}) cannot be used "
            "with calibration traces because the calibration traces are added "
            "outside the sweep."
        )

    if sequence_set not in ("xy3", "allxy21"):
        raise ValueError(
            f"Unsupported sequence_set: {sequence_set}. Choose xy3 or allxy21."
        )
    qscaling_sweep_pars = [
        SweepParameter(f"beta_{q.uid}", q_qscales, axis_name=f"{q.uid}")
        for q, q_qscales in zip(qubits, q_scalings)
    ]

    # We will fix the length of the measure section to the longest section among
    # the qubits to allow the qubits to have different readout and/or
    # integration lengths.
    max_measure_section_length = qpu.measure_section_length(qubits)
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
            name="drag_q_scaling_sweep",
            parameter=qscaling_sweep_pars,
        ):
            if sequence_set == "xy3":
                sequence_definitions = XY3_SEQUENCE_DEFINITIONS
            else:
                sequence_definitions = ALLXY_SEQUENCE_DEFINITIONS
            for sequence_definition in sequence_definitions:
                if opts.active_reset:
                    qop.active_reset(
                        qubits,
                        active_reset_states=opts.active_reset_states,
                        number_resets=opts.active_reset_repetitions,
                        measure_section_length=max_measure_section_length,
                    )
                with dsl.section(name="main", alignment=SectionAlignment.RIGHT):
                    with dsl.section(
                        name="main_drive", alignment=SectionAlignment.RIGHT
                    ):
                        for q, beta in zip(qubits, qscaling_sweep_pars):
                            pulse_id = sequence_definition[0]
                            qop.prepare_state.omit_section(q, opts.transition[0])
                            if sequence_set == "xy3":
                                op_id = sequence_definition[1]
                                phase = sequence_definition[2]
                                sec = qop.x90(
                                    q, pulse={"beta": beta}, transition=opts.transition
                                )
                                sec.alignment = SectionAlignment.RIGHT
                                sec = qop[op_id](
                                    q,
                                    pulse={"beta": beta},
                                    phase=phase,
                                    transition=opts.transition,
                                )
                                sec.alignment = SectionAlignment.RIGHT
                            else:
                                gates = sequence_definition[1]
                                _play_allxy_gate(
                                    qop,
                                    q,
                                    gates[0],
                                    beta,
                                    opts.transition,
                                )
                                _play_allxy_gate(
                                    qop,
                                    q,
                                    gates[1],
                                    beta,
                                    opts.transition,
                                )
                    with dsl.section(
                        name="main_measure", alignment=SectionAlignment.LEFT
                    ):
                        for q in qubits:
                            sec = qop.measure(
                                q, dsl.handles.result_handle(q.uid, suffix=pulse_id)
                            )
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

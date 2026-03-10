# Copyright 2026 AHNYALGAE
# SPDX-License-Identifier: Apache-2.0

"""Residual ZZ extraction via conditional Ramsey-echo.

This module runs conditional Ramsey-echo measurements across ctrl/targ pairs.
By default, mapping is all-pairs (cartesian product):
    ctrl=[q0,q3], targ=[q1,q2] -> q0->q1, q0->q2, q3->q1, q3->q2

Each pair is executed sequentially in one workflow run.
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

from qubit_experiment.analysis.residual_zz_echo import (
    analysis_workflow,
    build_pair_plan,
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
)

if TYPE_CHECKING:
    from laboneq.dsl.quantum import QuantumParameters
    from laboneq.dsl.quantum.qpu import QPU
    from laboneq.dsl.session import Session
    from laboneq_applications.typing import QuantumElements, QubitSweepPoints


def _validate_ctrl_states(ctrl_states: Sequence[str]) -> tuple[str, str]:
    states = tuple(str(s).lower() for s in ctrl_states)
    if states != ("g", "e"):
        raise ValueError(
            "ResidualZZEchoExperimentOptions.ctrl_states must be exactly ('g', 'e')."
        )
    return ("g", "e")


def _unique_qubits_in_order(qubits: Sequence[object]) -> list[object]:
    out: list[object] = []
    seen: set[str] = set()
    for q in qubits:
        uid = getattr(q, "uid", None)
        if uid in seen:
            continue
        seen.add(uid)
        out.append(q)
    return out


@workflow.task(save=False)
def _select_qubit_by_uid(
    qubits: QuantumElements,
    uid: str,
):
    qubits = validation.validate_and_convert_qubits_sweeps(qubits)
    for q in qubits:
        if q.uid == uid:
            return q
    raise ValueError(f"Qubit uid {uid!r} was not found in provided qubits.")


@workflow.task(save=False)
def _build_run_record(pair_key: str, result: object) -> dict[str, object]:
    return {"pair_key": pair_key, "result": result}


@workflow.task(save=False)
def _filter_ctrl_for_pair(
    ctrl_qubits: QuantumElements,
    targ_uid: str,
) -> list:
    ctrl_qubits = validation.validate_and_convert_qubits_sweeps(ctrl_qubits)
    run_ctrl = [q_c for q_c in ctrl_qubits if q_c.uid != targ_uid]
    if len(run_ctrl) == 0:
        raise ValueError(
            "No control qubits remain for target "
            f"{targ_uid!r} after excluding target from ctrl."
        )
    return run_ctrl


@workflow.task(save=False)
def _append_item(items: list, item: object) -> None:
    items.append(item)


@workflow.task(save=False)
def _records_to_pair_results(
    records: list[dict[str, object]],
) -> dict[str, object]:
    out: dict[str, object] = {}
    for rec in records:
        pair_key = str(rec["pair_key"])
        out[pair_key] = rec["result"]
    return out


@workflow.task_options(base_class=TuneupExperimentOptions)
class ResidualZZEchoExperimentOptions:
    """Options for conditional residual-ZZ echo experiment."""

    refocus_qop: str = workflow.option_field(
        "y180",
        description="Refocusing operation inserted between x90 pulses in echo Ramsey.",
    )
    ctrl_states: tuple[Literal["g"], Literal["e"]] = workflow.option_field(
        ("g", "e"),
        description="Control preparation states, fixed to ('g','e') order.",
    )


@workflow.workflow_options(base_class=TuneUpWorkflowOptions)
class ResidualZZEchoWorkflowOptions:
    """Workflow options for residual ZZ all-pairs/pairwise execution."""

    mapping_mode: Literal["all_pairs", "pairwise"] = workflow.option_field(
        "all_pairs",
        description="How ctrl/targ are mapped: cartesian all-pairs or 1:1 pairwise.",
    )


@workflow.workflow(name="residual_zz_echo")
def experiment_workflow(
    session: Session,
    qpu: QPU,
    ctrl: QuantumElements,
    targ: QuantumElements,
    delays: QubitSweepPoints,
    detunings: float | Sequence[float] | None = None,
    temporary_parameters: dict[str | tuple[str, str, str], dict | QuantumParameters]
    | None = None,
    options: ResidualZZEchoWorkflowOptions | None = None,
) -> None:
    """Run conditional Ramsey-echo to extract residual ZZ interaction."""
    opts = ResidualZZEchoWorkflowOptions() if options is None else options
    mapping_mode = opts.mapping_mode

    temp_qpu = temporary_qpu(qpu, temporary_parameters)
    ctrl = temporary_quantum_elements_from_qpu(temp_qpu, ctrl)
    targ = temporary_quantum_elements_from_qpu(temp_qpu, targ)
    ctrl_all = validation.validate_and_convert_qubits_sweeps(ctrl)

    pair_plan = build_pair_plan(
        ctrl=ctrl_all,
        targ=targ,
        delays=delays,
        detunings=detunings,
        mapping_mode=mapping_mode,
    )

    run_records: list = []
    with workflow.for_(pair_plan, lambda pair: pair["pair_key"]) as pair:
        run_ctrl = _filter_ctrl_for_pair(
            ctrl_qubits=ctrl_all,
            targ_uid=pair["targ_uid"],
        )
        pair_targ = _select_qubit_by_uid(
            qubits=targ,
            uid=pair["targ_uid"],
        )
        exp = create_experiment(
            qpu=temp_qpu,
            ctrl=run_ctrl,
            targ=pair_targ,
            delays=pair["delay_values"],
            detunings=pair["detuning_hz"],
            active_ctrl_uid=pair["ctrl_uid"],
        )
        compiled_exp = compile_experiment(session, exp)
        result = run_experiment(session, compiled_exp)
        run_record = _build_run_record(pair_key=pair["pair_key"], result=result)
        _append_item(run_records, run_record)

    results_by_pair = _records_to_pair_results(run_records)

    with workflow.if_(opts.do_analysis):
        _ = analysis_workflow(
            result=results_by_pair,
            ctrl=ctrl_all,
            targ=targ,
            delays=delays,
            detunings=detunings,
            mapping_mode=mapping_mode,
        )

    workflow.return_(results_by_pair)


@workflow.task
@dsl.qubit_experiment
def create_experiment(
    qpu: QPU,
    ctrl: QuantumElements,
    targ: QuantumElements,
    delays: QubitSweepPoints,
    detunings: float | Sequence[float] | None = None,
    active_ctrl_uid: str | None = None,
    options: ResidualZZEchoExperimentOptions | None = None,
) -> Experiment:
    """Create conditional echo-Ramsey experiment for one or more target qubits.

    Sweep dimensions:
    - outer: control prepared state in g/e
    - inner: delay (and matching phase sweep for Ramsey detuning) per target

    If `active_ctrl_uid` is provided, only that control qubit is toggled g/e and
    all other control qubits are fixed to g in both branches.
    """
    opts = ResidualZZEchoExperimentOptions() if options is None else options

    ctrl = validation.validate_and_convert_qubits_sweeps(ctrl)
    targ, delays = validation.validate_and_convert_qubits_sweeps(targ, delays)
    detunings = validate_and_convert_detunings(targ, detunings)
    ctrl_states = _validate_ctrl_states(opts.ctrl_states)

    if active_ctrl_uid is not None and active_ctrl_uid not in {q.uid for q in ctrl}:
        raise ValueError(
            f"active_ctrl_uid={active_ctrl_uid!r} is not present in ctrl qubits."
        )

    if (
        opts.use_cal_traces
        and AveragingMode(opts.averaging_mode) == AveragingMode.SEQUENTIAL
    ):
        raise ValueError(
            "'AveragingMode.SEQUENTIAL' cannot be used with calibration traces "
            "because calibration traces are added outside the sweep."
        )

    swp_ctrl_state = SweepParameter(
        uid="ctrl_state_index",
        values=np.array([0, 1], dtype=int),
        axis_name="ctrl_state",
    )
    swp_delays: list[SweepParameter] = []
    swp_phases: list[SweepParameter] = []
    for i, q_t in enumerate(targ):
        q_delays = np.asarray(delays[i], dtype=float).ravel()
        swp_delays.append(
            SweepParameter(
                uid=f"wait_time_{q_t.uid}",
                values=q_delays,
                axis_name=f"{q_t.uid}_tau",
            )
        )
        phase_values = ((q_delays - q_delays[0]) * detunings[i] * 2 * np.pi) % (
            2 * np.pi
        )
        swp_phases.append(
            SweepParameter(
                uid=f"x90_phase_{q_t.uid}",
                values=phase_values,
            )
        )

    qop = qpu.quantum_operations
    max_measure_section_length = qop.measure_section_length(targ)
    active_reset_qubits = _unique_qubits_in_order([*ctrl, *targ])

    with dsl.acquire_loop_rt(
        count=opts.count,
        averaging_mode=opts.averaging_mode,
        acquisition_type=opts.acquisition_type,
        repetition_mode=opts.repetition_mode,
        repetition_time=opts.repetition_time,
        reset_oscillator_phase=opts.reset_oscillator_phase,
    ):
        with dsl.sweep(
            name="ctrl_state_sweep",
            parameter=swp_ctrl_state,
            auto_chunking=False,
            alignment=SectionAlignment.LEFT,
        ) as ctrl_state_index:
            with dsl.sweep(
                name="residual_zz_echo_sweep",
                parameter=swp_delays + swp_phases,
                auto_chunking=True,
            ):
                if opts.active_reset:
                    qop.active_reset(
                        active_reset_qubits,
                        active_reset_states=opts.active_reset_states,
                        number_resets=opts.active_reset_repetitions,
                        measure_section_length=max_measure_section_length,
                    )

                with dsl.match(
                    name="ctrl_prepare",
                    sweep_parameter=ctrl_state_index,
                ) as ctrl_prepare:
                    with dsl.case(0):
                        for q_c in ctrl:
                            qop.prepare_state(q_c, ctrl_states[0])
                    with dsl.case(1):
                        for q_c in ctrl:
                            if active_ctrl_uid is None:
                                state = ctrl_states[1]
                            else:
                                state = (
                                    ctrl_states[1]
                                    if q_c.uid == active_ctrl_uid
                                    else ctrl_states[0]
                                )
                            qop.prepare_state(q_c, state)

                with dsl.section(
                    name="main",
                    alignment=SectionAlignment.RIGHT,
                    play_after=ctrl_prepare.uid,
                ):
                    with dsl.section(
                        name="main_drive",
                        alignment=SectionAlignment.RIGHT,
                    ):
                        for q_t, wait_time, phase in zip(targ, swp_delays, swp_phases):
                            qop.prepare_state.omit_section(q_t, opts.transition[0])
                            qop.ramsey.omit_section(
                                q_t,
                                wait_time,
                                phase,
                                echo_pulse=opts.refocus_qop,
                                transition=opts.transition,
                            )

                    with dsl.section(
                        name="main_measure",
                        alignment=SectionAlignment.LEFT,
                    ):
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

# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Local coherence-tracking workflow.

Runs local T1, T2*, and echo-based T2 measurements in one suite.
Composes only repo-owned experiment and analysis modules.
Persists trend history through the tracking analysis when enabled.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from analysis.coherence_tracking import analysis_workflow as tracking_analysis_workflow
from laboneq import workflow
from laboneq.workflow.tasks import (
    compile_experiment,
    run_experiment,
)
from laboneq_applications.experiments.options import TuneUpWorkflowOptions
from laboneq_applications.tasks.parameter_updating import (
    temporary_qpu,
    temporary_quantum_elements_from_qpu,
    update_qpu,
)

from experiments import lifetime_measurement, ramsey

if TYPE_CHECKING:
    from laboneq.dsl.quantum import QuantumParameters
    from laboneq.dsl.quantum.qpu import QPU
    from laboneq.dsl.session import Session
    from laboneq.workflow.tasks.run_experiment import RunExperimentResults
    from laboneq_applications.typing import QuantumElements, QubitSweepPoints


def _resolve_metric_flags_value(
    *,
    run_t1: bool,
    run_t2_star: bool,
    run_t2: bool,
    t1_delays: object,
    t2_star_delays: object,
    t2_delays: object,
) -> dict[str, bool]:
    flags = {
        "run_t1": bool(run_t1),
        "run_t2_star": bool(run_t2_star),
        "run_t2": bool(run_t2),
    }
    if not any(flags.values()):
        raise ValueError("At least one of run_t1, run_t2_star, or run_t2 must be True.")
    if flags["run_t1"] and t1_delays is None:
        raise ValueError("t1_delays must be provided when run_t1 is enabled.")
    if flags["run_t2_star"] and t2_star_delays is None:
        raise ValueError("t2_star_delays must be provided when run_t2_star is enabled.")
    if flags["run_t2"] and t2_delays is None:
        raise ValueError("t2_delays must be provided when run_t2 is enabled.")
    return flags


def _validate_transition_value(transition: object) -> str:
    resolved = "ge" if transition is None else str(transition)
    if resolved != "ge":
        raise ValueError(
            "coherence_tracking currently supports only transition='ge'. "
            f"Got {resolved!r}."
        )
    return resolved


@workflow.workflow_options(base_class=TuneUpWorkflowOptions)
class CoherenceTrackingWorkflowOptions:
    """Workflow options for local coherence tracking."""

    run_t1: bool = workflow.option_field(
        True,
        description="Whether to run the local lifetime-measurement branch.",
    )
    run_t2_star: bool = workflow.option_field(
        True,
        description="Whether to run the local Ramsey T2* branch.",
    )
    run_t2: bool = workflow.option_field(
        True,
        description="Whether to run the local echo-based T2 branch.",
    )
    refocus_qop: str = workflow.option_field(
        "y180",
        description="Refocusing operation used by the T2 echo branch.",
    )
    history_path: str = workflow.option_field(
        "laboneq_output/tracking/coherence_tracking.jsonl",
        description="JSONL path used for cross-run coherence tracking history.",
    )
    transition: str = workflow.option_field(
        "ge",
        description="Tracking currently supports only the ge transition.",
    )


@workflow.task(save=False)
def _resolve_metric_flags(
    run_t1: bool,
    run_t2_star: bool,
    run_t2: bool,
    t1_delays: object,
    t2_star_delays: object,
    t2_delays: object,
) -> dict[str, bool]:
    return _resolve_metric_flags_value(
        run_t1=run_t1,
        run_t2_star=run_t2_star,
        run_t2=run_t2,
        t1_delays=t1_delays,
        t2_star_delays=t2_star_delays,
        t2_delays=t2_delays,
    )


@workflow.task(save=False)
def _validate_transition(transition: object) -> str:
    return _validate_transition_value(transition)


@workflow.task(save=False)
def _empty_run_result():
    return None


@workflow.task(save=False)
def _empty_analysis_output():
    return None


@workflow.task(save=False)
def _build_workflow_output(
    t1_result: RunExperimentResults | None,
    t2_star_result: RunExperimentResults | None,
    t2_result: RunExperimentResults | None,
    analysis_output: dict[str, Any] | None,
) -> dict[str, object]:
    return {
        "results": {
            "t1": t1_result,
            "t2_star": t2_star_result,
            "t2": t2_result,
        },
        "analysis": analysis_output,
    }


@workflow.workflow(name="coherence_tracking")
def experiment_workflow(
    session: Session,
    qpu: QPU,
    qubits: QuantumElements,
    t1_delays: QubitSweepPoints | None = None,
    t2_star_delays: QubitSweepPoints | None = None,
    t2_delays: QubitSweepPoints | None = None,
    t2_star_detunings: float | Sequence[float] | None = None,
    temporary_parameters: dict[str | tuple[str, str, str], dict | QuantumParameters]
    | None = None,
    options: CoherenceTrackingWorkflowOptions | None = None,
) -> None:
    """Run local T1, T2*, and T2 measurements and aggregate their analysis."""
    metric_flags = _resolve_metric_flags(
        options.run_t1,
        options.run_t2_star,
        options.run_t2,
        t1_delays,
        t2_star_delays,
        t2_delays,
    )
    _validate_transition(options.transition)

    temp_qpu = temporary_qpu(qpu, temporary_parameters)
    qubits = temporary_quantum_elements_from_qpu(temp_qpu, qubits)

    t1_result = _empty_run_result()
    t2_star_result = _empty_run_result()
    t2_result = _empty_run_result()
    analysis_output = _empty_analysis_output()

    with workflow.if_(metric_flags["run_t1"]):
        t1_exp = lifetime_measurement.create_experiment(
            temp_qpu,
            qubits,
            delays=t1_delays,
        )
        t1_compiled_exp = compile_experiment(session, t1_exp)
        t1_result = run_experiment(session, t1_compiled_exp)

    with workflow.if_(metric_flags["run_t2_star"]):
        t2_star_exp = ramsey.create_experiment(
            temp_qpu,
            qubits,
            delays=t2_star_delays,
            detunings=t2_star_detunings,
            echo=False,
            refocus_qop=options.refocus_qop,
        )
        t2_star_compiled_exp = compile_experiment(session, t2_star_exp)
        t2_star_result = run_experiment(session, t2_star_compiled_exp)

    with workflow.if_(metric_flags["run_t2"]):
        t2_exp = ramsey.create_experiment(
            temp_qpu,
            qubits,
            delays=t2_delays,
            detunings=None,
            echo=True,
            refocus_qop=options.refocus_qop,
        )
        t2_compiled_exp = compile_experiment(session, t2_exp)
        t2_result = run_experiment(session, t2_compiled_exp)

    with workflow.if_(options.do_analysis):
        tracking_analysis = tracking_analysis_workflow(
            qubits=qubits,
            t1_result=t1_result,
            t2_star_result=t2_star_result,
            t2_result=t2_result,
            t1_delays=t1_delays,
            t2_star_delays=t2_star_delays,
            t2_delays=t2_delays,
            t2_star_detunings=t2_star_detunings,
            run_t1=metric_flags["run_t1"],
            run_t2_star=metric_flags["run_t2_star"],
            run_t2=metric_flags["run_t2"],
            history_path=options.history_path,
        )
        analysis_output = tracking_analysis.output
        with workflow.if_(options.update):
            update_qpu(qpu, analysis_output["new_parameter_values"])

    workflow.return_(
        _build_workflow_output(
            t1_result=t1_result,
            t2_star_result=t2_star_result,
            t2_result=t2_result,
            analysis_output=analysis_output,
        )
    )

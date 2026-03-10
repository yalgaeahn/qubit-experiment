# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Local coherence-tracking workflow.

Runs selected local T1, T2*, and echo-based T2 measurements in one suite and can
own a long-running monitoring window. Iterations are scheduled in workflow space,
history is materialized after each successful iteration through the tracking analysis,
and persistent updates apply only to the selected tracked parameters.
"""

from __future__ import annotations

import time
import traceback
from collections.abc import Sequence
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from qubit_experiment.analysis.coherence_tracking import (
    CoherenceTrackingAnalysisWorkflowOptions,
    load_tracking_history,
    plot_tracking_history,
    resolve_history_path,
    run_iteration_analysis_value,
)
from laboneq import workflow
from laboneq.workflow.opts.options_core import task_options
from laboneq_applications.experiments.options import (
    TuneupExperimentOptions,
    TuneUpWorkflowOptions,
)
from laboneq_applications.tasks.parameter_updating import (
    temporary_qpu,
    temporary_quantum_elements_from_qpu,
    update_qpu,
)

from . import lifetime_measurement, ramsey

if TYPE_CHECKING:
    from laboneq.dsl.quantum import QuantumParameters
    from laboneq.dsl.quantum.qpu import QPU
    from laboneq.dsl.session import Session
    from laboneq_applications.typing import QuantumElements, QubitSweepPoints


_TRACKED_PARAMETER_ALIASES = {
    "coherence": ("ge_T1", "ge_T2_star", "ge_T2"),
    "all": ("ge_T1", "ge_T2_star", "ge_T2", "resonance_frequency_ge"),
}
_TRACKED_PARAMETER_TO_METRIC = {
    "ge_T1": "t1",
    "ge_T2_star": "t2_star",
    "ge_T2": "t2",
    "resonance_frequency_ge": "t2_star",
}


def _normalize_tracked_parameters_value(
    tracked_parameters: Sequence[str] | str,
) -> tuple[str, ...]:
    raw_items: list[str] = []
    if isinstance(tracked_parameters, str):
        candidates = [tracked_parameters]
    else:
        candidates = list(tracked_parameters)

    for candidate in candidates:
        if not isinstance(candidate, str):
            raise ValueError(
                "tracked_parameters must contain only strings or be a single string."
            )
        raw_items.extend(part.strip() for part in candidate.split(",") if part.strip())

    if not raw_items:
        raise ValueError("tracked_parameters must contain at least one supported key.")

    canonical_by_lower = {
        key.lower(): key for key in _TRACKED_PARAMETER_TO_METRIC
    }
    normalized: list[str] = []
    for item in raw_items:
        lowered = item.lower()
        expanded = _TRACKED_PARAMETER_ALIASES.get(lowered)
        if expanded is not None:
            for key in expanded:
                if key not in normalized:
                    normalized.append(key)
            continue

        canonical = canonical_by_lower.get(lowered)
        if canonical is None:
            supported = sorted(_TRACKED_PARAMETER_TO_METRIC)
            aliases = sorted(_TRACKED_PARAMETER_ALIASES)
            raise ValueError(
                "Unsupported tracked parameter. "
                f"Got {item!r}. Supported keys: {supported}. Aliases: {aliases}."
            )
        if canonical not in normalized:
            normalized.append(canonical)

    if not normalized:
        raise ValueError("tracked_parameters must resolve to at least one key.")
    return tuple(normalized)


def _resolve_tracking_plan_value(
    tracked_parameters: Sequence[str] | str,
    t1_delays: object,
    t2_star_delays: object,
    t2_delays: object,
) -> dict[str, object]:
    normalized = _normalize_tracked_parameters_value(tracked_parameters)
    need_t1 = any(_TRACKED_PARAMETER_TO_METRIC[param] == "t1" for param in normalized)
    need_t2_star = any(
        _TRACKED_PARAMETER_TO_METRIC[param] == "t2_star" for param in normalized
    )
    need_t2 = any(_TRACKED_PARAMETER_TO_METRIC[param] == "t2" for param in normalized)

    if need_t1 and t1_delays is None:
        raise ValueError(
            "t1_delays must be provided when tracking ge_T1 or alias coherence/all."
        )
    if need_t2_star and t2_star_delays is None:
        raise ValueError(
            "t2_star_delays must be provided when tracking ge_T2_star or "
            "resonance_frequency_ge."
        )
    if need_t2 and t2_delays is None:
        raise ValueError("t2_delays must be provided when tracking ge_T2.")

    required_metrics = []
    if need_t1:
        required_metrics.append("t1")
    if need_t2_star:
        required_metrics.append("t2_star")
    if need_t2:
        required_metrics.append("t2")

    return {
        "tracked_parameters": list(normalized),
        "need_t1": need_t1,
        "need_t2_star": need_t2_star,
        "need_t2": need_t2,
        "required_metrics": required_metrics,
    }


def _resolve_iteration_schedule_value(
    total_duration_s: float | None,
    interval_s: float,
) -> list[dict[str, float | int]]:
    interval = float(interval_s)
    if not interval > 0 or not interval < float("inf"):
        raise ValueError("interval_s must be positive and finite.")

    if total_duration_s is None:
        return [{"iteration_index": 0, "scheduled_offset_s": 0.0}]

    duration = float(total_duration_s)
    if not duration > 0 or not duration < float("inf"):
        raise ValueError("total_duration_s must be positive and finite when provided.")

    schedule = [{"iteration_index": 0, "scheduled_offset_s": 0.0}]
    index = 1
    while index * interval < duration:
        schedule.append(
            {
                "iteration_index": index,
                "scheduled_offset_s": float(index * interval),
            }
        )
        index += 1
    return schedule


def _validate_transition_value(transition: object) -> str:
    resolved = "ge" if transition is None else str(transition)
    if resolved != "ge":
        raise ValueError(
            "coherence_tracking currently supports only transition='ge'. "
            f"Got {resolved!r}."
        )
    return resolved


def _resolve_plot_modes_value(
    total_duration_s: float | None,
    do_plotting: bool,
) -> dict[str, bool]:
    long_running = total_duration_s is not None
    enabled = bool(do_plotting)
    return {
        "render_metric_plots": enabled and not long_running,
        "render_history_plot": enabled and not long_running,
        "render_final_history_plot": enabled and long_running,
    }


@workflow.workflow_options(base_class=TuneUpWorkflowOptions)
class CoherenceTrackingWorkflowOptions:
    """Workflow options for local coherence tracking."""

    do_plotting: bool = workflow.option_field(
        True,
        description=(
            "Whether to render tracking plots. In long-running mode this controls "
            "the final history artifact; per-iteration plots stay suppressed."
        ),
    )
    tracked_parameters: str | tuple[str, ...] = workflow.option_field(
        ("ge_T1", "ge_T2_star", "ge_T2"),
        description=(
            "Tracked QPU parameter keys. Supported keys are 'ge_T1', 'ge_T2_star', "
            "'ge_T2', and 'resonance_frequency_ge'. Aliases: 'coherence', 'all'."
        ),
    )
    refocus_qop: str = workflow.option_field(
        "y180",
        description="Refocusing operation used by the T2 echo branch.",
    )
    history_path: str = workflow.option_field(
        "tracking/coherence_tracking.jsonl",
        description="JSONL path used for cross-run coherence tracking history.",
    )
    total_duration_s: float | None = workflow.option_field(
        None,
        description=(
            "When provided, run long-running mode for this total monitoring window in "
            "seconds. The first iteration starts immediately."
        ),
    )
    interval_s: float = workflow.option_field(
        300.0,
        description="Target spacing in seconds between iteration start times.",
    )
    continue_on_iteration_error: bool = workflow.option_field(
        True,
        description="Whether to continue later scheduled iterations after an execution failure.",
    )
    transition: str = workflow.option_field(
        "ge",
        description="Tracking currently supports only the ge transition.",
    )


@task_options
class CoherenceTrackingIterationAnalysisTaskOptions:
    """Notebook-facing analysis plotting options for iteration-level tracking."""

    do_raw_data_plotting: bool = workflow.option_field(
        True,
        description="Whether raw-data plots are enabled for single-pass tracking.",
    )
    do_qubit_population_plotting: bool = workflow.option_field(
        True,
        description="Whether qubit-population plots are enabled for single-pass tracking.",
    )


@workflow.task(save=False)
def _resolve_tracking_plan(
    tracked_parameters: Sequence[str] | str,
    t1_delays: object,
    t2_star_delays: object,
    t2_delays: object,
) -> dict[str, object]:
    return _resolve_tracking_plan_value(
        tracked_parameters,
        t1_delays,
        t2_star_delays,
        t2_delays,
    )


@workflow.task(save=False)
def _resolve_iteration_schedule(
    total_duration_s: float | None,
    interval_s: float,
) -> list[dict[str, float | int]]:
    return _resolve_iteration_schedule_value(total_duration_s, interval_s)


@workflow.task(save=False)
def _validate_transition(transition: object) -> str:
    return _validate_transition_value(transition)


@workflow.task(save=False)
def _resolve_plot_modes(
    total_duration_s: float | None,
    do_plotting: bool,
) -> dict[str, bool]:
    return _resolve_plot_modes_value(total_duration_s, do_plotting)


@workflow.task(save=False)
def _start_tracking_runtime(total_duration_s: float | None) -> dict[str, object]:
    started_utc = datetime.now(timezone.utc)
    started_monotonic = time.monotonic()
    deadline_monotonic = None
    if total_duration_s is not None:
        deadline_monotonic = started_monotonic + float(total_duration_s)
    return {
        "run_started_utc": started_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_started_monotonic": started_monotonic,
        "deadline_monotonic": deadline_monotonic,
    }


def _wait_for_iteration_slot_value(
    runtime: dict[str, object],
    iteration_plan: dict[str, object],
) -> dict[str, object]:
    start_monotonic = float(runtime["run_started_monotonic"])
    deadline_monotonic = runtime.get("deadline_monotonic")
    scheduled_offset_s = float(iteration_plan["scheduled_offset_s"])
    target_monotonic = start_monotonic + scheduled_offset_s

    now_monotonic = time.monotonic()
    waited_s = max(0.0, target_monotonic - now_monotonic)
    if waited_s > 0:
        time.sleep(waited_s)
    started_monotonic = time.monotonic()

    started_utc_dt = datetime.now(timezone.utc)
    scheduled_start_utc = datetime.fromisoformat(
        str(runtime["run_started_utc"]).replace("Z", "+00:00")
    ) + timedelta(seconds=scheduled_offset_s)

    if deadline_monotonic is not None and started_monotonic > float(deadline_monotonic):
        return {
            "run_iteration": False,
            "skip_reason": "deadline_passed",
            "waited_s": waited_s,
            "scheduled_start_utc": scheduled_start_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "started_utc": None,
        }

    return {
        "run_iteration": True,
        "skip_reason": None,
        "waited_s": waited_s,
        "scheduled_start_utc": scheduled_start_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "started_utc": started_utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


@workflow.task(save=False)
def _wait_for_iteration_slot(
    runtime: dict[str, object],
    iteration_plan: dict[str, object],
) -> dict[str, object]:
    return _wait_for_iteration_slot_value(runtime, iteration_plan)


def _empty_execution_bundle_value(metric_id: str) -> dict[str, object]:
    return {
        "metric_id": metric_id,
        "ok": False,
        "status": "skipped",
        "error_stage": None,
        "error_type": None,
        "error_message": None,
        "result": None,
    }


@workflow.task(save=False)
def _empty_execution_bundle(metric_id: str) -> dict[str, object]:
    return _empty_execution_bundle_value(metric_id)


@workflow.task(save=False)
def _materialize_execution_bundle(bundle: dict[str, object]) -> dict[str, object]:
    return dict(bundle)


@workflow.task(save=False)
def _extract_execution_result(bundle: dict[str, object]) -> object:
    return bundle.get("result")


def _execute_experiment_safe_value(
    session: Session,
    experiment,
    metric_id: str,
) -> dict[str, object]:
    try:
        compiled_experiment = session.compile(experiment=experiment)
    except BaseException as exc:  # noqa: BLE001
        return {
            "metric_id": metric_id,
            "ok": False,
            "status": "failed",
            "error_stage": "compile",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
            "result": None,
        }

    try:
        result = session.run(compiled_experiment)
    except BaseException as exc:  # noqa: BLE001
        return {
            "metric_id": metric_id,
            "ok": False,
            "status": "failed",
            "error_stage": "run",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
            "result": None,
        }

    return {
        "metric_id": metric_id,
        "ok": True,
        "status": "completed",
        "error_stage": None,
        "error_type": None,
        "error_message": None,
        "result": result,
    }


@workflow.task(save=False)
def _execute_experiment_safe(
    session: Session,
    experiment,
    metric_id: str,
) -> dict[str, object]:
    return _execute_experiment_safe_value(session, experiment, metric_id)


def _empty_last_successful_analysis_value() -> dict[str, object]:
    return {
        "metric_outputs": {},
        "old_parameter_values": {},
        "new_parameter_values": {},
        "history_path": None,
        "history_entries": [],
        "history_rows": [],
        "tracked_parameters": [],
    }


@workflow.task(save=False)
def _empty_last_successful_analysis() -> dict[str, object]:
    return _empty_last_successful_analysis_value()


def _summarize_iteration_execution_value(
    iteration_plan: dict[str, object],
    slot_bundle: dict[str, object],
    tracking_plan: dict[str, object],
    t1_bundle: dict[str, object],
    t2_star_bundle: dict[str, object],
    t2_bundle: dict[str, object],
) -> dict[str, object]:
    bundles = {
        "t1": t1_bundle,
        "t2_star": t2_star_bundle,
        "t2": t2_bundle,
    }
    required_metrics = list(tracking_plan["required_metrics"])
    failed_metric = None
    error_stage = None
    error_message = None
    per_metric_status: dict[str, dict[str, object]] = {}

    for metric_id in required_metrics:
        bundle = bundles[metric_id]
        per_metric_status[metric_id] = {
            "status": bundle["status"],
            "error_stage": bundle["error_stage"],
            "error_message": bundle["error_message"],
        }
        if failed_metric is None and not bool(bundle["ok"]):
            failed_metric = metric_id
            error_stage = bundle["error_stage"]
            error_message = bundle["error_message"]

    return {
        "iteration_index": int(iteration_plan["iteration_index"]),
        "scheduled_offset_s": float(iteration_plan["scheduled_offset_s"]),
        "scheduled_start_utc": slot_bundle["scheduled_start_utc"],
        "started_utc": slot_bundle["started_utc"],
        "status": "completed" if failed_metric is None else "failed_execution",
        "successful": failed_metric is None,
        "tracked_parameters": list(tracking_plan["tracked_parameters"]),
        "required_metrics": required_metrics,
        "per_metric_status": per_metric_status,
        "failed_metric": failed_metric,
        "error_stage": error_stage,
        "error_message": error_message,
        "waited_s": float(slot_bundle["waited_s"]),
    }


@workflow.task(save=False)
def _summarize_iteration_execution(
    iteration_plan: dict[str, object],
    slot_bundle: dict[str, object],
    tracking_plan: dict[str, object],
    t1_bundle: dict[str, object],
    t2_star_bundle: dict[str, object],
    t2_bundle: dict[str, object],
) -> dict[str, object]:
    return _summarize_iteration_execution_value(
        iteration_plan,
        slot_bundle,
        tracking_plan,
        t1_bundle,
        t2_star_bundle,
        t2_bundle,
    )


def _summarize_skipped_iteration_value(
    iteration_plan: dict[str, object],
    tracking_plan: dict[str, object],
    *,
    scheduled_start_utc: str | None,
    status: str,
    skip_reason: str,
) -> dict[str, object]:
    return {
        "iteration_index": int(iteration_plan["iteration_index"]),
        "scheduled_offset_s": float(iteration_plan["scheduled_offset_s"]),
        "scheduled_start_utc": scheduled_start_utc,
        "started_utc": None,
        "status": status,
        "successful": False,
        "tracked_parameters": list(tracking_plan["tracked_parameters"]),
        "required_metrics": list(tracking_plan["required_metrics"]),
        "per_metric_status": {},
        "failed_metric": None,
        "error_stage": None,
        "error_message": None,
        "skip_reason": skip_reason,
        "waited_s": 0.0,
    }


@workflow.task(save=False)
def _summarize_skipped_iteration(
    iteration_plan: dict[str, object],
    tracking_plan: dict[str, object],
    *,
    scheduled_start_utc: str | None,
    status: str,
    skip_reason: str,
) -> dict[str, object]:
    return _summarize_skipped_iteration_value(
        iteration_plan,
        tracking_plan,
        scheduled_start_utc=scheduled_start_utc,
        status=status,
        skip_reason=skip_reason,
    )


def _attach_iteration_analysis_value(
    iteration_summary: dict[str, object],
    analysis_output: dict[str, object],
) -> dict[str, object]:
    summary = dict(iteration_summary)
    updated_parameter_keys: set[str] = set()
    new_parameter_values = analysis_output.get("new_parameter_values", {})
    if isinstance(new_parameter_values, dict):
        for values in new_parameter_values.values():
            if isinstance(values, dict):
                updated_parameter_keys.update(str(key) for key in values)
    summary["history_row_count"] = len(list(analysis_output.get("history_entries", [])))
    summary["updated_parameter_keys"] = sorted(updated_parameter_keys)
    return summary


@workflow.task(save=False)
def _attach_iteration_analysis(
    iteration_summary: dict[str, object],
    analysis_output: dict[str, object],
) -> dict[str, object]:
    return _attach_iteration_analysis_value(iteration_summary, analysis_output)


def _build_final_analysis_output_value(
    last_successful_analysis: dict[str, object],
    iteration_summaries: Sequence[dict[str, object]],
    history_entries: Sequence[dict[str, object]],
    tracked_parameters: Sequence[str],
    history_path: str | None,
) -> dict[str, object]:
    output = dict(last_successful_analysis)
    output.pop("history_rows", None)
    output["tracked_parameters"] = list(tracked_parameters)
    output["history_path"] = history_path
    output["history_entries"] = list(history_entries)
    output["iteration_summaries"] = list(iteration_summaries)

    successful_iterations = 0
    failed_iterations = 0
    last_successful_iteration_index = None
    for summary in iteration_summaries:
        status = str(summary.get("status"))
        if status == "completed":
            successful_iterations += 1
            last_successful_iteration_index = int(summary["iteration_index"])
        elif status == "failed_execution":
            failed_iterations += 1

    output["successful_iterations"] = successful_iterations
    output["failed_iterations"] = failed_iterations
    output["last_successful_iteration_index"] = last_successful_iteration_index
    output.setdefault("metric_outputs", {})
    output.setdefault("old_parameter_values", {})
    output.setdefault("new_parameter_values", {})
    return output


@workflow.task(save=False)
def _build_final_analysis_output(
    last_successful_analysis: dict[str, object],
    iteration_summaries: Sequence[dict[str, object]],
    history_entries: Sequence[dict[str, object]],
    tracked_parameters: Sequence[str],
    history_path: str | None,
) -> dict[str, object]:
    return _build_final_analysis_output_value(
        last_successful_analysis,
        iteration_summaries,
        history_entries,
        tracked_parameters,
        history_path,
    )


def _build_workflow_output_value(
    analysis_output: dict[str, object] | None,
    iteration_summaries: Sequence[dict[str, object]],
    tracked_parameters: Sequence[str],
) -> dict[str, object]:
    return {
        "analysis": analysis_output,
        "iteration_summaries": list(iteration_summaries),
        "tracked_parameters": list(tracked_parameters),
    }


@workflow.task(save=False)
def _build_workflow_output(
    analysis_output: dict[str, object] | None,
    iteration_summaries: Sequence[dict[str, object]],
    tracked_parameters: Sequence[str],
) -> dict[str, object]:
    return _build_workflow_output_value(
        analysis_output,
        iteration_summaries,
        tracked_parameters,
    )


def _resolve_create_experiment_options_value(
    workflow_options: object | None,
) -> TuneupExperimentOptions:
    if workflow_options is None:
        return TuneupExperimentOptions()
    task_options = getattr(workflow_options, "_task_options", None)
    if isinstance(task_options, dict):
        create_options = task_options.get("create_experiment")
        if create_options is not None:
            return deepcopy(create_options)
    return TuneupExperimentOptions()


@workflow.task(save=False)
def _clone_create_experiment_options(
    options: TuneupExperimentOptions | None = None,
) -> TuneupExperimentOptions:
    return deepcopy(TuneupExperimentOptions() if options is None else options)


@workflow.task(save=False)
def _clone_iteration_analysis_options(
    options: CoherenceTrackingAnalysisWorkflowOptions | None = None,
) -> CoherenceTrackingAnalysisWorkflowOptions:
    return deepcopy(
        CoherenceTrackingAnalysisWorkflowOptions() if options is None else options
    )


@workflow.task(save=False)
def _capture_iteration_analysis_task_options(
    options: CoherenceTrackingIterationAnalysisTaskOptions | None = None,
) -> dict[str, bool]:
    opts = (
        CoherenceTrackingIterationAnalysisTaskOptions()
        if options is None
        else options
    )
    return {
        "do_raw_data_plotting": bool(opts.do_raw_data_plotting),
        "do_qubit_population_plotting": bool(opts.do_qubit_population_plotting),
    }


@workflow.task(save=False)
def _initialize_tracking_state() -> dict[str, object]:
    return {
        "continue_running": True,
        "iteration_summaries": [],
        "history_entries": [],
        "last_successful_analysis": _empty_last_successful_analysis_value(),
    }


@workflow.task(save=False)
def _perform_tracking_iteration(
    state: dict[str, object],
    session: Session,
    qpu: QPU,
    qubits: QuantumElements,
    iteration_plan: dict[str, object],
    runtime: dict[str, object],
    tracking_plan: dict[str, object],
    experiment_options: TuneupExperimentOptions,
    t1_delays: QubitSweepPoints | None = None,
    t2_star_delays: QubitSweepPoints | None = None,
    t2_delays: QubitSweepPoints | None = None,
    t2_star_detunings: float | Sequence[float] | None = None,
    temporary_parameters: dict[str | tuple[str, str, str], dict | QuantumParameters]
    | None = None,
    *,
    history_path: str,
    refocus_qop: str,
    do_analysis: bool,
    update: bool,
    continue_on_iteration_error: bool,
) -> dict[str, object]:
    iteration_summaries = state["iteration_summaries"]
    history_entries = state["history_entries"]

    if not bool(state["continue_running"]):
        summary = _summarize_skipped_iteration_value(
            iteration_plan,
            tracking_plan,
            scheduled_start_utc=None,
            status="skipped_after_failure",
            skip_reason="previous_iteration_failed",
        )
        iteration_summaries.append(summary)
        return summary

    slot_bundle = _wait_for_iteration_slot_value(runtime, iteration_plan)
    if not bool(slot_bundle["run_iteration"]):
        summary = _summarize_skipped_iteration_value(
            iteration_plan,
            tracking_plan,
            scheduled_start_utc=slot_bundle["scheduled_start_utc"],
            status="skipped_deadline",
            skip_reason=str(slot_bundle["skip_reason"]),
        )
        iteration_summaries.append(summary)
        return summary

    temp_qpu = temporary_qpu.func(qpu, temporary_parameters)
    iteration_qubits = temporary_quantum_elements_from_qpu.func(temp_qpu, qubits)

    t1_bundle = _empty_execution_bundle_value("t1")
    if bool(tracking_plan["need_t1"]):
        t1_experiment = lifetime_measurement.create_experiment.func(
            temp_qpu,
            iteration_qubits,
            delays=t1_delays,
            options=experiment_options,
        )
        t1_bundle = _execute_experiment_safe_value(session, t1_experiment, "t1")

    t2_star_bundle = _empty_execution_bundle_value("t2_star")
    if bool(tracking_plan["need_t2_star"]):
        t2_star_experiment = ramsey.create_experiment.func(
            temp_qpu,
            iteration_qubits,
            delays=t2_star_delays,
            detunings=t2_star_detunings,
            options=experiment_options,
            echo=False,
            refocus_qop=refocus_qop,
        )
        t2_star_bundle = _execute_experiment_safe_value(
            session,
            t2_star_experiment,
            "t2_star",
        )

    t2_bundle = _empty_execution_bundle_value("t2")
    if bool(tracking_plan["need_t2"]):
        t2_experiment = ramsey.create_experiment.func(
            temp_qpu,
            iteration_qubits,
            delays=t2_delays,
            detunings=None,
            options=experiment_options,
            echo=True,
            refocus_qop=refocus_qop,
        )
        t2_bundle = _execute_experiment_safe_value(session, t2_experiment, "t2")

    summary = _summarize_iteration_execution_value(
        iteration_plan,
        slot_bundle,
        tracking_plan,
        t1_bundle,
        t2_star_bundle,
        t2_bundle,
    )
    if not bool(summary["successful"]):
        if not continue_on_iteration_error:
            state["continue_running"] = False
        iteration_summaries.append(summary)
        return summary

    if do_analysis:
        analysis_output = run_iteration_analysis_value(
            qubits=iteration_qubits,
            tracked_parameters=tracking_plan["tracked_parameters"],
            t1_result=t1_bundle["result"],
            t2_star_result=t2_star_bundle["result"],
            t2_result=t2_bundle["result"],
            t1_delays=t1_delays,
            t2_star_delays=t2_star_delays,
            t2_delays=t2_delays,
            t2_star_detunings=t2_star_detunings,
            need_t1=bool(tracking_plan["need_t1"]),
            need_t2_star=bool(tracking_plan["need_t2_star"]),
            need_t2=bool(tracking_plan["need_t2"]),
            history_path=history_path,
            timestamp_utc=slot_bundle["started_utc"],
        )
        history_entries.extend(list(analysis_output["history_entries"]))
        state["last_successful_analysis"] = dict(analysis_output)
        summary = _attach_iteration_analysis_value(summary, analysis_output)
        if update:
            update_qpu.func(qpu, analysis_output["new_parameter_values"])

    iteration_summaries.append(summary)
    return summary


@workflow.task(save=False)
def _finalize_tracking_analysis(
    state: dict[str, object],
    tracked_parameters: Sequence[str],
    history_path: str | None,
    do_analysis: bool,
) -> dict[str, object] | None:
    if not do_analysis:
        return None
    return _build_final_analysis_output_value(
        state["last_successful_analysis"],
        state["iteration_summaries"],
        state["history_entries"],
        tracked_parameters,
        history_path,
    )


@workflow.task(save=False)
def _finalize_tracking_output(
    state: dict[str, object],
    tracked_parameters: Sequence[str],
    history_path: str | None,
    do_analysis: bool,
) -> dict[str, object]:
    analysis_output = None
    if do_analysis:
        analysis_output = _build_final_analysis_output_value(
            state["last_successful_analysis"],
            state["iteration_summaries"],
            state["history_entries"],
            tracked_parameters,
            history_path,
        )
    return _build_workflow_output_value(
        analysis_output,
        state["iteration_summaries"],
        tracked_parameters,
    )


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
    """Run selected local coherence measurements in single-pass or long-running mode."""
    options = CoherenceTrackingWorkflowOptions() if options is None else options
    _validate_transition(options.transition)
    tracking_plan = _resolve_tracking_plan(
        options.tracked_parameters,
        t1_delays,
        t2_star_delays,
        t2_delays,
    )
    iteration_schedule = _resolve_iteration_schedule(
        options.total_duration_s,
        options.interval_s,
    )
    runtime = _start_tracking_runtime(options.total_duration_s)
    experiment_options = _clone_create_experiment_options()
    _clone_iteration_analysis_options()
    _capture_iteration_analysis_task_options()
    resolved_history_path = resolve_history_path(options.history_path)
    tracking_state = _initialize_tracking_state()

    with workflow.for_(
        iteration_schedule,
        lambda item: str(item["iteration_index"]),
    ) as iteration_plan:
        _perform_tracking_iteration(
            tracking_state,
            session,
            qpu,
            qubits,
            iteration_plan,
            runtime,
            tracking_plan,
            experiment_options,
            t1_delays=t1_delays,
            t2_star_delays=t2_star_delays,
            t2_delays=t2_delays,
            t2_star_detunings=t2_star_detunings,
            temporary_parameters=temporary_parameters,
            history_path=resolved_history_path,
            refocus_qop=options.refocus_qop,
            do_analysis=options.do_analysis,
            update=options.update,
            continue_on_iteration_error=options.continue_on_iteration_error,
        )

    with workflow.if_(options.do_analysis):
        with workflow.if_(options.do_plotting):
            history_bundle = load_tracking_history(resolved_history_path)
            plot_tracking_history(
                qubits,
                history_bundle["history_rows"],
                tracking_plan["tracked_parameters"],
            )

    workflow.return_(
        _finalize_tracking_output(
            tracking_state,
            tracking_plan["tracked_parameters"],
            resolved_history_path,
            options.do_analysis,
        )
    )

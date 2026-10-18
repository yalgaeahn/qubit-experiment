# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Local coherence-tracking workflow.

Runs local T1, Ramsey T2*, and echo T2 independently at scheduled start times.
Long runs persist raw results before analysis, diagnostics, events, and local trends.
Optional Ramsey feedback changes only a run-local QPU; input updates are forbidden
in that mode. Missed slots are skipped; operator interrupts propagate immediately.
"""

from __future__ import annotations

import math
import time
import traceback
from collections.abc import Sequence
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import numpy as np
from laboneq import workflow
from laboneq.workflow.logbook import LogbookStore
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

from qubit_experiment.analysis.coherence_tracking import (
    CoherenceTrackingAnalysisWorkflowOptions,
    _build_history_rows,
    _filter_merged_output_value,
    _merge_metric_outputs,
    analyze_monitoring_metric_value,
    load_tracking_history,
    plot_tracking_history,
    resolve_history_path,
)
from qubit_experiment.analysis.lifetime_measurement import FitDataT1Options
from qubit_experiment.workflow import coherence_monitoring as monitor

from . import lifetime_measurement, ramsey

if TYPE_CHECKING:
    from laboneq.dsl.quantum import QuantumParameters
    from laboneq.dsl.quantum.qpu import QPU
    from laboneq.dsl.session import Session
    from laboneq_applications.typing import QubitSweepPoints


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
            raise ValueError("tracked_parameters must contain only strings or be a single string.")
        raw_items.extend(part.strip() for part in candidate.split(",") if part.strip())

    if not raw_items:
        raise ValueError("tracked_parameters must contain at least one supported key.")

    canonical_by_lower = {key.lower(): key for key in _TRACKED_PARAMETER_TO_METRIC}
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
    need_t2_star = any(_TRACKED_PARAMETER_TO_METRIC[param] == "t2_star" for param in normalized)
    need_t2 = any(_TRACKED_PARAMETER_TO_METRIC[param] == "t2" for param in normalized)

    if need_t1 and t1_delays is None:
        raise ValueError("t1_delays must be provided when tracking ge_T1 or alias coherence/all.")
    if need_t2_star and t2_star_delays is None:
        raise ValueError(
            "t2_star_delays must be provided when tracking ge_T2_star or resonance_frequency_ge."
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
            f"coherence_tracking currently supports only transition='ge'. Got {resolved!r}."
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

    logstore: list[LogbookStore] | LogbookStore | None = workflow.option_field(
        factory=list,
        description="No automatic execution log folders by default; explicitly select stores to enable.",
        exclude=True,
        repr=False,
        converter=lambda value: [value] if isinstance(value, LogbookStore) else value,
    )

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
    total_duration_s: float | int | None = workflow.option_field(
        None,
        description=(
            "When provided, run long-running mode for this total monitoring window in "
            "seconds. The first iteration starts immediately."
        ),
    )
    interval_s: float | int = workflow.option_field(
        300.0,
        description="Target spacing in seconds between iteration start times.",
    )
    continue_on_iteration_error: bool = workflow.option_field(
        True,
        description="Whether to continue later scheduled iterations after an execution failure.",
    )
    track_frequency: bool = workflow.option_field(
        False,
        description="Track Ramsey frequency on a run-local QPU; never update the input QPU.",
    )
    monitoring_directory: str | None = workflow.option_field(
        None,
        description="New run directory for durable records; automatic for long-running mode.",
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
    if deadline_monotonic is not None and now_monotonic >= float(deadline_monotonic):
        return {
            "run_iteration": False,
            "skip_reason": "deadline_passed",
            "waited_s": 0.0,
            "scheduled_start_utc": None,
            "started_utc": None,
        }
    previous_finish = runtime.get("last_iteration_finished_monotonic")
    if (
        int(iteration_plan["iteration_index"]) > 0
        and previous_finish is not None
        and target_monotonic < float(previous_finish)
    ):
        return {
            "run_iteration": False,
            "skip_reason": "missed_slot",
            "waited_s": 0.0,
            "scheduled_start_utc": None,
            "started_utc": None,
        }
    if waited_s > 0:
        time.sleep(waited_s)
    started_monotonic = time.monotonic()

    started_utc_dt = datetime.now(timezone.utc)
    scheduled_start_utc = datetime.fromisoformat(
        str(runtime["run_started_utc"]).replace("Z", "+00:00")
    ) + timedelta(seconds=scheduled_offset_s)

    if deadline_monotonic is not None and started_monotonic >= float(deadline_monotonic):
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
    except Exception as exc:  # noqa: BLE001
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
    except Exception as exc:  # noqa: BLE001
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
        elif status in ("failed_execution", "partial_success"):
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
    return deepcopy(CoherenceTrackingAnalysisWorkflowOptions() if options is None else options)


@workflow.task(save=False)
def _capture_iteration_analysis_task_options(
    options: CoherenceTrackingIterationAnalysisTaskOptions | None = None,
) -> dict[str, bool]:
    opts = CoherenceTrackingIterationAnalysisTaskOptions() if options is None else options
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
        "run_id": uuid4().hex,
        "monitoring_directory": None,
        "metric_counts": {
            key: {"success": 0, "failure": 0, "quality_warning": 0}
            for key in ("t1", "t2_star", "t2")
        },
        "initial_frequencies": {},
        "frequencies": {},
        "frequency_history": [],
    }


@workflow.task(save=False)
def _perform_tracking_iteration(
    state: dict[str, object],
    session: Session,
    qpu: QPU,
    qubits: list[str] | str,
    iteration_plan: dict[str, object],
    runtime: dict[str, object],
    tracking_plan: dict[str, object],
    experiment_options: TuneupExperimentOptions,
    t1_delays: QubitSweepPoints | None = None,
    t2_star_delays: QubitSweepPoints | None = None,
    t2_delays: QubitSweepPoints | None = None,
    t2_star_detunings: float | Sequence[float] | None = None,
    temporary_parameters: dict[str | tuple[str, str, str], dict | QuantumParameters] | None = None,
    *,
    history_path: str,
    refocus_qop: str,
    do_analysis: bool,
    update: bool,
    continue_on_iteration_error: bool,
) -> dict[str, object]:
    try:
        return _perform_iteration_value(
            state,
            session,
            qpu,
            qubits,
            iteration_plan,
            runtime,
            tracking_plan,
            experiment_options,
            t1_delays,
            t2_star_delays,
            t2_delays,
            t2_star_detunings,
            temporary_parameters,
            history_path,
            refocus_qop,
            do_analysis,
            update,
            continue_on_iteration_error,
        )
    except (KeyboardInterrupt, SystemExit):
        monitor.record_terminal(
            state, "interrupted", iteration_index=iteration_plan["iteration_index"]
        )
        raise
    except monitor.MonitoringStorageError as exc:
        monitor.record_terminal(state, "storage_failed", error_message=str(exc))
        raise
    finally:
        runtime["last_iteration_finished_monotonic"] = time.monotonic()


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
    if analysis_output is not None:
        analysis_output["tracked_parameters"] = list(state["tracked_parameters"])
    output = _build_workflow_output_value(
        analysis_output,
        state["iteration_summaries"],
        state["tracked_parameters"],
    )
    output.update(
        {
            key: state[key]
            for key in ("monitoring_directory", "run_id", "metric_counts", "frequency_history")
        }
    )
    output["final_frequencies"] = dict(state["frequencies"])
    return output


@workflow.workflow(name="coherence_tracking")
def experiment_workflow(
    session: Session,
    qpu: QPU,
    qubits: list[str] | str,
    t1_delays: QubitSweepPoints | None = None,
    t2_star_delays: QubitSweepPoints | None = None,
    t2_delays: QubitSweepPoints | None = None,
    t2_star_detunings: float | Sequence[float] | None = None,
    temporary_parameters: dict[str | tuple[str, str, str], dict | QuantumParameters] | None = None,
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
    experiment_options = _clone_create_experiment_options()
    analysis_options = _clone_monitoring_analysis_options()
    _capture_iteration_analysis_task_options()
    resolved_history_path = resolve_history_path(options.history_path)
    tracking_state = _initialize_tracking_state()
    _prepare_monitoring(
        tracking_state,
        qpu,
        qubits,
        temporary_parameters,
        tracking_plan,
        experiment_options,
        analysis_options,
        options.track_frequency,
        options.monitoring_directory,
        options.total_duration_s,
        options.interval_s,
        resolved_history_path,
        t1_delays,
        t2_star_delays,
        t2_delays,
        t2_star_detunings,
        options.do_analysis,
        options.update,
        options.do_plotting,
        options.continue_on_iteration_error,
        options.refocus_qop,
    )
    runtime = _start_tracking_runtime(options.total_duration_s)

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
            _plot_final_history_safe(tracking_state, qubits, resolved_history_path)
    _complete_monitoring(tracking_state)

    workflow.return_(
        _finalize_tracking_output(
            tracking_state,
            tracking_plan["tracked_parameters"],
            resolved_history_path,
            options.do_analysis,
        )
    )


@workflow.task(save=False)
def _clone_monitoring_analysis_options(
    options: FitDataT1Options | None = None,
):
    return deepcopy(FitDataT1Options() if options is None else options)


def _qubit_list(qubits):
    return list(qubits) if isinstance(qubits, (tuple, list)) else [qubits]


@workflow.task(save=False)
def _prepare_monitoring(
    state,
    qpu,
    qubits,
    temporary_parameters,
    tracking_plan,
    experiment_options,
    analysis_options,
    track_frequency,
    monitoring_directory,
    total_duration_s,
    interval_s,
    history_path,
    t1_delays,
    t2_star_delays,
    t2_delays,
    detunings,
    do_analysis,
    update,
    do_plotting,
    continue_on_iteration_error,
    refocus_qop,
):
    from laboneq.simple import AcquisitionType, AveragingMode

    from qubit_experiment.analysis.ramsey import validate_and_convert_detunings

    _validate_transition_value(experiment_options.transition)
    if AcquisitionType(experiment_options.acquisition_type) not in (
        AcquisitionType.INTEGRATION,
        AcquisitionType.SPECTROSCOPY,
    ):
        raise ValueError("Coherence monitoring requires integration or spectroscopy acquisition.")
    if (
        experiment_options.use_cal_traces
        and AveragingMode(experiment_options.averaging_mode) == AveragingMode.SEQUENTIAL
    ):
        raise ValueError("SEQUENTIAL averaging cannot be used with calibration traces.")
    if track_frequency and (
        update
        or not do_analysis
        or not analysis_options.do_fitting
        or not tracking_plan["need_t2_star"]
    ):
        raise ValueError("track_frequency requires Ramsey analysis and update=False.")
    # The supported helper always copies quantum elements, including when overrides are None.
    runtime_qpu = temporary_qpu.func(qpu, temporary_parameters)
    runtime_qubits = _qubit_list(temporary_quantum_elements_from_qpu.func(runtime_qpu, qubits))
    ds = validate_and_convert_detunings(runtime_qubits, detunings)
    if track_frequency and any(not math.isfinite(float(d)) or float(d) == 0 for d in ds):
        raise ValueError("Frequency tracking requires finite, nonzero detunings for every qubit.")
    frequencies = {q.uid: float(q.parameters.resonance_frequency_ge) for q in runtime_qubits}
    if any(not math.isfinite(f) for f in frequencies.values()):
        raise ValueError("Initial resonance frequencies must be finite.")
    state.update(
        {
            "runtime_qpu": runtime_qpu,
            "runtime_qubits": runtime_qubits,
            "analysis_options": analysis_options,
            "track_frequency": track_frequency,
            "detunings": dict(zip(frequencies, ds)),
            "initial_frequencies": dict(frequencies),
            "frequencies": dict(frequencies),
            "do_plotting": do_plotting,
            "tracked_parameters": list(tracking_plan["tracked_parameters"]),
        }
    )
    if track_frequency and "resonance_frequency_ge" not in state["tracked_parameters"]:
        state["tracked_parameters"].append("resonance_frequency_ge")
    if monitoring_directory is not None or total_duration_s is not None:
        directory = (
            Path(monitoring_directory).expanduser().resolve()
            if monitoring_directory is not None
            else Path(history_path).parent / "monitoring" / state["run_id"]
        )
        try:
            directory.mkdir(parents=True, exist_ok=False)
        except Exception as exc:
            raise monitor.MonitoringStorageError(
                f"Monitoring directory must be new: {directory}"
            ) from exc
        state["monitoring_directory"] = str(directory)
        try:
            monitor.atomic_write(
                directory / "manifest.json",
                {
                    "run_id": state["run_id"],
                    "created_utc": monitor.utc_now(),
                    "versions": {
                        p: version(p)
                        for p in ("laboneq", "laboneq-applications", "qubit-experiment")
                    },
                    "input_qpu": {
                        q.uid: {
                            "parameters": q.parameters,
                            "signals": q.signals,
                            "type": f"{type(q).__module__}.{type(q).__qualname__}",
                        }
                        for q in qpu.quantum_elements
                    },
                    "runtime_qpu": {
                        q.uid: {
                            "parameters": q.parameters,
                            "signals": q.signals,
                            "type": f"{type(q).__module__}.{type(q).__qualname__}",
                        }
                        for q in runtime_qpu.quantum_elements
                    },
                    "settings": {
                        "tracked_parameters": state["tracked_parameters"],
                        "track_frequency": track_frequency,
                        "update": update,
                        "do_analysis": do_analysis,
                        "do_plotting": do_plotting,
                        "continue_on_iteration_error": continue_on_iteration_error,
                        "refocus_qop": refocus_qop,
                        "total_duration_s": total_duration_s,
                        "interval_s": interval_s,
                        "history_path": history_path,
                        "t1_delays": t1_delays,
                        "t2_star_delays": t2_star_delays,
                        "t2_delays": t2_delays,
                        "detunings": ds,
                        "experiment_options": experiment_options,
                        "analysis_options": analysis_options,
                    },
                },
            )
            monitor.event(state, "run_started")
            monitor.save_status(state, "running")
        except monitor.MonitoringStorageError as exc:
            monitor.record_terminal(state, "storage_failed", error_message=str(exc))
            raise


def _frequency_updates(state, metric_outputs, iteration_index):
    if not state["track_frequency"]:
        return
    ramsey_output = metric_outputs.get("t2_star", {})
    updates = {}
    for uid, current in state["frequencies"].items():
        estimate = (
            ramsey_output.get("new_parameter_values", {}).get(uid, {}).get("resonance_frequency_ge")
        )
        quality = ramsey_output.get("quality", {}).get(uid, {})
        value = getattr(estimate, "nominal_value", None)
        error = getattr(estimate, "std_dev", None)
        detuning = abs(state["detunings"][uid])
        reason = "accepted"
        if not quality.get("fit_success"):
            reason = "fit_unsuccessful"
        elif value is None or error is None or not math.isfinite(value) or not math.isfinite(error):
            reason = "frequency_uncertainty_missing"
        elif abs(value - current) > detuning / 2:
            reason = "frequency_step_exceeded"
        elif error > detuning / 4:
            reason = "frequency_uncertainty_exceeded"
        record = {
            "iteration_index": iteration_index,
            "qubit_uid": uid,
            "initial_frequency_hz": state["initial_frequencies"][uid],
            "drive_frequency_hz": current,
            "estimated_frequency_hz": value,
            "estimated_std_hz": error,
            "accepted": reason == "accepted",
            "reason": reason,
            "next_frequency_hz": value if reason == "accepted" else current,
        }
        record["cumulative_shift_hz"] = (
            record["next_frequency_hz"] - state["initial_frequencies"][uid]
        )
        monitor.event(state, "frequency_update", **record)
        state["frequency_history"].append(record)
        if reason == "accepted":
            updates[uid] = {"resonance_frequency_ge": estimate}
    if updates:
        update_qpu.func(state["runtime_qpu"], updates)
        for uid, values in updates.items():
            state["frequencies"][uid] = values["resonance_frequency_ge"].nominal_value


def _finish_iteration(state, summary):
    state["iteration_summaries"].append(summary)
    monitor.event(state, "iteration_finished", **summary)
    monitor.save_status(state, "running", last_iteration=summary)
    if state["do_plotting"]:
        try:
            monitor.render_trends(state, state["tracked_parameters"])
        except Exception as exc:
            monitor.event(
                state, "plot_failed", error_type=type(exc).__name__, error_message=str(exc)
            )
    return summary


def _perform_iteration_value(
    state,
    session,
    qpu,
    qubits,
    iteration_plan,
    runtime,
    tracking_plan,
    experiment_options,
    t1_delays,
    t2_star_delays,
    t2_delays,
    t2_star_detunings,
    temporary_parameters,
    history_path,
    refocus_qop,
    do_analysis,
    update,
    continue_on_iteration_error,
):
    index = iteration_plan["iteration_index"]
    if not state["continue_running"]:
        return _finish_iteration(
            state,
            _summarize_skipped_iteration_value(
                iteration_plan,
                tracking_plan,
                scheduled_start_utc=None,
                status="skipped_after_failure",
                skip_reason="previous_iteration_failed",
            ),
        )
    slot = _wait_for_iteration_slot_value(runtime, iteration_plan)
    if not slot["run_iteration"]:
        return _finish_iteration(
            state,
            _summarize_skipped_iteration_value(
                iteration_plan,
                tracking_plan,
                scheduled_start_utc=slot["scheduled_start_utc"],
                status="skipped_deadline"
                if slot["skip_reason"] == "deadline_passed"
                else "skipped_missed_slot",
                skip_reason=slot["skip_reason"],
            ),
        )
    monitor.event(state, "iteration_started", iteration_index=index, **slot)
    monitor.save_status(state, "running", iteration_index=index, stage="iteration_started")
    runtime_qpu = state["runtime_qpu"]
    iteration_qubits = state["runtime_qubits"]
    used_frequencies = dict(state["frequencies"])
    metric_outputs = {}
    iteration_rows = []
    per_metric = {}
    for metric, delays in (("t1", t1_delays), ("t2_star", t2_star_delays), ("t2", t2_delays)):
        if metric not in tracking_plan["required_metrics"]:
            continue
        deadline = runtime["deadline_monotonic"]
        if deadline is not None and time.monotonic() >= deadline:
            per_metric[metric] = {"status": "skipped_deadline"}
            monitor.event(state, "metric_skipped_deadline", iteration_index=index, metric=metric)
            continue
        prefix = (
            Path(state["monitoring_directory"]) / f"iteration_{index:06d}" / metric
            if state["monitoring_directory"]
            else None
        )
        record = {"status": "failed", "started_utc": monitor.utc_now()}
        stage = "create_experiment"
        monitor.event(
            state,
            "metric_started",
            iteration_index=index,
            metric=metric,
            drive_frequencies=used_frequencies,
        )
        monitor.save_status(state, "running", iteration_index=index, metric=metric, stage=stage)
        try:
            if len(iteration_qubits) == 1 and np.ndim(delays) == 1:
                delays = [delays]
            if metric == "t1":
                experiment = lifetime_measurement.create_experiment.func(
                    runtime_qpu, iteration_qubits, delays=delays, options=experiment_options
                )
            else:
                experiment = ramsey.create_experiment.func(
                    runtime_qpu,
                    iteration_qubits,
                    delays=delays,
                    detunings=t2_star_detunings if metric == "t2_star" else None,
                    options=experiment_options,
                    echo=metric == "t2",
                    refocus_qop=refocus_qop,
                )
            stage = "compile"
            _check_metric_deadline(deadline)
            compiled = session.compile(experiment=experiment)
            stage = "run"
            _check_metric_deadline(deadline)
            result = session.run(compiled)
            # Core writes are fatal and occur before any population/fit work.
            if prefix:
                monitor.atomic_write(
                    prefix / "context.json",
                    {
                        "run_id": state["run_id"],
                        "iteration_index": index,
                        "metric": metric,
                        "started_utc": record["started_utc"],
                        "drive_frequencies": used_frequencies,
                        "delays": delays,
                        "detunings": t2_star_detunings if metric == "t2_star" else None,
                        "experiment_options": experiment_options,
                        "analysis_options": state["analysis_options"],
                        "qubit_parameters": {q.uid: q.parameters for q in iteration_qubits},
                    },
                )
                monitor.atomic_write(prefix / "raw.json", result, laboneq=True)
                record["raw_path"] = str(prefix / "raw.json")
                monitor.event(
                    state,
                    "raw_saved",
                    iteration_index=index,
                    metric=metric,
                    path=record["raw_path"],
                )
            if getattr(result, "execution_errors", None):
                raise RuntimeError(
                    f"Acquisition reported execution errors: {result.execution_errors}"
                )
            stage = "analysis"
            if do_analysis:
                payload = analyze_monitoring_metric_value(
                    metric,
                    result,
                    iteration_qubits,
                    delays,
                    t2_star_detunings if metric == "t2_star" else None,
                    experiment_options,
                    state["analysis_options"],
                )
                metric_outputs[metric] = payload
                if prefix:
                    monitor.atomic_write(prefix / "analysis.json", payload)
                rows = _build_history_rows(
                    {metric: payload}, state["tracked_parameters"], record["started_utc"]
                )
                for row in rows:
                    quality = payload["quality"][row["qubit_uid"]]
                    row.update(
                        {
                            "run_id": state["run_id"],
                            "iteration_index": index,
                            "quality_reasons": quality["reasons"],
                            "fit_success": quality["fit_success"],
                            "drive_frequency_hz": used_frequencies[row["qubit_uid"]],
                        }
                    )
                monitor.append_records(history_path, rows)
                state["history_entries"].extend(rows)
                iteration_rows.extend(rows)
                qualities = list(payload["quality"].values())
                failures = sum(q["failed"] for q in qualities)
                warnings = sum(bool(q["reasons"]) for q in qualities)
                state["metric_counts"][metric]["quality_warning"] += warnings
                record.update(
                    {
                        "quality": payload["quality"],
                        "status": "failed_analysis"
                        if failures == len(qualities)
                        else "partial_success"
                        if failures
                        else "completed",
                    }
                )
            else:
                record["status"] = "completed"
        except monitor.MonitoringStorageError:
            raise
        except _MetricDeadlineReached:
            record["status"] = "skipped_deadline"
        except Exception as exc:
            record.update(
                {
                    "error_stage": stage,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
        record["finished_utc"] = monitor.utc_now()
        per_metric[metric] = record
        counts = state["metric_counts"][metric]
        if record["status"] != "skipped_deadline":
            counts["success" if record["status"] == "completed" else "failure"] += 1
        monitor.event(state, "metric_finished", iteration_index=index, metric=metric, **record)

    empty = {"old_parameter_values": {}, "new_parameter_values": {}}
    merged = _merge_metric_outputs.func(
        *(metric_outputs.get(m, empty) for m in ("t1", "t2_star", "t2"))
    )
    selected = state["tracked_parameters"]
    if do_analysis:
        rows = iteration_rows
        analysis = _filter_merged_output_value(merged, selected)
        analysis.update(
            {"history_entries": rows, "history_path": str(history_path), "history_rows": rows}
        )
        if metric_outputs:
            state["last_successful_analysis"] = analysis
        stage = "frequency_update"
        try:
            _frequency_updates(state, metric_outputs, index)
            if update:
                valid_updates = {}
                for metric, payload in metric_outputs.items():
                    for uid, values in payload["new_parameter_values"].items():
                        if payload["quality"][uid]["valid"]:
                            valid_updates.setdefault(uid, {}).update(
                                {k: v for k, v in values.items() if k in selected}
                            )
                update_qpu.func(qpu, valid_updates)
                update_qpu.func(runtime_qpu, valid_updates)
                state["frequencies"] = {
                    q.uid: float(q.parameters.resonance_frequency_ge) for q in iteration_qubits
                }
        except monitor.MonitoringStorageError:
            raise
        except Exception as exc:
            per_metric[stage] = {"status": "failed", "error_message": str(exc)}
            monitor.event(state, "update_failed", iteration_index=index, error_message=str(exc))
    successes = sum(r["status"] in ("completed", "partial_success") for r in per_metric.values())
    failed = any(r["status"] not in ("completed", "skipped_deadline") for r in per_metric.values())
    incomplete = any(r["status"] != "completed" for r in per_metric.values())
    status = (
        "partial_success"
        if incomplete and successes
        else "failed_execution"
        if failed
        else "completed"
        if successes
        else "skipped_deadline"
    )
    summary = {
        "iteration_index": index,
        "scheduled_offset_s": iteration_plan["scheduled_offset_s"],
        "scheduled_start_utc": slot["scheduled_start_utc"],
        "started_utc": slot["started_utc"],
        "status": status,
        "successful": status == "completed",
        "per_metric_status": per_metric,
        "tracked_parameters": selected,
        "required_metrics": tracking_plan["required_metrics"],
        "waited_s": slot["waited_s"],
        "drive_frequencies": used_frequencies,
    }
    if failed and not continue_on_iteration_error:
        state["continue_running"] = False
    return _finish_iteration(state, summary)


@workflow.task(save=False)
def _complete_monitoring(state):
    status = "completed" if state["continue_running"] else "stopped_after_failure"
    try:
        monitor.event(state, "run_finished", status=status)
        monitor.save_status(state, status)
    except monitor.MonitoringStorageError as exc:
        monitor.record_terminal(state, "storage_failed", error_message=str(exc))
        raise


@workflow.task(save=False)
def _plot_final_history_safe(state, qubits, history_path):
    try:
        try:
            bundle = load_tracking_history.func(history_path)
            plot_tracking_history.func(qubits, bundle["history_rows"], state["tracked_parameters"])
        except Exception as exc:
            monitor.event(
                state, "plot_failed", error_type=type(exc).__name__, error_message=str(exc)
            )
    except (KeyboardInterrupt, SystemExit):
        monitor.record_terminal(state, "interrupted")
        raise
    except monitor.MonitoringStorageError as exc:
        monitor.record_terminal(state, "storage_failed", error_message=str(exc))
        raise


class _MetricDeadlineReached(Exception):
    """A compile or acquisition was not started because the monitoring window ended."""


def _check_metric_deadline(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise _MetricDeadlineReached()

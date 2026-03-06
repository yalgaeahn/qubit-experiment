# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Coherence-tracking analysis for local T1, T2, and T2* runs.

Aggregates per-iteration local analysis outputs into selected tracked parameters,
persists JSONL history after each successful iteration, and renders per-qubit trend
artifacts from accumulated history. Supports long-running workflow ownership while
keeping local lifetime, Ramsey, and echo analysis contracts unchanged.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from analysis.echo import analysis_workflow as echo_analysis_workflow
from analysis.lifetime_measurement import analysis_workflow as lifetime_analysis_workflow
from analysis.plot_theme import with_plot_theme
from analysis.plotting_helpers import timestamped_title
from analysis.ramsey import analysis_workflow as ramsey_analysis_workflow
from laboneq import workflow
from laboneq_applications.analysis.calibration_traces_rotation import (
    calculate_qubit_population,
)
from laboneq_applications.analysis.options import (
    BasePlottingOptions,
    TuneUpAnalysisWorkflowOptions,
)
from laboneq_applications.core.validation import validate_and_convert_qubits_sweeps

from analysis import echo as echo_analysis_module
from analysis import lifetime_measurement as lifetime_analysis_module
from analysis import ramsey as ramsey_analysis_module

if TYPE_CHECKING:
    import matplotlib as mpl
    from laboneq.workflow.tasks.run_experiment import RunExperimentResults
    from laboneq_applications.typing import QuantumElements, QubitSweepPoints


_TRACKED_PARAMETER_METADATA: dict[str, dict[str, object]] = {
    "ge_T1": {
        "metric_id": "t1",
        "label": "T1",
        "unit": "s",
        "display_scale": 1e6,
        "display_unit": "us",
    },
    "ge_T2_star": {
        "metric_id": "t2_star",
        "label": "T2*",
        "unit": "s",
        "display_scale": 1e6,
        "display_unit": "us",
    },
    "ge_T2": {
        "metric_id": "t2",
        "label": "T2",
        "unit": "s",
        "display_scale": 1e6,
        "display_unit": "us",
    },
    "resonance_frequency_ge": {
        "metric_id": "t2_star",
        "label": "f_ge",
        "unit": "Hz",
        "display_scale": 1e-9,
        "display_unit": "GHz",
    },
}
_OLD_METRIC_TO_PARAMETER_KEY = {
    "t1": "ge_T1",
    "t2": "ge_T2",
    "t2_star": "ge_T2_star",
}
_TRACKING_HISTORY_VERSION = 2


def _tracked_parameters_tuple(tracked_parameters: Sequence[str] | str) -> tuple[str, ...]:
    if isinstance(tracked_parameters, str):
        return (tracked_parameters,)
    return tuple(str(param) for param in tracked_parameters)


def _active_folder_store_root() -> Path | None:
    for store in workflow.logbook.active_logbook_stores():
        folder = getattr(store, "_folder", None)
        if folder is None:
            continue
        path = Path(folder)
        if path.exists():
            return path
    return None


def _empty_analysis_payload_value() -> dict[str, dict[str, dict[str, object]]]:
    return {
        "old_parameter_values": {},
        "new_parameter_values": {},
    }


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    if hasattr(value, "nominal_value"):
        try:
            return float(value.nominal_value)
        except (TypeError, ValueError):
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_uncertainty(value: object) -> tuple[float | None, float | None]:
    if value is None:
        return (None, None)
    if hasattr(value, "nominal_value"):
        nominal = _safe_float(value)
        std_dev = getattr(value, "std_dev", None)
        return nominal, _safe_float(std_dev)
    return _safe_float(value), None


def _resolve_history_path_value(history_path: str | Path) -> Path:
    path = Path(history_path).expanduser()
    if not str(path):
        raise ValueError("history_path must not be empty.")
    if path.is_absolute():
        return path.resolve()

    folder_store_root = _active_folder_store_root()
    if folder_store_root is not None:
        path = folder_store_root / path
    else:
        path = Path.cwd() / path
    return path.resolve()


def _sort_history_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(
        rows,
        key=lambda row: (
            str(row.get("timestamp_utc", "")),
            str(row.get("qubit_uid", "")),
            str(row.get("parameter_key", row.get("metric", ""))),
        ),
    )


def _load_history_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []

    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                rows.append(record)
    return rows


def _append_history_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def _parameter_value_from_row(
    row: dict[str, object],
    parameter_key: str,
) -> tuple[float | None, float | None]:
    row_parameter_key = row.get("parameter_key")
    if row_parameter_key == parameter_key:
        value = _safe_float(row.get("value"))
        std_dev = _safe_float(row.get("std_dev"))
        if value is None:
            unit = _TRACKED_PARAMETER_METADATA[parameter_key]["unit"]
            if unit == "s":
                value = _safe_float(row.get("value_s"))
                std_dev = _safe_float(row.get("std_dev_s"))
            elif unit == "Hz":
                value = _safe_float(row.get("value_hz"))
                std_dev = _safe_float(row.get("std_dev_hz"))
        return value, std_dev

    if parameter_key == "resonance_frequency_ge":
        return (
            _safe_float(row.get("resonance_frequency_ge_hz")),
            _safe_float(row.get("resonance_frequency_ge_std_hz")),
        )

    legacy_key = _OLD_METRIC_TO_PARAMETER_KEY.get(str(row.get("metric", "")))
    if legacy_key != parameter_key:
        return (None, None)
    return (
        _safe_float(row.get("value_s", row.get("value"))),
        _safe_float(row.get("std_dev_s", row.get("std_dev"))),
    )


def _parse_timestamp_utc(timestamp_utc: object) -> datetime | None:
    if not isinstance(timestamp_utc, str) or not timestamp_utc:
        return None
    try:
        return datetime.fromisoformat(timestamp_utc.replace("Z", "+00:00"))
    except ValueError:
        return None


def _history_points_for_parameter(
    history_rows: list[dict[str, object]],
    parameter_key: str,
) -> list[dict[str, object]]:
    points: list[dict[str, object]] = []
    for row in _sort_history_rows(list(history_rows)):
        timestamp = _parse_timestamp_utc(row.get("timestamp_utc"))
        if timestamp is None:
            continue
        value, std_dev = _parameter_value_from_row(row, parameter_key)
        if value is None:
            continue
        points.append(
            {
                "timestamp": timestamp,
                "value": value,
                "std_dev": std_dev,
            }
        )
    return points


def _filter_merged_output_value(
    merged_output: dict[str, object],
    tracked_parameters: Sequence[str] | str,
) -> dict[str, object]:
    selected = _tracked_parameters_tuple(tracked_parameters)
    old_parameter_values: dict[str, dict[str, object]] = {}
    new_parameter_values: dict[str, dict[str, object]] = {}

    metric_outputs = merged_output.get("metric_outputs", {})
    if not isinstance(metric_outputs, dict):
        metric_outputs = {}

    merged_old = merged_output.get("old_parameter_values", {})
    if isinstance(merged_old, dict):
        for qubit_uid, values in merged_old.items():
            if not isinstance(values, dict):
                continue
            selected_values = {
                key: value for key, value in values.items() if key in selected
            }
            if selected_values:
                old_parameter_values[str(qubit_uid)] = selected_values

    merged_new = merged_output.get("new_parameter_values", {})
    if isinstance(merged_new, dict):
        for qubit_uid, values in merged_new.items():
            if not isinstance(values, dict):
                continue
            selected_values = {
                key: value for key, value in values.items() if key in selected
            }
            if selected_values:
                new_parameter_values[str(qubit_uid)] = selected_values

    return {
        "metric_outputs": metric_outputs,
        "old_parameter_values": old_parameter_values,
        "new_parameter_values": new_parameter_values,
        "tracked_parameters": list(selected),
    }


def _analyze_t1_metric_value(
    result: object,
    qubits: QuantumElements,
    delays: QubitSweepPoints,
) -> dict[str, object]:
    processed_data = calculate_qubit_population(qubits, result, delays)
    fit_results = lifetime_analysis_module.fit_data.func(qubits, processed_data)
    return lifetime_analysis_module.extract_qubit_parameters.func(qubits, fit_results)


def _analyze_t2_star_metric_value(
    result: object,
    qubits: QuantumElements,
    delays: QubitSweepPoints,
    detunings: float | Sequence[float] | None,
) -> dict[str, object]:
    processed_data = calculate_qubit_population(qubits, result, delays)
    fit_results = ramsey_analysis_module.fit_data.func(qubits, processed_data)
    return ramsey_analysis_module.extract_qubit_parameters.func(
        qubits,
        fit_results,
        detunings,
    )


def _analyze_t2_metric_value(
    result: object,
    qubits: QuantumElements,
    delays: QubitSweepPoints,
) -> dict[str, object]:
    processed_data = calculate_qubit_population(qubits, result, delays)
    fit_results = echo_analysis_module.fit_data.func(qubits, processed_data)
    return echo_analysis_module.extract_qubit_parameters.func(qubits, fit_results)


def run_iteration_analysis_value(
    qubits: QuantumElements,
    tracked_parameters: Sequence[str] | str,
    t1_result: object | None = None,
    t2_star_result: object | None = None,
    t2_result: object | None = None,
    t1_delays: QubitSweepPoints | None = None,
    t2_star_delays: QubitSweepPoints | None = None,
    t2_delays: QubitSweepPoints | None = None,
    t2_star_detunings: float | Sequence[float] | None = None,
    *,
    need_t1: bool = False,
    need_t2_star: bool = False,
    need_t2: bool = False,
    history_path: str | Path = "tracking/coherence_tracking.jsonl",
    timestamp_utc: str | None = None,
) -> dict[str, object]:
    t1_output = _empty_analysis_payload_value()
    t2_star_output = _empty_analysis_payload_value()
    t2_output = _empty_analysis_payload_value()

    if need_t1 and t1_result is not None and t1_delays is not None:
        t1_output = _analyze_t1_metric_value(t1_result, qubits, t1_delays)
    if need_t2_star and t2_star_result is not None and t2_star_delays is not None:
        t2_star_output = _analyze_t2_star_metric_value(
            t2_star_result,
            qubits,
            t2_star_delays,
            t2_star_detunings,
        )
    if need_t2 and t2_result is not None and t2_delays is not None:
        t2_output = _analyze_t2_metric_value(t2_result, qubits, t2_delays)

    merged_output = _merge_metric_outputs(t1_output, t2_star_output, t2_output)
    filtered_output = _filter_merged_output_value(merged_output, tracked_parameters)
    resolved_path = _resolve_history_path_value(history_path)
    existing_rows = _sort_history_rows(_load_history_rows(resolved_path))
    new_rows = _build_history_rows(
        merged_output["metric_outputs"],
        tracked_parameters,
        timestamp_utc,
    )
    _append_history_rows(resolved_path, new_rows)
    filtered_output["history_path"] = str(resolved_path)
    filtered_output["history_entries"] = new_rows
    filtered_output["history_rows"] = _sort_history_rows([*existing_rows, *new_rows])
    return filtered_output


def _build_history_rows(
    metric_outputs: dict[str, dict[str, object]],
    tracked_parameters: Sequence[str] | str,
    timestamp_utc: str | None = None,
) -> list[dict[str, object]]:
    timestamp = timestamp_utc or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    selected = _tracked_parameters_tuple(tracked_parameters)
    rows: list[dict[str, object]] = []

    for parameter_key in selected:
        metadata = _TRACKED_PARAMETER_METADATA.get(parameter_key)
        if metadata is None:
            continue
        metric_id = str(metadata["metric_id"])
        metric_output = metric_outputs.get(metric_id) or {}
        old_values = metric_output.get("old_parameter_values", {})
        new_values = metric_output.get("new_parameter_values", {})
        if not isinstance(old_values, dict) or not isinstance(new_values, dict):
            continue

        for qubit_uid, qubit_updates in new_values.items():
            if not isinstance(qubit_updates, dict) or parameter_key not in qubit_updates:
                continue

            value, std_dev = _safe_uncertainty(qubit_updates.get(parameter_key))
            if value is None:
                continue

            previous_value = None
            old_qubit_values = old_values.get(qubit_uid, {})
            if isinstance(old_qubit_values, dict):
                previous_value = _safe_float(old_qubit_values.get(parameter_key))

            row: dict[str, object] = {
                "timestamp_utc": timestamp,
                "qubit_uid": str(qubit_uid),
                "parameter_key": parameter_key,
                "source_metric": metric_id,
                "value": value,
                "std_dev": std_dev,
                "previous_value": previous_value,
                "unit": str(metadata["unit"]),
                "history_version": _TRACKING_HISTORY_VERSION,
            }
            if metadata["unit"] == "s":
                row["value_s"] = value
                row["std_dev_s"] = std_dev
                row["previous_value_s"] = previous_value
            elif metadata["unit"] == "Hz":
                row["value_hz"] = value
                row["std_dev_hz"] = std_dev
                row["previous_value_hz"] = previous_value

            if parameter_key == "ge_T2_star" and "resonance_frequency_ge" in selected:
                freq_hz, freq_std_hz = _safe_uncertainty(
                    qubit_updates.get("resonance_frequency_ge")
                )
                if freq_hz is not None:
                    row["resonance_frequency_ge_hz"] = freq_hz
                if freq_std_hz is not None:
                    row["resonance_frequency_ge_std_hz"] = freq_std_hz

            rows.append(row)

    return rows


def _plot_suppressed_analysis_options_value(workflow_builder) -> Any:
    opts = workflow_builder.options()
    if hasattr(opts, "base"):
        opts = opts.base
    opts.do_plotting = False
    return opts


@workflow.task(save=False)
def _suppressed_t1_analysis_options() -> object:
    return _plot_suppressed_analysis_options_value(lifetime_analysis_workflow)


@workflow.task(save=False)
def _suppressed_t2_star_analysis_options() -> object:
    return _plot_suppressed_analysis_options_value(ramsey_analysis_workflow)


@workflow.task(save=False)
def _suppressed_t2_analysis_options() -> object:
    return _plot_suppressed_analysis_options_value(echo_analysis_workflow)


@workflow.task(save=False)
def _select_metric_analysis_options(
    render_metric_plots: bool,
    suppressed_options: object,
) -> object | None:
    return None if render_metric_plots else suppressed_options


@workflow.workflow_options(base_class=TuneUpAnalysisWorkflowOptions)
class CoherenceTrackingAnalysisWorkflowOptions:
    """Options for per-iteration coherence-tracking analysis."""


@workflow.task_options(base_class=BasePlottingOptions)
class PlotCoherenceTrackingOptions:
    """Options for coherence-tracking trend plots."""


@workflow.task(save=False)
def _empty_analysis_payload():
    return _empty_analysis_payload_value()


@workflow.task(save=False)
def _materialize_analysis_payload(payload: dict[str, object]) -> dict[str, object]:
    """Resolve a branch-selected analysis payload to a stable task output."""

    return dict(payload)


@workflow.task(save=False)
def _materialize_metric_input(value: object) -> object:
    """Resolve parent workflow inputs to stable task outputs before child analysis calls."""

    return value


@workflow.task(save=False)
def _merge_metric_outputs(
    t1_output: dict[str, object],
    t2_star_output: dict[str, object],
    t2_output: dict[str, object],
) -> dict[str, object]:
    metric_outputs = {
        "t1": dict(t1_output),
        "t2_star": dict(t2_star_output),
        "t2": dict(t2_output),
    }
    old_parameter_values: dict[str, dict[str, object]] = {}
    new_parameter_values: dict[str, dict[str, object]] = {}

    for metric_output in metric_outputs.values():
        metric_old = metric_output.get("old_parameter_values", {})
        metric_new = metric_output.get("new_parameter_values", {})
        if isinstance(metric_old, dict):
            for qubit_uid, values in metric_old.items():
                if isinstance(values, dict):
                    old_parameter_values.setdefault(str(qubit_uid), {}).update(dict(values))
        if isinstance(metric_new, dict):
            for qubit_uid, values in metric_new.items():
                if isinstance(values, dict):
                    new_parameter_values.setdefault(str(qubit_uid), {}).update(dict(values))

    return {
        "metric_outputs": metric_outputs,
        "old_parameter_values": old_parameter_values,
        "new_parameter_values": new_parameter_values,
    }


@workflow.task(save=False)
def _filter_merged_output(
    merged_output: dict[str, object],
    tracked_parameters: Sequence[str] | str,
) -> dict[str, object]:
    return _filter_merged_output_value(merged_output, tracked_parameters)


@workflow.task
def resolve_history_path(history_path: str | Path) -> str:
    return str(_resolve_history_path_value(history_path))


@workflow.task
def load_tracking_history(history_path: str | Path) -> dict[str, object]:
    resolved_path = _resolve_history_path_value(history_path)
    rows = _sort_history_rows(_load_history_rows(resolved_path))
    return {
        "history_path": str(resolved_path),
        "history_rows": rows,
    }


@workflow.task
def materialize_tracking_history(
    metric_outputs: dict[str, dict[str, object]],
    tracked_parameters: Sequence[str] | str,
    history_path: str | Path,
    timestamp_utc: str | None = None,
) -> dict[str, object]:
    """Load existing history, append current rows, and return combined history."""
    resolved_path = _resolve_history_path_value(history_path)
    existing_rows = _sort_history_rows(_load_history_rows(resolved_path))
    new_rows = _build_history_rows(metric_outputs, tracked_parameters, timestamp_utc)
    _append_history_rows(resolved_path, new_rows)
    combined_rows = _sort_history_rows([*existing_rows, *new_rows])
    return {
        "history_path": str(resolved_path),
        "history_entries": new_rows,
        "history_rows": combined_rows,
    }


@workflow.task
@with_plot_theme
def plot_tracking_history(
    qubits: QuantumElements,
    history_rows: list[dict[str, object]],
    tracked_parameters: Sequence[str] | str,
    options: PlotCoherenceTrackingOptions | None = None,
) -> dict[str, mpl.figure.Figure]:
    """Create per-qubit trend figures from JSONL tracking history."""
    opts = PlotCoherenceTrackingOptions() if options is None else options
    qubits = validate_and_convert_qubits_sweeps(qubits)
    selected = _tracked_parameters_tuple(tracked_parameters)
    figures: dict[str, mpl.figure.Figure] = {}
    if not selected:
        return figures

    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)

    for q in qubits:
        q_rows = [row for row in history_rows if row.get("qubit_uid") == q.uid]
        if not q_rows:
            continue

        fig_height = max(3.2, 2.35 * len(selected) + 1.1)
        fig, axes = plt.subplots(len(selected), 1, figsize=(9.2, fig_height), sharex=True)
        axes_array = np.atleast_1d(axes)
        fig.suptitle(timestamped_title(f"Coherence tracking {q.uid}"))

        for axis, parameter_key in zip(axes_array, selected):
            metadata = _TRACKED_PARAMETER_METADATA[parameter_key]
            points = _history_points_for_parameter(q_rows, parameter_key)
            axis.grid(alpha=0.22)
            axis.set_ylabel(
                f"{metadata['label']} ({metadata['display_unit']})"
            )
            if not points:
                axis.text(
                    0.5,
                    0.5,
                    "no data",
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                    alpha=0.7,
                )
                continue

            x_values = [point["timestamp"] for point in points]
            y_values = np.asarray(
                [float(point["value"]) for point in points],
                dtype=float,
            ) * float(metadata["display_scale"])
            std_values = np.asarray(
                [
                    np.nan
                    if point["std_dev"] is None
                    else float(point["std_dev"])
                    for point in points
                ],
                dtype=float,
            ) * float(metadata["display_scale"])

            if np.isfinite(std_values).any():
                axis.errorbar(
                    x_values,
                    y_values,
                    yerr=std_values,
                    fmt="o-",
                    capsize=3,
                    linewidth=1.4,
                )
            else:
                axis.plot(x_values, y_values, "o-", linewidth=1.4)
            axis.xaxis.set_major_locator(locator)
            axis.xaxis.set_major_formatter(formatter)

        axes_array[-1].set_xlabel("Time (UTC)")
        fig.tight_layout()

        if opts.save_figures:
            workflow.save_artifact(f"Coherence_tracking_{q.uid}", fig)

        if opts.close_figures:
            plt.close(fig)

        figures[q.uid] = fig

    return figures


@workflow.task(save=False)
def _attach_history_bundle(
    merged_output: dict[str, object],
    history_bundle: dict[str, object],
) -> dict[str, object]:
    output = dict(merged_output)
    output["history_path"] = history_bundle.get("history_path")
    output["history_entries"] = list(history_bundle.get("history_entries", []))
    output["history_rows"] = list(history_bundle.get("history_rows", []))
    return output


@workflow.workflow
def analysis_workflow(
    qubits: QuantumElements,
    tracked_parameters: Sequence[str] | str,
    t1_result: RunExperimentResults | None = None,
    t2_star_result: RunExperimentResults | None = None,
    t2_result: RunExperimentResults | None = None,
    t1_delays: QubitSweepPoints | None = None,
    t2_star_delays: QubitSweepPoints | None = None,
    t2_delays: QubitSweepPoints | None = None,
    t2_star_detunings: float | Sequence[float] | None = None,
    *,
    need_t1: bool = False,
    need_t2_star: bool = False,
    need_t2: bool = False,
    render_metric_plots: bool = True,
    render_history_plot: bool = True,
    history_path: str | Path = "tracking/coherence_tracking.jsonl",
    timestamp_utc: str | None = None,
    options: CoherenceTrackingAnalysisWorkflowOptions | None = None,
) -> None:
    """Aggregate selected per-iteration T1/T2/T2* analyses and persist history."""
    suppressed_t1_options = _suppressed_t1_analysis_options()
    suppressed_t2_star_options = _suppressed_t2_star_analysis_options()
    suppressed_t2_options = _suppressed_t2_analysis_options()
    t1_analysis_options = _select_metric_analysis_options(
        render_metric_plots,
        suppressed_t1_options,
    )
    t2_star_analysis_options = _select_metric_analysis_options(
        render_metric_plots,
        suppressed_t2_star_options,
    )
    t2_analysis_options = _select_metric_analysis_options(
        render_metric_plots,
        suppressed_t2_options,
    )
    materialized_t1_result = _materialize_metric_input(t1_result)
    materialized_t2_star_result = _materialize_metric_input(t2_star_result)
    materialized_t2_result = _materialize_metric_input(t2_result)

    t1_output = _empty_analysis_payload()
    t2_star_output = _empty_analysis_payload()
    t2_output = _empty_analysis_payload()

    with workflow.if_(need_t1):
        t1_output = _materialize_analysis_payload(
            lifetime_analysis_workflow(
                materialized_t1_result,
                qubits,
                t1_delays,
                options=t1_analysis_options,
            ).output
        )
    with workflow.if_(need_t2_star):
        t2_star_output = _materialize_analysis_payload(
            ramsey_analysis_workflow(
                materialized_t2_star_result,
                qubits,
                t2_star_delays,
                t2_star_detunings,
                options=t2_star_analysis_options,
            ).output
        )
    with workflow.if_(need_t2):
        t2_output = _materialize_analysis_payload(
            echo_analysis_workflow(
                materialized_t2_result,
                qubits,
                t2_delays,
                options=t2_analysis_options,
            ).output
        )

    materialized_t1_output = _materialize_analysis_payload(t1_output)
    materialized_t2_star_output = _materialize_analysis_payload(t2_star_output)
    materialized_t2_output = _materialize_analysis_payload(t2_output)

    merged_output = _merge_metric_outputs(
        materialized_t1_output,
        materialized_t2_star_output,
        materialized_t2_output,
    )
    filtered_output = _filter_merged_output(merged_output, tracked_parameters)
    history_bundle = materialize_tracking_history(
        merged_output["metric_outputs"],
        tracked_parameters,
        history_path,
        timestamp_utc,
    )

    with workflow.if_(render_history_plot):
        plot_tracking_history(
            qubits,
            history_bundle["history_rows"],
            tracked_parameters,
        )

    workflow.return_(_attach_history_bundle(filtered_output, history_bundle))

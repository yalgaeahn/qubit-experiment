# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Coherence-tracking analysis for local T1, T2, and T2* runs.

Aggregates per-metric local analysis outputs, persists JSONL history,
and saves per-qubit trend artifacts for `ge_T1`, `ge_T2`, and `ge_T2_star`.
Keeps Ramsey frequency-update payloads intact when T2* is enabled.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from analysis.echo import analysis_workflow as echo_analysis_workflow
from analysis.lifetime_measurement import analysis_workflow as lifetime_analysis_workflow
from analysis.plot_theme import with_plot_theme
from analysis.plotting_helpers import timestamped_title
from analysis.ramsey import analysis_workflow as ramsey_analysis_workflow
from laboneq import workflow
from laboneq_applications.analysis.options import (
    BasePlottingOptions,
    TuneUpAnalysisWorkflowOptions,
)
from laboneq_applications.core.validation import validate_and_convert_qubits_sweeps

if TYPE_CHECKING:
    import matplotlib as mpl
    from laboneq.workflow.tasks.run_experiment import RunExperimentResults
    from laboneq_applications.typing import QuantumElements, QubitSweepPoints


_TRACKED_PARAM_KEYS = {
    "t1": "ge_T1",
    "t2": "ge_T2",
    "t2_star": "ge_T2_star",
}
_RAMSEY_FREQUENCY_KEY = "resonance_frequency_ge"
_TRACKING_HISTORY_VERSION = 1


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
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def _sort_history_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(
        rows,
        key=lambda row: (
            str(row.get("timestamp_utc", "")),
            str(row.get("qubit_uid", "")),
            str(row.get("metric", "")),
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


def _build_history_rows(
    metric_outputs: dict[str, dict[str, object]],
    timestamp_utc: str | None = None,
) -> list[dict[str, object]]:
    timestamp = timestamp_utc or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows: list[dict[str, object]] = []
    for metric, param_key in _TRACKED_PARAM_KEYS.items():
        metric_output = metric_outputs.get(metric) or {}
        old_values = metric_output.get("old_parameter_values", {})
        new_values = metric_output.get("new_parameter_values", {})
        if not isinstance(old_values, dict) or not isinstance(new_values, dict):
            continue

        for qubit_uid, qubit_updates in new_values.items():
            if not isinstance(qubit_updates, dict) or param_key not in qubit_updates:
                continue

            value_s, std_dev_s = _safe_uncertainty(qubit_updates.get(param_key))
            if value_s is None:
                continue

            previous_value_s = None
            old_qubit_values = old_values.get(qubit_uid, {})
            if isinstance(old_qubit_values, dict):
                previous_value_s = _safe_float(old_qubit_values.get(param_key))

            row: dict[str, object] = {
                "timestamp_utc": timestamp,
                "qubit_uid": str(qubit_uid),
                "metric": metric,
                "value_s": value_s,
                "std_dev_s": std_dev_s,
                "previous_value_s": previous_value_s,
                "history_version": _TRACKING_HISTORY_VERSION,
            }
            if metric == "t2_star":
                freq_hz, freq_std_hz = _safe_uncertainty(
                    qubit_updates.get(_RAMSEY_FREQUENCY_KEY)
                )
                if freq_hz is not None:
                    row["resonance_frequency_ge_hz"] = freq_hz
                if freq_std_hz is not None:
                    row["resonance_frequency_ge_std_hz"] = freq_std_hz
            rows.append(row)
    return rows


@workflow.workflow_options(base_class=TuneUpAnalysisWorkflowOptions)
class CoherenceTrackingAnalysisWorkflowOptions:
    """Options for coherence-tracking aggregation and trend plotting."""


@workflow.task_options(base_class=BasePlottingOptions)
class PlotCoherenceTrackingOptions:
    """Options for coherence-tracking trend plots."""


@workflow.task(save=False)
def _empty_analysis_payload():
    return _empty_analysis_payload_value()


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


@workflow.task
def materialize_tracking_history(
    metric_outputs: dict[str, dict[str, object]],
    history_path: str | Path,
) -> dict[str, object]:
    """Load existing history, append current rows, and return combined history."""
    resolved_path = _resolve_history_path_value(history_path)
    existing_rows = _sort_history_rows(_load_history_rows(resolved_path))
    new_rows = _build_history_rows(metric_outputs)
    _append_history_rows(resolved_path, new_rows)
    combined_rows = _sort_history_rows([*existing_rows, *new_rows])
    return {
        "history_path": str(resolved_path),
        "history_entries": new_rows,
        "history_rows": combined_rows,
    }


def _metric_label(metric: str) -> str:
    if metric == "t1":
        return "T1"
    if metric == "t2":
        return "T2"
    if metric == "t2_star":
        return "T2*"
    return metric


@workflow.task
@with_plot_theme
def plot_tracking_history(
    qubits: QuantumElements,
    history_rows: list[dict[str, object]],
    options: PlotCoherenceTrackingOptions | None = None,
) -> dict[str, mpl.figure.Figure]:
    """Create per-qubit trend figures from JSONL tracking history."""
    opts = PlotCoherenceTrackingOptions() if options is None else options
    qubits = validate_and_convert_qubits_sweeps(qubits)
    history_rows = _sort_history_rows(list(history_rows))
    figures: dict[str, mpl.figure.Figure] = {}

    for q in qubits:
        q_rows = [row for row in history_rows if row.get("qubit_uid") == q.uid]
        if not q_rows:
            continue

        fig, axes = plt.subplots(3, 1, figsize=(8.8, 7.4), sharex=True)
        fig.suptitle(timestamped_title(f"Coherence tracking {q.uid}"))

        for axis, metric in zip(axes, ("t1", "t2", "t2_star")):
            metric_rows = [row for row in q_rows if row.get("metric") == metric]
            axis.set_ylabel(f"{_metric_label(metric)} ($\\mu$s)")
            axis.grid(alpha=0.22)
            if not metric_rows:
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

            y_values = np.asarray(
                [float(row["value_s"]) for row in metric_rows],
                dtype=float,
            ) * 1e6
            x_values = np.arange(1, len(metric_rows) + 1, dtype=int)
            std_values = np.asarray(
                [
                    np.nan if row.get("std_dev_s") is None else float(row["std_dev_s"])
                    for row in metric_rows
                ],
                dtype=float,
            ) * 1e6

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

        axes[-1].set_xlabel("Run index")
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
    return output


@workflow.workflow
def analysis_workflow(
    qubits: QuantumElements,
    t1_result: RunExperimentResults | None = None,
    t2_star_result: RunExperimentResults | None = None,
    t2_result: RunExperimentResults | None = None,
    t1_delays: QubitSweepPoints | None = None,
    t2_star_delays: QubitSweepPoints | None = None,
    t2_delays: QubitSweepPoints | None = None,
    t2_star_detunings: float | Sequence[float] | None = None,
    *,
    run_t1: bool = True,
    run_t2_star: bool = True,
    run_t2: bool = True,
    history_path: str | Path = "laboneq_output/tracking/coherence_tracking.jsonl",
    options: CoherenceTrackingAnalysisWorkflowOptions | None = None,
) -> None:
    """Aggregate local T1/T2/T2* analyses and persist tracking history."""
    opts = CoherenceTrackingAnalysisWorkflowOptions() if options is None else options

    t1_output = _empty_analysis_payload()
    t2_star_output = _empty_analysis_payload()
    t2_output = _empty_analysis_payload()

    with workflow.if_(run_t1):
        t1_analysis = lifetime_analysis_workflow(t1_result, qubits, t1_delays)
        t1_output = t1_analysis.output
    with workflow.if_(run_t2_star):
        t2_star_analysis = ramsey_analysis_workflow(
            t2_star_result,
            qubits,
            t2_star_delays,
            t2_star_detunings,
        )
        t2_star_output = t2_star_analysis.output
    with workflow.if_(run_t2):
        t2_analysis = echo_analysis_workflow(t2_result, qubits, t2_delays)
        t2_output = t2_analysis.output

    merged_output = _merge_metric_outputs(t1_output, t2_star_output, t2_output)
    history_bundle = materialize_tracking_history(
        merged_output["metric_outputs"],
        history_path,
    )

    with workflow.if_(opts.do_plotting):
        plot_tracking_history(qubits, history_bundle["history_rows"])

    workflow.return_(_attach_history_bundle(merged_output, history_bundle))

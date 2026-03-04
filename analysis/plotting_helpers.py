"""Local plotting helpers for analysis workflows."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from laboneq import workflow
from laboneq.simple import dsl

from analysis.plot_theme import with_plot_theme
from laboneq_applications.analysis.options import BasePlottingOptions
from laboneq_applications.core.validation import (
    validate_and_convert_qubits_sweeps,
    validate_result,
)

if TYPE_CHECKING:
    import matplotlib as mpl
    from laboneq.workflow.tasks.run_experiment import RunExperimentResults
    from numpy.typing import ArrayLike

    from laboneq_applications.typing import QuantumElements, QubitSweepPoints


@workflow.task_options(base_class=BasePlottingOptions)
class PlotRawDataOptions:
    """Options for local raw complex 1D plotting."""

    use_cal_traces: bool = workflow.option_field(
        True,
        description="Whether to include calibration-trace reference lines.",
    )
    cal_states: str | tuple = workflow.option_field(
        "ge",
        description="Calibration states used for calibration traces.",
    )


def timestamped_title(title: str, dt: datetime | None = None) -> str:
    """Append local timestamp to title with a deterministic format."""
    timestamp = (dt or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
    return f"{title} ({timestamp})"


def _unwrap_result_like(result_like):
    current = result_like
    for _ in range(8):
        if current is None:
            return None
        if isinstance(current, tuple) and current:
            current = current[0]
            continue
        if hasattr(current, "output"):
            current = current.output
            continue
        if isinstance(current, dict) and "result" in current:
            current = current["result"]
            continue
        return current
    return current


def _normalize_cal_states(cal_states: str | tuple | list) -> list[str]:
    if isinstance(cal_states, str):
        return [str(ch) for ch in cal_states]
    if isinstance(cal_states, Iterable):
        return [str(cs) for cs in cal_states]
    return []


def _extract_suffixes(result, qubit_uid: str) -> list[str]:
    handle = dsl.handles.result_handle(qubit_uid)
    node = result[handle]
    if hasattr(node, "keys"):
        return [str(k) for k in node.keys()]
    if isinstance(node, (list, tuple, set)):
        return [str(v) for v in node]
    if hasattr(node, "__iter__") and not hasattr(node, "data"):
        return [str(v) for v in node]
    return []


def _to_1d_trace(data: ArrayLike, expected_len: int) -> np.ndarray:
    arr = np.asarray(data, dtype=complex)
    if arr.ndim == 1 and arr.size == expected_len:
        return arr
    if arr.ndim == 0:
        return np.repeat(arr.reshape(1), expected_len)

    if arr.ndim >= 1:
        for axis, size in enumerate(arr.shape):
            if int(size) == int(expected_len):
                moved = np.moveaxis(arr, axis, 0)
                return moved.reshape(expected_len, -1).mean(axis=1)

    flat = arr.reshape(-1)
    if expected_len > 0 and flat.size % expected_len == 0:
        return flat.reshape(expected_len, -1).mean(axis=1)

    if flat.size >= expected_len:
        return flat[:expected_len]

    padded = np.full(expected_len, np.nan + 0j, dtype=complex)
    padded[: flat.size] = flat
    return padded


def _read_result_data_by_suffix(
    result,
    qubit_uid: str,
    expected_len: int,
) -> dict[str, np.ndarray]:
    suffixes = _extract_suffixes(result, qubit_uid)
    if suffixes:
        out = {}
        for suffix in suffixes:
            raw = result[dsl.handles.result_handle(qubit_uid, suffix=suffix)].data
            out[suffix] = _to_1d_trace(raw, expected_len)
        return out

    raw = result[dsl.handles.result_handle(qubit_uid)].data
    return {"data": _to_1d_trace(raw, expected_len)}


def _raw_artifact_name(qubit_uid: str, trace_labels: list[str]) -> str:
    if len(trace_labels) <= 1:
        return f"Raw_data_{qubit_uid}"
    labels = "_".join(label.replace(" ", "-") for label in trace_labels)
    return f"Raw_data_{qubit_uid}_{labels}"


@workflow.task
@with_plot_theme
def plot_raw_complex_data_1d(
    qubits: QuantumElements,
    result: RunExperimentResults,
    sweep_points: QubitSweepPoints,
    xlabel: str = "Sweep Points",
    xscaling: float = 1.0,
    options: PlotRawDataOptions | None = None,
) -> dict[str, mpl.figure.Figure]:
    """Plot raw integrated complex data in I/Q panels for each qubit."""
    opts = PlotRawDataOptions() if options is None else options
    result = _unwrap_result_like(result)
    validate_result(result)
    qubits, sweep_points = validate_and_convert_qubits_sweeps(qubits, sweep_points)

    figures: dict[str, mpl.figure.Figure] = {}
    cal_states = _normalize_cal_states(opts.cal_states)

    for q, swpts in zip(qubits, sweep_points):
        swpts_arr = np.asarray(swpts, dtype=float).reshape(-1)
        traces = _read_result_data_by_suffix(
            result=result,
            qubit_uid=q.uid,
            expected_len=len(swpts_arr),
        )

        fig, axes = plt.subplots(2, 1, figsize=(8.2, 5.6), sharex=True)
        ax_i, ax_q = axes

        multi_trace = len(traces) > 1
        for trace_label, trace in traces.items():
            label_i = f"{trace_label} I" if multi_trace else "I"
            label_q = f"{trace_label} Q" if multi_trace else "Q"
            ax_i.plot(swpts_arr * xscaling, np.real(trace), marker="o", label=label_i)
            ax_q.plot(swpts_arr * xscaling, np.imag(trace), marker="o", label=label_q)

        if opts.use_cal_traces:
            for cal_state in cal_states:
                try:
                    cal_data = np.asarray(
                        result[dsl.handles.calibration_trace_handle(q.uid, cal_state)].data,
                        dtype=complex,
                    ).reshape(-1)
                except Exception:  # noqa: BLE001
                    continue
                if cal_data.size == 0:
                    continue
                ax_i.axhline(
                    np.mean(np.real(cal_data)),
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.55,
                    label=f"cal {cal_state} I",
                )
                ax_q.axhline(
                    np.mean(np.imag(cal_data)),
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.55,
                    label=f"cal {cal_state} Q",
                )

        ax_i.set_title(timestamped_title(f"Raw data {q.uid}"))
        ax_i.set_ylabel("I (a.u.)")
        ax_q.set_ylabel("Q (a.u.)")
        ax_q.set_xlabel(xlabel)
        ax_i.grid(alpha=0.22)
        ax_q.grid(alpha=0.22)
        ax_i.legend(loc="best", frameon=False)
        ax_q.legend(loc="best", frameon=False)
        fig.tight_layout()

        artifact_name = _raw_artifact_name(q.uid, list(traces.keys()))
        if opts.save_figures:
            workflow.save_artifact(artifact_name, fig)

        if opts.close_figures:
            plt.close(fig)

        figure_key = q.uid if len(traces) == 1 else f"{q.uid}_multi"
        figures[figure_key] = fig

    return figures


__all__ = [
    "PlotRawDataOptions",
    "plot_raw_complex_data_1d",
    "timestamped_title",
]

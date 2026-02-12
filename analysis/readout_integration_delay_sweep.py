"""Analysis workflow for joint readout integration delay/length optimization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from laboneq import workflow

from analysis.readout_sweep_common import (
    calibration_shots_by_state_and_sweep,
    evaluate_iq_binary,
    unwrap_result_like,
)
from laboneq_applications.core.validation import validate_result

if TYPE_CHECKING:
    from collections.abc import Sequence

    import matplotlib as mpl
    from laboneq.dsl.quantum.quantum_element import QuantumElement
    from laboneq.workflow.tasks.run_experiment import RunExperimentResults
    from numpy.typing import ArrayLike


@workflow.workflow_options
class ReadoutIntegrationDelaySweepAnalysisOptions:
    """Options for integration-window optimization analysis."""

    do_plotting: bool = workflow.option_field(
        True, description="Whether to create plots."
    )
    do_plotting_heatmap: bool = workflow.option_field(
        True, description="Whether to plot fidelity/SNR heatmaps."
    )
    ridge_target_condition: float = workflow.option_field(
        1e6,
        description="Target covariance condition number for ridge regularization.",
    )
    fidelity_tolerance: float = workflow.option_field(
        5e-4,
        description="Tolerance from max fidelity for tie candidates.",
    )


@workflow.workflow(name="analysis_readout_integration_delay_sweep")
def analysis_workflow(
    results: Sequence[RunExperimentResults],
    qubits: Sequence[QuantumElement],
    delays: ArrayLike,
    integration_lengths: ArrayLike,
    reference_qubit: QuantumElement,
    options: ReadoutIntegrationDelaySweepAnalysisOptions | None = None,
) -> None:
    """Analyze a delay sweep repeated across integration-length candidates."""
    options = (
        ReadoutIntegrationDelaySweepAnalysisOptions() if options is None else options
    )

    metrics = calculate_metrics(
        results=results,
        qubits=qubits,
        delays=delays,
        integration_lengths=integration_lengths,
        reference_qubit=reference_qubit,
        ridge_target_condition=options.ridge_target_condition,
        fidelity_tolerance=options.fidelity_tolerance,
    )
    with workflow.if_(options.do_plotting):
        with workflow.if_(options.do_plotting_heatmap):
            plot_heatmaps(metrics=metrics)
    workflow.return_(metrics)


@workflow.task
def calculate_metrics(
    results: Sequence[RunExperimentResults],
    qubits: Sequence[QuantumElement],
    delays: ArrayLike,
    integration_lengths: ArrayLike,
    reference_qubit: QuantumElement,
    ridge_target_condition: float = 1e6,
    fidelity_tolerance: float = 5e-4,
) -> dict:
    """Compute fidelity/SNR metric grid and choose best integration window."""
    delay_points = np.asarray(delays, dtype=float).reshape(-1)
    length_points = np.asarray(integration_lengths, dtype=float).reshape(-1)
    if delay_points.size < 1:
        raise ValueError("delays must contain at least one point.")
    if length_points.size < 1:
        raise ValueError("integration_lengths must contain at least one point.")
    if len(results) != len(length_points):
        raise ValueError(
            "results and integration_lengths size mismatch: "
            f"{len(results)} != {len(length_points)}."
        )
    if len(qubits) != len(length_points):
        raise ValueError(
            "qubits and integration_lengths size mismatch: "
            f"{len(qubits)} != {len(length_points)}."
        )

    n_lengths = len(length_points)
    n_delays = len(delay_points)
    fidelity_grid = np.zeros((n_lengths, n_delays), dtype=float)
    snr_grid = np.zeros((n_lengths, n_delays), dtype=float)

    for li, (result_like, qubit) in enumerate(zip(results, qubits)):
        result = unwrap_result_like(result_like)
        validate_result(result)
        state_shots = calibration_shots_by_state_and_sweep(
            result=result,
            qubit_uid=qubit.uid,
            states=("g", "e"),
            n_points=n_delays,
        )
        g_shots = state_shots["g"]
        e_shots = state_shots["e"]
        for di in range(n_delays):
            metric = evaluate_iq_binary(
                shots_g=g_shots[di],
                shots_e=e_shots[di],
                target_condition=float(ridge_target_condition),
            )
            fidelity_grid[li, di] = float(metric["assignment_fidelity"])
            snr_grid[li, di] = float(metric["delta_mu_over_sigma"])

    max_fid = float(np.max(fidelity_grid))
    tol = max(0.0, float(fidelity_tolerance))
    candidate_idx = np.argwhere(fidelity_grid >= max_fid - tol)
    if candidate_idx.size == 0:
        candidate_idx = np.argwhere(fidelity_grid == max_fid)

    records = []
    for li, di in candidate_idx:
        records.append(
            (
                int(li),
                int(di),
                float(snr_grid[li, di]),
                float(length_points[li]),
                float(delay_points[di]),
            )
        )
    records.sort(key=lambda x: (-x[2], x[3], x[4]))
    best_li, best_di = int(records[0][0]), int(records[0][1])

    sorted_flat = np.sort(fidelity_grid.reshape(-1))
    second_best = float(sorted_flat[-2]) if sorted_flat.size >= 2 else -np.inf
    low_margin = bool(max_fid - second_best <= tol)
    flat_optimum = bool(len(records) > 1)
    edge_hit = bool(
        best_li in (0, n_lengths - 1) or best_di in (0, n_delays - 1)
    )
    if edge_hit:
        quality_flag = "edge_hit"
    elif low_margin:
        quality_flag = "low_margin"
    elif flat_optimum:
        quality_flag = "flat_optimum"
    else:
        quality_flag = "ok"

    old_delay = reference_qubit.parameters.readout_integration_delay
    old_length = reference_qubit.parameters.readout_integration_length
    best_delay = float(delay_points[best_di])
    best_length = float(length_points[best_li])

    return {
        "sweep_parameter": "readout_integration_window",
        "sweep_points": {
            "readout_integration_length": length_points,
            "readout_integration_delay": delay_points,
        },
        "metrics_vs_sweep": {
            "assignment_fidelity": fidelity_grid,
            "delta_mu_over_sigma": snr_grid,
        },
        "best_point": {
            "index_length": int(best_li),
            "index_delay": int(best_di),
            "readout_integration_length": best_length,
            "readout_integration_delay": best_delay,
            "assignment_fidelity": float(fidelity_grid[best_li, best_di]),
            "delta_mu_over_sigma": float(snr_grid[best_li, best_di]),
        },
        "quality_flag": quality_flag,
        "old_parameter_values": {
            reference_qubit.uid: {
                "readout_integration_delay": old_delay,
                "readout_integration_length": old_length,
            }
        },
        "new_parameter_values": {
            reference_qubit.uid: {
                "readout_integration_delay": best_delay,
                "readout_integration_length": best_length,
            }
        },
    }


@workflow.task
def plot_heatmaps(metrics: dict) -> mpl.figure.Figure:
    """Plot assignment fidelity and SNR over (integration_length, delay)."""
    lengths = np.asarray(
        metrics["sweep_points"]["readout_integration_length"], dtype=float
    )
    delays = np.asarray(metrics["sweep_points"]["readout_integration_delay"], dtype=float)
    fid = np.asarray(metrics["metrics_vs_sweep"]["assignment_fidelity"], dtype=float)
    snr = np.asarray(metrics["metrics_vs_sweep"]["delta_mu_over_sigma"], dtype=float)
    best = metrics["best_point"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    x_ns = delays * 1e9
    y_us = lengths * 1e6
    extent = [float(np.min(x_ns)), float(np.max(x_ns)), float(np.min(y_us)), float(np.max(y_us))]

    im0 = axes[0].imshow(fid, aspect="auto", origin="lower", extent=extent)
    axes[0].scatter(
        [best["readout_integration_delay"] * 1e9],
        [best["readout_integration_length"] * 1e6],
        c="w",
        s=45,
        edgecolors="k",
    )
    axes[0].set_title("Assignment fidelity")
    axes[0].set_xlabel("Integration delay (ns)")
    axes[0].set_ylabel("Integration length (us)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(snr, aspect="auto", origin="lower", extent=extent)
    axes[1].scatter(
        [best["readout_integration_delay"] * 1e9],
        [best["readout_integration_length"] * 1e6],
        c="w",
        s=45,
        edgecolors="k",
    )
    axes[1].set_title("SNR (delta_mu_over_sigma)")
    axes[1].set_xlabel("Integration delay (ns)")
    axes[1].set_ylabel("Integration length (us)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Readout integration window sweep (quality={metrics['quality_flag']})",
        fontsize=12,
    )
    workflow.save_artifact("readout_integration_window_heatmaps", fig)
    return fig

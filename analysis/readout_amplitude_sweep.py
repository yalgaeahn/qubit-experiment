"""Analysis workflow for readout amplitude optimization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from laboneq import workflow

from analysis.readout_sweep_common import (
    calibration_shots_by_state_and_sweep,
    evaluate_iq_binary,
    select_best_index,
    unwrap_result_like,
)
from laboneq_applications.core.validation import (
    validate_and_convert_single_qubit_sweeps,
    validate_result,
)

if TYPE_CHECKING:
    import matplotlib as mpl
    from laboneq.dsl.quantum.quantum_element import QuantumElement
    from laboneq.workflow.tasks.run_experiment import RunExperimentResults
    from numpy.typing import ArrayLike


@workflow.workflow_options
class ReadoutAmplitudeSweepAnalysisOptions:
    """Options for readout amplitude sweep analysis."""

    do_plotting: bool = workflow.option_field(
        True, description="Whether to create plots."
    )
    do_plotting_metrics: bool = workflow.option_field(
        True, description="Whether to plot metric curves."
    )
    ridge_target_condition: float = workflow.option_field(
        1e6,
        description="Target covariance condition number for ridge regularization.",
    )
    fidelity_tolerance: float = workflow.option_field(
        5e-4,
        description="Tolerance from max fidelity for tie candidates.",
    )
    avoid_edge_fraction: float = workflow.option_field(
        0.1,
        description="Preferred interior fraction to avoid amplitude-edge operation.",
    )


@workflow.workflow(name="analysis_readout_amplitude_sweep")
def analysis_workflow(
    result: RunExperimentResults,
    qubit: QuantumElement,
    amplitudes: ArrayLike,
    options: ReadoutAmplitudeSweepAnalysisOptions | None = None,
) -> None:
    """Analyze a single experiment with hardware readout-amplitude sweep."""
    options = ReadoutAmplitudeSweepAnalysisOptions() if options is None else options
    metrics = calculate_metrics(
        result=result,
        qubit=qubit,
        amplitudes=amplitudes,
        ridge_target_condition=options.ridge_target_condition,
        fidelity_tolerance=options.fidelity_tolerance,
        avoid_edge_fraction=options.avoid_edge_fraction,
    )
    with workflow.if_(options.do_plotting):
        with workflow.if_(options.do_plotting_metrics):
            plot_metrics(metrics=metrics)
    workflow.return_(metrics)


@workflow.task
def calculate_metrics(
    result: RunExperimentResults,
    qubit: QuantumElement,
    amplitudes: ArrayLike,
    ridge_target_condition: float = 1e6,
    fidelity_tolerance: float = 5e-4,
    avoid_edge_fraction: float = 0.1,
) -> dict:
    """Compute fidelity/SNR curves and select best readout amplitude."""
    qubit, amplitude_points = validate_and_convert_single_qubit_sweeps(qubit, amplitudes)
    amplitude_points = np.asarray(amplitude_points, dtype=float).reshape(-1)
    if amplitude_points.size < 1:
        raise ValueError("amplitudes must contain at least one point.")

    result = unwrap_result_like(result)
    validate_result(result)

    state_shots = calibration_shots_by_state_and_sweep(
        result=result,
        qubit_uid=qubit.uid,
        states=("g", "e"),
        n_points=len(amplitude_points),
    )
    g_shots = state_shots["g"]
    e_shots = state_shots["e"]

    fidelity = np.zeros(len(amplitude_points), dtype=float)
    snr = np.zeros(len(amplitude_points), dtype=float)
    for i in range(len(amplitude_points)):
        metric = evaluate_iq_binary(
            shots_g=g_shots[i],
            shots_e=e_shots[i],
            target_condition=float(ridge_target_condition),
        )
        fidelity[i] = float(metric["assignment_fidelity"])
        snr[i] = float(metric["delta_mu_over_sigma"])

    global_best = select_best_index(
        assignment_fidelity=fidelity,
        delta_mu_over_sigma=snr,
        fidelity_tolerance=float(fidelity_tolerance),
        prefer_smallest=False,
    )
    best_idx = int(global_best["index"])
    quality_flag = str(global_best["quality_flag"])

    frac = max(0.0, min(0.45, float(avoid_edge_fraction)))
    n = len(amplitude_points)
    lo = int(np.floor(frac * n))
    hi = int(np.ceil((1.0 - frac) * n))
    interior = np.arange(lo, hi, dtype=int)
    if interior.size > 0:
        interior_sel = select_best_index(
            assignment_fidelity=fidelity[interior],
            delta_mu_over_sigma=snr[interior],
            fidelity_tolerance=float(fidelity_tolerance),
            prefer_smallest=False,
        )
        interior_best_idx = int(interior[int(interior_sel["index"])])
        if fidelity[interior_best_idx] >= fidelity[best_idx] - float(fidelity_tolerance):
            best_idx = interior_best_idx
            if best_idx in (0, n - 1):
                quality_flag = "edge_hit"
            elif quality_flag == "edge_hit":
                quality_flag = "possible_saturation"
        elif best_idx in (0, n - 1):
            quality_flag = "possible_saturation"

    return {
        "sweep_parameter": "readout_amplitude",
        "sweep_points": amplitude_points,
        "metrics_vs_sweep": {
            "assignment_fidelity": fidelity,
            "delta_mu_over_sigma": snr,
        },
        "best_point": {
            "index": int(best_idx),
            "readout_amplitude": float(amplitude_points[best_idx]),
            "assignment_fidelity": float(fidelity[best_idx]),
            "delta_mu_over_sigma": float(snr[best_idx]),
        },
        "quality_flag": quality_flag,
        "old_parameter_values": {
            qubit.uid: {
                "readout_amplitude": qubit.parameters.readout_amplitude,
            }
        },
        "new_parameter_values": {
            qubit.uid: {
                "readout_amplitude": float(amplitude_points[best_idx]),
            }
        },
    }


@workflow.task
def plot_metrics(metrics: dict) -> mpl.figure.Figure:
    """Plot readout-amplitude sweep metrics."""
    amplitudes = np.asarray(metrics["sweep_points"], dtype=float)
    fidelity = np.asarray(metrics["metrics_vs_sweep"]["assignment_fidelity"], dtype=float)
    snr = np.asarray(metrics["metrics_vs_sweep"]["delta_mu_over_sigma"], dtype=float)
    best = metrics["best_point"]

    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    axes[0].plot(amplitudes, fidelity, marker="o")
    axes[0].axvline(best["readout_amplitude"], linestyle="--", color="gray")
    axes[0].set_ylabel("Assignment fidelity")
    axes[0].grid(alpha=0.25)

    axes[1].plot(amplitudes, snr, marker="o")
    axes[1].axvline(best["readout_amplitude"], linestyle="--", color="gray")
    axes[1].set_ylabel("SNR (delta_mu_over_sigma)")
    axes[1].set_xlabel("Readout amplitude (a.u.)")
    axes[1].grid(alpha=0.25)

    fig.suptitle(
        f"Readout amplitude sweep (quality={metrics['quality_flag']})",
        fontsize=12,
    )
    workflow.save_artifact("readout_amplitude_sweep_metrics", fig)
    return fig

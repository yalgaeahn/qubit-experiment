"""Analysis workflow for readout pulse-length optimization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from laboneq import workflow

from analysis.readout_sweep_common import (
    calibration_shots_by_state,
    evaluate_iq_binary,
    select_best_index,
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
class ReadoutLengthSweepAnalysisOptions:
    """Options for readout length sweep analysis."""

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
    prefer_shorter_within_tolerance: bool = workflow.option_field(
        True,
        description="Prefer shortest readout length among near-optimal points.",
    )


@workflow.workflow(name="analysis_readout_length_sweep")
def analysis_workflow(
    results: Sequence[RunExperimentResults],
    qubits: Sequence[QuantumElement],
    readout_lengths: ArrayLike,
    reference_qubit: QuantumElement,
    options: ReadoutLengthSweepAnalysisOptions | None = None,
) -> None:
    """Analyze repeated IQ-cloud runs across readout-length candidates."""
    options = ReadoutLengthSweepAnalysisOptions() if options is None else options
    metrics = calculate_metrics(
        results=results,
        qubits=qubits,
        readout_lengths=readout_lengths,
        reference_qubit=reference_qubit,
        ridge_target_condition=options.ridge_target_condition,
        fidelity_tolerance=options.fidelity_tolerance,
        prefer_shorter_within_tolerance=options.prefer_shorter_within_tolerance,
    )
    with workflow.if_(options.do_plotting):
        with workflow.if_(options.do_plotting_metrics):
            plot_metrics(metrics=metrics)
    workflow.return_(metrics)


@workflow.task
def calculate_metrics(
    results: Sequence[RunExperimentResults],
    qubits: Sequence[QuantumElement],
    readout_lengths: ArrayLike,
    reference_qubit: QuantumElement,
    ridge_target_condition: float = 1e6,
    fidelity_tolerance: float = 5e-4,
    prefer_shorter_within_tolerance: bool = True,
) -> dict:
    """Compute metric curves over readout-length candidates and choose optimum."""
    length_points = np.asarray(readout_lengths, dtype=float).reshape(-1)
    if length_points.size < 1:
        raise ValueError("readout_lengths must contain at least one point.")
    if len(results) != len(length_points):
        raise ValueError(
            "results and readout_lengths size mismatch: "
            f"{len(results)} != {len(length_points)}."
        )
    if len(qubits) != len(length_points):
        raise ValueError(
            "qubits and readout_lengths size mismatch: "
            f"{len(qubits)} != {len(length_points)}."
        )

    fidelity = np.zeros(len(length_points), dtype=float)
    snr = np.zeros(len(length_points), dtype=float)
    for i, (result_like, qubit) in enumerate(zip(results, qubits)):
        result = unwrap_result_like(result_like)
        validate_result(result)
        shots = calibration_shots_by_state(
            result=result,
            qubit_uid=qubit.uid,
            states=("g", "e"),
        )
        metric = evaluate_iq_binary(
            shots_g=shots["g"],
            shots_e=shots["e"],
            target_condition=float(ridge_target_condition),
        )
        fidelity[i] = float(metric["assignment_fidelity"])
        snr[i] = float(metric["delta_mu_over_sigma"])

    best = select_best_index(
        assignment_fidelity=fidelity,
        delta_mu_over_sigma=snr,
        fidelity_tolerance=float(fidelity_tolerance),
        prefer_smallest=bool(prefer_shorter_within_tolerance),
    )
    best_idx = int(best["index"])
    best_length = float(length_points[best_idx])
    quality_flag = str(best["quality_flag"])

    argmax_idx = int(np.argmax(fidelity))
    if prefer_shorter_within_tolerance and best_idx != argmax_idx and quality_flag == "ok":
        quality_flag = "latency_tradeoff"

    return {
        "sweep_parameter": "readout_length",
        "sweep_points": length_points,
        "metrics_vs_sweep": {
            "assignment_fidelity": fidelity,
            "delta_mu_over_sigma": snr,
            "effective_latency_s": length_points,
        },
        "best_point": {
            "index": best_idx,
            "readout_length": best_length,
            "assignment_fidelity": float(fidelity[best_idx]),
            "delta_mu_over_sigma": float(snr[best_idx]),
        },
        "quality_flag": quality_flag,
        "old_parameter_values": {
            reference_qubit.uid: {
                "readout_length": reference_qubit.parameters.readout_length,
            }
        },
        "new_parameter_values": {
            reference_qubit.uid: {
                "readout_length": best_length,
            }
        },
    }


@workflow.task
def plot_metrics(metrics: dict) -> mpl.figure.Figure:
    """Plot readout-length sweep metrics."""
    length_points = np.asarray(metrics["sweep_points"], dtype=float)
    fidelity = np.asarray(metrics["metrics_vs_sweep"]["assignment_fidelity"], dtype=float)
    snr = np.asarray(metrics["metrics_vs_sweep"]["delta_mu_over_sigma"], dtype=float)
    best = metrics["best_point"]

    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    x_us = length_points * 1e6

    axes[0].plot(x_us, fidelity, marker="o")
    axes[0].axvline(best["readout_length"] * 1e6, linestyle="--", color="gray")
    axes[0].set_ylabel("Assignment fidelity")
    axes[0].grid(alpha=0.25)

    axes[1].plot(x_us, snr, marker="o")
    axes[1].axvline(best["readout_length"] * 1e6, linestyle="--", color="gray")
    axes[1].set_ylabel("SNR (delta_mu_over_sigma)")
    axes[1].set_xlabel("Readout length (us)")
    axes[1].grid(alpha=0.25)

    fig.suptitle(f"Readout length sweep (quality={metrics['quality_flag']})", fontsize=12)
    workflow.save_artifact("readout_length_sweep_metrics", fig)
    return fig

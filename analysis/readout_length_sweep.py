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
    do_plotting_error_bars: bool = workflow.option_field(
        True, description="Whether to include bootstrap error bars in metric plots."
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
    bootstrap_samples: int = workflow.option_field(
        400,
        description="Bootstrap sample count per sweep point for uncertainty.",
    )
    bootstrap_confidence_level: float = workflow.option_field(
        0.95,
        description="Confidence level for bootstrap intervals.",
    )
    bootstrap_seed: int | None = workflow.option_field(
        None,
        description="Random seed for bootstrap resampling.",
    )


@workflow.workflow(name="analysis_readout_length_sweep")
def analysis_workflow(
    results: Sequence[RunExperimentResults],
    qubits: Sequence[QuantumElement],
    readout_lengths: ArrayLike,
    reference_qubit: QuantumElement,
    options: ReadoutLengthSweepAnalysisOptions | None = None,
) -> None:
    """Analyze repeated IQ-cloud acquisitions across readout-length candidates."""
    options = ReadoutLengthSweepAnalysisOptions() if options is None else options
    metrics = calculate_metrics(
        results=results,
        qubits=qubits,
        readout_lengths=readout_lengths,
        reference_qubit=reference_qubit,
        ridge_target_condition=options.ridge_target_condition,
        fidelity_tolerance=options.fidelity_tolerance,
        prefer_shorter_within_tolerance=options.prefer_shorter_within_tolerance,
        bootstrap_samples=options.bootstrap_samples,
        bootstrap_confidence_level=options.bootstrap_confidence_level,
        bootstrap_seed=options.bootstrap_seed,
    )
    with workflow.if_(options.do_plotting):
        with workflow.if_(options.do_plotting_metrics):
            plot_metrics(
                metrics=metrics,
                include_error_bars=options.do_plotting_error_bars,
            )
    workflow.return_(metrics)


def _bootstrap_ci(
    shots_g: np.ndarray,
    shots_e: np.ndarray,
    *,
    target_condition: float,
    samples: int,
    confidence_level: float,
    rng: np.random.Generator,
) -> dict[str, float]:
    point = evaluate_iq_binary(
        shots_g=shots_g,
        shots_e=shots_e,
        target_condition=float(target_condition),
    )
    if int(samples) < 2 or shots_g.size < 2 or shots_e.size < 2:
        return {
            "fidelity_ci_low": float(point["assignment_fidelity"]),
            "fidelity_ci_high": float(point["assignment_fidelity"]),
            "snr_ci_low": float(point["delta_mu_over_sigma"]),
            "snr_ci_high": float(point["delta_mu_over_sigma"]),
        }

    n_g = shots_g.size
    n_e = shots_e.size
    f_vals = np.empty(int(samples), dtype=float)
    s_vals = np.empty(int(samples), dtype=float)
    for i in range(int(samples)):
        idx_g = rng.integers(0, n_g, size=n_g)
        idx_e = rng.integers(0, n_e, size=n_e)
        metric = evaluate_iq_binary(
            shots_g=shots_g[idx_g],
            shots_e=shots_e[idx_e],
            target_condition=float(target_condition),
        )
        f_vals[i] = float(metric["assignment_fidelity"])
        s_vals[i] = float(metric["delta_mu_over_sigma"])

    cl = float(np.clip(confidence_level, 1e-6, 1.0 - 1e-6))
    alpha = 0.5 * (1.0 - cl)
    return {
        "fidelity_ci_low": float(np.quantile(f_vals, alpha)),
        "fidelity_ci_high": float(np.quantile(f_vals, 1.0 - alpha)),
        "snr_ci_low": float(np.quantile(s_vals, alpha)),
        "snr_ci_high": float(np.quantile(s_vals, 1.0 - alpha)),
    }


@workflow.task(save=False)
def calculate_metrics(
    results: Sequence[RunExperimentResults],
    qubits: Sequence[QuantumElement],
    readout_lengths: ArrayLike,
    reference_qubit: QuantumElement,
    ridge_target_condition: float = 1e6,
    fidelity_tolerance: float = 5e-4,
    prefer_shorter_within_tolerance: bool = True,
    bootstrap_samples: int = 400,
    bootstrap_confidence_level: float = 0.95,
    bootstrap_seed: int | None = None,
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
    fidelity_ci_low = np.zeros(len(length_points), dtype=float)
    fidelity_ci_high = np.zeros(len(length_points), dtype=float)
    snr_ci_low = np.zeros(len(length_points), dtype=float)
    snr_ci_high = np.zeros(len(length_points), dtype=float)
    rng = np.random.default_rng(bootstrap_seed)

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
        ci = _bootstrap_ci(
            shots_g=np.asarray(shots["g"], dtype=complex).reshape(-1),
            shots_e=np.asarray(shots["e"], dtype=complex).reshape(-1),
            target_condition=float(ridge_target_condition),
            samples=int(max(0, bootstrap_samples)),
            confidence_level=float(bootstrap_confidence_level),
            rng=rng,
        )
        fidelity_ci_low[i] = float(ci["fidelity_ci_low"])
        fidelity_ci_high[i] = float(ci["fidelity_ci_high"])
        snr_ci_low[i] = float(ci["snr_ci_low"])
        snr_ci_high[i] = float(ci["snr_ci_high"])

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
        "bootstrap": {
            "assignment_fidelity": {
                "ci_low": fidelity_ci_low,
                "ci_high": fidelity_ci_high,
                "confidence_level": float(bootstrap_confidence_level),
            },
            "delta_mu_over_sigma": {
                "ci_low": snr_ci_low,
                "ci_high": snr_ci_high,
                "confidence_level": float(bootstrap_confidence_level),
            },
            "settings": {
                "bootstrap_samples": int(max(0, bootstrap_samples)),
                "seed": bootstrap_seed,
            },
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
                "readout_integration_length": (
                    reference_qubit.parameters.readout_integration_length
                ),
            }
        },
        "new_parameter_values": {
            reference_qubit.uid: {
                "readout_length": best_length,
                "readout_integration_length": best_length,
            }
        },
    }


@workflow.task(save=False)
def plot_metrics(
    metrics: dict,
    include_error_bars: bool = True,
) -> mpl.figure.Figure:
    """Plot readout-length sweep metrics."""
    length_points = np.asarray(metrics["sweep_points"], dtype=float)
    fidelity = np.asarray(metrics["metrics_vs_sweep"]["assignment_fidelity"], dtype=float)
    snr = np.asarray(metrics["metrics_vs_sweep"]["delta_mu_over_sigma"], dtype=float)
    bootstrap = metrics.get("bootstrap", {})
    f_ci = bootstrap.get("assignment_fidelity", {})
    s_ci = bootstrap.get("delta_mu_over_sigma", {})
    f_lo = np.asarray(f_ci.get("ci_low", []), dtype=float).reshape(-1)
    f_hi = np.asarray(f_ci.get("ci_high", []), dtype=float).reshape(-1)
    s_lo = np.asarray(s_ci.get("ci_low", []), dtype=float).reshape(-1)
    s_hi = np.asarray(s_ci.get("ci_high", []), dtype=float).reshape(-1)
    best = metrics["best_point"]

    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    x_us = length_points * 1e6

    if include_error_bars and f_lo.size == fidelity.size and f_hi.size == fidelity.size:
        f_err = np.vstack(
            [
                np.maximum(0.0, fidelity - f_lo),
                np.maximum(0.0, f_hi - fidelity),
            ]
        )
        axes[0].errorbar(x_us, fidelity, yerr=f_err, marker="o", capsize=3)
    else:
        axes[0].plot(x_us, fidelity, marker="o")
    axes[0].axvline(best["readout_length"] * 1e6, linestyle="--", color="gray")
    axes[0].set_ylabel("Assignment fidelity")
    axes[0].grid(alpha=0.25)

    if include_error_bars and s_lo.size == snr.size and s_hi.size == snr.size:
        s_err = np.vstack(
            [
                np.maximum(0.0, snr - s_lo),
                np.maximum(0.0, s_hi - snr),
            ]
        )
        axes[1].errorbar(x_us, snr, yerr=s_err, marker="o", capsize=3)
    else:
        axes[1].plot(x_us, snr, marker="o")
    axes[1].axvline(best["readout_length"] * 1e6, linestyle="--", color="gray")
    axes[1].set_ylabel("SNR (delta_mu_over_sigma)")
    axes[1].set_xlabel("Readout length (us)")
    axes[1].grid(alpha=0.25)

    fig.suptitle(f"Readout length sweep (quality={metrics['quality_flag']})", fontsize=12)
    workflow.save_artifact("readout_length_sweep_metrics", fig)
    return fig

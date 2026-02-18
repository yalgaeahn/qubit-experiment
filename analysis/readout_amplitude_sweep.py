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


@workflow.task
def calculate_metrics(
    result: RunExperimentResults,
    qubit: QuantumElement,
    amplitudes: ArrayLike,
    ridge_target_condition: float = 1e6,
    fidelity_tolerance: float = 5e-4,
    bootstrap_samples: int = 400,
    bootstrap_confidence_level: float = 0.95,
    bootstrap_seed: int | None = None,
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
    fidelity_ci_low = np.zeros(len(amplitude_points), dtype=float)
    fidelity_ci_high = np.zeros(len(amplitude_points), dtype=float)
    snr_ci_low = np.zeros(len(amplitude_points), dtype=float)
    snr_ci_high = np.zeros(len(amplitude_points), dtype=float)
    rng = np.random.default_rng(bootstrap_seed)
    for i in range(len(amplitude_points)):
        metric = evaluate_iq_binary(
            shots_g=g_shots[i],
            shots_e=e_shots[i],
            target_condition=float(ridge_target_condition),
        )
        fidelity[i] = float(metric["assignment_fidelity"])
        snr[i] = float(metric["delta_mu_over_sigma"])
        ci = _bootstrap_ci(
            shots_g=np.asarray(g_shots[i], dtype=complex).reshape(-1),
            shots_e=np.asarray(e_shots[i], dtype=complex).reshape(-1),
            target_condition=float(ridge_target_condition),
            samples=int(max(0, bootstrap_samples)),
            confidence_level=float(bootstrap_confidence_level),
            rng=rng,
        )
        fidelity_ci_low[i] = float(ci["fidelity_ci_low"])
        fidelity_ci_high[i] = float(ci["fidelity_ci_high"])
        snr_ci_low[i] = float(ci["snr_ci_low"])
        snr_ci_high[i] = float(ci["snr_ci_high"])

    global_best = select_best_index(
        assignment_fidelity=fidelity,
        delta_mu_over_sigma=snr,
        fidelity_tolerance=float(fidelity_tolerance),
        prefer_smallest=False,
    )
    best_idx = int(global_best["index"])
    quality_flag = str(global_best["quality_flag"])

    return {
        "sweep_parameter": "readout_amplitude",
        "sweep_points": amplitude_points,
        "metrics_vs_sweep": {
            "assignment_fidelity": fidelity,
            "delta_mu_over_sigma": snr,
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
def plot_metrics(
    metrics: dict,
    include_error_bars: bool = True,
) -> mpl.figure.Figure:
    """Plot readout-amplitude sweep metrics."""
    amplitudes = np.asarray(metrics["sweep_points"], dtype=float)
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
    if include_error_bars and f_lo.size == fidelity.size and f_hi.size == fidelity.size:
        f_err = np.vstack(
            [
                np.maximum(0.0, fidelity - f_lo),
                np.maximum(0.0, f_hi - fidelity),
            ]
        )
        axes[0].errorbar(amplitudes, fidelity, yerr=f_err, marker="o", capsize=3)
    else:
        axes[0].plot(amplitudes, fidelity, marker="o")
    axes[0].axvline(best["readout_amplitude"], linestyle="--", color="gray")
    axes[0].set_ylabel("Assignment fidelity")
    axes[0].grid(alpha=0.25)

    if include_error_bars and s_lo.size == snr.size and s_hi.size == snr.size:
        s_err = np.vstack(
            [
                np.maximum(0.0, snr - s_lo),
                np.maximum(0.0, s_hi - snr),
            ]
        )
        axes[1].errorbar(amplitudes, snr, yerr=s_err, marker="o", capsize=3)
    else:
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

"""Analysis workflow for readout resonator frequency optimization."""

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
from analysis.iq_cloud import (
    bootstrap_metrics,
    calculate_confusion_and_fidelity,
    calculate_separation_metrics,
    fit_decision_models,
    plot_assignment_matrices,
    plot_iq_clouds,
    select_assignment_fidelity,
    select_confusion_matrices,
)
from laboneq_applications.core.validation import validate_result

if TYPE_CHECKING:
    from collections.abc import Sequence

    import matplotlib as mpl
    from laboneq.dsl.quantum.quantum_element import QuantumElement
    from laboneq.workflow.tasks.run_experiment import RunExperimentResults
    from numpy.typing import ArrayLike


@workflow.workflow_options
class ReadoutFrequencySweepAnalysisOptions:
    """Options for readout-frequency sweep analysis."""

    do_plotting: bool = workflow.option_field(
        True, description="Whether to create plots."
    )
    do_plotting_metrics: bool = workflow.option_field(
        True, description="Whether to plot metric curves."
    )
    do_plotting_error_bars: bool = workflow.option_field(
        True, description="Whether to include bootstrap error bars in metric plots."
    )
    do_plotting_optimal_iq_cloud: bool = workflow.option_field(
        True,
        description=(
            "Whether to run IQ-cloud style plots at the selected optimal frequency."
        ),
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
    prefer_lower_frequency_within_tolerance: bool = workflow.option_field(
        False,
        description=(
            "If true, choose the lowest-frequency candidate among near-optimal points. "
            "If false, choose center-most candidate."
        ),
    )
    mid_weight: float = workflow.option_field(
        0.0,
        description=(
            "Weight for MID score in composite selection. "
            "0 disables MID penalty, 1 prioritizes MID only."
        ),
    )
    mid_penalty_scale: float = workflow.option_field(
        1.0,
        description="Scale factor used in MID-score normalization.",
    )


@workflow.workflow(name="analysis_readout_frequency_sweep")
def analysis_workflow(
    results: Sequence[RunExperimentResults],
    qubits: Sequence[QuantumElement],
    frequencies: ArrayLike,
    reference_qubit: QuantumElement,
    mid_penalty: ArrayLike | None = None,
    options: ReadoutFrequencySweepAnalysisOptions | None = None,
) -> None:
    """Analyze repeated IQ-cloud acquisitions across readout-frequency candidates."""
    options = ReadoutFrequencySweepAnalysisOptions() if options is None else options
    metrics = calculate_metrics(
        results=results,
        qubits=qubits,
        frequencies=frequencies,
        reference_qubit=reference_qubit,
        ridge_target_condition=options.ridge_target_condition,
        fidelity_tolerance=options.fidelity_tolerance,
        bootstrap_samples=options.bootstrap_samples,
        bootstrap_confidence_level=options.bootstrap_confidence_level,
        bootstrap_seed=options.bootstrap_seed,
        prefer_lower_frequency_within_tolerance=options.prefer_lower_frequency_within_tolerance,
        mid_penalty=mid_penalty,
        mid_weight=options.mid_weight,
        mid_penalty_scale=options.mid_penalty_scale,
    )
    with workflow.if_(options.do_plotting):
        with workflow.if_(options.do_plotting_metrics):
            plot_metrics(metrics=metrics, include_error_bars=options.do_plotting_error_bars)
        with workflow.if_(options.do_plotting_optimal_iq_cloud):
            best_index = extract_best_index(metrics=metrics)
            optimal_result = select_result_by_index(results=results, index=best_index)
            processed_data_opt = build_processed_data_for_optimal_frequency(
                result=optimal_result,
                qubit_uid=reference_qubit.uid,
            )
            plot_optimal_iq_cloud_bundle(
                processed_data=processed_data_opt,
                reference_qubit=reference_qubit,
                metrics=metrics,
                ridge_target_condition=options.ridge_target_condition,
                bootstrap_samples=options.bootstrap_samples,
                bootstrap_confidence_level=options.bootstrap_confidence_level,
                bootstrap_seed=options.bootstrap_seed,
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
            "fidelity_mean": float(point["assignment_fidelity"]),
            "fidelity_ci_low": float(point["assignment_fidelity"]),
            "fidelity_ci_high": float(point["assignment_fidelity"]),
            "snr_mean": float(point["delta_mu_over_sigma"]),
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
        "fidelity_mean": float(np.mean(f_vals)),
        "fidelity_ci_low": float(np.quantile(f_vals, alpha)),
        "fidelity_ci_high": float(np.quantile(f_vals, 1.0 - alpha)),
        "snr_mean": float(np.mean(s_vals)),
        "snr_ci_low": float(np.quantile(s_vals, alpha)),
        "snr_ci_high": float(np.quantile(s_vals, 1.0 - alpha)),
    }


@workflow.task(save=False)
def calculate_metrics(
    results: Sequence[RunExperimentResults],
    qubits: Sequence[QuantumElement],
    frequencies: ArrayLike,
    reference_qubit: QuantumElement,
    ridge_target_condition: float = 1e6,
    fidelity_tolerance: float = 5e-4,
    bootstrap_samples: int = 400,
    bootstrap_confidence_level: float = 0.95,
    bootstrap_seed: int | None = None,
    prefer_lower_frequency_within_tolerance: bool = False,
    mid_penalty: ArrayLike | None = None,
    mid_weight: float = 0.0,
    mid_penalty_scale: float = 1.0,
) -> dict:
    """Compute metric curves over frequency candidates and choose optimum."""
    frequency_points = np.asarray(frequencies, dtype=float).reshape(-1)
    if frequency_points.size < 1:
        raise ValueError("frequencies must contain at least one point.")
    if len(results) != len(frequency_points):
        raise ValueError(
            "results and frequencies size mismatch: "
            f"{len(results)} != {len(frequency_points)}."
        )
    if len(qubits) != len(frequency_points):
        raise ValueError(
            "qubits and frequencies size mismatch: "
            f"{len(qubits)} != {len(frequency_points)}."
        )

    fidelity = np.zeros(len(frequency_points), dtype=float)
    snr = np.zeros(len(frequency_points), dtype=float)
    fidelity_ci_low = np.zeros(len(frequency_points), dtype=float)
    fidelity_ci_high = np.zeros(len(frequency_points), dtype=float)
    snr_ci_low = np.zeros(len(frequency_points), dtype=float)
    snr_ci_high = np.zeros(len(frequency_points), dtype=float)
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

    fidelity_threshold = float(fidelity_tolerance)
    mid_penalty_values = (
        np.asarray(mid_penalty, dtype=float).reshape(-1)
        if mid_penalty is not None
        else None
    )
    use_mid = (
        mid_penalty_values is not None
        and mid_penalty_values.size == frequency_points.size
        and float(mid_weight) > 0.0
    )
    if use_mid:
        mid_penalty_values = np.where(
            np.isfinite(mid_penalty_values), mid_penalty_values, np.inf
        )

    best = select_best_index(
        assignment_fidelity=fidelity,
        delta_mu_over_sigma=snr,
        fidelity_tolerance=float(fidelity_tolerance),
        prefer_smallest=bool(prefer_lower_frequency_within_tolerance),
    )
    best_idx = int(best["index"])
    if use_mid and np.isfinite(mid_penalty_values).any():
        candidate_mask = np.isfinite(fidelity) & np.isfinite(mid_penalty_values)
        candidates = np.where(candidate_mask)[0]
        if candidates.size == 0:
            candidates = np.array([best_idx], dtype=int)

        # start from fidelity-first shortlist, then apply MID-aware scoring
        # allow ties by fidelity tolerance (same behavior as base selection)
        max_fid = float(np.max(fidelity))
        candidate_mask = fidelity >= max_fid - fidelity_threshold
        fidelity_set = np.where(candidate_mask & np.isfinite(mid_penalty_values))[0]
        if fidelity_set.size > 0:
            candidates = fidelity_set

        mid_finite = mid_penalty_values[candidates]
        snr_sub = snr[candidates]
        valid_mid = np.isfinite(mid_finite)
        if np.count_nonzero(valid_mid) > 0:
            finite_mid = mid_finite[valid_mid]
            finite_snr = snr_sub[valid_mid]
            if finite_mid.size > 1:
                mid_min = float(np.min(finite_mid))
                mid_max = float(np.max(finite_mid))
                if float(mid_max - mid_min) > 0.0:
                    mid_norm = (finite_mid - mid_min) / (mid_max - mid_min)
                else:
                    mid_norm = np.zeros_like(finite_mid, dtype=float)
            else:
                mid_norm = np.zeros_like(finite_mid, dtype=float)
            # larger mid score is better (lower MID)
            mid_score = 1.0 - mid_norm
            mid_score = np.maximum(0.0, np.minimum(1.0, mid_score / float(mid_penalty_scale)))

            if finite_snr.size > 1:
                snr_min = float(np.min(finite_snr))
                snr_max = float(np.max(finite_snr))
                if float(snr_max - snr_min) > 0.0:
                    snr_norm = (finite_snr - snr_min) / (snr_max - snr_min)
                else:
                    snr_norm = np.zeros_like(finite_snr, dtype=float)
            else:
                snr_norm = np.zeros_like(finite_snr, dtype=float)
            composite = (1.0 - float(mid_weight)) * snr_norm + float(mid_weight) * mid_score
            best_local = int(np.argmax(composite))
            best_mid_idx = int(candidates[valid_mid][best_local])
            best_idx = int(best_mid_idx)

            if 0 <= best_mid_idx < len(fidelity):
                best = {
                    "index": best_idx,
                    "max_assignment_fidelity": float(np.max(fidelity)),
                    "quality_flag": "composite_mid",
                    "num_candidates": int(candidates.size),
                }

    best_frequency = float(frequency_points[best_idx])

    return {
        "sweep_parameter": "readout_resonator_frequency",
        "sweep_points": frequency_points,
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
            "index": best_idx,
            "readout_resonator_frequency": best_frequency,
            "assignment_fidelity": float(fidelity[best_idx]),
            "delta_mu_over_sigma": float(snr[best_idx]),
        },
        "quality_flag": str(best["quality_flag"]),
        "old_parameter_values": {
            reference_qubit.uid: {
                "readout_resonator_frequency": (
                    reference_qubit.parameters.readout_resonator_frequency
                ),
            }
        },
        "new_parameter_values": {
            reference_qubit.uid: {
                "readout_resonator_frequency": best_frequency,
            }
        },
        "mid_penalty": None if mid_penalty is None else np.asarray(mid_penalty, dtype=float).tolist(),
    }


@workflow.task(save=False)
def extract_best_index(metrics: dict) -> int:
    """Extract best sweep index from metric payload."""
    return int(metrics["best_point"]["index"])


@workflow.task(save=False)
def select_result_by_index(results: Sequence[RunExperimentResults], index: int):
    """Select one result from the sweep-result sequence."""
    idx = int(index)
    if idx < 0 or idx >= len(results):
        raise IndexError(f"best-point index {idx} out of range for {len(results)} results.")
    return results[idx]


@workflow.task(save=False)
def build_processed_data_for_optimal_frequency(result, qubit_uid: str) -> dict:
    """Build iq_cloud-compatible payload from optimal-frequency calibration traces."""
    res = unwrap_result_like(result)
    validate_result(res)
    shots = calibration_shots_by_state(
        result=res,
        qubit_uid=str(qubit_uid),
        states=("g", "e"),
    )
    return {
        "prepared_labels": ["g", "e"],
        "shots_per_qubit": {
            str(qubit_uid): {
                "g": np.asarray(shots["g"], dtype=complex).reshape(-1).tolist(),
                "e": np.asarray(shots["e"], dtype=complex).reshape(-1).tolist(),
            }
        },
    }


@workflow.task(save=False)
def annotate_optimal_frequency_figures(
    iq_cloud_figures: dict,
    assignment_figures: dict,
    metrics: dict,
) -> None:
    """Annotate and save IQ-cloud artifacts with optimal-frequency context."""
    best = metrics["best_point"]
    f_hz = float(best["readout_resonator_frequency"])
    f_ghz = f_hz / 1e9
    text = f"Optimal readout frequency: {f_ghz:.6f} GHz"

    for uid, fig in (iq_cloud_figures or {}).items():
        fig.suptitle(text, fontsize=11, y=1.02)
        fig.tight_layout()
        workflow.save_artifact(f"iq_cloud_{uid}_optimal_frequency", fig)

    assignment_fig = (assignment_figures or {}).get("assignment_matrices")
    if assignment_fig is not None:
        assignment_fig.suptitle(text, fontsize=11, y=1.02)
        assignment_fig.tight_layout()
        workflow.save_artifact("iq_cloud_assignment_matrices_optimal_frequency", assignment_fig)


@workflow.task(save=False)
def plot_optimal_iq_cloud_bundle(
    processed_data: dict,
    reference_qubit,
    metrics: dict,
    ridge_target_condition: float = 1e6,
    bootstrap_samples: int = 400,
    bootstrap_confidence_level: float = 0.95,
    bootstrap_seed: int | None = None,
) -> None:
    """Run optimal-point IQ-cloud analysis in one save-disabled task to avoid serializer issues."""
    qubits = [reference_qubit]
    decision_model = fit_decision_models.func(
        processed_data=processed_data,
        qubits=qubits,
        ridge_target_condition=ridge_target_condition,
    )
    assignment_bundle = calculate_confusion_and_fidelity.func(
        processed_data=processed_data,
        decision_model=decision_model,
        qubits=qubits,
    )
    confusion_matrices = select_confusion_matrices.func(
        assignment_bundle=assignment_bundle
    )
    assignment_fidelity = select_assignment_fidelity.func(
        assignment_bundle=assignment_bundle
    )
    separation_metrics = calculate_separation_metrics.func(
        decision_model=decision_model
    )
    optimal_bootstrap = bootstrap_metrics.func(
        processed_data=processed_data,
        qubits=qubits,
        ridge_target_condition=ridge_target_condition,
        bootstrap_samples=bootstrap_samples,
        bootstrap_confidence_level=bootstrap_confidence_level,
        bootstrap_seed=bootstrap_seed,
    )
    iq_cloud_figs = plot_iq_clouds.func(
        processed_data=processed_data,
        decision_model=decision_model,
        qubits=qubits,
        bootstrap=optimal_bootstrap,
    )
    assignment_figs = plot_assignment_matrices.func(
        confusion_matrices=confusion_matrices,
        assignment_fidelity=assignment_fidelity,
        qubits=qubits,
        separation_metrics=separation_metrics,
        bootstrap=optimal_bootstrap,
    )
    annotate_optimal_frequency_figures.func(
        iq_cloud_figures=iq_cloud_figs,
        assignment_figures=assignment_figs,
        metrics=metrics,
    )


@workflow.task(save=False)
def plot_metrics(metrics: dict, include_error_bars: bool = True) -> mpl.figure.Figure:
    """Plot readout-frequency sweep metrics."""
    frequency_points = np.asarray(metrics["sweep_points"], dtype=float)
    fidelity = np.asarray(metrics["metrics_vs_sweep"]["assignment_fidelity"], dtype=float)
    snr = np.asarray(metrics["metrics_vs_sweep"]["delta_mu_over_sigma"], dtype=float)
    bootstrap = metrics.get("bootstrap", {})
    f_ci = bootstrap.get("assignment_fidelity", {})
    s_ci = bootstrap.get("delta_mu_over_sigma", {})
    f_low = np.asarray(f_ci.get("ci_low", fidelity), dtype=float)
    f_high = np.asarray(f_ci.get("ci_high", fidelity), dtype=float)
    s_low = np.asarray(s_ci.get("ci_low", snr), dtype=float)
    s_high = np.asarray(s_ci.get("ci_high", snr), dtype=float)
    best = metrics["best_point"]

    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    x_ghz = frequency_points / 1e9

    if include_error_bars:
        f_err = np.vstack([np.maximum(0.0, fidelity - f_low), np.maximum(0.0, f_high - fidelity)])
        axes[0].errorbar(x_ghz, fidelity, yerr=f_err, fmt="o-", capsize=3)
    else:
        axes[0].plot(x_ghz, fidelity, marker="o")
    axes[0].axvline(best["readout_resonator_frequency"] / 1e9, linestyle="--", color="gray")
    axes[0].set_ylabel("Assignment fidelity")
    axes[0].grid(alpha=0.25)

    if include_error_bars:
        s_err = np.vstack([np.maximum(0.0, snr - s_low), np.maximum(0.0, s_high - snr)])
        axes[1].errorbar(x_ghz, snr, yerr=s_err, fmt="o-", capsize=3)
    else:
        axes[1].plot(x_ghz, snr, marker="o")
    axes[1].axvline(best["readout_resonator_frequency"] / 1e9, linestyle="--", color="gray")
    axes[1].set_ylabel("SNR (delta_mu_over_sigma)")
    axes[1].set_xlabel("Readout resonator frequency (GHz)")
    axes[1].grid(alpha=0.25)
    axes[1].text(
        0.01,
        0.02,
        (
            f"Optimal f_ro = {best['readout_resonator_frequency']/1e9:.6f} GHz, "
            f"F={best['assignment_fidelity']:.4f}, "
            f"SNR={best['delta_mu_over_sigma']:.3f}"
        ),
        transform=axes[1].transAxes,
        fontsize=9,
    )

    fig.suptitle(
        f"Readout frequency sweep (quality={metrics['quality_flag']})",
        fontsize=12,
    )
    workflow.save_artifact("readout_frequency_sweep_metrics", fig)
    return fig

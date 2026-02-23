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
    fidelity_floor: float = workflow.option_field(
        0.95,
        description=(
            "Minimum assignment fidelity floor used to select best readout length. "
            "The smallest length with fidelity >= fidelity_floor is selected."
        ),
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
    requested_readout_lengths: ArrayLike | None = None,
    dropped_points: ArrayLike | None = None,
    dropped_reasons: Sequence[str] | None = None,
    point_metrics: Sequence[dict] | None = None,
    options: ReadoutLengthSweepAnalysisOptions | None = None,
) -> None:
    """Analyze repeated IQ-cloud acquisitions across readout-length candidates."""
    options = ReadoutLengthSweepAnalysisOptions() if options is None else options
    metrics = calculate_metrics(
        results=results,
        qubits=qubits,
        readout_lengths=readout_lengths,
        reference_qubit=reference_qubit,
        requested_readout_lengths=requested_readout_lengths,
        dropped_points=dropped_points,
        dropped_reasons=dropped_reasons,
        point_metrics=point_metrics,
        ridge_target_condition=options.ridge_target_condition,
        fidelity_floor=options.fidelity_floor,
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


def _finite_float_or_none(value) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _require_finite_float(value, *, key: str, index: int) -> float:
    out = _finite_float_or_none(value)
    if out is None:
        raise ValueError(
            "point_metrics contains non-finite or missing value for "
            f"key={key!r} at point index={index}."
        )
    return float(out)


def _integration_window_from_qubit(qubit: QuantumElement) -> tuple[float, float]:
    delay = _finite_float_or_none(
        getattr(qubit.parameters, "readout_integration_delay", None)
    )
    length = _finite_float_or_none(
        getattr(qubit.parameters, "readout_integration_length", None)
    )
    delay = 0.0 if delay is None else float(delay)
    if length is None or length <= 0.0:
        raise ValueError(
            "readout_integration_length must be positive and finite for "
            f"qubit={qubit.uid!r}, got {length!r}."
        )
    return delay, float(length)


def _result_supports_calibration_shots(result_obj, qubit_uid: str) -> bool:
    try:
        calibration_shots_by_state(
            result=result_obj,
            qubit_uid=qubit_uid,
            states=("g", "e"),
        )
    except Exception:
        return False
    return True


def _coerce_run_result_for_qubit(result_like, qubit_uid: str):
    """Find an object that supports calibration handle access for the target qubit."""
    queue = [result_like]
    seen: set[int] = set()
    seen_types: set[str] = set()
    attrs = ("output", "result", "analysis_result", "analysis_workflow_result", "tasks")

    while queue:
        current = queue.pop(0)
        if current is None:
            continue
        oid = id(current)
        if oid in seen:
            continue
        seen.add(oid)
        seen_types.add(type(current).__name__)

        if _result_supports_calibration_shots(current, qubit_uid):
            return current

        if isinstance(current, dict):
            for k in attrs:
                if k in current:
                    queue.append(current[k])
            queue.extend(current.values())
            continue

        if isinstance(current, (list, tuple, set)):
            queue.extend(list(current))
            continue

        for a in attrs:
            if hasattr(current, a):
                try:
                    queue.append(getattr(current, a))
                except Exception:
                    pass

        d = getattr(current, "__dict__", None)
        if isinstance(d, dict):
            queue.extend(d.values())

    raise TypeError(
        "Could not resolve a RunExperimentResults-like object with calibration handles "
        f"for qubit={qubit_uid!r}. Encountered types={sorted(seen_types)!r}."
    )


@workflow.task(save=False)
def calculate_metrics(
    results: Sequence[RunExperimentResults],
    qubits: Sequence[QuantumElement],
    readout_lengths: ArrayLike,
    reference_qubit: QuantumElement,
    requested_readout_lengths: ArrayLike | None = None,
    dropped_points: ArrayLike | None = None,
    dropped_reasons: Sequence[str] | None = None,
    point_metrics: Sequence[dict] | None = None,
    ridge_target_condition: float = 1e6,
    fidelity_floor: float = 0.95,
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

    requested_points = (
        np.asarray(requested_readout_lengths, dtype=float).reshape(-1)
        if requested_readout_lengths is not None
        else np.asarray(length_points, dtype=float).reshape(-1)
    )
    dropped_points_arr = (
        np.asarray(dropped_points, dtype=float).reshape(-1)
        if dropped_points is not None
        else np.asarray([], dtype=float)
    )
    dropped_reasons_list = (
        list(dropped_reasons)
        if dropped_reasons is not None
        else ["flat_window_detection_failed"] * int(dropped_points_arr.size)
    )
    if len(dropped_reasons_list) != int(dropped_points_arr.size):
        raise ValueError(
            "dropped_points and dropped_reasons size mismatch: "
            f"{int(dropped_points_arr.size)} != {len(dropped_reasons_list)}."
        )

    fidelity = np.zeros(len(length_points), dtype=float)
    snr = np.zeros(len(length_points), dtype=float)
    integration_delay = np.zeros(len(length_points), dtype=float)
    integration_length = np.zeros(len(length_points), dtype=float)
    fidelity_ci_low = np.zeros(len(length_points), dtype=float)
    fidelity_ci_high = np.zeros(len(length_points), dtype=float)
    snr_ci_low = np.zeros(len(length_points), dtype=float)
    snr_ci_high = np.zeros(len(length_points), dtype=float)

    if point_metrics is not None:
        if len(point_metrics) != len(length_points):
            raise ValueError(
                "point_metrics and readout_lengths size mismatch: "
                f"{len(point_metrics)} != {len(length_points)}."
            )

        for i, metric in enumerate(point_metrics):
            if not isinstance(metric, dict):
                raise ValueError(
                    "Each point_metrics entry must be a dict, got "
                    f"type={type(metric)!r} at index={i}."
                )

            point_length = _finite_float_or_none(metric.get("readout_length"))
            if point_length is not None:
                atol = max(1e-15, abs(float(length_points[i])) * 1e-9)
                if not np.isclose(point_length, float(length_points[i]), rtol=0.0, atol=atol):
                    raise ValueError(
                        "point_metrics readout_length mismatch at index "
                        f"{i}: {point_length:.3e}s != {float(length_points[i]):.3e}s."
                    )

            fidelity[i] = _require_finite_float(
                metric.get("assignment_fidelity"),
                key="assignment_fidelity",
                index=i,
            )
            snr[i] = _require_finite_float(
                metric.get("delta_mu_over_sigma"),
                key="delta_mu_over_sigma",
                index=i,
            )
            integration_delay[i] = _require_finite_float(
                metric.get("readout_integration_delay"),
                key="readout_integration_delay",
                index=i,
            )
            integration_length[i] = _require_finite_float(
                metric.get("readout_integration_length"),
                key="readout_integration_length",
                index=i,
            )
            if integration_length[i] <= 0.0:
                raise ValueError(
                    "point_metrics readout_integration_length must be positive at "
                    f"index={i}, got {integration_length[i]!r}."
                )

            f_lo = _finite_float_or_none(metric.get("fidelity_ci_low"))
            f_hi = _finite_float_or_none(metric.get("fidelity_ci_high"))
            s_lo = _finite_float_or_none(metric.get("snr_ci_low"))
            s_hi = _finite_float_or_none(metric.get("snr_ci_high"))
            fidelity_ci_low[i] = fidelity[i] if f_lo is None else float(f_lo)
            fidelity_ci_high[i] = fidelity[i] if f_hi is None else float(f_hi)
            snr_ci_low[i] = snr[i] if s_lo is None else float(s_lo)
            snr_ci_high[i] = snr[i] if s_hi is None else float(s_hi)

            if not np.isfinite(fidelity_ci_low[i]):
                fidelity_ci_low[i] = fidelity[i]
            if not np.isfinite(fidelity_ci_high[i]):
                fidelity_ci_high[i] = fidelity[i]
            if not np.isfinite(snr_ci_low[i]):
                snr_ci_low[i] = snr[i]
            if not np.isfinite(snr_ci_high[i]):
                snr_ci_high[i] = snr[i]

    else:
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

        rng = np.random.default_rng(bootstrap_seed)
        for i, (result_like, qubit) in enumerate(zip(results, qubits)):
            delay, length = _integration_window_from_qubit(qubit)
            integration_delay[i] = delay
            integration_length[i] = length

            result = _coerce_run_result_for_qubit(result_like, qubit.uid)
            try:
                validate_result(result)
            except TypeError:
                pass
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

    floor = float(fidelity_floor)
    if not np.isfinite(floor):
        raise ValueError(f"fidelity_floor must be finite, got {fidelity_floor!r}.")
    if floor < 0.0 or floor > 1.0:
        raise ValueError(
            f"fidelity_floor must be within [0, 1], got {fidelity_floor!r}."
        )

    floor_mask = np.isfinite(fidelity) & (fidelity >= floor)
    if np.any(floor_mask):
        candidate_idx = np.where(floor_mask)[0]
        candidate_lengths = length_points[candidate_idx]
        min_length = float(np.min(candidate_lengths))
        min_len_idx = candidate_idx[np.isclose(candidate_lengths, min_length)]
        if min_len_idx.size > 1:
            # Tie-break at same minimum length by fidelity, then SNR.
            local_fid = fidelity[min_len_idx]
            best_local = int(np.nanargmax(local_fid))
            tied = min_len_idx[np.isclose(local_fid, local_fid[best_local])]
            if tied.size > 1:
                local_snr = snr[tied]
                best_local_snr = int(np.nanargmax(local_snr))
                best_idx = int(tied[best_local_snr])
            else:
                best_idx = int(tied[0])
        else:
            best_idx = int(min_len_idx[0])
        quality_flag = "fidelity_floor_met"
    else:
        best = select_best_index(
            assignment_fidelity=fidelity,
            delta_mu_over_sigma=snr,
            fidelity_tolerance=float(fidelity_tolerance),
            prefer_smallest=bool(prefer_shorter_within_tolerance),
        )
        best_idx = int(best["index"])
        quality_flag = f"fidelity_floor_unmet|{best['quality_flag']}"

    best_length = float(length_points[best_idx])
    best_delay = float(integration_delay[best_idx])
    best_int_length = float(integration_length[best_idx])

    return {
        "sweep_parameter": "readout_length",
        "requested_sweep_points": requested_points,
        "sweep_points": length_points,
        "dropped_points": dropped_points_arr,
        "dropped_reasons": dropped_reasons_list,
        "num_dropped": int(dropped_points_arr.size),
        "metrics_vs_sweep": {
            "assignment_fidelity": fidelity,
            "delta_mu_over_sigma": snr,
            "readout_integration_delay": integration_delay,
            "readout_integration_length": integration_length,
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
            "readout_integration_delay": best_delay,
            "readout_integration_length": best_int_length,
            "assignment_fidelity": float(fidelity[best_idx]),
            "delta_mu_over_sigma": float(snr[best_idx]),
        },
        "quality_flag": quality_flag,
        "old_parameter_values": {
            reference_qubit.uid: {
                "readout_length": reference_qubit.parameters.readout_length,
                "readout_integration_delay": (
                    reference_qubit.parameters.readout_integration_delay
                ),
                "readout_integration_length": (
                    reference_qubit.parameters.readout_integration_length
                ),
            }
        },
        "new_parameter_values": {
            reference_qubit.uid: {
                "readout_length": best_length,
                "readout_integration_delay": best_delay,
                "readout_integration_length": best_int_length,
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

"""Canonical three-qubit QST analysis under the `threeq_qst` name.

This workflow keeps the INTEGRATION + SINGLE_SHOT contract and returns a
single plain analysis payload for one tomography run. Convergence and shot
sweep helpers are re-exported for the split experiment workflows.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from laboneq import workflow

from .plot_theme import with_plot_theme
from .three_qubit_state_tomography import (
    calculate_state_metrics,
    collect_tomography_counts,
    evaluate_optimization_convergence,
    extract_assignment_matrix,
    fit_discriminator_from_readout_calibration,
    maximum_likelihood_reconstruct,
    plot_counts,
    plot_density_matrix,
)

if TYPE_CHECKING:
    import matplotlib as mpl
    from laboneq.workflow.tasks.run_experiment import RunExperimentResults


DEFAULT_PRODUCT_SUITE_STATES: tuple[str, ...] = (
    "000",
    "001",
    "010",
    "011",
    "100",
    "101",
    "110",
    "111",
    "+++",
    "++-",
    "+-+",
    "+--",
    "-++",
    "-+-",
    "--+",
    "---",
)
DEFAULT_SHOT_SWEEP_LOG2_VALUES: tuple[int, ...] = tuple(range(3, 13))
SHOT_SWEEP_EPS: float = 1e-12
SHOT_SWEEP_INFID_TOL: float = 1e-9


@workflow.workflow_options
class ThreeQQstAnalysisOptions:
    """Options for canonical single-run 3Q QST analysis."""

    do_plotting: bool = workflow.option_field(
        True,
        description="Whether to generate density-matrix and counts plots.",
    )
    max_mle_iterations: int = workflow.option_field(
        2000,
        description="Maximum iterations for MLE optimization.",
    )


def _build_analysis_payload_impl(
    tomography_result: RunExperimentResults,
    q0_uid: str,
    q1_uid: str,
    q2_uid: str,
    readout_calibration_result: RunExperimentResults | None = None,
    target_state=None,
    max_iterations: int = 2000,
) -> dict[str, object]:
    discriminator = fit_discriminator_from_readout_calibration.func(
        readout_calibration_result=readout_calibration_result,
        q0_uid=q0_uid,
        q1_uid=q1_uid,
        q2_uid=q2_uid,
    )
    assignment = extract_assignment_matrix.func(
        readout_calibration_result=readout_calibration_result,
        q0_uid=q0_uid,
        q1_uid=q1_uid,
        q2_uid=q2_uid,
        discriminator=discriminator,
    )
    tomography_counts = collect_tomography_counts.func(
        tomography_result=tomography_result,
        q0_uid=q0_uid,
        q1_uid=q1_uid,
        q2_uid=q2_uid,
        discriminator=discriminator,
    )
    mle_result = maximum_likelihood_reconstruct.func(
        tomography_counts=tomography_counts,
        assignment=assignment,
        max_iterations=max_iterations,
    )
    state_metrics = calculate_state_metrics.func(
        rho_hat_real=mle_result["rho_hat_real"],
        rho_hat_imag=mle_result["rho_hat_imag"],
        target_state=target_state,
    )
    optimization_convergence = evaluate_optimization_convergence.func(
        tomography_counts=tomography_counts,
        predicted_counts=mle_result["predicted_counts"],
        optimizer_success=mle_result["optimizer_success"],
        optimizer_message=mle_result["optimizer_message"],
        negative_log_likelihood=mle_result["negative_log_likelihood"],
        rho_hat_real=mle_result["rho_hat_real"],
        rho_hat_imag=mle_result["rho_hat_imag"],
    )

    return {
        "assignment_matrix": assignment["assignment_matrix"],
        "assignment_counts": assignment["counts_matrix_soft"],
        "assignment_counts_soft": assignment["counts_matrix_soft"],
        "assignment_counts_hard": assignment["counts_matrix_hard"],
        "tomography_counts": tomography_counts["counts"],
        "tomography_counts_hard": tomography_counts["counts_hard"],
        "setting_labels": tomography_counts["setting_labels"],
        "shots_per_setting": tomography_counts["shots_per_setting"],
        "rho_hat_real": mle_result["rho_hat_real"],
        "rho_hat_imag": mle_result["rho_hat_imag"],
        "predicted_probabilities": mle_result["predicted_probabilities"],
        "predicted_counts": mle_result["predicted_counts"],
        "optimizer_success": mle_result["optimizer_success"],
        "optimizer_message": mle_result["optimizer_message"],
        "negative_log_likelihood": mle_result["negative_log_likelihood"],
        "metrics": state_metrics,
        "discriminator_model": discriminator["model"],
        "classification_diagnostics": discriminator["diagnostics"],
        "optimization_convergence": optimization_convergence,
    }


@workflow.task
def analyze_tomography_run(
    tomography_result: RunExperimentResults,
    q0_uid: str,
    q1_uid: str,
    q2_uid: str,
    readout_calibration_result: RunExperimentResults | None = None,
    target_state=None,
    max_iterations: int = 2000,
) -> dict[str, object]:
    """Build the full single-run analysis payload without plotting."""
    return _build_analysis_payload_impl(
        tomography_result=tomography_result,
        q0_uid=q0_uid,
        q1_uid=q1_uid,
        q2_uid=q2_uid,
        readout_calibration_result=readout_calibration_result,
        target_state=target_state,
        max_iterations=max_iterations,
    )


@workflow.workflow(name="analysis_threeq_qst")
def analysis_workflow(
    tomography_result: RunExperimentResults,
    q0,
    q1,
    q2,
    readout_calibration_result: RunExperimentResults | None = None,
    target_state=None,
    options: ThreeQQstAnalysisOptions | None = None,
) -> None:
    """Run readout-mitigated MLE analysis for one 3Q QST dataset."""
    opts = ThreeQQstAnalysisOptions() if options is None else options

    analysis_payload = analyze_tomography_run(
        tomography_result=tomography_result,
        q0_uid=q0.uid,
        q1_uid=q1.uid,
        q2_uid=q2.uid,
        readout_calibration_result=readout_calibration_result,
        target_state=target_state,
        max_iterations=opts.max_mle_iterations,
    )

    with workflow.if_(opts.do_plotting):
        plot_density_matrix(
            rho_hat_real=analysis_payload["rho_hat_real"],
            rho_hat_imag=analysis_payload["rho_hat_imag"],
        )
        plot_counts(
            observed_counts=analysis_payload["tomography_counts"],
            predicted_counts=analysis_payload["predicted_counts"],
            setting_labels=analysis_payload["setting_labels"],
        )

    workflow.return_(analysis_payload)


def _unwrap_analysis_output(result_like):
    current = result_like
    for _ in range(8):
        if current is None:
            return None
        if hasattr(current, "output"):
            current = current.output
            continue
        return current
    return current


def _materialize_analysis_output(result_like) -> dict[str, object] | None:
    current = _unwrap_analysis_output(result_like)
    return dict(current) if isinstance(current, dict) else None


def _safe_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_bool(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return bool(value)
    return False


def _to_float_or_nan(value) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return numeric if np.isfinite(numeric) else float("nan")


@workflow.task
def collect_convergence_run_record(
    state_label: str,
    repeat_index: int,
    analysis_result,
) -> dict[str, object]:
    """Extract compact per-run convergence record from a 3Q analysis output."""
    out = _materialize_analysis_output(analysis_result)
    if not isinstance(out, dict):
        return {
            "state_label": str(state_label),
            "repeat_index": int(repeat_index),
            "fidelity_to_target": None,
            "optimizer_success": False,
            "negative_log_likelihood": None,
            "rho_min_eigenvalue": None,
            "nll_finite": False,
            "nll_per_shot": None,
            "mae_counts": None,
            "max_abs_counts_error": None,
            "normalized_mae_counts": None,
        }

    metrics = out.get("metrics", {}) if isinstance(out.get("metrics"), dict) else {}
    opt = (
        out.get("optimization_convergence", {})
        if isinstance(out.get("optimization_convergence"), dict)
        else {}
    )
    negative_log_likelihood = out.get(
        "negative_log_likelihood",
        opt.get("negative_log_likelihood"),
    )
    rho_min_eigenvalue = metrics.get("min_eigenvalue", opt.get("min_eigenvalue"))
    optimizer_success = out.get("optimizer_success", opt.get("optimizer_success", False))

    return {
        "state_label": str(state_label),
        "repeat_index": int(repeat_index),
        "fidelity_to_target": _safe_float(metrics.get("fidelity_to_target")),
        "optimizer_success": _safe_bool(optimizer_success),
        "negative_log_likelihood": _safe_float(negative_log_likelihood),
        "rho_min_eigenvalue": _safe_float(rho_min_eigenvalue),
        "nll_finite": _safe_bool(opt.get("nll_finite", False)),
        "nll_per_shot": _safe_float(opt.get("nll_per_shot")),
        "mae_counts": _safe_float(opt.get("mae_counts")),
        "max_abs_counts_error": _safe_float(opt.get("max_abs_counts_error")),
        "normalized_mae_counts": _safe_float(opt.get("normalized_mae_counts")),
    }


@workflow.task
def extract_main_run_optimization_convergence(
    analysis_result,
) -> dict[str, object] | None:
    """Extract optimization convergence payload from the main analysis run."""
    out = _materialize_analysis_output(analysis_result)
    if not isinstance(out, dict):
        return None
    convergence = out.get("optimization_convergence")
    return convergence if isinstance(convergence, dict) else None


def _finite_stats(values) -> dict[str, float | int | None]:
    finite_values = []
    for value in values:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric):
            finite_values.append(numeric)
    arr = np.asarray(finite_values, dtype=float)
    n = int(arr.size)
    if n == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "sem": None,
            "ci95": None,
        }
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    sem = float(std / np.sqrt(n)) if n > 1 else 0.0
    ci95 = float(1.96 * sem) if n > 1 else 0.0
    return {
        "count": n,
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci95": ci95,
    }


def _summarize_statistical_convergence_impl(
    run_records: list[dict[str, object]],
) -> dict[str, object]:
    records = run_records or []
    grouped: dict[str, list[dict[str, object]]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        state_label = str(record.get("state_label", "unknown"))
        grouped.setdefault(state_label, []).append(record)

    per_state: dict[str, dict[str, object]] = {}
    all_fidelities: list[float] = []
    success_flags: list[float] = []
    min_eigs: list[float] = []

    for state, items in grouped.items():
        fidelities: list[float] = []
        nlls: list[float] = []
        state_min_eigs: list[float] = []
        state_success: list[float] = []

        for item in items:
            fidelity = item.get("fidelity_to_target")
            if fidelity is not None and np.isfinite(float(fidelity)):
                fidelities.append(float(fidelity))
                all_fidelities.append(float(fidelity))

            nll = item.get("negative_log_likelihood")
            if nll is not None and np.isfinite(float(nll)):
                nlls.append(float(nll))

            eig = item.get("rho_min_eigenvalue")
            if eig is not None and np.isfinite(float(eig)):
                state_min_eigs.append(float(eig))
                min_eigs.append(float(eig))

            success_value = 1.0 if bool(item.get("optimizer_success", False)) else 0.0
            state_success.append(success_value)
            success_flags.append(success_value)

        fidelity_stats = _finite_stats(fidelities)
        nll_stats = _finite_stats(nlls)
        eig_stats = _finite_stats(state_min_eigs)
        per_state[state] = {
            "num_runs": len(items),
            "num_valid_fidelity_runs": int(fidelity_stats["count"]),
            "optimizer_success_rate": (
                float(np.mean(state_success)) if state_success else 0.0
            ),
            "fidelity_mean": fidelity_stats["mean"],
            "fidelity_std": fidelity_stats["std"],
            "fidelity_sem": fidelity_stats["sem"],
            "fidelity_ci95": fidelity_stats["ci95"],
            "nll_mean": nll_stats["mean"],
            "nll_std": nll_stats["std"],
            "rho_min_eigenvalue_mean": eig_stats["mean"],
            "rho_min_eigenvalue_min": (
                float(np.min(state_min_eigs)) if state_min_eigs else None
            ),
        }

    all_fidelity_stats = _finite_stats(all_fidelities)
    aggregate = {
        "num_total_runs": len(records),
        "overall_optimizer_success_rate": (
            float(np.mean(success_flags)) if success_flags else 0.0
        ),
        "pooled_fidelity_mean": all_fidelity_stats["mean"],
        "pooled_fidelity_std": all_fidelity_stats["std"],
        "pooled_fidelity_sem": all_fidelity_stats["sem"],
        "pooled_fidelity_ci95": all_fidelity_stats["ci95"],
        "worst_rho_min_eigenvalue": float(np.min(min_eigs)) if min_eigs else None,
    }
    return {
        "per_state": per_state,
        "aggregate": aggregate,
    }


@workflow.task
def summarize_statistical_convergence(
    run_records: list[dict[str, object]],
) -> dict[str, object]:
    """Aggregate repeated-run convergence statistics across a state suite."""
    return _summarize_statistical_convergence_impl(run_records)


@workflow.task
def collect_shot_sweep_run_record(
    state_label: str,
    log2_shots: int,
    shots: int,
    repeat: int,
    tomography_result: RunExperimentResults,
    q0_uid: str,
    q1_uid: str,
    q2_uid: str,
    readout_calibration_result: RunExperimentResults | None,
    target_state=None,
    max_iterations: int = 2000,
    eps: float = SHOT_SWEEP_EPS,
) -> dict[str, object]:
    """Analyze one 3Q shot-sweep run and return a row plus failure metadata."""
    failure = None
    try:
        payload = analyze_tomography_run.func(
            tomography_result=tomography_result,
            q0_uid=q0_uid,
            q1_uid=q1_uid,
            q2_uid=q2_uid,
            readout_calibration_result=readout_calibration_result,
            target_state=target_state,
            max_iterations=max_iterations,
        )
        metrics = payload.get("metrics", {}) if isinstance(payload, dict) else {}
        fid_raw = metrics.get("fidelity_to_target")
        nll_raw = payload.get("negative_log_likelihood")
        min_eig_raw = metrics.get("min_eigenvalue")

        fid = _to_float_or_nan(fid_raw)
        if np.isfinite(fid):
            infid = max(float(eps), 1.0 - fid)
            log10_infid = float(np.log10(infid))
        else:
            infid = float("nan")
            log10_infid = float("nan")
            failure = {
                "state": str(state_label),
                "log2_shots": int(log2_shots),
                "shots": int(shots),
                "repeat": int(repeat),
                "reason": f"invalid fidelity: {fid_raw!r}",
            }

        row = {
            "state": str(state_label),
            "log2_shots": int(log2_shots),
            "shots": int(shots),
            "repeat": int(repeat),
            "fidelity": fid,
            "infidelity": infid,
            "log10_infidelity": log10_infid,
            "nll": _to_float_or_nan(nll_raw),
            "min_eig": _to_float_or_nan(min_eig_raw),
        }
    except Exception as exc:
        row = {
            "state": str(state_label),
            "log2_shots": int(log2_shots),
            "shots": int(shots),
            "repeat": int(repeat),
            "fidelity": float("nan"),
            "infidelity": float("nan"),
            "log10_infidelity": float("nan"),
            "nll": float("nan"),
            "min_eig": float("nan"),
        }
        failure = {
            "state": str(state_label),
            "log2_shots": int(log2_shots),
            "shots": int(shots),
            "repeat": int(repeat),
            "reason": repr(exc),
        }

    return {"record": row, "failure": failure}


def _validate_shot_sweep_run_records_impl(
    run_records: list[dict[str, object]],
    suite_states: tuple[str, ...] | list[str],
    shot_log2_values: tuple[int, ...] | list[int],
    repeats_per_point: int,
    eps: float = SHOT_SWEEP_EPS,
    infid_tol: float = SHOT_SWEEP_INFID_TOL,
) -> dict[str, object]:
    counts_by_group: dict[tuple[str, int], int] = defaultdict(int)
    violations: list[dict[str, object]] = []
    for row in run_records or []:
        if not isinstance(row, dict):
            continue
        state = str(row.get("state", ""))
        log2_shots = int(row.get("log2_shots", -1))
        counts_by_group[(state, log2_shots)] += 1

        infidelity = _to_float_or_nan(row.get("infidelity"))
        if np.isfinite(infidelity) and (
            infidelity < (float(eps) - 1e-15)
            or infidelity > (1.0 + float(infid_tol))
        ):
            violations.append(dict(row))

    expected_pairs = [
        (str(state), int(log2_shots))
        for state in suite_states
        for log2_shots in shot_log2_values
    ]
    missing_groups = [
        {"state": state, "log2_shots": log2_shots}
        for state, log2_shots in expected_pairs
        if (state, log2_shots) not in counts_by_group
    ]
    bad_repeat_groups = [
        {
            "state": state,
            "log2_shots": log2_shots,
            "observed_repeats": int(count),
            "expected_repeats": int(repeats_per_point),
        }
        for (state, log2_shots), count in sorted(counts_by_group.items())
        if int(count) != int(repeats_per_point)
    ]
    return {
        "expected_group_count": int(len(expected_pairs)),
        "observed_group_count": int(len(counts_by_group)),
        "missing_groups": missing_groups,
        "bad_repeat_groups": bad_repeat_groups,
        "infidelity_range_violations": violations,
    }


@workflow.task
def validate_shot_sweep_run_records(
    run_records: list[dict[str, object]],
    suite_states: tuple[str, ...] | list[str],
    shot_log2_values: tuple[int, ...] | list[int],
    repeats_per_point: int,
    eps: float = SHOT_SWEEP_EPS,
    infid_tol: float = SHOT_SWEEP_INFID_TOL,
) -> dict[str, object]:
    """Validate shot-sweep coverage and infidelity range constraints."""
    return _validate_shot_sweep_run_records_impl(
        run_records=run_records,
        suite_states=suite_states,
        shot_log2_values=shot_log2_values,
        repeats_per_point=repeats_per_point,
        eps=eps,
        infid_tol=infid_tol,
    )


def _aggregate_shot_sweep_statistics_impl(
    run_records: list[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in run_records or []:
        if not isinstance(row, dict):
            continue
        grouped[(str(row.get("state", "")), int(row.get("log2_shots", -1)))].append(
            row
        )

    aggregated_rows: list[dict[str, object]] = []
    for (state, log2_shots), items in sorted(
        grouped.items(), key=lambda item: (item[0][0], item[0][1])
    ):
        if not items:
            continue
        inf_stats = _finite_stats(
            [
                row.get("infidelity")
                for row in items
                if np.isfinite(_to_float_or_nan(row.get("infidelity")))
            ]
        )
        log_stats = _finite_stats(
            [
                row.get("log10_infidelity")
                for row in items
                if np.isfinite(_to_float_or_nan(row.get("log10_infidelity")))
            ]
        )
        aggregated_rows.append(
            {
                "state": state,
                "log2_shots": int(log2_shots),
                "shots": int(items[0].get("shots", 0)),
                "n_total": int(len(items)),
                "n_valid_infidelity": int(inf_stats["count"]),
                "infid_mean": inf_stats["mean"],
                "infid_std": inf_stats["std"],
                "infid_sem": inf_stats["sem"],
                "infid_ci95": inf_stats["ci95"],
                "log10_infid_mean": log_stats["mean"],
                "log10_infid_std": log_stats["std"],
                "log10_infid_sem": log_stats["sem"],
                "log10_infid_ci95": log_stats["ci95"],
            }
        )
    return aggregated_rows


@workflow.task
def aggregate_shot_sweep_statistics(
    run_records: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Aggregate shot-sweep rows into per-state and per-shot summaries."""
    return _aggregate_shot_sweep_statistics_impl(run_records)


def _summarize_final_shot_sweep_impl(
    aggregated_stats: list[dict[str, object]],
    shot_log2_values: tuple[int, ...] | list[int],
) -> list[dict[str, object]]:
    if not shot_log2_values:
        return []
    last_log2 = max(int(value) for value in shot_log2_values)
    rows = []
    for row in aggregated_stats or []:
        if not isinstance(row, dict):
            continue
        if int(row.get("log2_shots", -1)) != last_log2:
            continue
        rows.append(
            {
                "state": str(row.get("state", "")),
                "n_total": int(row.get("n_total", 0)),
                "n_valid_infidelity": int(row.get("n_valid_infidelity", 0)),
                "infid_mean": row.get("infid_mean"),
                "infid_ci95": row.get("infid_ci95"),
                "log10_infid_mean": row.get("log10_infid_mean"),
                "log10_infid_ci95": row.get("log10_infid_ci95"),
            }
        )
    rows.sort(key=lambda item: item["state"])
    return rows


@workflow.task
def summarize_final_shot_sweep(
    aggregated_stats: list[dict[str, object]],
    shot_log2_values: tuple[int, ...] | list[int],
) -> list[dict[str, object]]:
    """Summarize the largest-shot operating point from shot-sweep stats."""
    return _summarize_final_shot_sweep_impl(
        aggregated_stats=aggregated_stats,
        shot_log2_values=shot_log2_values,
    )


@workflow.task
@with_plot_theme
def plot_convergence_suite_fidelity(
    statistical_convergence: dict[str, object],
) -> dict[str, mpl.figure.Figure]:
    """Plot product-state suite fidelity mean ± 95% CI."""
    per_state = (
        statistical_convergence.get("per_state", {})
        if isinstance(statistical_convergence, dict)
        else {}
    )
    states = sorted(per_state.keys())
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    if states:
        means = np.array(
            [
                per_state[state].get("fidelity_mean")
                if per_state[state].get("fidelity_mean") is not None
                else np.nan
                for state in states
            ],
            dtype=float,
        )
        errs = np.array(
            [
                per_state[state].get("fidelity_ci95")
                if per_state[state].get("fidelity_ci95") is not None
                else np.nan
                for state in states
            ],
            dtype=float,
        )
        x = np.arange(len(states))
        ax.errorbar(x, means, yerr=errs, fmt="o", capsize=4)
        ax.set_xticks(x, states, rotation=45)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Fidelity")
    ax.set_title("3Q product-state suite fidelity mean ± 95% CI")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    workflow.save_artifact("threeq_qst_convergence_fidelity", fig)
    return {"convergence_fidelity": fig}


@workflow.task
@with_plot_theme
def plot_shot_sweep_summary(
    aggregated_stats: list[dict[str, object]],
    suite_states: tuple[str, ...] | list[str],
) -> dict[str, mpl.figure.Figure]:
    """Plot infidelity and log10-infidelity versus log2(shots)."""
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in aggregated_stats or []:
        if not isinstance(row, dict):
            continue
        grouped[str(row.get("state", ""))].append(row)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    for state in suite_states:
        state_rows = sorted(
            grouped.get(str(state), []),
            key=lambda row: int(row.get("log2_shots", -1)),
        )
        if not state_rows:
            continue
        x = np.asarray(
            [row.get("log2_shots", np.nan) for row in state_rows],
            dtype=float,
        )

        y_infid = np.asarray(
            [row.get("infid_mean", np.nan) for row in state_rows],
            dtype=float,
        )
        e_infid = np.asarray(
            [row.get("infid_ci95", np.nan) for row in state_rows],
            dtype=float,
        )
        mask_infid = np.isfinite(y_infid)
        if np.any(mask_infid):
            axes[0].plot(
                x[mask_infid],
                y_infid[mask_infid],
                "o--",
                label=f"|{state}><{state}|",
            )
            low = y_infid[mask_infid] - np.nan_to_num(e_infid[mask_infid], nan=0.0)
            high = y_infid[mask_infid] + np.nan_to_num(e_infid[mask_infid], nan=0.0)
            axes[0].fill_between(x[mask_infid], low, high, alpha=0.2)

        y_log = np.asarray(
            [row.get("log10_infid_mean", np.nan) for row in state_rows],
            dtype=float,
        )
        e_log = np.asarray(
            [row.get("log10_infid_ci95", np.nan) for row in state_rows],
            dtype=float,
        )
        mask_log = np.isfinite(y_log)
        if np.any(mask_log):
            axes[1].plot(
                x[mask_log],
                y_log[mask_log],
                "o--",
                label=f"|{state}><{state}|",
            )
            low = y_log[mask_log] - np.nan_to_num(e_log[mask_log], nan=0.0)
            high = y_log[mask_log] + np.nan_to_num(e_log[mask_log], nan=0.0)
            axes[1].fill_between(x[mask_log], low, high, alpha=0.2)

    axes[0].set_title("Infidelity vs Log(Number of shots)")
    axes[0].set_xlabel("Log(Number of shots) = log2(shots)")
    axes[0].set_ylabel("Infidelity")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=9)

    axes[1].set_title("Log10(Infidelity) vs Log(Number of shots)")
    axes[1].set_xlabel("Log(Number of shots) = log2(shots)")
    axes[1].set_ylabel("Log10(Infidelity)")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=9)

    fig.suptitle("3Q QST Product-State Suite Shot Sweep", y=1.02, fontsize=14)
    fig.tight_layout()
    workflow.save_artifact("threeq_qst_shot_sweep_summary", fig)
    return {"shot_sweep_summary": fig}


__all__ = [
    "DEFAULT_PRODUCT_SUITE_STATES",
    "DEFAULT_SHOT_SWEEP_LOG2_VALUES",
    "SHOT_SWEEP_EPS",
    "SHOT_SWEEP_INFID_TOL",
    "ThreeQQstAnalysisOptions",
    "analysis_workflow",
    "analyze_tomography_run",
    "collect_convergence_run_record",
    "collect_shot_sweep_run_record",
    "extract_main_run_optimization_convergence",
    "summarize_statistical_convergence",
    "validate_shot_sweep_run_records",
    "aggregate_shot_sweep_statistics",
    "summarize_final_shot_sweep",
    "plot_convergence_suite_fidelity",
    "plot_shot_sweep_summary",
    "_summarize_statistical_convergence_impl",
    "_validate_shot_sweep_run_records_impl",
    "_aggregate_shot_sweep_statistics_impl",
    "_summarize_final_shot_sweep_impl",
]

"""Three-qubit virtual-Z validation analysis helpers.

This module evaluates tomography results from
`experiments.three_qubit_virtual_z_validation` for two validation stages:
product-state local phase tracking and GHZ-tail coherence phase tracking.
It reuses the canonical 3Q MLE tomography path, extracts phase-sensitive
observables, aggregates per-target tracking metrics, and saves summary plots.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np
from laboneq import workflow

from qubit_experiment.experiments.three_qubit_tomography_common import (
    canonical_three_qubit_state_label,
)

from .plot_theme import with_plot_theme
from .three_qubit_state_tomography import analyze_tomography_run

if TYPE_CHECKING:
    import matplotlib as mpl
    from laboneq.workflow.tasks.run_experiment import RunExperimentResults


PHASE_TARGETS: tuple[str, ...] = ("q0", "q1", "q2")
VALIDATION_STAGES: tuple[str, ...] = ("product", "ghz")


@workflow.workflow_options
class ThreeQVZValidationAnalysisOptions:
    """Options for 3Q virtual-Z validation summaries."""

    do_plotting: bool = workflow.option_field(
        True,
        description="Whether to generate phase-tracking summary plots.",
    )
    max_mle_iterations: int = workflow.option_field(
        2000,
        description="Maximum iterations for shared tomography MLE reconstruction.",
    )


def _coerce_analysis_options(
    analysis_options: ThreeQVZValidationAnalysisOptions | None,
) -> ThreeQVZValidationAnalysisOptions:
    if analysis_options is None:
        return ThreeQVZValidationAnalysisOptions()
    if isinstance(analysis_options, ThreeQVZValidationAnalysisOptions):
        return analysis_options

    base = getattr(analysis_options, "_base", None)
    if isinstance(base, ThreeQVZValidationAnalysisOptions):
        return base

    raise TypeError(
        "analysis_options must be a ThreeQVZValidationAnalysisOptions instance or "
        "an options() builder."
    )


@workflow.task(save=False)
def resolve_analysis_max_mle_iterations(
    analysis_options: ThreeQVZValidationAnalysisOptions | None = None,
) -> int:
    """Resolve the analysis iteration count in a workflow-safe helper."""
    return int(_coerce_analysis_options(analysis_options).max_mle_iterations)


@workflow.task(save=False)
def resolve_do_plotting(
    analysis_options: ThreeQVZValidationAnalysisOptions | None = None,
) -> bool:
    """Resolve the plotting flag in a workflow-safe helper."""
    return bool(_coerce_analysis_options(analysis_options).do_plotting)


def _normalize_stage(stage: str) -> Literal["product", "ghz"]:
    text = str(stage).strip().lower()
    if text not in VALIDATION_STAGES:
        raise ValueError(
            "stage must be one of ('product', 'ghz'). "
            f"Received {stage!r}."
        )
    return text  # type: ignore[return-value]


def _phase_target_index(phase_target: str) -> int:
    text = str(phase_target).strip().lower()
    try:
        return PHASE_TARGETS.index(text)
    except ValueError as exc:
        raise ValueError(
            "phase_target must be one of ('q0', 'q1', 'q2'). "
            f"Received {phase_target!r}."
        ) from exc


def _token_statevector(token: str) -> np.ndarray:
    if token == "0":
        return np.array([1.0, 0.0], dtype=complex)
    if token == "1":
        return np.array([0.0, 1.0], dtype=complex)
    if token == "+":
        return np.array([1.0, 1.0], dtype=complex) / np.sqrt(2.0)
    if token == "-":
        return np.array([1.0, -1.0], dtype=complex) / np.sqrt(2.0)
    raise ValueError(f"Unsupported single-qubit token {token!r}.")


def _apply_virtual_z_to_statevector(statevector: np.ndarray, phase_value: float) -> np.ndarray:
    rz = np.diag(
        [
            np.exp(-0.5j * float(phase_value)),
            np.exp(0.5j * float(phase_value)),
        ]
    )
    return rz @ np.asarray(statevector, dtype=complex)


def _product_target_statevector(
    phase_target: str,
    phase_value: float,
    product_initial_state: str,
) -> np.ndarray:
    canonical = canonical_three_qubit_state_label(product_initial_state)
    target_index = _phase_target_index(phase_target)

    vectors: list[np.ndarray] = []
    for index, token in enumerate(canonical):
        vector = _token_statevector(token)
        if index == target_index:
            vector = _apply_virtual_z_to_statevector(vector, float(phase_value))
        vectors.append(vector)

    psi = np.kron(np.kron(vectors[0], vectors[1]), vectors[2])
    norm = np.linalg.norm(psi)
    if norm == 0.0:
        raise ValueError("Product target statevector has zero norm.")
    return psi / norm


def _ghz_target_statevector(phase_value: float) -> np.ndarray:
    psi = np.zeros(8, dtype=complex)
    psi[0] = 1.0 / np.sqrt(2.0)
    psi[7] = np.exp(1j * float(phase_value)) / np.sqrt(2.0)
    return psi


def _rho_from_payload(payload: dict[str, object]) -> np.ndarray:
    rho_real = np.asarray(payload["rho_hat_real"], dtype=float)
    rho_imag = np.asarray(payload["rho_hat_imag"], dtype=float)
    rho = rho_real + 1j * rho_imag
    rho = (rho + rho.conj().T) / 2.0
    trace = np.trace(rho)
    if np.abs(trace) > 0:
        rho = rho / trace
    return rho


def _bits_to_index(bits: list[int]) -> int:
    return (bits[0] << 2) | (bits[1] << 1) | bits[2]


def _single_qubit_reduced_density_matrix(rho: np.ndarray, target_index: int) -> np.ndarray:
    others = [index for index in range(3) if index != int(target_index)]
    reduced = np.zeros((2, 2), dtype=complex)
    for bra_target in (0, 1):
        for ket_target in (0, 1):
            total = 0.0j
            for bit0 in (0, 1):
                for bit1 in (0, 1):
                    row_bits = [0, 0, 0]
                    col_bits = [0, 0, 0]
                    row_bits[target_index] = bra_target
                    col_bits[target_index] = ket_target
                    row_bits[others[0]] = bit0
                    col_bits[others[0]] = bit0
                    row_bits[others[1]] = bit1
                    col_bits[others[1]] = bit1
                    total += rho[_bits_to_index(row_bits), _bits_to_index(col_bits)]
            reduced[bra_target, ket_target] = total
    return reduced


def _wrapped_phase_difference(measured_phase: float, expected_phase: float) -> float:
    return float(
        np.angle(
            np.exp(
                1j * (float(measured_phase) - float(expected_phase))
            )
        )
    )


def _to_float_or_nan(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _extract_product_phase_impl(
    rho: np.ndarray,
    phase_target: str,
) -> tuple[float, float]:
    reduced = _single_qubit_reduced_density_matrix(rho, _phase_target_index(phase_target))
    coherence = complex(reduced[1, 0])
    return float(np.angle(coherence)), float(np.abs(coherence))


def _extract_ghz_phase_impl(rho: np.ndarray) -> tuple[float, float]:
    coherence = complex(rho[7, 0])
    return float(np.angle(coherence)), float(np.abs(coherence))


def _virtual_z_target_statevector(
    stage: str,
    phase_target: str,
    phase_value: float,
    product_initial_state: str,
) -> np.ndarray:
    normalized_stage = _normalize_stage(stage)
    if normalized_stage == "product":
        return _product_target_statevector(
            phase_target=phase_target,
            phase_value=phase_value,
            product_initial_state=product_initial_state,
        )
    return _ghz_target_statevector(phase_value)


def _record_failure(
    *,
    stage: str,
    phase_target: str,
    phase_value: float,
    repeat: int,
    reason: str,
) -> dict[str, object]:
    return {
        "stage": str(stage),
        "phase_target": str(phase_target),
        "phase_value": float(phase_value),
        "repeat": int(repeat),
        "reason": str(reason),
    }


def collect_virtual_z_run_record_impl(
    *,
    stage: str,
    phase_target: str,
    phase_value: float,
    repeat: int,
    tomography_result: RunExperimentResults,
    q0_uid: str,
    q1_uid: str,
    q2_uid: str,
    readout_calibration_result: RunExperimentResults | None,
    product_initial_state: str = "+++",
    max_iterations: int = 2000,
) -> dict[str, object]:
    """Analyze one validation run and return a record plus optional failure metadata."""
    stage_name = _normalize_stage(stage)
    expected_phase = float(phase_value)
    failure = None

    try:
        target_state = _virtual_z_target_statevector(
            stage=stage_name,
            phase_target=phase_target,
            phase_value=expected_phase,
            product_initial_state=product_initial_state,
        )
        payload = analyze_tomography_run.func(
            tomography_result=tomography_result,
            q0_uid=q0_uid,
            q1_uid=q1_uid,
            q2_uid=q2_uid,
            readout_calibration_result=readout_calibration_result,
            target_state=target_state,
            max_iterations=max_iterations,
        )
        rho = _rho_from_payload(payload)
        metrics = payload.get("metrics", {}) if isinstance(payload, dict) else {}
        optimization = (
            payload.get("optimization_convergence", {})
            if isinstance(payload, dict)
            else {}
        )

        if stage_name == "product":
            measured_phase, coherence_magnitude = _extract_product_phase_impl(
                rho=rho,
                phase_target=phase_target,
            )
        else:
            measured_phase, coherence_magnitude = _extract_ghz_phase_impl(rho)

        wrapped_phase_error = _wrapped_phase_difference(measured_phase, expected_phase)
        fidelity = _to_float_or_nan(metrics.get("fidelity_to_target"))
        optimizer_success = bool(
            optimization.get("optimizer_success", payload.get("optimizer_success", False))
        )
        negative_log_likelihood = _to_float_or_nan(
            payload.get("negative_log_likelihood")
        )
        min_eig = _to_float_or_nan(metrics.get("min_eigenvalue"))

        row = {
            "stage": stage_name,
            "phase_target": str(phase_target),
            "phase_value": expected_phase,
            "expected_phase": expected_phase,
            "repeat": int(repeat),
            "measured_phase": float(measured_phase),
            "wrapped_phase_error": float(wrapped_phase_error),
            "fidelity": fidelity,
            "coherence_magnitude": float(coherence_magnitude),
            "optimizer_success": optimizer_success,
            "nll": negative_log_likelihood,
            "min_eig": min_eig,
        }
        if not np.isfinite(measured_phase):
            failure = _record_failure(
                stage=stage_name,
                phase_target=phase_target,
                phase_value=expected_phase,
                repeat=repeat,
                reason="measured_phase is not finite",
            )
    except Exception as exc:  # pragma: no cover - exercised via task wrapper tests
        row = {
            "stage": stage_name,
            "phase_target": str(phase_target),
            "phase_value": expected_phase,
            "expected_phase": expected_phase,
            "repeat": int(repeat),
            "measured_phase": float("nan"),
            "wrapped_phase_error": float("nan"),
            "fidelity": float("nan"),
            "coherence_magnitude": float("nan"),
            "optimizer_success": False,
            "nll": float("nan"),
            "min_eig": float("nan"),
        }
        failure = _record_failure(
            stage=stage_name,
            phase_target=phase_target,
            phase_value=expected_phase,
            repeat=repeat,
            reason=repr(exc),
        )

    return {"record": row, "failure": failure}


@workflow.task
def collect_virtual_z_run_record(
    stage: str,
    phase_target: str,
    phase_value: float,
    repeat: int,
    tomography_result: RunExperimentResults,
    q0_uid: str,
    q1_uid: str,
    q2_uid: str,
    readout_calibration_result: RunExperimentResults | None,
    product_initial_state: str = "+++",
    max_iterations: int = 2000,
) -> dict[str, object]:
    """Workflow task wrapper for one analyzed validation run."""
    return collect_virtual_z_run_record_impl(
        stage=stage,
        phase_target=phase_target,
        phase_value=phase_value,
        repeat=repeat,
        tomography_result=tomography_result,
        q0_uid=q0_uid,
        q1_uid=q1_uid,
        q2_uid=q2_uid,
        readout_calibration_result=readout_calibration_result,
        product_initial_state=product_initial_state,
        max_iterations=max_iterations,
    )


def _summarize_virtual_z_validation_impl(
    *,
    stage: str,
    run_records: list[dict[str, object]],
) -> dict[str, object]:
    stage_name = _normalize_stage(stage)
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in run_records or []:
        if not isinstance(row, dict):
            continue
        if str(row.get("stage", "")).lower() != stage_name:
            continue
        grouped[str(row.get("phase_target", ""))].append(row)

    per_target: dict[str, dict[str, object]] = {}
    for phase_target in PHASE_TARGETS:
        rows = sorted(
            grouped.get(phase_target, []),
            key=lambda item: (
                _to_float_or_nan(item.get("phase_value")),
                int(item.get("repeat", 0)),
            ),
        )
        x_values = np.asarray(
            [row.get("phase_value", np.nan) for row in rows],
            dtype=float,
        )
        measured = np.asarray(
            [row.get("measured_phase", np.nan) for row in rows],
            dtype=float,
        )
        wrapped_errors = np.asarray(
            [row.get("wrapped_phase_error", np.nan) for row in rows],
            dtype=float,
        )
        fidelities = np.asarray(
            [row.get("fidelity", np.nan) for row in rows],
            dtype=float,
        )
        coherences = np.asarray(
            [row.get("coherence_magnitude", np.nan) for row in rows],
            dtype=float,
        )

        fit_mask = np.isfinite(x_values) & np.isfinite(measured)
        if np.count_nonzero(fit_mask) >= 2:
            y_unwrapped = np.unwrap(measured[fit_mask])
            slope, intercept = np.polyfit(x_values[fit_mask], y_unwrapped, deg=1)
        else:
            slope = np.nan
            intercept = np.nan

        error_mask = np.isfinite(wrapped_errors)
        if np.any(error_mask):
            phase_rmse = float(
                np.sqrt(np.mean(np.square(wrapped_errors[error_mask])))
            )
        else:
            phase_rmse = np.nan

        fidelity_mask = np.isfinite(fidelities)
        coherence_mask = np.isfinite(coherences)
        per_target[phase_target] = {
            "slope": None if np.isnan(slope) else float(slope),
            "intercept": None if np.isnan(intercept) else float(intercept),
            "phase_rmse": None if np.isnan(phase_rmse) else float(phase_rmse),
            "mean_fidelity": (
                None if not np.any(fidelity_mask) else float(np.mean(fidelities[fidelity_mask]))
            ),
            "min_fidelity": (
                None if not np.any(fidelity_mask) else float(np.min(fidelities[fidelity_mask]))
            ),
            "mean_coherence_magnitude": (
                None
                if not np.any(coherence_mask)
                else float(np.mean(coherences[coherence_mask]))
            ),
            "n_valid_points": int(np.count_nonzero(fit_mask)),
            "n_total_runs": int(len(rows)),
        }

    return {
        "stage": stage_name,
        "per_target": per_target,
        "aggregate": {
            "num_total_runs": int(len(run_records or [])),
            "num_targets": int(len(per_target)),
        },
    }


@workflow.task
def summarize_virtual_z_validation(
    stage: str,
    run_records: list[dict[str, object]],
) -> dict[str, object]:
    """Aggregate per-target phase tracking metrics for one validation stage."""
    return _summarize_virtual_z_validation_impl(stage=stage, run_records=run_records)


def _iter_grouped_points(
    run_records: list[dict[str, object]],
    *,
    phase_target: str,
    key: str,
) -> tuple[np.ndarray, np.ndarray]:
    rows = sorted(
        [
            row
            for row in run_records or []
            if isinstance(row, dict) and str(row.get("phase_target", "")) == phase_target
        ],
        key=lambda item: (
            _to_float_or_nan(item.get("phase_value")),
            int(item.get("repeat", 0)),
        ),
    )
    if not rows:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    x = np.asarray([row.get("phase_value", np.nan) for row in rows], dtype=float)
    y = np.asarray([row.get(key, np.nan) for row in rows], dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


@workflow.task
@with_plot_theme
def plot_phase_tracking(
    product_run_records: list[dict[str, object]],
    ghz_run_records: list[dict[str, object]],
) -> dict[str, mpl.figure.Figure]:
    """Plot programmed versus measured phase for product and GHZ stages."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    for ax, stage_name, records in (
        (axes[0], "Product Stage", product_run_records),
        (axes[1], "GHZ Tail Stage", ghz_run_records),
    ):
        for phase_target in PHASE_TARGETS:
            x, y = _iter_grouped_points(records, phase_target=phase_target, key="measured_phase")
            if x.size == 0:
                continue
            ax.plot(x, np.unwrap(y), "o-", label=phase_target)
        ax.plot([-np.pi, np.pi], [-np.pi, np.pi], "--", color="black", linewidth=1)
        ax.set_title(stage_name)
        ax.set_xlabel("Programmed phase (rad)")
        ax.set_xlim(-np.pi, np.pi)
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=9)
    axes[0].set_ylabel("Measured phase (rad)")
    fig.tight_layout()
    workflow.save_artifact("three_qubit_virtual_z_validation_phase_tracking", fig)
    return {"phase_tracking": fig}


@workflow.task
@with_plot_theme
def plot_quality_summary(
    product_run_records: list[dict[str, object]],
    ghz_run_records: list[dict[str, object]],
) -> dict[str, mpl.figure.Figure]:
    """Plot fidelity and coherence magnitude versus programmed phase."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.0), sharex=True)
    layout = (
        (axes[0, 0], "Product Fidelity", product_run_records, "fidelity"),
        (axes[0, 1], "Product Coherence", product_run_records, "coherence_magnitude"),
        (axes[1, 0], "GHZ Fidelity", ghz_run_records, "fidelity"),
        (axes[1, 1], "GHZ Coherence", ghz_run_records, "coherence_magnitude"),
    )
    for ax, title, records, key in layout:
        for phase_target in PHASE_TARGETS:
            x, y = _iter_grouped_points(records, phase_target=phase_target, key=key)
            if x.size == 0:
                continue
            ax.plot(x, y, "o-", label=phase_target)
        ax.set_title(title)
        ax.set_xlim(-np.pi, np.pi)
        ax.grid(alpha=0.3)
        if key == "fidelity":
            ax.set_ylim(0.0, 1.05)
        else:
            ax.set_ylim(0.0, 0.55)
        ax.legend(frameon=False, fontsize=9)
    axes[0, 0].set_ylabel("Fidelity")
    axes[1, 0].set_ylabel("Fidelity")
    axes[0, 1].set_ylabel("|coherence|")
    axes[1, 1].set_ylabel("|coherence|")
    axes[1, 0].set_xlabel("Programmed phase (rad)")
    axes[1, 1].set_xlabel("Programmed phase (rad)")
    fig.tight_layout()
    workflow.save_artifact("three_qubit_virtual_z_validation_quality_summary", fig)
    return {"quality_summary": fig}


__all__ = [
    "PHASE_TARGETS",
    "VALIDATION_STAGES",
    "ThreeQVZValidationAnalysisOptions",
    "collect_virtual_z_run_record",
    "collect_virtual_z_run_record_impl",
    "plot_phase_tracking",
    "plot_quality_summary",
    "resolve_analysis_max_mle_iterations",
    "resolve_do_plotting",
    "summarize_virtual_z_validation",
    "_extract_ghz_phase_impl",
    "_extract_product_phase_impl",
    "_ghz_target_statevector",
    "_product_target_statevector",
    "_summarize_virtual_z_validation_impl",
]

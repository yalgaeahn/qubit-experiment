"""GHZ-specific three-qubit QST analysis under the `three_qubit_ghz` name.

This workflow reuses the canonical 3Q tomography MLE payload but fixes the
target state to GHZ+. It returns a plain single-run analysis payload and GHZ
repeat-convergence helpers for the split experiment workflows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from laboneq import workflow

from qubit_experiment.experiments.three_qubit_tomography_common import OUTCOME_LABELS

from .plot_theme import with_plot_theme
from .threeq_qst import (
    _build_analysis_payload_impl,
    collect_convergence_run_record,
    extract_main_run_optimization_convergence,
    summarize_statistical_convergence,
)

if TYPE_CHECKING:
    import matplotlib as mpl
    from laboneq.workflow.tasks.run_experiment import RunExperimentResults


@workflow.workflow_options
class ThreeQGhzAnalysisOptions:
    """Options for GHZ-specific single-run 3Q tomography analysis."""

    do_plotting: bool = workflow.option_field(
        True,
        description="Whether to generate density-matrix and counts plots.",
    )
    max_mle_iterations: int = workflow.option_field(
        2000,
        description="Maximum iterations for MLE optimization.",
    )


@workflow.task
def analyze_tomography_run(
    tomography_result: RunExperimentResults,
    q0_uid: str,
    q1_uid: str,
    q2_uid: str,
    readout_calibration_result: RunExperimentResults | None = None,
    max_iterations: int = 2000,
) -> dict[str, object]:
    """Build the full single-run GHZ analysis payload without plotting."""
    return _build_analysis_payload_impl(
        tomography_result=tomography_result,
        q0_uid=q0_uid,
        q1_uid=q1_uid,
        q2_uid=q2_uid,
        readout_calibration_result=readout_calibration_result,
        target_state="ghz",
        max_iterations=max_iterations,
    )


@workflow.task
@with_plot_theme
def plot_density_matrix(
    rho_hat_real: list[list[float]],
    rho_hat_imag: list[list[float]],
) -> dict[str, mpl.figure.Figure]:
    """Plot real/imaginary parts of reconstructed GHZ density matrix."""
    rho_real = np.asarray(rho_hat_real, dtype=float)
    rho_imag = np.asarray(rho_hat_imag, dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels = [f"|{s}>" for s in OUTCOME_LABELS]

    im0 = axes[0].imshow(rho_real, cmap="RdBu_r")
    axes[0].set_title("Re[rho]")
    axes[0].set_xticks(range(8), labels, rotation=45)
    axes[0].set_yticks(range(8), labels)
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(rho_imag, cmap="RdBu_r")
    axes[1].set_title("Im[rho]")
    axes[1].set_xticks(range(8), labels, rotation=45)
    axes[1].set_yticks(range(8), labels)
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    fig.tight_layout()
    workflow.save_artifact("three_qubit_ghz_density_matrix", fig)
    return {"density_matrix": fig}


@workflow.task
@with_plot_theme
def plot_counts(
    observed_counts: list[list[int]],
    predicted_counts: list[list[float]],
    setting_labels: list[str],
) -> dict[str, mpl.figure.Figure]:
    """Plot observed and MLE-predicted counts for each GHZ setting/outcome."""
    observed = np.asarray(observed_counts, dtype=float)
    predicted = np.asarray(predicted_counts, dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    im0 = axes[0].imshow(observed, aspect="auto", cmap="viridis")
    axes[0].set_title("Observed Counts")
    axes[0].set_xlabel("Outcome")
    axes[0].set_ylabel("Setting")
    axes[0].set_xticks(range(8), OUTCOME_LABELS, rotation=45)
    axes[0].set_yticks(range(len(setting_labels)), setting_labels)
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(predicted, aspect="auto", cmap="viridis")
    axes[1].set_title("Predicted Counts (MLE)")
    axes[1].set_xlabel("Outcome")
    axes[1].set_xticks(range(8), OUTCOME_LABELS, rotation=45)
    axes[1].set_yticks(range(len(setting_labels)), setting_labels)
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    fig.tight_layout()
    workflow.save_artifact("three_qubit_ghz_counts", fig)
    return {"counts": fig}


@workflow.workflow(name="analysis_three_qubit_ghz")
def analysis_workflow(
    tomography_result: RunExperimentResults,
    q0,
    q1,
    q2,
    readout_calibration_result: RunExperimentResults | None = None,
    options: ThreeQGhzAnalysisOptions | None = None,
) -> None:
    """Run readout-mitigated GHZ analysis for one 3Q tomography dataset."""
    opts = ThreeQGhzAnalysisOptions() if options is None else options

    analysis_payload = analyze_tomography_run(
        tomography_result=tomography_result,
        q0_uid=q0.uid,
        q1_uid=q1.uid,
        q2_uid=q2.uid,
        readout_calibration_result=readout_calibration_result,
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


@workflow.task
@with_plot_theme
def plot_convergence_fidelity(
    statistical_convergence: dict[str, object],
) -> dict[str, mpl.figure.Figure]:
    """Plot GHZ fidelity mean ± 95% CI for repeated runs."""
    per_state = (
        statistical_convergence.get("per_state", {})
        if isinstance(statistical_convergence, dict)
        else {}
    )
    states = sorted(per_state.keys())
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
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
        ax.set_xticks(x, states)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Fidelity")
    ax.set_title("3Q GHZ fidelity mean +/- 95% CI")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    workflow.save_artifact("three_qubit_ghz_convergence_fidelity", fig)
    return {"convergence_fidelity": fig}


__all__ = [
    "ThreeQGhzAnalysisOptions",
    "analysis_workflow",
    "analyze_tomography_run",
    "collect_convergence_run_record",
    "extract_main_run_optimization_convergence",
    "summarize_statistical_convergence",
    "plot_convergence_fidelity",
]

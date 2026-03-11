"""Canonical two-qubit QST analysis under the `twoq_qst` name.

This workflow keeps the INTEGRATION + SINGLE_SHOT contract and returns a
single plain analysis payload for one tomography run. Convergence and shot
sweep helpers are re-exported for the split experiment workflows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from laboneq import workflow

from .two_qubit_qst import (
    DEFAULT_PRODUCT_SUITE_STATES,
    DEFAULT_SHOT_SWEEP_LOG2_VALUES,
    SHOT_SWEEP_EPS,
    SHOT_SWEEP_INFID_TOL,
    TwoQQstAnalysisOptions,
    _aggregate_shot_sweep_statistics_impl,
    _summarize_final_shot_sweep_impl,
    _summarize_statistical_convergence_impl,
    _validate_shot_sweep_run_records_impl,
    aggregate_shot_sweep_statistics,
    analyze_tomography_run,
    collect_convergence_run_record,
    collect_shot_sweep_run_record,
    extract_main_run_optimization_convergence,
    plot_convergence_suite_fidelity,
    plot_counts,
    plot_density_matrix,
    plot_shot_sweep_summary,
    summarize_final_shot_sweep,
    summarize_statistical_convergence,
    validate_shot_sweep_run_records,
)

if TYPE_CHECKING:
    from laboneq.workflow.tasks.run_experiment import RunExperimentResults


@workflow.workflow(name="analysis_twoq_qst")
def analysis_workflow(
    tomography_result: RunExperimentResults,
    ctrl,
    targ,
    readout_calibration_result: RunExperimentResults | None = None,
    target_state=None,
    options: TwoQQstAnalysisOptions | None = None,
) -> None:
    """Run readout-mitigated MLE analysis for one 2Q QST dataset."""
    opts = TwoQQstAnalysisOptions() if options is None else options

    analysis_payload = analyze_tomography_run(
        tomography_result=tomography_result,
        ctrl_uid=ctrl.uid,
        targ_uid=targ.uid,
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


__all__ = [
    "DEFAULT_PRODUCT_SUITE_STATES",
    "DEFAULT_SHOT_SWEEP_LOG2_VALUES",
    "SHOT_SWEEP_EPS",
    "SHOT_SWEEP_INFID_TOL",
    "TwoQQstAnalysisOptions",
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

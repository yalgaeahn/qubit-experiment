"""Canonical two-qubit QST workflows under the `twoq_qst` name.

This module splits raw execution, convergence validation, and shot-sweep
studies into separate workflows. Each workflow returns only top-level payloads;
`run_bundle(...)` assembles notebook-friendly nested dictionaries in Python.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from laboneq import workflow
from laboneq.simple import Experiment, dsl
from laboneq.workflow.tasks import compile_experiment, run_experiment
from laboneq_applications.tasks.parameter_updating import (
    temporary_qpu,
    temporary_quantum_elements_from_qpu,
)

from qubit_experiment.analysis.twoq_qst import (
    SHOT_SWEEP_EPS,
    SHOT_SWEEP_INFID_TOL,
    TwoQQstAnalysisOptions,
    aggregate_shot_sweep_statistics,
    analysis_workflow,
    analyze_tomography_run,
    collect_convergence_run_record,
    collect_shot_sweep_run_record,
    plot_convergence_suite_fidelity,
    plot_shot_sweep_summary,
    summarize_final_shot_sweep,
    summarize_statistical_convergence,
    validate_shot_sweep_run_records,
)

from .two_qubit_qst import (
    TwoQQstExperimentOptions,
    TwoQQstWorkflowOptions,
    _append_item,
    _append_item_if_present,
    _bundle_readout_calibration_result,
    _create_experiment_impl,
    _extract_readout_calibration_result,
    _materialize_list,
    _materialize_readout_calibration_bundle,
    _normalize_two_qubits,
    _select_qubit_for_analysis,
    create_readout_calibration_experiment,
    resolve_convergence_repeat_indices,
    resolve_convergence_repeats_per_state,
    resolve_convergence_suite_states,
    resolve_shot_sweep_counts,
    resolve_shot_sweep_log2_values,
    resolve_shot_sweep_repeat_indices,
    resolve_shot_sweep_repeats_per_point,
    resolve_shot_sweep_suite_states,
    resolve_target_configuration,
    shot_count_from_log2,
    should_run_readout_calibration,
)

if TYPE_CHECKING:
    from laboneq.dsl.quantum import QuantumParameters
    from laboneq.dsl.quantum.qpu import QPU
    from laboneq.dsl.session import Session
    from laboneq_applications.typing import QuantumElements


def _coerce_workflow_options(
    options: TwoQQstWorkflowOptions | None,
) -> TwoQQstWorkflowOptions:
    if options is None:
        return TwoQQstWorkflowOptions()
    if isinstance(options, TwoQQstWorkflowOptions):
        return options

    base = getattr(options, "_base", None)
    if isinstance(base, TwoQQstWorkflowOptions):
        return base

    raise TypeError(
        "options must be a TwoQQstWorkflowOptions instance or an "
        "experiment_workflow.options() builder."
    )


def _coerce_analysis_options(
    analysis_options: TwoQQstAnalysisOptions | None,
) -> TwoQQstAnalysisOptions:
    if analysis_options is None:
        return TwoQQstAnalysisOptions()
    if isinstance(analysis_options, TwoQQstAnalysisOptions):
        return analysis_options

    base = getattr(analysis_options, "_base", None)
    if isinstance(base, TwoQQstAnalysisOptions):
        return base

    raise TypeError(
        "analysis_options must be a TwoQQstAnalysisOptions instance or an "
        "analysis_workflow.options() builder."
    )


@workflow.task
@dsl.qubit_experiment
def create_experiment(
    qpu: QPU,
    qubits: QuantumElements,
    bus: QuantumElements,
    custom_prep: bool = False,
    initial_state: str = "++",
    count_override: int | None = None,
    options: TwoQQstExperimentOptions | None = None,
) -> Experiment:
    """Create a 2Q QST experiment without RIP preparation."""
    return _create_experiment_impl(
        qpu=qpu,
        qubits=qubits,
        bus=bus,
        custom_prep=custom_prep,
        initial_state=initial_state,
        count_override=count_override,
        options=options,
    )


@workflow.task(save=False)
def _resolve_analysis_max_mle_iterations(
    analysis_options: TwoQQstAnalysisOptions | None = None,
) -> int:
    """Resolve analysis iterations without touching input References in the body."""
    return int(_coerce_analysis_options(analysis_options).max_mle_iterations)


@workflow.workflow(name="twoq_qst")
def experiment_workflow(
    session: Session,
    qpu: QPU,
    qubits: QuantumElements,
    bus: QuantumElements,
    readout_calibration_result=None,
    target_state=None,
    temporary_parameters: dict[str | tuple[str, str, str], dict | QuantumParameters]
    | None = None,
    options: TwoQQstWorkflowOptions | None = None,
) -> None:
    """Run the canonical raw 2Q QST workflow."""
    opts = TwoQQstWorkflowOptions() if options is None else options
    resolved_config = resolve_target_configuration(
        custom_prep=opts.custom_prep,
        initial_state=opts.initial_state,
        target_state=target_state,
    )

    temp_qpu = temporary_qpu(qpu, temporary_parameters)
    qubits = temporary_quantum_elements_from_qpu(temp_qpu, qubits)
    _ = bus

    run_readout_calibration = should_run_readout_calibration(
        do_readout_calibration=opts.do_readout_calibration,
        readout_calibration_result=readout_calibration_result,
    )
    calibration_bundle = _bundle_readout_calibration_result(readout_calibration_result)
    with workflow.if_(run_readout_calibration):
        readout_cal_exp = create_readout_calibration_experiment(temp_qpu, qubits)
        compiled_readout_cal = compile_experiment(session, readout_cal_exp)
        calibration_bundle = _bundle_readout_calibration_result(
            run_experiment(session, compiled_readout_cal)
        )

    stable_calibration_bundle = _materialize_readout_calibration_bundle(
        calibration_bundle
    )
    stable_calibration_result = _extract_readout_calibration_result(
        stable_calibration_bundle
    )

    exp = create_experiment(
        temp_qpu,
        qubits,
        bus,
        custom_prep=resolved_config["custom_prep"],
        initial_state=resolved_config["initial_state"],
    )
    compiled_exp = compile_experiment(session, exp)
    tomography_result = run_experiment(session, compiled_exp)

    workflow.return_(
        tomography_result=tomography_result,
        readout_calibration_result=stable_calibration_result,
        initial_state=resolved_config["initial_state"],
        target_state_effective=resolved_config["target_state_effective"],
        custom_prep=resolved_config["custom_prep"],
    )


@workflow.workflow(name="twoq_qst_convergence")
def convergence_validation_workflow(
    session: Session,
    qpu: QPU,
    qubits: QuantumElements,
    bus: QuantumElements,
    readout_calibration_result=None,
    main_run_optimization_convergence=None,
    temporary_parameters: dict[str | tuple[str, str, str], dict | QuantumParameters]
    | None = None,
    options: TwoQQstWorkflowOptions | None = None,
    analysis_options: TwoQQstAnalysisOptions | None = None,
) -> None:
    """Run repeated product-state convergence validation for 2Q QST."""
    opts = TwoQQstWorkflowOptions() if options is None else options

    temp_qpu = temporary_qpu(qpu, temporary_parameters)
    qubits = temporary_quantum_elements_from_qpu(temp_qpu, qubits)
    _ = bus

    run_readout_calibration = should_run_readout_calibration(
        do_readout_calibration=opts.do_readout_calibration,
        readout_calibration_result=readout_calibration_result,
    )
    calibration_bundle = _bundle_readout_calibration_result(readout_calibration_result)
    with workflow.if_(run_readout_calibration):
        readout_cal_exp = create_readout_calibration_experiment(temp_qpu, qubits)
        compiled_readout_cal = compile_experiment(session, readout_cal_exp)
        calibration_bundle = _bundle_readout_calibration_result(
            run_experiment(session, compiled_readout_cal)
        )

    stable_calibration_bundle = _materialize_readout_calibration_bundle(
        calibration_bundle
    )
    stable_calibration_result = _extract_readout_calibration_result(
        stable_calibration_bundle
    )

    ctrl = _select_qubit_for_analysis(
        qubits=qubits,
        index=0,
        expected_len=2,
        caller="twoq_qst",
    )
    targ = _select_qubit_for_analysis(
        qubits=qubits,
        index=1,
        expected_len=2,
        caller="twoq_qst",
    )
    suite_states = resolve_convergence_suite_states(
        suite_states=opts.convergence_suite_states,
        repeats_per_state=opts.convergence_repeats_per_state,
    )
    repeat_indices = resolve_convergence_repeat_indices(
        repeats_per_state=opts.convergence_repeats_per_state,
    )
    repeats_per_state = resolve_convergence_repeats_per_state(
        repeats_per_state=opts.convergence_repeats_per_state,
    )
    raw_run_records = []
    analysis_max_iterations = _resolve_analysis_max_mle_iterations(analysis_options)

    with workflow.for_(suite_states, lambda state: state) as state_label:
        with workflow.for_(repeat_indices, lambda idx: idx) as repeat_index:
            suite_exp = create_experiment(
                temp_qpu,
                qubits,
                bus,
                custom_prep=False,
                initial_state=state_label,
            )
            compiled_suite_exp = compile_experiment(session, suite_exp)
            suite_tomography_result = run_experiment(session, compiled_suite_exp)
            suite_analysis_result = analyze_tomography_run(
                tomography_result=suite_tomography_result,
                ctrl_uid=ctrl.uid,
                targ_uid=targ.uid,
                readout_calibration_result=stable_calibration_result,
                target_state=state_label,
                max_iterations=analysis_max_iterations,
            )
            record = collect_convergence_run_record(
                state_label=state_label,
                repeat_index=repeat_index,
                analysis_result=suite_analysis_result,
            )
            _append_item(raw_run_records, record)

    raw_run_records = _materialize_list(raw_run_records)
    statistical_convergence = summarize_statistical_convergence(raw_run_records)
    with workflow.if_(opts.convergence_do_plotting):
        plot_convergence_suite_fidelity(
            statistical_convergence=statistical_convergence
        )

    workflow.return_(
        suite_states=suite_states,
        repeats_per_state=repeats_per_state,
        raw_run_records=raw_run_records,
        statistical_convergence=statistical_convergence,
        main_run_optimization_convergence=main_run_optimization_convergence,
    )


@workflow.workflow(name="twoq_qst_shot_sweep")
def shot_sweep_workflow(
    session: Session,
    qpu: QPU,
    qubits: QuantumElements,
    bus: QuantumElements,
    readout_calibration_result=None,
    temporary_parameters: dict[str | tuple[str, str, str], dict | QuantumParameters]
    | None = None,
    options: TwoQQstWorkflowOptions | None = None,
    analysis_options: TwoQQstAnalysisOptions | None = None,
) -> None:
    """Run the product-state shot sweep convergence study for 2Q QST."""
    opts = TwoQQstWorkflowOptions() if options is None else options

    temp_qpu = temporary_qpu(qpu, temporary_parameters)
    qubits = temporary_quantum_elements_from_qpu(temp_qpu, qubits)
    _ = bus

    run_readout_calibration = should_run_readout_calibration(
        do_readout_calibration=opts.do_readout_calibration,
        readout_calibration_result=readout_calibration_result,
    )
    calibration_bundle = _bundle_readout_calibration_result(readout_calibration_result)
    with workflow.if_(run_readout_calibration):
        readout_cal_exp = create_readout_calibration_experiment(temp_qpu, qubits)
        compiled_readout_cal = compile_experiment(session, readout_cal_exp)
        calibration_bundle = _bundle_readout_calibration_result(
            run_experiment(session, compiled_readout_cal)
        )

    stable_calibration_bundle = _materialize_readout_calibration_bundle(
        calibration_bundle
    )
    stable_calibration_result = _extract_readout_calibration_result(
        stable_calibration_bundle
    )

    ctrl = _select_qubit_for_analysis(
        qubits=qubits,
        index=0,
        expected_len=2,
        caller="twoq_qst",
    )
    targ = _select_qubit_for_analysis(
        qubits=qubits,
        index=1,
        expected_len=2,
        caller="twoq_qst",
    )
    suite_states = resolve_shot_sweep_suite_states(
        suite_states=opts.shot_sweep_suite_states,
        repeats_per_point=opts.shot_sweep_repeats_per_point,
    )
    shot_log2_values = resolve_shot_sweep_log2_values(
        shot_log2_values=opts.shot_sweep_log2_values
    )
    shot_counts = resolve_shot_sweep_counts(shot_log2_values)
    repeats_per_point = resolve_shot_sweep_repeats_per_point(
        repeats_per_point=opts.shot_sweep_repeats_per_point
    )
    repeat_indices = resolve_shot_sweep_repeat_indices(
        repeats_per_point=opts.shot_sweep_repeats_per_point
    )
    raw_run_records = []
    failed_runs = []
    analysis_max_iterations = _resolve_analysis_max_mle_iterations(analysis_options)

    with workflow.for_(suite_states, lambda state: state) as state_label:
        with workflow.for_(shot_log2_values, lambda value: value) as log2_shots:
            shots = shot_count_from_log2(log2_shots)
            with workflow.for_(repeat_indices, lambda idx: idx) as repeat_index:
                sweep_exp = create_experiment(
                    temp_qpu,
                    qubits,
                    bus,
                    custom_prep=False,
                    initial_state=state_label,
                    count_override=shots,
                )
                compiled_sweep_exp = compile_experiment(session, sweep_exp)
                sweep_tomography_result = run_experiment(session, compiled_sweep_exp)
                sweep_record = collect_shot_sweep_run_record(
                    state_label=state_label,
                    log2_shots=log2_shots,
                    shots=shots,
                    repeat=repeat_index,
                    tomography_result=sweep_tomography_result,
                    ctrl_uid=ctrl.uid,
                    targ_uid=targ.uid,
                    readout_calibration_result=stable_calibration_result,
                    target_state=state_label,
                    max_iterations=analysis_max_iterations,
                    eps=SHOT_SWEEP_EPS,
                )
                _append_item(raw_run_records, sweep_record["record"])
                _append_item_if_present(failed_runs, sweep_record["failure"])

    raw_run_records = _materialize_list(raw_run_records)
    failed_runs = _materialize_list(failed_runs)
    validation_checks = validate_shot_sweep_run_records(
        run_records=raw_run_records,
        suite_states=suite_states,
        shot_log2_values=shot_log2_values,
        repeats_per_point=repeats_per_point,
        eps=SHOT_SWEEP_EPS,
        infid_tol=SHOT_SWEEP_INFID_TOL,
    )
    aggregated_stats = aggregate_shot_sweep_statistics(raw_run_records)
    final_summary = summarize_final_shot_sweep(
        aggregated_stats=aggregated_stats,
        shot_log2_values=shot_log2_values,
    )
    with workflow.if_(opts.shot_sweep_do_plotting):
        plot_shot_sweep_summary(
            aggregated_stats=aggregated_stats,
            suite_states=suite_states,
        )

    workflow.return_(
        suite_states=suite_states,
        shot_log2_values=shot_log2_values,
        shot_counts=shot_counts,
        repeats_per_point=repeats_per_point,
        raw_run_records=raw_run_records,
        failed_runs=failed_runs,
        validation_checks=validation_checks,
        aggregated_stats=aggregated_stats,
        final_summary=final_summary,
    )


def _validate_bundle_configuration(
    options: TwoQQstWorkflowOptions,
    readout_calibration_result=None,
) -> None:
    if not bool(options.do_analysis) and (
        bool(options.do_convergence_validation) or bool(options.do_shot_sweep_convergence)
    ):
        raise ValueError("Convergence validation and shot sweep require do_analysis=True.")
    if (
        bool(options.do_analysis)
        and not bool(options.do_readout_calibration)
        and readout_calibration_result is None
    ):
        raise ValueError(
            "Analysis-capable twoq_qst bundles require readout calibration. "
            "Provide `readout_calibration_result` or set `do_readout_calibration=True`."
        )


def _output_to_dict(output) -> dict[str, object]:
    if output is None:
        return {}
    if isinstance(output, dict):
        return dict(output)
    if hasattr(output, "__dict__"):
        return dict(vars(output))
    keys = getattr(output, "keys", None)
    if callable(keys):
        return {key: output[key] for key in keys()}
    return {}


def run_bundle(
    session: Session,
    qpu: QPU,
    qubits: QuantumElements,
    bus: QuantumElements,
    readout_calibration_result=None,
    target_state=None,
    temporary_parameters: dict[str | tuple[str, str, str], dict | QuantumParameters]
    | None = None,
    options: TwoQQstWorkflowOptions | None = None,
    analysis_options: TwoQQstAnalysisOptions | None = None,
) -> dict[str, object]:
    """Run the split workflows and assemble a notebook-friendly plain dict."""
    opts = _coerce_workflow_options(options)
    analysis_opts = _coerce_analysis_options(analysis_options)
    _validate_bundle_configuration(opts, readout_calibration_result)

    main_result = experiment_workflow(
        session=session,
        qpu=qpu,
        qubits=qubits,
        bus=bus,
        readout_calibration_result=readout_calibration_result,
        target_state=target_state,
        temporary_parameters=temporary_parameters,
        options=opts,
    ).run()
    main_output = _output_to_dict(main_result.output)

    analysis_result = None
    convergence_report = None
    shot_sweep_report = None

    if opts.do_analysis:
        temp_qpu = temporary_qpu(qpu, temporary_parameters)
        temp_qubits = temporary_quantum_elements_from_qpu(temp_qpu, qubits)
        ctrl, targ = _normalize_two_qubits(temp_qubits)
        analysis_result = analysis_workflow(
            tomography_result=main_output["tomography_result"],
            ctrl=ctrl,
            targ=targ,
            readout_calibration_result=main_output["readout_calibration_result"],
            target_state=main_output["target_state_effective"],
            options=analysis_opts,
        ).run().output

        main_run_optimization_convergence = None
        if isinstance(analysis_result, dict):
            main_run_optimization_convergence = analysis_result.get(
                "optimization_convergence"
            )

        if opts.do_convergence_validation:
            convergence_result = convergence_validation_workflow(
                session=session,
                qpu=qpu,
                qubits=qubits,
                bus=bus,
                readout_calibration_result=main_output["readout_calibration_result"],
                main_run_optimization_convergence=main_run_optimization_convergence,
                temporary_parameters=temporary_parameters,
                options=opts,
                analysis_options=analysis_opts,
            ).run()
            convergence_report = _output_to_dict(convergence_result.output)

        if opts.do_shot_sweep_convergence:
            shot_sweep_result = shot_sweep_workflow(
                session=session,
                qpu=qpu,
                qubits=qubits,
                bus=bus,
                readout_calibration_result=main_output["readout_calibration_result"],
                temporary_parameters=temporary_parameters,
                options=opts,
                analysis_options=analysis_opts,
            ).run()
            shot_sweep_report = _output_to_dict(shot_sweep_result.output)

    return {
        "tomography_result": main_output.get("tomography_result"),
        "readout_calibration_result": main_output.get("readout_calibration_result"),
        "analysis_result": analysis_result,
        "convergence_report": convergence_report,
        "shot_sweep_report": shot_sweep_report,
        "initial_state": main_output.get("initial_state"),
        "target_state_effective": main_output.get("target_state_effective"),
        "custom_prep": bool(main_output.get("custom_prep", False)),
    }


__all__ = [
    "TwoQQstExperimentOptions",
    "TwoQQstWorkflowOptions",
    "TwoQQstAnalysisOptions",
    "create_experiment",
    "experiment_workflow",
    "convergence_validation_workflow",
    "shot_sweep_workflow",
    "run_bundle",
]

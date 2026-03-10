"""Two-qubit QST experiment workflow focused on product-state preparation.

This workflow is a simplified 2Q tomography entrypoint derived from the
validation path of `two_qubit_state_tomography`. It supports only
INTEGRATION-based, single-shot tomography and links to the dedicated
`analysis.two_qubit_qst` module for single-run analysis, product-state
convergence validation, and shot-sweep convergence summaries.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from laboneq import workflow
from laboneq.dsl.enums import AcquisitionType, AveragingMode
from laboneq.simple import Experiment, SectionAlignment, dsl
from laboneq.workflow.tasks import compile_experiment, run_experiment
from laboneq_applications.core import validation
from laboneq_applications.experiments.options import BaseExperimentOptions
from laboneq_applications.tasks.parameter_updating import (
    temporary_qpu,
    temporary_quantum_elements_from_qpu,
)

from qubit_experiment.analysis.two_qubit_qst import (
    DEFAULT_PRODUCT_SUITE_STATES,
    DEFAULT_SHOT_SWEEP_LOG2_VALUES,
    SHOT_SWEEP_EPS,
    SHOT_SWEEP_INFID_TOL,
    TwoQQstAnalysisOptions,
    aggregate_shot_sweep_statistics,
    analysis_workflow,
    analyze_tomography_run,
    collect_convergence_run_record,
    collect_shot_sweep_run_record,
    extract_main_run_optimization_convergence,
    plot_convergence_suite_fidelity,
    plot_shot_sweep_summary,
    summarize_final_shot_sweep,
    summarize_statistical_convergence,
    validate_shot_sweep_run_records,
)

from .two_qubit_readout_calibration import (
    create_experiment as create_readout_calibration_experiment,
)
from .two_qubit_tomography_common import (
    TOMOGRAPHY_SETTINGS,
    canonical_two_qubit_state_label,
    state_token_for_section_name,
    tomography_handle,
)

if TYPE_CHECKING:
    from laboneq.dsl.quantum import QuantumParameters
    from laboneq.dsl.quantum.qpu import QPU
    from laboneq.dsl.session import Session
    from laboneq_applications.typing import QuantumElements


@workflow.task_options(base_class=BaseExperimentOptions)
class TwoQQstExperimentOptions:
    """Options for 2Q QST experiment creation."""

    count: int = workflow.option_field(
        4096,
        description="Number of shots per tomography setting.",
    )
    acquisition_type: AcquisitionType = workflow.option_field(
        AcquisitionType.INTEGRATION,
        description="Acquire integrated complex IQ outcomes for QST.",
    )
    averaging_mode: AveragingMode = workflow.option_field(
        AveragingMode.SINGLE_SHOT,
        description="Use single-shot integrated outcomes for QST counts.",
    )


@workflow.workflow_options
class TwoQQstWorkflowOptions:
    """Workflow options for 2Q QST."""

    do_analysis: bool = workflow.option_field(
        True,
        description="Whether to run the main single-run QST analysis workflow.",
    )
    do_readout_calibration: bool = workflow.option_field(
        True,
        description="Whether to run readout calibration before QST.",
    )
    initial_state: str = workflow.option_field(
        "++",
        description=(
            "Initial 2-qubit product state used when custom_prep=False. "
            "Supported labels: '++', '+-', '-+', '--', "
            "'00', '01', '10', '11', 'gg', 'ge', 'eg', 'ee'."
        ),
    )
    custom_prep: bool = workflow.option_field(
        False,
        description=(
            "Whether to use the future internal custom preparation block instead "
            "of initial_state-driven preparation."
        ),
    )
    do_convergence_validation: bool = workflow.option_field(
        False,
        description="Whether to run repeated product-state convergence validation.",
    )
    convergence_repeats_per_state: int = workflow.option_field(
        2,
        description="Number of repeated runs per state in convergence validation.",
    )
    convergence_suite_states: tuple[str, ...] = workflow.option_field(
        DEFAULT_PRODUCT_SUITE_STATES,
        description="State labels used for product-state convergence validation.",
    )
    convergence_do_plotting: bool = workflow.option_field(
        False,
        description="Whether to generate the convergence summary plot artifact.",
    )
    do_shot_sweep_convergence: bool = workflow.option_field(
        False,
        description="Whether to run the product-state shot sweep convergence study.",
    )
    shot_sweep_log2_values: tuple[int, ...] = workflow.option_field(
        DEFAULT_SHOT_SWEEP_LOG2_VALUES,
        description="Shot-sweep grid expressed as log2(shots).",
    )
    shot_sweep_suite_states: tuple[str, ...] = workflow.option_field(
        DEFAULT_PRODUCT_SUITE_STATES,
        description="State labels used for the product-state shot sweep.",
    )
    shot_sweep_repeats_per_point: int = workflow.option_field(
        2,
        description="Repeated runs per (state, shot count) point in the shot sweep.",
    )
    shot_sweep_do_plotting: bool = workflow.option_field(
        False,
        description="Whether to generate the shot-sweep summary plot artifact.",
    )


@workflow.task(save=False)
def _append_item(items: list, item) -> None:
    items.append(item)


@workflow.task(save=False)
def _append_item_if_present(items: list, item) -> None:
    if item is not None:
        items.append(item)


@workflow.task(save=False)
def _materialize_list(items: list) -> list:
    return list(items)


@workflow.task(save=False)
def _select_qubit_for_analysis(
    qubits: QuantumElements,
    index: int,
    expected_len: int,
    caller: str,
):
    """Select one qubit by index after runtime validation."""
    qlist = list(validation.validate_and_convert_qubits_sweeps(qubits))
    if len(qlist) != int(expected_len):
        raise ValueError(
            f"{caller} expects exactly {expected_len} qubits in `qubits`."
            f" Received {len(qlist)}."
        )
    idx = int(index)
    if idx < 0 or idx >= int(expected_len):
        raise ValueError(
            f"Invalid qubit index {idx} for expected_len={expected_len}."
        )
    return validation.validate_and_convert_single_qubit_sweeps(qlist[idx])


@workflow.task(save=False)
def resolve_convergence_repeat_indices(repeats_per_state: int) -> tuple[int, ...]:
    repeats = int(repeats_per_state)
    if repeats < 1:
        raise ValueError("convergence_repeats_per_state must be >= 1.")
    return tuple(range(repeats))


@workflow.task(save=False)
def resolve_convergence_repeats_per_state(repeats_per_state: int) -> int:
    repeats = int(repeats_per_state)
    if repeats < 1:
        raise ValueError("convergence_repeats_per_state must be >= 1.")
    return repeats


@workflow.task(save=False)
def resolve_shot_sweep_repeats_per_point(repeats_per_point: int) -> int:
    repeats = int(repeats_per_point)
    if repeats < 1:
        raise ValueError("shot_sweep_repeats_per_point must be >= 1.")
    return repeats


@workflow.task(save=False)
def resolve_shot_sweep_repeat_indices(repeats_per_point: int) -> tuple[int, ...]:
    repeats = int(repeats_per_point)
    if repeats < 1:
        raise ValueError("shot_sweep_repeats_per_point must be >= 1.")
    return tuple(range(1, repeats + 1))


@workflow.task(save=False)
def resolve_shot_sweep_log2_values(
    shot_log2_values: tuple[int, ...] | list[int],
) -> tuple[int, ...]:
    if shot_log2_values is None:
        raise ValueError("shot_sweep_log2_values cannot be None.")
    normalized = []
    seen = set()
    for raw in shot_log2_values:
        value = int(raw)
        if value < 0:
            raise ValueError("shot_sweep_log2_values must contain only non-negative integers.")
        if value not in seen:
            seen.add(value)
            normalized.append(value)
    if not normalized:
        raise ValueError("shot_sweep_log2_values must contain at least one value.")
    return tuple(normalized)


@workflow.task(save=False)
def resolve_shot_sweep_counts(
    shot_log2_values: tuple[int, ...] | list[int],
) -> tuple[int, ...]:
    return tuple(2 ** int(v) for v in shot_log2_values)


@workflow.task(save=False)
def shot_count_from_log2(log2_shots: int) -> int:
    value = int(log2_shots)
    if value < 0:
        raise ValueError("log2_shots must be non-negative.")
    return 2 ** value


def _resolve_target_configuration_impl(
    custom_prep: bool,
    initial_state: str,
    target_state=None,
) -> dict[str, object]:
    if custom_prep:
        raise NotImplementedError(
            "custom_prep=True is not implemented yet. A future custom prep block "
            "will define preparation pulses independently of initial_state."
        )

    canonical_initial_state = _canonical_initial_state_label(initial_state)
    effective_target_state = target_state
    if target_state is None:
        effective_target_state = canonical_initial_state
    elif _canonical_target_state_label(target_state) != canonical_initial_state:
        raise ValueError(
            "When custom_prep=False, target_state must match initial_state. "
            f"Got target_state={target_state!r}, initial_state={initial_state!r}."
        )

    return {
        "custom_prep": False,
        "initial_state": canonical_initial_state,
        "target_state_effective": effective_target_state,
    }


@workflow.task
def resolve_target_configuration(
    custom_prep: bool,
    initial_state: str,
    target_state=None,
) -> dict[str, object]:
    """Resolve initial-state and target-state semantics for the supported path."""
    return _resolve_target_configuration_impl(
        custom_prep=custom_prep,
        initial_state=initial_state,
        target_state=target_state,
    )


@workflow.task
def validate_analysis_prerequisites(
    do_analysis: bool,
    do_readout_calibration: bool,
    readout_calibration_result,
) -> None:
    """Validate that analysis-capable paths have readout calibration input."""
    if do_analysis and not do_readout_calibration and readout_calibration_result is None:
        raise ValueError(
            "Analysis requires readout calibration. Provide "
            "`readout_calibration_result` or set `do_readout_calibration=True`."
        )


@workflow.task
def validate_workflow_configuration(
    do_analysis: bool,
    do_convergence_validation: bool,
    do_shot_sweep_convergence: bool,
) -> None:
    """Validate coupled workflow options that require analysis outputs."""
    if not bool(do_analysis) and (
        bool(do_convergence_validation) or bool(do_shot_sweep_convergence)
    ):
        raise ValueError(
            "Convergence validation and shot sweep require do_analysis=True."
        )


@workflow.task
def resolve_convergence_suite_states(
    suite_states: tuple[str, ...] | list[str],
    repeats_per_state: int,
) -> tuple[str, ...]:
    """Validate and normalize state labels for convergence checks."""
    if int(repeats_per_state) < 1:
        raise ValueError("convergence_repeats_per_state must be >= 1.")
    if suite_states is None:
        raise ValueError("convergence_suite_states cannot be None.")
    normalized = []
    seen = set()
    for label in suite_states:
        canonical = _canonical_initial_state_label(str(label))
        if canonical not in seen:
            seen.add(canonical)
            normalized.append(canonical)
    if not normalized:
        raise ValueError("convergence_suite_states must contain at least one valid state.")
    return tuple(normalized)


@workflow.task
def resolve_shot_sweep_suite_states(
    suite_states: tuple[str, ...] | list[str],
    repeats_per_point: int,
) -> tuple[str, ...]:
    """Validate and normalize state labels for the shot-sweep suite."""
    if int(repeats_per_point) < 1:
        raise ValueError("shot_sweep_repeats_per_point must be >= 1.")
    if suite_states is None:
        raise ValueError("shot_sweep_suite_states cannot be None.")
    normalized = []
    seen = set()
    for label in suite_states:
        canonical = _canonical_initial_state_label(str(label))
        if canonical not in seen:
            seen.add(canonical)
            normalized.append(canonical)
    if not normalized:
        raise ValueError("shot_sweep_suite_states must contain at least one valid state.")
    return tuple(normalized)


@workflow.workflow(name="two_qubit_qst")
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
    """Run simplified 2Q QST with optional convergence analyses."""
    options = TwoQQstWorkflowOptions() if options is None else options

    validate_workflow_configuration(
        do_analysis=options.do_analysis,
        do_convergence_validation=options.do_convergence_validation,
        do_shot_sweep_convergence=options.do_shot_sweep_convergence,
    )

    resolved_config = resolve_target_configuration(
        custom_prep=options.custom_prep,
        initial_state=options.initial_state,
        target_state=target_state,
    )

    temp_qpu = temporary_qpu(qpu, temporary_parameters)
    qubits = temporary_quantum_elements_from_qpu(temp_qpu, qubits)
    _ = bus

    calibration_result = readout_calibration_result
    with workflow.if_(
        options.do_readout_calibration and readout_calibration_result is None
    ):
        readout_cal_exp = create_readout_calibration_experiment(
            temp_qpu,
            qubits,
        )
        compiled_readout_cal = compile_experiment(session, readout_cal_exp)
        calibration_result = run_experiment(session, compiled_readout_cal)

    validate_analysis_prerequisites(
        do_analysis=options.do_analysis,
        do_readout_calibration=options.do_readout_calibration,
        readout_calibration_result=calibration_result,
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

    ctrl = _select_qubit_for_analysis(
        qubits=qubits,
        index=0,
        expected_len=2,
        caller="two_qubit_qst",
    )
    targ = _select_qubit_for_analysis(
        qubits=qubits,
        index=1,
        expected_len=2,
        caller="two_qubit_qst",
    )

    analysis_result = None
    with workflow.if_(options.do_analysis):
        analysis_result = analysis_workflow(
            tomography_result=tomography_result,
            ctrl=ctrl,
            targ=targ,
            readout_calibration_result=calibration_result,
            target_state=resolved_config["target_state_effective"],
        )

    convergence_report = None
    with workflow.if_(options.do_analysis and options.do_convergence_validation):
        suite_states = resolve_convergence_suite_states(
            suite_states=options.convergence_suite_states,
            repeats_per_state=options.convergence_repeats_per_state,
        )
        repeat_indices = resolve_convergence_repeat_indices(
            repeats_per_state=options.convergence_repeats_per_state,
        )
        repeats_per_state = resolve_convergence_repeats_per_state(
            repeats_per_state=options.convergence_repeats_per_state,
        )
        raw_run_records = []
        analysis_max_iterations = int(TwoQQstAnalysisOptions().max_mle_iterations)

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
                    readout_calibration_result=calibration_result,
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
        main_run_convergence = extract_main_run_optimization_convergence(
            analysis_result=analysis_result
        )
        with workflow.if_(options.convergence_do_plotting):
            plot_convergence_suite_fidelity(
                statistical_convergence=statistical_convergence
            )
        convergence_report = {
            "suite_states": suite_states,
            "repeats_per_state": repeats_per_state,
            "main_run_optimization_convergence": main_run_convergence,
            "statistical_convergence": statistical_convergence,
            "raw_run_records": raw_run_records,
        }

    shot_sweep_report = None
    with workflow.if_(options.do_analysis and options.do_shot_sweep_convergence):
        suite_states = resolve_shot_sweep_suite_states(
            suite_states=options.shot_sweep_suite_states,
            repeats_per_point=options.shot_sweep_repeats_per_point,
        )
        shot_log2_values = resolve_shot_sweep_log2_values(
            shot_log2_values=options.shot_sweep_log2_values
        )
        shot_counts = resolve_shot_sweep_counts(shot_log2_values)
        repeats_per_point = resolve_shot_sweep_repeats_per_point(
            repeats_per_point=options.shot_sweep_repeats_per_point
        )
        repeat_indices = resolve_shot_sweep_repeat_indices(
            repeats_per_point=options.shot_sweep_repeats_per_point
        )
        raw_run_records = []
        failed_runs = []
        analysis_max_iterations = int(TwoQQstAnalysisOptions().max_mle_iterations)

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
                        readout_calibration_result=calibration_result,
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
        with workflow.if_(options.shot_sweep_do_plotting):
            plot_shot_sweep_summary(
                aggregated_stats=aggregated_stats,
                suite_states=suite_states,
            )
        shot_sweep_report = {
            "suite_states": suite_states,
            "shot_log2_values": shot_log2_values,
            "shot_counts": shot_counts,
            "repeats_per_point": repeats_per_point,
            "raw_run_records": raw_run_records,
            "failed_runs": failed_runs,
            "validation_checks": validation_checks,
            "aggregated_stats": aggregated_stats,
            "final_summary": final_summary,
        }

    workflow.return_(
        {
            "tomography_result": tomography_result,
            "readout_calibration_result": calibration_result,
            "analysis_result": analysis_result,
            "convergence_report": convergence_report,
            "shot_sweep_report": shot_sweep_report,
            "initial_state": resolved_config["initial_state"],
            "target_state_effective": resolved_config["target_state_effective"],
            "custom_prep": resolved_config["custom_prep"],
        }
    )


def _canonical_initial_state_label(state: str) -> str:
    """Normalize a supported 2Q product-state label."""
    try:
        return canonical_two_qubit_state_label(state)
    except ValueError as exc:
        raise ValueError(
            "Unsupported initial_state. Use one of "
            "'++', '+-', '-+', '--', "
            "'00', '01', '10', '11', 'gg', 'ge', 'eg', 'ee'."
        ) from exc


def _canonical_target_state_label(target_state) -> str:
    """Normalize target-state string labels used for matching checks."""
    if not isinstance(target_state, str):
        return str(target_state)
    s = target_state.strip().lower().replace(" ", "")
    try:
        return canonical_two_qubit_state_label(s)
    except ValueError:
        return s


def _single_qubit_state_token(label: str, *, qubit_role: str) -> str:
    idx = 0 if qubit_role == "ctrl" else 1
    token = label[idx]
    if token in {"+", "-"}:
        return token
    if token == "0":
        return "g"
    if token == "1":
        return "e"
    raise ValueError(f"Unsupported token {token!r} in initial_state {label!r}.")


def _state_token_for_section_name(token: str) -> str:
    return state_token_for_section_name(token)


def _validate_tomography_qop_contract(qop) -> None:
    required_methods = (
        "prepare_tomography_state",
        "apply_tomography_prerotation",
    )
    missing = [
        name for name in required_methods if not callable(getattr(qop, name, None))
    ]
    if missing:
        missing_display = ", ".join(missing)
        raise TypeError(
            "The current quantum_operations class does not define required "
            "tomography methods for two_qubit_qst. "
            f"class={type(qop).__name__!r}, missing=[{missing_display}]."
        )


def _create_experiment_impl(
    qpu: QPU,
    qubits: QuantumElements,
    bus: QuantumElements,
    custom_prep: bool = False,
    initial_state: str = "++",
    count_override: int | None = None,
    options: TwoQQstExperimentOptions | None = None,
) -> Experiment:
    opts = TwoQQstExperimentOptions() if options is None else options
    try:
        acquisition_type = AcquisitionType(opts.acquisition_type)
    except Exception as exc:
        raise ValueError(
            "two_qubit_qst only supports AcquisitionType.INTEGRATION."
        ) from exc
    if acquisition_type != AcquisitionType.INTEGRATION:
        raise ValueError(
            "two_qubit_qst only supports AcquisitionType.INTEGRATION."
        )
    try:
        averaging_mode = AveragingMode(opts.averaging_mode)
    except Exception as exc:
        raise ValueError(
            "two_qubit_qst only supports AveragingMode.SINGLE_SHOT."
        ) from exc
    if averaging_mode != AveragingMode.SINGLE_SHOT:
        raise ValueError(
            "two_qubit_qst only supports AveragingMode.SINGLE_SHOT."
        )
    if custom_prep:
        raise NotImplementedError(
            "custom_prep=True is not implemented yet. A future custom prep block "
            "will define preparation pulses independently of initial_state."
        )
    count = int(opts.count if count_override is None else count_override)
    if count < 1:
        raise ValueError("count must be >= 1.")

    ctrl, targ = _normalize_two_qubits(qubits)
    canonical_initial_state = _canonical_initial_state_label(initial_state)
    ctrl_token = _single_qubit_state_token(canonical_initial_state, qubit_role="ctrl")
    targ_token = _single_qubit_state_token(canonical_initial_state, qubit_role="targ")
    ctrl_token_name = _state_token_for_section_name(ctrl_token)
    targ_token_name = _state_token_for_section_name(targ_token)

    qop = qpu.quantum_operations
    _validate_tomography_qop_contract(qop)
    max_measure_section_length = qop.measure_section_length([ctrl, targ])
    _ = bus

    with dsl.acquire_loop_rt(
        count=count,
        averaging_mode=opts.averaging_mode,
        acquisition_type=opts.acquisition_type,
        repetition_mode=opts.repetition_mode,
        repetition_time=opts.repetition_time,
        reset_oscillator_phase=opts.reset_oscillator_phase,
    ):
        for setting_label, (ctrl_axis, targ_axis) in TOMOGRAPHY_SETTINGS:
            with dsl.section(
                name=f"tomo_{setting_label}",
                alignment=SectionAlignment.LEFT,
            ):
                prep_play_after = None
                if opts.active_reset:
                    active_reset_sec = qop.active_reset(
                        [ctrl, targ],
                        active_reset_states=opts.active_reset_states,
                        number_resets=opts.active_reset_repetitions,
                        measure_section_length=max_measure_section_length,
                    )
                    prep_play_after = active_reset_sec.uid

                prep_section_kwargs = {
                    "name": f"prep_{setting_label}",
                    "alignment": SectionAlignment.LEFT,
                }
                if prep_play_after is not None:
                    prep_section_kwargs["play_after"] = prep_play_after

                with dsl.section(**prep_section_kwargs) as prep_sec:
                    with dsl.section(
                        name=f"prep_ctrl_{ctrl_token_name}_{setting_label}",
                        alignment=SectionAlignment.LEFT,
                    ) as prep_ctrl:
                        qop.prepare_tomography_state(ctrl, ctrl_token)

                    with dsl.section(
                        name=f"prep_targ_{targ_token_name}_{setting_label}",
                        alignment=SectionAlignment.LEFT,
                        play_after=prep_ctrl.uid,
                    ):
                        qop.prepare_tomography_state(targ, targ_token)

                # Future custom prep block goes here and will define preparation
                # pulses independently of initial_state when custom_prep=True.

                with dsl.section(
                    name=f"basis_{setting_label}",
                    alignment=SectionAlignment.LEFT,
                    play_after=prep_sec.uid,
                ) as basis_sec:
                    with dsl.section(
                        name=f"basis_ctrl_{setting_label}",
                        alignment=SectionAlignment.LEFT,
                    ) as basis_ctrl:
                        qop.apply_tomography_prerotation(ctrl, ctrl_axis)

                    with dsl.section(
                        name=f"basis_targ_{setting_label}",
                        alignment=SectionAlignment.LEFT,
                        play_after=basis_ctrl.uid,
                    ):
                        qop.apply_tomography_prerotation(targ, targ_axis)

                with dsl.section(
                    name=f"measure_{setting_label}",
                    alignment=SectionAlignment.LEFT,
                    play_after=basis_sec.uid,
                ):
                    sec_ctrl = qop.measure(
                        ctrl,
                        handle=tomography_handle(ctrl.uid, setting_label),
                    )
                    sec_targ = qop.measure(
                        targ,
                        handle=tomography_handle(targ.uid, setting_label),
                    )
                    sec_ctrl.length = max_measure_section_length
                    sec_targ.length = max_measure_section_length
                    qop.passive_reset(ctrl)
                    qop.passive_reset(targ)


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


def _normalize_two_qubits(qubits: QuantumElements):
    """Validate qubits input and return exactly two single-qubit elements."""
    qlist = list(validation.validate_and_convert_qubits_sweeps(qubits))
    if len(qlist) != 2:
        raise ValueError(
            "two_qubit_qst expects exactly 2 qubits in `qubits`."
            f" Received {len(qlist)}."
        )
    ctrl = validation.validate_and_convert_single_qubit_sweeps(qlist[0])
    targ = validation.validate_and_convert_single_qubit_sweeps(qlist[1])
    return ctrl, targ

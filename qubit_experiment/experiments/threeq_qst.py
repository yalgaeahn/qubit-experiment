"""Canonical three-qubit QST workflows under the `threeq_qst` name.

This module splits raw execution, convergence validation, and shot-sweep
studies into separate workflows. Each workflow returns only top-level payloads;
`run_bundle(...)` assembles notebook-friendly nested dictionaries in Python.
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

from qubit_experiment.analysis.threeq_qst import (
    DEFAULT_PRODUCT_SUITE_STATES,
    DEFAULT_SHOT_SWEEP_LOG2_VALUES,
    SHOT_SWEEP_EPS,
    SHOT_SWEEP_INFID_TOL,
    ThreeQQstAnalysisOptions,
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

from .three_qubit_readout_calibration import (
    create_experiment as create_readout_calibration_experiment,
)
from .three_qubit_tomography_common import (
    TOMOGRAPHY_SETTINGS,
    canonical_three_qubit_state_label,
    state_token_for_section_name,
    tomography_handle,
)

if TYPE_CHECKING:
    from laboneq.dsl.quantum import QuantumParameters
    from laboneq.dsl.quantum.qpu import QPU
    from laboneq.dsl.session import Session
    from laboneq_applications.typing import QuantumElements


@workflow.task_options(base_class=BaseExperimentOptions)
class ThreeQQstExperimentOptions:
    """Options for 3Q QST experiment creation."""

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
class ThreeQQstWorkflowOptions:
    """Workflow options for canonical 3Q QST."""

    do_analysis: bool = workflow.option_field(
        True,
        description="Whether to run the main single-run QST analysis workflow.",
    )
    do_readout_calibration: bool = workflow.option_field(
        True,
        description="Whether to run readout calibration before QST.",
    )
    initial_state: str = workflow.option_field(
        "+++",
        description=(
            "Initial 3-qubit product state used when custom_prep=False. "
            "Supported labels include binary ('000'..'111'), +/- labels, "
            "and g/e labels ('ggg'..'eee')."
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


def _coerce_workflow_options(
    options: ThreeQQstWorkflowOptions | None,
) -> ThreeQQstWorkflowOptions:
    if options is None:
        return ThreeQQstWorkflowOptions()
    if isinstance(options, ThreeQQstWorkflowOptions):
        return options

    base = getattr(options, "_base", None)
    if isinstance(base, ThreeQQstWorkflowOptions):
        return base

    raise TypeError(
        "options must be a ThreeQQstWorkflowOptions instance or an "
        "experiment_workflow.options() builder."
    )


def _coerce_analysis_options(
    analysis_options: ThreeQQstAnalysisOptions | None,
) -> ThreeQQstAnalysisOptions:
    if analysis_options is None:
        return ThreeQQstAnalysisOptions()
    if isinstance(analysis_options, ThreeQQstAnalysisOptions):
        return analysis_options

    base = getattr(analysis_options, "_base", None)
    if isinstance(base, ThreeQQstAnalysisOptions):
        return base

    raise TypeError(
        "analysis_options must be a ThreeQQstAnalysisOptions instance or an "
        "analysis_workflow.options() builder."
    )


def _canonical_initial_state_label(state: str) -> str:
    """Normalize 3Q product-state labels for the canonical path."""
    try:
        return canonical_three_qubit_state_label(state)
    except ValueError as exc:
        raise ValueError(
            "Unsupported initial_state. Use binary labels ('000'..'111'), "
            "+/- labels, or g/e labels ('ggg'..'eee')."
        ) from exc


def _canonical_target_state_label(target_state) -> str:
    """Normalize target-state string labels used for matching."""
    if not isinstance(target_state, str):
        return str(target_state)
    text = target_state.strip().lower().replace(" ", "")
    try:
        return canonical_three_qubit_state_label(text)
    except ValueError:
        return text


def _single_qubit_state_token(label: str, *, qubit_index: int) -> str:
    """Extract and map one qubit token from a canonical 3Q product-state label."""
    token = label[qubit_index]
    if token in {"+", "-"}:
        return token
    if token == "0":
        return "g"
    if token == "1":
        return "e"
    raise ValueError(f"Unsupported token {token!r} in initial_state {label!r}.")


def _normalize_three_qubits(qubits: QuantumElements):
    """Validate qubits input and return exactly three single-qubit elements."""
    qlist = list(validation.validate_and_convert_qubits_sweeps(qubits))
    if len(qlist) != 3:
        raise ValueError(
            "threeq_qst expects exactly 3 qubits in `qubits`."
            f" Received {len(qlist)}."
        )
    q0 = validation.validate_and_convert_single_qubit_sweeps(qlist[0])
    q1 = validation.validate_and_convert_single_qubit_sweeps(qlist[1])
    q2 = validation.validate_and_convert_single_qubit_sweeps(qlist[2])
    return q0, q1, q2


def _validate_tomography_qop_contract(qop) -> None:
    """Ensure required tomography qop methods exist on the current operation set."""
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
            "tomography methods for threeq_qst. "
            f"class={type(qop).__name__!r}, missing=[{missing_display}]."
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
def _bundle_readout_calibration_result(result) -> dict[str, object]:
    """Wrap optional calibration outputs for stable conditional branching."""
    return {"readout_calibration_result": result}


@workflow.task(save=False)
def _materialize_readout_calibration_bundle(
    bundle: dict[str, object],
) -> dict[str, object]:
    """Copy branch-local calibration bundles into a stable runtime value."""
    return dict(bundle)


@workflow.task(save=False)
def _extract_readout_calibration_result(bundle: dict[str, object]):
    """Extract the concrete calibration result from a stable bundle."""
    return dict(bundle).get("readout_calibration_result")


@workflow.task(save=False)
def _resolve_analysis_max_mle_iterations(
    analysis_options: ThreeQQstAnalysisOptions | None = None,
) -> int:
    """Resolve analysis iterations without touching input References in the body."""
    return int(_coerce_analysis_options(analysis_options).max_mle_iterations)


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
            raise ValueError(
                "shot_sweep_log2_values must contain only non-negative integers."
            )
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
    return tuple(2 ** int(value) for value in shot_log2_values)


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
        raise ValueError(
            "convergence_suite_states must contain at least one valid state."
        )
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
        raise ValueError(
            "shot_sweep_suite_states must contain at least one valid state."
        )
    return tuple(normalized)


@workflow.task(save=False)
def should_run_readout_calibration(
    do_readout_calibration: bool,
    readout_calibration_result,
) -> bool:
    """Resolve whether the workflow should acquire a fresh readout calibration."""
    return bool(do_readout_calibration) and readout_calibration_result is None


def _create_experiment_impl(
    qpu: QPU,
    qubits: QuantumElements,
    bus: QuantumElements,
    custom_prep: bool = False,
    initial_state: str = "+++",
    count_override: int | None = None,
    options: ThreeQQstExperimentOptions | None = None,
) -> Experiment:
    opts = ThreeQQstExperimentOptions() if options is None else options
    try:
        acquisition_type = AcquisitionType(opts.acquisition_type)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "threeq_qst only supports AcquisitionType.INTEGRATION."
        ) from exc
    if acquisition_type != AcquisitionType.INTEGRATION:
        raise ValueError(
            "threeq_qst only supports AcquisitionType.INTEGRATION."
        )
    try:
        averaging_mode = AveragingMode(opts.averaging_mode)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "threeq_qst only supports AveragingMode.SINGLE_SHOT."
        ) from exc
    if averaging_mode != AveragingMode.SINGLE_SHOT:
        raise ValueError(
            "threeq_qst only supports AveragingMode.SINGLE_SHOT."
        )
    if custom_prep:
        raise NotImplementedError(
            "custom_prep=True is not implemented yet. A future custom prep block "
            "will define preparation pulses independently of initial_state."
        )

    q0, q1, q2 = _normalize_three_qubits(qubits)
    _ = bus

    canonical_initial_state = _canonical_initial_state_label(initial_state)
    q0_token = _single_qubit_state_token(canonical_initial_state, qubit_index=0)
    q1_token = _single_qubit_state_token(canonical_initial_state, qubit_index=1)
    q2_token = _single_qubit_state_token(canonical_initial_state, qubit_index=2)
    q0_token_name = state_token_for_section_name(q0_token)
    q1_token_name = state_token_for_section_name(q1_token)
    q2_token_name = state_token_for_section_name(q2_token)

    qop = qpu.quantum_operations
    _validate_tomography_qop_contract(qop)
    max_measure_section_length = qop.measure_section_length([q0, q1, q2])
    count = int(count_override) if count_override is not None else int(opts.count)

    with dsl.acquire_loop_rt(
        count=count,
        averaging_mode=opts.averaging_mode,
        acquisition_type=opts.acquisition_type,
        repetition_mode=opts.repetition_mode,
        repetition_time=opts.repetition_time,
        reset_oscillator_phase=opts.reset_oscillator_phase,
    ):
        for setting_label, (q0_axis, q1_axis, q2_axis) in TOMOGRAPHY_SETTINGS:
            with dsl.section(
                name=f"tomo_{setting_label}",
                alignment=SectionAlignment.LEFT,
            ):
                prep_play_after = None
                if opts.active_reset:
                    active_reset_sec = qop.active_reset(
                        [q0, q1, q2],
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
                        name=f"prep_q0_{q0_token_name}_{setting_label}",
                        alignment=SectionAlignment.LEFT,
                    ) as prep_q0:
                        qop.prepare_tomography_state(q0, q0_token)

                    with dsl.section(
                        name=f"prep_q1_{q1_token_name}_{setting_label}",
                        alignment=SectionAlignment.LEFT,
                        play_after=prep_q0.uid,
                    ) as prep_q1:
                        qop.prepare_tomography_state(q1, q1_token)

                    with dsl.section(
                        name=f"prep_q2_{q2_token_name}_{setting_label}",
                        alignment=SectionAlignment.LEFT,
                        play_after=prep_q1.uid,
                    ):
                        qop.prepare_tomography_state(q2, q2_token)

                with dsl.section(
                    name=f"basis_{setting_label}",
                    alignment=SectionAlignment.LEFT,
                    play_after=prep_sec.uid,
                ) as basis_sec:
                    with dsl.section(
                        name=f"basis_q0_{setting_label}",
                        alignment=SectionAlignment.LEFT,
                    ) as basis_q0:
                        qop.apply_tomography_prerotation(q0, q0_axis)

                    with dsl.section(
                        name=f"basis_q1_{setting_label}",
                        alignment=SectionAlignment.LEFT,
                        play_after=basis_q0.uid,
                    ) as basis_q1:
                        qop.apply_tomography_prerotation(q1, q1_axis)

                    with dsl.section(
                        name=f"basis_q2_{setting_label}",
                        alignment=SectionAlignment.LEFT,
                        play_after=basis_q1.uid,
                    ):
                        qop.apply_tomography_prerotation(q2, q2_axis)

                with dsl.section(
                    name=f"measure_{setting_label}",
                    alignment=SectionAlignment.LEFT,
                    play_after=basis_sec.uid,
                ):
                    sec_q0 = qop.measure(
                        q0,
                        handle=tomography_handle(q0.uid, setting_label),
                    )
                    sec_q1 = qop.measure(
                        q1,
                        handle=tomography_handle(q1.uid, setting_label),
                    )
                    sec_q2 = qop.measure(
                        q2,
                        handle=tomography_handle(q2.uid, setting_label),
                    )
                    sec_q0.length = max_measure_section_length
                    sec_q1.length = max_measure_section_length
                    sec_q2.length = max_measure_section_length
                    qop.passive_reset(q0)
                    qop.passive_reset(q1)
                    qop.passive_reset(q2)


@workflow.task
@dsl.qubit_experiment
def create_experiment(
    qpu: QPU,
    qubits: QuantumElements,
    bus: QuantumElements,
    custom_prep: bool = False,
    initial_state: str = "+++",
    count_override: int | None = None,
    options: ThreeQQstExperimentOptions | None = None,
) -> Experiment:
    """Create a 3Q product-state QST experiment without RIP preparation."""
    return _create_experiment_impl(
        qpu=qpu,
        qubits=qubits,
        bus=bus,
        custom_prep=custom_prep,
        initial_state=initial_state,
        count_override=count_override,
        options=options,
    )


@workflow.workflow(name="threeq_qst")
def experiment_workflow(
    session: Session,
    qpu: QPU,
    qubits: QuantumElements,
    bus: QuantumElements,
    readout_calibration_result=None,
    target_state=None,
    temporary_parameters: dict[str | tuple[str, str, str], dict | QuantumParameters]
    | None = None,
    options: ThreeQQstWorkflowOptions | None = None,
) -> None:
    """Run the canonical raw 3Q product-state QST workflow."""
    opts = ThreeQQstWorkflowOptions() if options is None else options
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

    validate_analysis_prerequisites(
        do_analysis=opts.do_analysis,
        do_readout_calibration=opts.do_readout_calibration,
        readout_calibration_result=stable_calibration_result,
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


@workflow.workflow(name="threeq_qst_convergence")
def convergence_validation_workflow(
    session: Session,
    qpu: QPU,
    qubits: QuantumElements,
    bus: QuantumElements,
    readout_calibration_result=None,
    main_run_optimization_convergence=None,
    temporary_parameters: dict[str | tuple[str, str, str], dict | QuantumParameters]
    | None = None,
    options: ThreeQQstWorkflowOptions | None = None,
    analysis_options: ThreeQQstAnalysisOptions | None = None,
) -> None:
    """Run repeated product-state convergence validation for 3Q QST."""
    opts = ThreeQQstWorkflowOptions() if options is None else options

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

    q0 = _select_qubit_for_analysis(
        qubits=qubits,
        index=0,
        expected_len=3,
        caller="threeq_qst",
    )
    q1 = _select_qubit_for_analysis(
        qubits=qubits,
        index=1,
        expected_len=3,
        caller="threeq_qst",
    )
    q2 = _select_qubit_for_analysis(
        qubits=qubits,
        index=2,
        expected_len=3,
        caller="threeq_qst",
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
                q0_uid=q0.uid,
                q1_uid=q1.uid,
                q2_uid=q2.uid,
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


@workflow.workflow(name="threeq_qst_shot_sweep")
def shot_sweep_workflow(
    session: Session,
    qpu: QPU,
    qubits: QuantumElements,
    bus: QuantumElements,
    readout_calibration_result=None,
    temporary_parameters: dict[str | tuple[str, str, str], dict | QuantumParameters]
    | None = None,
    options: ThreeQQstWorkflowOptions | None = None,
    analysis_options: ThreeQQstAnalysisOptions | None = None,
) -> None:
    """Run the product-state shot sweep convergence study for 3Q QST."""
    opts = ThreeQQstWorkflowOptions() if options is None else options

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

    q0 = _select_qubit_for_analysis(
        qubits=qubits,
        index=0,
        expected_len=3,
        caller="threeq_qst",
    )
    q1 = _select_qubit_for_analysis(
        qubits=qubits,
        index=1,
        expected_len=3,
        caller="threeq_qst",
    )
    q2 = _select_qubit_for_analysis(
        qubits=qubits,
        index=2,
        expected_len=3,
        caller="threeq_qst",
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
                    q0_uid=q0.uid,
                    q1_uid=q1.uid,
                    q2_uid=q2.uid,
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
    options: ThreeQQstWorkflowOptions,
    readout_calibration_result=None,
) -> None:
    if not bool(options.do_analysis) and (
        bool(options.do_convergence_validation)
        or bool(options.do_shot_sweep_convergence)
    ):
        raise ValueError("Convergence validation and shot sweep require do_analysis=True.")
    if (
        bool(options.do_analysis)
        and not bool(options.do_readout_calibration)
        and readout_calibration_result is None
    ):
        raise ValueError(
            "Analysis-capable threeq_qst bundles require readout calibration. "
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
    options: ThreeQQstWorkflowOptions | None = None,
    analysis_options: ThreeQQstAnalysisOptions | None = None,
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
        q0, q1, q2 = _normalize_three_qubits(temp_qubits)
        analysis_result = analysis_workflow(
            tomography_result=main_output["tomography_result"],
            q0=q0,
            q1=q1,
            q2=q2,
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
    "ThreeQQstExperimentOptions",
    "ThreeQQstWorkflowOptions",
    "ThreeQQstAnalysisOptions",
    "create_experiment",
    "experiment_workflow",
    "convergence_validation_workflow",
    "shot_sweep_workflow",
    "run_bundle",
]

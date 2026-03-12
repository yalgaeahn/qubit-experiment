"""Three-qubit virtual-Z validation workflows under the `three_qubit_virtual_z_validation` name.

This module validates `qop.rz` in two tomography-based stages:
product-state phase tracking and GHZ-tail coherence tracking.
It supports only `INTEGRATION + SINGLE_SHOT`, expects ordered
`qubits=[q0, q1, q2]` and `bus=[b0, b1, b2]`, reuses one readout calibration
run across the full sweep, and returns only top-level summary payloads.
"""

from __future__ import annotations

from numbers import Real
from typing import TYPE_CHECKING, Literal

import numpy as np
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

from qubit_experiment.analysis.three_qubit_virtual_z_validation import (
    ThreeQVZValidationAnalysisOptions,
    collect_virtual_z_run_record,
    plot_phase_tracking,
    plot_quality_summary,
    resolve_analysis_max_mle_iterations,
    resolve_do_plotting,
    summarize_virtual_z_validation,
)

from . import three_qubit_ghz as ghz_experiment
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
    from collections.abc import Sequence

    from laboneq.dsl.quantum import QuantumParameters
    from laboneq.dsl.quantum.qpu import QPU
    from laboneq.dsl.session import Session
    from laboneq_applications.typing import QuantumElements


VALIDATION_STAGES: tuple[str, ...] = ("product", "ghz")
PHASE_TARGETS: tuple[str, ...] = ("q0", "q1", "q2")


@workflow.task_options(base_class=BaseExperimentOptions)
class ThreeQVZValidationExperimentOptions:
    """Options for 3Q virtual-Z validation experiment creation."""

    count: int = workflow.option_field(
        1024,
        description="Number of shots per tomography setting.",
    )
    acquisition_type: AcquisitionType = workflow.option_field(
        AcquisitionType.INTEGRATION,
        description="Acquire integrated complex IQ outcomes for validation tomography.",
    )
    averaging_mode: AveragingMode = workflow.option_field(
        AveragingMode.SINGLE_SHOT,
        description="Use single-shot integrated outcomes for validation tomography.",
    )


@workflow.workflow_options
class ThreeQVZValidationWorkflowOptions:
    """Workflow options for 3Q virtual-Z validation."""

    do_readout_calibration: bool = workflow.option_field(
        True,
        description="Whether to run readout calibration before validation tomography.",
    )
    phase_targets: tuple[str, ...] = workflow.option_field(
        PHASE_TARGETS,
        description="One-hot virtual-Z targets to validate.",
    )
    product_initial_state: str = workflow.option_field(
        "+++",
        description="Initial product state used by the product-stage validation.",
    )
    repeats_per_phase: int = workflow.option_field(
        1,
        description="Number of repeated runs per (stage, target, phase) point.",
    )


def _coerce_workflow_options(
    options: ThreeQVZValidationWorkflowOptions | None,
) -> ThreeQVZValidationWorkflowOptions:
    if options is None:
        return ThreeQVZValidationWorkflowOptions()
    if isinstance(options, ThreeQVZValidationWorkflowOptions):
        return options

    base = getattr(options, "_base", None)
    if isinstance(base, ThreeQVZValidationWorkflowOptions):
        return base

    raise TypeError(
        "options must be a ThreeQVZValidationWorkflowOptions instance or an "
        "experiment_workflow.options() builder."
    )


def _normalize_validation_stage(stage: str) -> Literal["product", "ghz"]:
    text = str(stage).strip().lower()
    if text not in VALIDATION_STAGES:
        raise ValueError(
            "stage must be one of ('product', 'ghz'). "
            f"Received {stage!r}."
        )
    return text  # type: ignore[return-value]


def _normalize_three_qubits(qubits: QuantumElements):
    """Validate qubits input and return exactly three single-qubit elements."""
    qlist = list(validation.validate_and_convert_qubits_sweeps(qubits))
    if len(qlist) != 3:
        raise ValueError(
            "three_qubit_virtual_z_validation expects exactly 3 qubits in `qubits`."
            f" Received {len(qlist)}."
        )
    q0 = validation.validate_and_convert_single_qubit_sweeps(qlist[0])
    q1 = validation.validate_and_convert_single_qubit_sweeps(qlist[1])
    q2 = validation.validate_and_convert_single_qubit_sweeps(qlist[2])
    return q0, q1, q2


def _normalize_three_buses(bus: QuantumElements) -> tuple:
    """Validate bus input and return exactly three bus elements."""
    bus_list = list(bus) if isinstance(bus, (list, tuple)) else [bus]
    if len(bus_list) != 3:
        raise ValueError(
            "three_qubit_virtual_z_validation expects exactly 3 bus elements in `bus`."
            f" Received {len(bus_list)}."
        )

    normalized = []
    seen_uids: set[str] = set()
    for item in bus_list:
        bus_elem = validation.validate_and_convert_single_qubit_sweeps(item)
        uid = getattr(bus_elem, "uid", None)
        if not isinstance(uid, str):
            raise TypeError(f"Invalid bus element type: {type(bus_elem)!r}.")
        if uid in seen_uids:
            raise ValueError(f"Duplicate bus uid in input: {uid!r}.")
        signals = getattr(bus_elem, "signals", {})
        if "drive" not in signals:
            raise ValueError(f"Bus {uid!r} does not define the 'drive' logical signal.")
        if "drive_p" not in signals:
            raise ValueError(
                f"Bus {uid!r} does not define the 'drive_p' logical signal."
            )
        seen_uids.add(uid)
        normalized.append(bus_elem)
    return tuple(normalized)


def _single_qubit_state_token(label: str, *, qubit_index: int) -> str:
    token = label[qubit_index]
    if token in {"+", "-"}:
        return token
    if token == "0":
        return "g"
    if token == "1":
        return "e"
    raise ValueError(f"Unsupported token {token!r} in product_initial_state {label!r}.")


def _normalize_phase_tuple(phase_tuple) -> tuple[float, float, float]:
    if isinstance(phase_tuple, (str, bytes)):
        raise ValueError("phase_tuple must be an iterable of exactly 3 numeric values.")

    try:
        values = tuple(phase_tuple)
    except TypeError as exc:
        raise ValueError(
            "phase_tuple must be an iterable of exactly 3 numeric values."
        ) from exc

    if len(values) != 3:
        raise ValueError(
            "phase_tuple must contain exactly 3 values for (q0, q1, q2). "
            f"Received {len(values)}."
        )

    normalized: list[float] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(
                "phase_tuple entries must be numeric. "
                f"Received index {index}: {value!r}."
            )
        normalized.append(float(value))
    return tuple(normalized)  # type: ignore[return-value]


def _validate_product_qop_contract(qop) -> None:
    required_methods = (
        "active_reset",
        "apply_tomography_prerotation",
        "measure",
        "measure_section_length",
        "passive_reset",
        "prepare_tomography_state",
        "rz",
    )
    missing = [
        name for name in required_methods if not callable(getattr(qop, name, None))
    ]
    if missing:
        missing_display = ", ".join(missing)
        raise TypeError(
            "The current quantum_operations class does not define required "
            "product-stage validation methods. "
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
    return {"readout_calibration_result": result}


@workflow.task(save=False)
def _materialize_readout_calibration_bundle(
    bundle: dict[str, object],
) -> dict[str, object]:
    return dict(bundle)


@workflow.task(save=False)
def _extract_readout_calibration_result(bundle: dict[str, object]):
    return dict(bundle).get("readout_calibration_result")


@workflow.task(save=False)
def _select_qubit_for_analysis(
    qubits: QuantumElements,
    index: int,
    expected_len: int,
    caller: str,
):
    qlist = list(validation.validate_and_convert_qubits_sweeps(qubits))
    if len(qlist) != int(expected_len):
        raise ValueError(
            f"{caller} expects exactly {expected_len} qubits in `qubits`."
            f" Received {len(qlist)}."
        )
    idx = int(index)
    if idx < 0 or idx >= int(expected_len):
        raise ValueError(f"Invalid qubit index {idx} for expected_len={expected_len}.")
    return validation.validate_and_convert_single_qubit_sweeps(qlist[idx])


@workflow.task(save=False)
def resolve_phase_targets(
    phase_targets: Sequence[str],
) -> tuple[str, ...]:
    if phase_targets is None:
        raise ValueError("phase_targets cannot be None.")
    normalized = []
    seen = set()
    for raw in phase_targets:
        target = str(raw).strip().lower()
        if target not in PHASE_TARGETS:
            raise ValueError(
                "phase_targets must contain only 'q0', 'q1', or 'q2'. "
                f"Received {raw!r}."
            )
        if target not in seen:
            seen.add(target)
            normalized.append(target)
    if not normalized:
        raise ValueError("phase_targets must contain at least one target.")
    return tuple(normalized)


@workflow.task(save=False)
def resolve_phase_values(phase_values) -> tuple[float, ...]:
    if isinstance(phase_values, (str, bytes)):
        raise ValueError("phase_values must be an iterable of numeric values.")
    try:
        values = tuple(phase_values)
    except TypeError as exc:
        raise ValueError("phase_values must be an iterable of numeric values.") from exc
    if not values:
        raise ValueError("phase_values must contain at least one value.")

    normalized = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(
                "phase_values entries must be numeric. "
                f"Received index {index}: {value!r}."
            )
        normalized.append(float(value))
    return tuple(normalized)


@workflow.task(save=False)
def resolve_repeats_per_phase(repeats_per_phase: int) -> int:
    repeats = int(repeats_per_phase)
    if repeats < 1:
        raise ValueError("repeats_per_phase must be >= 1.")
    return repeats


@workflow.task(save=False)
def resolve_repeat_indices(repeats_per_phase: int) -> tuple[int, ...]:
    repeats = int(repeats_per_phase)
    if repeats < 1:
        raise ValueError("repeats_per_phase must be >= 1.")
    return tuple(range(1, repeats + 1))


def _resolve_phase_tuple_for_target_impl(
    phase_target: str,
    phase_value: float,
) -> tuple[float, float, float]:
    target = str(phase_target).strip().lower()
    if target == "q0":
        return (float(phase_value), 0.0, 0.0)
    if target == "q1":
        return (0.0, float(phase_value), 0.0)
    if target == "q2":
        return (0.0, 0.0, float(phase_value))
    raise ValueError(
        "phase_target must be one of ('q0', 'q1', 'q2'). "
        f"Received {phase_target!r}."
    )


@workflow.task(save=False)
def resolve_phase_tuple_for_target(
    phase_target: str,
    phase_value: float,
) -> tuple[float, float, float]:
    return _resolve_phase_tuple_for_target_impl(
        phase_target=phase_target,
        phase_value=phase_value,
    )


@workflow.task(save=False)
def should_run_readout_calibration(
    do_readout_calibration: bool,
    readout_calibration_result,
) -> bool:
    return bool(do_readout_calibration) and readout_calibration_result is None


@workflow.task
def validate_analysis_prerequisites(
    do_readout_calibration: bool,
    readout_calibration_result,
) -> None:
    if not do_readout_calibration and readout_calibration_result is None:
        raise ValueError(
            "three_qubit_virtual_z_validation requires readout calibration. Provide "
            "`readout_calibration_result` or set `do_readout_calibration=True`."
        )


def _create_product_stage_experiment_impl(
    qpu: QPU,
    qubits: QuantumElements,
    bus: QuantumElements,
    *,
    phase_tuple: tuple[float, float, float],
    product_initial_state: str,
    options: ThreeQVZValidationExperimentOptions | None = None,
) -> Experiment:
    opts = ThreeQVZValidationExperimentOptions() if options is None else options
    try:
        acquisition_type = AcquisitionType(opts.acquisition_type)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "three_qubit_virtual_z_validation only supports AcquisitionType.INTEGRATION."
        ) from exc
    if acquisition_type != AcquisitionType.INTEGRATION:
        raise ValueError(
            "three_qubit_virtual_z_validation only supports AcquisitionType.INTEGRATION."
        )
    try:
        averaging_mode = AveragingMode(opts.averaging_mode)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "three_qubit_virtual_z_validation only supports AveragingMode.SINGLE_SHOT."
        ) from exc
    if averaging_mode != AveragingMode.SINGLE_SHOT:
        raise ValueError(
            "three_qubit_virtual_z_validation only supports AveragingMode.SINGLE_SHOT."
        )

    q0, q1, q2 = _normalize_three_qubits(qubits)
    _normalize_three_buses(bus)
    phase_tuple = _normalize_phase_tuple(phase_tuple)
    canonical_initial_state = canonical_three_qubit_state_label(product_initial_state)

    q0_token = _single_qubit_state_token(canonical_initial_state, qubit_index=0)
    q1_token = _single_qubit_state_token(canonical_initial_state, qubit_index=1)
    q2_token = _single_qubit_state_token(canonical_initial_state, qubit_index=2)
    q0_token_name = state_token_for_section_name(q0_token)
    q1_token_name = state_token_for_section_name(q1_token)
    q2_token_name = state_token_for_section_name(q2_token)

    qop = qpu.quantum_operations
    _validate_product_qop_contract(qop)
    max_measure_section_length = qop.measure_section_length([q0, q1, q2])

    with dsl.acquire_loop_rt(
        count=opts.count,
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

                with dsl.section(**prep_section_kwargs):
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
                    ) as prep_q2:
                        qop.prepare_tomography_state(q2, q2_token)

                    prep_tail_uid = prep_q2.uid
                    if any(phase != 0.0 for phase in phase_tuple):
                        with dsl.section(
                            name=f"product_virtual_z_{setting_label}",
                            alignment=SectionAlignment.LEFT,
                            play_after=prep_q2.uid,
                        ) as virtual_z_sec:
                            for qubit, phase in zip((q0, q1, q2), phase_tuple):
                                if phase != 0.0:
                                    qop.rz(qubit, angle=phase)
                        prep_tail_uid = virtual_z_sec.uid

                with dsl.section(
                    name=f"basis_{setting_label}",
                    alignment=SectionAlignment.LEFT,
                    play_after=prep_tail_uid,
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


def _create_experiment_impl(
    qpu: QPU,
    qubits: QuantumElements,
    bus: QuantumElements,
    phase_tuple: tuple[float, float, float],
    stage: Literal["product", "ghz"],
    product_initial_state: str = "+++",
    options: ThreeQVZValidationExperimentOptions | None = None,
) -> Experiment:
    stage_name = _normalize_validation_stage(stage)
    if stage_name == "product":
        return _create_product_stage_experiment_impl(
            qpu=qpu,
            qubits=qubits,
            bus=bus,
            phase_tuple=phase_tuple,
            product_initial_state=product_initial_state,
            options=options,
        )

    ghz_options = ThreeQVZValidationExperimentOptions() if options is None else options
    return ghz_experiment._create_experiment_impl(
        qpu=qpu,
        qubits=qubits,
        bus=bus,
        ghz_prep=True,
        final_virtual_z_phases=_normalize_phase_tuple(phase_tuple),
        options=ghz_options,
    )


@workflow.task
@dsl.qubit_experiment
def create_experiment(
    qpu: QPU,
    qubits: QuantumElements,
    bus: QuantumElements,
    phase_tuple: tuple[float, float, float],
    stage: Literal["product", "ghz"],
    product_initial_state: str = "+++",
    options: ThreeQVZValidationExperimentOptions | None = None,
) -> Experiment:
    """Create a 3Q virtual-Z validation experiment for one stage and phase tuple."""
    return _create_experiment_impl(
        qpu=qpu,
        qubits=qubits,
        bus=bus,
        phase_tuple=phase_tuple,
        stage=stage,
        product_initial_state=product_initial_state,
        options=options,
    )


@workflow.workflow(name="three_qubit_virtual_z_validation")
def experiment_workflow(
    session: Session,
    qpu: QPU,
    qubits: QuantumElements,
    bus: QuantumElements,
    phase_values,
    readout_calibration_result=None,
    temporary_parameters: dict[str | tuple[str, str, str], dict | QuantumParameters]
    | None = None,
    options: ThreeQVZValidationWorkflowOptions | None = None,
    analysis_options: ThreeQVZValidationAnalysisOptions | None = None,
) -> None:
    """Run the product and GHZ virtual-Z validation sweeps."""
    opts = ThreeQVZValidationWorkflowOptions() if options is None else options

    temp_qpu = temporary_qpu(qpu, temporary_parameters)
    qubits = temporary_quantum_elements_from_qpu(temp_qpu, qubits)
    bus = temporary_quantum_elements_from_qpu(temp_qpu, bus)

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
        do_readout_calibration=opts.do_readout_calibration,
        readout_calibration_result=stable_calibration_result,
    )

    q0 = _select_qubit_for_analysis(
        qubits=qubits,
        index=0,
        expected_len=3,
        caller="three_qubit_virtual_z_validation",
    )
    q1 = _select_qubit_for_analysis(
        qubits=qubits,
        index=1,
        expected_len=3,
        caller="three_qubit_virtual_z_validation",
    )
    q2 = _select_qubit_for_analysis(
        qubits=qubits,
        index=2,
        expected_len=3,
        caller="three_qubit_virtual_z_validation",
    )
    resolved_phase_targets = resolve_phase_targets(opts.phase_targets)
    resolved_phase_values = resolve_phase_values(phase_values)
    repeats_per_phase = resolve_repeats_per_phase(opts.repeats_per_phase)
    repeat_indices = resolve_repeat_indices(opts.repeats_per_phase)
    max_iterations = resolve_analysis_max_mle_iterations(analysis_options)
    do_plotting = resolve_do_plotting(analysis_options)

    product_raw_run_records = []
    product_failed_runs = []
    with workflow.for_(resolved_phase_targets, lambda target: target) as phase_target:
        with workflow.for_(resolved_phase_values, lambda phase: phase) as phase_value:
            phase_tuple = resolve_phase_tuple_for_target(phase_target, phase_value)
            with workflow.for_(repeat_indices, lambda idx: idx) as repeat_index:
                product_exp = create_experiment(
                    temp_qpu,
                    qubits,
                    bus,
                    phase_tuple=phase_tuple,
                    stage="product",
                    product_initial_state=opts.product_initial_state,
                )
                compiled_product_exp = compile_experiment(session, product_exp)
                product_tomography_result = run_experiment(
                    session,
                    compiled_product_exp,
                )
                product_record_bundle = collect_virtual_z_run_record(
                    stage="product",
                    phase_target=phase_target,
                    phase_value=phase_value,
                    repeat=repeat_index,
                    tomography_result=product_tomography_result,
                    q0_uid=q0.uid,
                    q1_uid=q1.uid,
                    q2_uid=q2.uid,
                    readout_calibration_result=stable_calibration_result,
                    product_initial_state=opts.product_initial_state,
                    max_iterations=max_iterations,
                )
                _append_item(product_raw_run_records, product_record_bundle["record"])
                _append_item_if_present(product_failed_runs, product_record_bundle["failure"])

    product_raw_run_records = _materialize_list(product_raw_run_records)
    product_failed_runs = _materialize_list(product_failed_runs)
    product_summary = summarize_virtual_z_validation(
        stage="product",
        run_records=product_raw_run_records,
    )

    ghz_raw_run_records = []
    ghz_failed_runs = []
    with workflow.for_(resolved_phase_targets, lambda target: target) as phase_target:
        with workflow.for_(resolved_phase_values, lambda phase: phase) as phase_value:
            phase_tuple = resolve_phase_tuple_for_target(phase_target, phase_value)
            with workflow.for_(repeat_indices, lambda idx: idx) as repeat_index:
                ghz_exp = create_experiment(
                    temp_qpu,
                    qubits,
                    bus,
                    phase_tuple=phase_tuple,
                    stage="ghz",
                    product_initial_state=opts.product_initial_state,
                )
                compiled_ghz_exp = compile_experiment(session, ghz_exp)
                ghz_tomography_result = run_experiment(session, compiled_ghz_exp)
                ghz_record_bundle = collect_virtual_z_run_record(
                    stage="ghz",
                    phase_target=phase_target,
                    phase_value=phase_value,
                    repeat=repeat_index,
                    tomography_result=ghz_tomography_result,
                    q0_uid=q0.uid,
                    q1_uid=q1.uid,
                    q2_uid=q2.uid,
                    readout_calibration_result=stable_calibration_result,
                    product_initial_state=opts.product_initial_state,
                    max_iterations=max_iterations,
                )
                _append_item(ghz_raw_run_records, ghz_record_bundle["record"])
                _append_item_if_present(ghz_failed_runs, ghz_record_bundle["failure"])

    ghz_raw_run_records = _materialize_list(ghz_raw_run_records)
    ghz_failed_runs = _materialize_list(ghz_failed_runs)
    ghz_summary = summarize_virtual_z_validation(
        stage="ghz",
        run_records=ghz_raw_run_records,
    )

    with workflow.if_(do_plotting):
        plot_phase_tracking(
            product_run_records=product_raw_run_records,
            ghz_run_records=ghz_raw_run_records,
        )
        plot_quality_summary(
            product_run_records=product_raw_run_records,
            ghz_run_records=ghz_raw_run_records,
        )

    workflow.return_(
        readout_calibration_result=stable_calibration_result,
        product_raw_run_records=product_raw_run_records,
        product_failed_runs=product_failed_runs,
        product_summary=product_summary,
        ghz_raw_run_records=ghz_raw_run_records,
        ghz_failed_runs=ghz_failed_runs,
        ghz_summary=ghz_summary,
        phase_values=resolved_phase_values,
        phase_targets=resolved_phase_targets,
        repeats_per_phase=repeats_per_phase,
    )


def _validate_bundle_configuration(
    options: ThreeQVZValidationWorkflowOptions,
    readout_calibration_result=None,
) -> None:
    if (
        not bool(options.do_readout_calibration)
        and readout_calibration_result is None
    ):
        raise ValueError(
            "three_qubit_virtual_z_validation requires readout calibration. "
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
    phase_values=None,
    readout_calibration_result=None,
    temporary_parameters: dict[str | tuple[str, str, str], dict | QuantumParameters]
    | None = None,
    options: ThreeQVZValidationWorkflowOptions | None = None,
    analysis_options: ThreeQVZValidationAnalysisOptions | None = None,
) -> dict[str, object]:
    """Run the validation workflow and return a plain notebook-friendly dict."""
    opts = _coerce_workflow_options(options)
    _validate_bundle_configuration(opts, readout_calibration_result)

    if phase_values is None:
        phase_values = tuple(float(value) for value in np.linspace(-np.pi, np.pi, 13))

    result = experiment_workflow(
        session=session,
        qpu=qpu,
        qubits=qubits,
        bus=bus,
        phase_values=phase_values,
        readout_calibration_result=readout_calibration_result,
        temporary_parameters=temporary_parameters,
        options=opts,
        analysis_options=analysis_options,
    ).run()
    return _output_to_dict(result.output)


__all__ = [
    "ThreeQVZValidationAnalysisOptions",
    "ThreeQVZValidationExperimentOptions",
    "ThreeQVZValidationWorkflowOptions",
    "create_experiment",
    "experiment_workflow",
    "run_bundle",
    "resolve_phase_targets",
    "resolve_phase_tuple_for_target",
    "resolve_phase_values",
]

"""This module defines the experiments for 3-qubit state tomography with RIP state preparation."""

from __future__ import annotations

from typing import TYPE_CHECKING

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

from analysis.three_qubit_state_tomography import analysis_workflow
from experiments.three_qubit_readout_calibration import (
    create_experiment as create_readout_calibration_experiment,
)
from experiments.three_qubit_tomography_common import (
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
class ThreeQStateTomographyExperimentOptions:
    """Options for 3Q state tomography experiment."""

    count: int = workflow.option_field(
        4096,
        description="Number of shots per tomography setting.",
    )
    acquisition_type: AcquisitionType = workflow.option_field(
        AcquisitionType.INTEGRATION,
        description="Acquire integrated complex IQ outcomes for state tomography.",
    )
    averaging_mode: AveragingMode = workflow.option_field(
        AveragingMode.SINGLE_SHOT,
        description="Use single-shot integrated outcomes for state tomography counts.",
    )


@workflow.workflow_options
class ThreeQStateTomographyWorkflowOptions:
    """Workflow options for 3Q state tomography."""

    do_analysis: bool = workflow.option_field(
        True,
        description="Whether to run tomography analysis workflow.",
    )
    do_readout_calibration: bool = workflow.option_field(
        True,
        description="Whether to run readout calibration before tomography.",
    )
    validation_mode: bool = workflow.option_field(
        False,
        description=(
            "If True, skip RIP entangling pulse and run tomography directly after "
            "initial product-state preparation."
        ),
    )
    use_rip: bool = workflow.option_field(
        True,
        description=(
            "Whether to apply RIP entangling pulse during state preparation. "
            "Ignored (forced False) when validation_mode=True."
        ),
    )
    initial_state: str = workflow.option_field(
        "+++",
        description=(
            "Initial 3-qubit product state for tomography experiment. "
            "Supported labels include binary ('000'..'111'), +/- labels, "
            "and g/e labels ('ggg'..'eee')."
        ),
    )
    enforce_target_match: bool = workflow.option_field(
        True,
        description=(
            "If validation_mode=True, enforce target_state to match initial_state."
        ),
    )


@workflow.workflow(name="three_qubit_state_tomography")
def experiment_workflow(
    session: Session,
    qpu: QPU,
    qubits: QuantumElements,
    bus: QuantumElements,
    readout_calibration_result=None,
    target_state=None,
    temporary_parameters: dict[str | tuple[str, str, str], dict | QuantumParameters]
    | None = None,
    options: ThreeQStateTomographyWorkflowOptions | None = None,
) -> None:
    """Run 3Q tomography with optional RIP preparation.

    Args:
        bus:
            Either a single bus element or a list/tuple of bus elements.
            When multiple buses are provided, RIP drives are played simultaneously
            inside the same RIP section.
    """
    options = ThreeQStateTomographyWorkflowOptions() if options is None else options

    temp_qpu = temporary_qpu(qpu, temporary_parameters)
    qubits = temporary_quantum_elements_from_qpu(temp_qpu, qubits)
    bus = temporary_quantum_elements_from_qpu(temp_qpu, bus)

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

    resolved_config = resolve_validation_configuration(
        validation_mode=options.validation_mode,
        use_rip=options.use_rip,
        initial_state=options.initial_state,
        target_state=target_state,
        enforce_target_match=options.enforce_target_match,
    )

    exp = create_experiment(
        temp_qpu,
        qubits,
        bus,
        use_rip=resolved_config["used_rip"],
        initial_state=resolved_config["initial_state"],
    )
    compiled_exp = compile_experiment(session, exp)
    tomography_result = run_experiment(session, compiled_exp)

    q0 = _select_qubit_for_analysis(
        qubits=qubits,
        index=0,
        expected_len=3,
        caller="three_qubit_state_tomography",
    )
    q1 = _select_qubit_for_analysis(
        qubits=qubits,
        index=1,
        expected_len=3,
        caller="three_qubit_state_tomography",
    )
    q2 = _select_qubit_for_analysis(
        qubits=qubits,
        index=2,
        expected_len=3,
        caller="three_qubit_state_tomography",
    )

    analysis_result = None
    with workflow.if_(options.do_analysis):
        analysis_result = analysis_workflow(
            tomography_result=tomography_result,
            q0=q0,
            q1=q1,
            q2=q2,
            readout_calibration_result=calibration_result,
            target_state=resolved_config["target_state_effective"],
        ).output

    workflow.return_(
        {
            "tomography_result": tomography_result,
            "readout_calibration_result": calibration_result,
            "analysis_result": analysis_result,
            "validation_mode": resolved_config["validation_mode"],
            "initial_state": resolved_config["initial_state"],
            "used_rip": resolved_config["used_rip"],
            "target_state_effective": resolved_config["target_state_effective"],
        }
    )


@workflow.task
def resolve_validation_configuration(
    validation_mode: bool,
    use_rip: bool,
    initial_state: str,
    target_state=None,
    enforce_target_match: bool = True,
) -> dict[str, object]:
    """Resolve RIP usage and target-state policy for validation mode."""
    canonical_initial_state = _canonical_initial_state_label(initial_state)
    used_rip = bool(use_rip) and not bool(validation_mode)

    effective_target_state = target_state
    if validation_mode:
        if target_state is None:
            effective_target_state = canonical_initial_state
        elif enforce_target_match:
            if _canonical_target_state_label(target_state) != canonical_initial_state:
                raise ValueError(
                    "In validation_mode, target_state must match initial_state. "
                    f"Got target_state={target_state!r}, initial_state={initial_state!r}."
                )

    return {
        "validation_mode": bool(validation_mode),
        "used_rip": bool(used_rip),
        "initial_state": canonical_initial_state,
        "target_state_effective": effective_target_state,
    }


@workflow.task
def validate_analysis_prerequisites(
    do_analysis: bool,
    do_readout_calibration: bool,
    readout_calibration_result,
) -> None:
    """Validate that analysis has the required readout calibration input."""
    if do_analysis and not do_readout_calibration and readout_calibration_result is None:
        raise ValueError(
            "Analysis requires readout calibration. Provide "
            "`readout_calibration_result` or set `do_readout_calibration=True`."
        )


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


def _canonical_initial_state_label(state: str) -> str:
    """Normalize 3Q product-state label."""
    try:
        return canonical_three_qubit_state_label(state)
    except ValueError as exc:
        raise ValueError(
            "Unsupported initial_state. Use binary labels ('000'..'111'), "
            "+/- labels, or g/e labels ('ggg'..'eee')."
        ) from exc


def _canonical_target_state_label(target_state) -> str:
    """Normalize target-state string labels used for validation matching."""
    if not isinstance(target_state, str):
        return str(target_state)
    s = target_state.strip().lower().replace(" ", "")
    try:
        return canonical_three_qubit_state_label(s)
    except ValueError:
        return s


def _single_qubit_state_token(label: str, *, qubit_index: int) -> str:
    """Extract and map a single-qubit token from canonical 3Q label."""
    token = label[qubit_index]
    if token in {"+", "-"}:
        return token
    if token == "0":
        return "g"
    if token == "1":
        return "e"
    raise ValueError(f"Unsupported token {token!r} in initial_state {label!r}.")


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
            "tomography methods for three_qubit_state_tomography. "
            f"class={type(qop).__name__!r}, missing=[{missing_display}]."
        )


def _normalize_bus_elements(bus: QuantumElements) -> list:
    """Normalize bus input to a non-empty list of unique bus elements."""
    bus_list = list(bus) if isinstance(bus, (list, tuple)) else [bus]
    if len(bus_list) == 0:
        raise ValueError("bus cannot be an empty list.")

    normalized = []
    seen_uids: set[str] = set()
    for item in bus_list:
        bus_elem = validation.validate_and_convert_single_qubit_sweeps(item)
        uid = getattr(bus_elem, "uid", None)
        if not isinstance(uid, str):
            raise TypeError(f"Invalid bus element type: {type(bus_elem)!r}.")
        if uid in seen_uids:
            raise ValueError(f"Duplicate bus uid in input: {uid!r}.")
        seen_uids.add(uid)
        normalized.append(bus_elem)
    return normalized


def _resolve_bus_rf_frequency(bus_elem) -> float:
    """Resolve bus RF frequency from bus parameters."""
    resonance = getattr(bus_elem.parameters, "resonance_frequency_bus", None)
    if resonance is None:
        raise ValueError(
            f"Bus {bus_elem.uid!r} requires parameters.resonance_frequency_bus "
            "when use_rip=True."
        )
    detuning = getattr(bus_elem.parameters, "rip_detuning", None) or 0.0
    return float(resonance + detuning)


def _resolve_bus_rip_phase(bus_elem) -> float:
    """Resolve RIP phase from bus parameters with a pi/2 fallback."""
    phase = getattr(bus_elem.parameters, "rip_phase", None)
    if phase is None:
        return float(np.pi / 2)
    return float(phase)


@workflow.task
@dsl.qubit_experiment
def create_experiment(
    qpu: QPU,
    qubits: QuantumElements,
    bus: QuantumElements,
    use_rip: bool = True,
    initial_state: str = "+++",
    options: ThreeQStateTomographyExperimentOptions | None = None,
) -> Experiment:
    """Create 3Q tomography experiment with optional RIP state preparation.

    RIP parameters are resolved per bus from element parameters:
    `resonance_frequency_bus + rip_detuning`, `rip_amplitude`, `rip_length`,
    and `rip_phase`.
    """
    opts = ThreeQStateTomographyExperimentOptions() if options is None else options
    if AcquisitionType(opts.acquisition_type) != AcquisitionType.INTEGRATION:
        raise ValueError(
            "three_qubit_state_tomography only supports AcquisitionType.INTEGRATION."
        )
    if AveragingMode(opts.averaging_mode) != AveragingMode.SINGLE_SHOT:
        raise ValueError(
            "three_qubit_state_tomography only supports AveragingMode.SINGLE_SHOT."
        )

    q0, q1, q2 = _normalize_three_qubits(qubits)
    buses = _normalize_bus_elements(bus)

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

    with dsl.acquire_loop_rt(
        count=opts.count,
        averaging_mode=opts.averaging_mode,
        acquisition_type=opts.acquisition_type,
        repetition_mode=opts.repetition_mode,
        repetition_time=opts.repetition_time,
        reset_oscillator_phase=opts.reset_oscillator_phase,
    ):
        if use_rip:
            # set_bus_frequency can only be called once per signal in one experiment
            for bus_elem in buses:
                qop.set_bus_frequency(
                    bus_elem,
                    frequency=_resolve_bus_rf_frequency(bus_elem),
                )

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

                basis_play_after = prep_sec.uid
                if use_rip:
                    with dsl.section(
                        name=f"rip_{setting_label}",
                        alignment=SectionAlignment.LEFT,
                        play_after=prep_sec.uid,
                    ) as rip_sec:
                        for bus_elem in buses:
                            qop.rip(
                                bus_elem,
                                phase=_resolve_bus_rip_phase(bus_elem),
                            )
                    basis_play_after = rip_sec.uid

                with dsl.section(
                    name=f"basis_{setting_label}",
                    alignment=SectionAlignment.LEFT,
                    play_after=basis_play_after,
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


def _normalize_three_qubits(qubits: QuantumElements):
    """Validate qubits input and return exactly three single-qubit elements."""
    qlist = list(validation.validate_and_convert_qubits_sweeps(qubits))
    if len(qlist) != 3:
        raise ValueError(
            "three_qubit_state_tomography expects exactly 3 qubits in `qubits`."
            f" Received {len(qlist)}."
        )
    q0 = validation.validate_and_convert_single_qubit_sweeps(qlist[0])
    q1 = validation.validate_and_convert_single_qubit_sweeps(qlist[1])
    q2 = validation.validate_and_convert_single_qubit_sweeps(qlist[2])
    return q0, q1, q2

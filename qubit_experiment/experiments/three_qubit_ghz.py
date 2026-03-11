"""GHZ-specific three-qubit tomography workflows under the `three_qubit_ghz` name.

This module creates a fixed GHZ+ preparation circuit followed by 3Q tomography.
It supports only `INTEGRATION + SINGLE_SHOT`, expects ordered
`qubits=[q0, q1, q2]` and `bus=[b0, b1, b2]`, and uses `drive`/`drive_p`
RIP sections for the two CZ blocks.
"""

from __future__ import annotations

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

from qubit_experiment.analysis.three_qubit_ghz import (
    ThreeQGhzAnalysisOptions,
    analysis_workflow,
    analyze_tomography_run,
    collect_convergence_run_record,
    plot_convergence_fidelity,
    summarize_statistical_convergence,
)

from .three_qubit_readout_calibration import (
    create_experiment as create_readout_calibration_experiment,
)
from .three_qubit_tomography_common import (
    TOMOGRAPHY_SETTINGS,
    tomography_handle,
)

if TYPE_CHECKING:
    from laboneq.dsl.quantum import QuantumParameters
    from laboneq.dsl.quantum.qpu import QPU
    from laboneq.dsl.session import Session
    from laboneq_applications.typing import QuantumElements


@workflow.task_options(base_class=BaseExperimentOptions)
class ThreeQGhzExperimentOptions:
    """Options for GHZ-specific 3Q experiment creation."""

    count: int = workflow.option_field(
        4096,
        description="Number of shots per tomography setting.",
    )
    acquisition_type: AcquisitionType = workflow.option_field(
        AcquisitionType.INTEGRATION,
        description="Acquire integrated complex IQ outcomes for GHZ tomography.",
    )
    averaging_mode: AveragingMode = workflow.option_field(
        AveragingMode.SINGLE_SHOT,
        description="Use single-shot integrated outcomes for GHZ tomography counts.",
    )


@workflow.workflow_options
class ThreeQGhzWorkflowOptions:
    """Workflow options for GHZ-specific 3Q tomography."""

    do_analysis: bool = workflow.option_field(
        True,
        description="Whether to run the main single-run GHZ analysis workflow.",
    )
    do_readout_calibration: bool = workflow.option_field(
        True,
        description="Whether to run readout calibration before tomography.",
    )
    ghz_prep: bool = workflow.option_field(
        True,
        description="Whether to run the fixed internal GHZ preparation block.",
    )
    do_convergence_validation: bool = workflow.option_field(
        False,
        description="Whether to run repeated GHZ convergence validation.",
    )
    convergence_repeats: int = workflow.option_field(
        2,
        description="Number of repeated GHZ runs in convergence validation.",
    )
    convergence_do_plotting: bool = workflow.option_field(
        False,
        description="Whether to generate the GHZ convergence summary plot artifact.",
    )


def _coerce_workflow_options(
    options: ThreeQGhzWorkflowOptions | None,
) -> ThreeQGhzWorkflowOptions:
    if options is None:
        return ThreeQGhzWorkflowOptions()
    if isinstance(options, ThreeQGhzWorkflowOptions):
        return options

    base = getattr(options, "_base", None)
    if isinstance(base, ThreeQGhzWorkflowOptions):
        return base

    raise TypeError(
        "options must be a ThreeQGhzWorkflowOptions instance or an "
        "experiment_workflow.options() builder."
    )


def _coerce_analysis_options(
    analysis_options: ThreeQGhzAnalysisOptions | None,
) -> ThreeQGhzAnalysisOptions:
    if analysis_options is None:
        return ThreeQGhzAnalysisOptions()
    if isinstance(analysis_options, ThreeQGhzAnalysisOptions):
        return analysis_options

    base = getattr(analysis_options, "_base", None)
    if isinstance(base, ThreeQGhzAnalysisOptions):
        return base

    raise TypeError(
        "analysis_options must be a ThreeQGhzAnalysisOptions instance or an "
        "analysis_workflow.options() builder."
    )


def _normalize_three_qubits(qubits: QuantumElements):
    """Validate qubits input and return exactly three single-qubit elements."""
    qlist = list(validation.validate_and_convert_qubits_sweeps(qubits))
    if len(qlist) != 3:
        raise ValueError(
            "three_qubit_ghz expects exactly 3 qubits in `qubits`."
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
            "three_qubit_ghz expects exactly 3 bus elements in `bus`."
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


def _resolve_bus_rf_frequency(
    bus_elem,
    *,
    line: Literal["drive", "drive_p"],
) -> float:
    """Resolve RF frequency for the requested bus line from element parameters."""
    if line == "drive":
        resonance = getattr(bus_elem.parameters, "resonance_frequency_bus", None)
        detuning = getattr(bus_elem.parameters, "rip_detuning", None) or 0.0
        field = "resonance_frequency_bus"
    else:
        resonance = getattr(bus_elem.parameters, "resonance_frequency_bus_p", None)
        detuning = getattr(bus_elem.parameters, "rip_p_detuning", None) or 0.0
        field = "resonance_frequency_bus_p"

    if resonance is None:
        raise ValueError(
            f"Bus {bus_elem.uid!r} requires parameters.{field} when line={line!r}."
        )
    return float(resonance + detuning)


def _validate_tomography_qop_contract(qop) -> None:
    """Ensure required GHZ tomography qop methods exist on the operation set."""
    required_methods = (
        "apply_tomography_prerotation",
        "measure_section_length",
        "rip",
        "ry",
        "set_bus_frequency",
    )
    missing = [
        name for name in required_methods if not callable(getattr(qop, name, None))
    ]
    if missing:
        missing_display = ", ".join(missing)
        raise TypeError(
            "The current quantum_operations class does not define required "
            "GHZ tomography methods for three_qubit_ghz. "
            f"class={type(qop).__name__!r}, missing=[{missing_display}]."
        )


@workflow.task(save=False)
def _append_item(items: list, item) -> None:
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
    analysis_options: ThreeQGhzAnalysisOptions | None = None,
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
def resolve_convergence_repeat_indices(repeats: int) -> tuple[int, ...]:
    repeats = int(repeats)
    if repeats < 1:
        raise ValueError("convergence_repeats must be >= 1.")
    return tuple(range(repeats))


@workflow.task(save=False)
def resolve_convergence_repeats(repeats: int) -> int:
    repeats = int(repeats)
    if repeats < 1:
        raise ValueError("convergence_repeats must be >= 1.")
    return repeats


@workflow.task(save=False)
def should_run_readout_calibration(
    do_readout_calibration: bool,
    readout_calibration_result,
) -> bool:
    """Resolve whether the workflow should acquire a fresh readout calibration."""
    return bool(do_readout_calibration) and readout_calibration_result is None


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
) -> None:
    """Validate coupled workflow options that require analysis outputs."""
    if not bool(do_analysis) and bool(do_convergence_validation):
        raise ValueError("Convergence validation requires do_analysis=True.")


def _create_experiment_impl(
    qpu: QPU,
    qubits: QuantumElements,
    bus: QuantumElements,
    ghz_prep: bool = True,
    options: ThreeQGhzExperimentOptions | None = None,
) -> Experiment:
    opts = ThreeQGhzExperimentOptions() if options is None else options
    try:
        acquisition_type = AcquisitionType(opts.acquisition_type)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "three_qubit_ghz only supports AcquisitionType.INTEGRATION."
        ) from exc
    if acquisition_type != AcquisitionType.INTEGRATION:
        raise ValueError(
            "three_qubit_ghz only supports AcquisitionType.INTEGRATION."
        )
    try:
        averaging_mode = AveragingMode(opts.averaging_mode)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "three_qubit_ghz only supports AveragingMode.SINGLE_SHOT."
        ) from exc
    if averaging_mode != AveragingMode.SINGLE_SHOT:
        raise ValueError(
            "three_qubit_ghz only supports AveragingMode.SINGLE_SHOT."
        )
    if not bool(ghz_prep):
        raise ValueError("three_qubit_ghz only supports ghz_prep=True.")

    q0, q1, q2 = _normalize_three_qubits(qubits)
    buses = _normalize_three_buses(bus)

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
        for bus_elem in buses:
            qop.set_bus_frequency(
                bus_elem,
                frequency=_resolve_bus_rf_frequency(bus_elem, line="drive"),
                line="drive",
            )
        for bus_elem in buses:
            qop.set_bus_frequency(
                bus_elem,
                frequency=_resolve_bus_rf_frequency(bus_elem, line="drive_p"),
                line="drive_p",
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
                    "name": f"ghz_prep_{setting_label}",
                    "alignment": SectionAlignment.LEFT,
                }
                if prep_play_after is not None:
                    prep_section_kwargs["play_after"] = prep_play_after

                with dsl.section(**prep_section_kwargs) as prep_sec:
                    with dsl.section(
                        name=f"ghz_q0_ry90_{setting_label}",
                        alignment=SectionAlignment.LEFT,
                    ) as prep_q0:
                        qop.ry(q0, angle=np.pi / 2)

                    with dsl.section(
                        name=f"ghz_q1_ry90_{setting_label}",
                        alignment=SectionAlignment.LEFT,
                        play_after=prep_q0.uid,
                    ) as prep_q1:
                        qop.ry(q1, angle=np.pi / 2)

                    with dsl.section(
                        name=f"ghz_cz1_{setting_label}",
                        alignment=SectionAlignment.LEFT,
                        play_after=prep_q1.uid,
                    ) as cz1:
                        for bus_elem in buses:
                            qop.rip(bus_elem, line="drive")

                    with dsl.section(
                        name=f"ghz_q1_ry_minus_90_{setting_label}",
                        alignment=SectionAlignment.LEFT,
                        play_after=cz1.uid,
                    ) as prep_q1_post:
                        qop.ry(q1, angle=-np.pi / 2)

                    with dsl.section(
                        name=f"ghz_q2_ry90_{setting_label}",
                        alignment=SectionAlignment.LEFT,
                        play_after=prep_q1_post.uid,
                    ) as prep_q2:
                        qop.ry(q2, angle=np.pi / 2)

                    with dsl.section(
                        name=f"ghz_cz2_{setting_label}",
                        alignment=SectionAlignment.LEFT,
                        play_after=prep_q2.uid,
                    ) as cz2:
                        for bus_elem in buses:
                            qop.rip(bus_elem, line="drive_p")

                    with dsl.section(
                        name=f"ghz_q2_ry_minus_90_{setting_label}",
                        alignment=SectionAlignment.LEFT,
                        play_after=cz2.uid,
                    ):
                        qop.ry(q2, angle=-np.pi / 2)

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
    ghz_prep: bool = True,
    options: ThreeQGhzExperimentOptions | None = None,
) -> Experiment:
    """Create a GHZ-specific 3Q tomography experiment."""
    return _create_experiment_impl(
        qpu=qpu,
        qubits=qubits,
        bus=bus,
        ghz_prep=ghz_prep,
        options=options,
    )


@workflow.workflow(name="three_qubit_ghz")
def experiment_workflow(
    session: Session,
    qpu: QPU,
    qubits: QuantumElements,
    bus: QuantumElements,
    readout_calibration_result=None,
    temporary_parameters: dict[str | tuple[str, str, str], dict | QuantumParameters]
    | None = None,
    options: ThreeQGhzWorkflowOptions | None = None,
) -> None:
    """Run the canonical raw GHZ-specific 3Q tomography workflow."""
    opts = ThreeQGhzWorkflowOptions() if options is None else options

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
        do_analysis=opts.do_analysis,
        do_readout_calibration=opts.do_readout_calibration,
        readout_calibration_result=stable_calibration_result,
    )

    exp = create_experiment(
        temp_qpu,
        qubits,
        bus,
        ghz_prep=opts.ghz_prep,
    )
    compiled_exp = compile_experiment(session, exp)
    tomography_result = run_experiment(session, compiled_exp)

    workflow.return_(
        tomography_result=tomography_result,
        readout_calibration_result=stable_calibration_result,
        ghz_prep=bool(opts.ghz_prep),
        target_state_effective="ghz",
    )


@workflow.workflow(name="three_qubit_ghz_convergence")
def convergence_validation_workflow(
    session: Session,
    qpu: QPU,
    qubits: QuantumElements,
    bus: QuantumElements,
    readout_calibration_result=None,
    main_run_optimization_convergence=None,
    temporary_parameters: dict[str | tuple[str, str, str], dict | QuantumParameters]
    | None = None,
    options: ThreeQGhzWorkflowOptions | None = None,
    analysis_options: ThreeQGhzAnalysisOptions | None = None,
) -> None:
    """Run repeated GHZ convergence validation for 3Q tomography."""
    opts = ThreeQGhzWorkflowOptions() if options is None else options

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

    q0 = _select_qubit_for_analysis(
        qubits=qubits,
        index=0,
        expected_len=3,
        caller="three_qubit_ghz",
    )
    q1 = _select_qubit_for_analysis(
        qubits=qubits,
        index=1,
        expected_len=3,
        caller="three_qubit_ghz",
    )
    q2 = _select_qubit_for_analysis(
        qubits=qubits,
        index=2,
        expected_len=3,
        caller="three_qubit_ghz",
    )
    repeat_indices = resolve_convergence_repeat_indices(opts.convergence_repeats)
    repeats = resolve_convergence_repeats(opts.convergence_repeats)
    raw_run_records = []
    analysis_max_iterations = _resolve_analysis_max_mle_iterations(analysis_options)

    with workflow.for_(repeat_indices, lambda idx: idx) as repeat_index:
        ghz_exp = create_experiment(
            temp_qpu,
            qubits,
            bus,
            ghz_prep=opts.ghz_prep,
        )
        compiled_ghz_exp = compile_experiment(session, ghz_exp)
        ghz_tomography_result = run_experiment(session, compiled_ghz_exp)
        ghz_analysis_result = analyze_tomography_run(
            tomography_result=ghz_tomography_result,
            q0_uid=q0.uid,
            q1_uid=q1.uid,
            q2_uid=q2.uid,
            readout_calibration_result=stable_calibration_result,
            max_iterations=analysis_max_iterations,
        )
        record = collect_convergence_run_record(
            state_label="ghz",
            repeat_index=repeat_index,
            analysis_result=ghz_analysis_result,
        )
        _append_item(raw_run_records, record)

    raw_run_records = _materialize_list(raw_run_records)
    statistical_convergence = summarize_statistical_convergence(raw_run_records)
    with workflow.if_(opts.convergence_do_plotting):
        plot_convergence_fidelity(statistical_convergence=statistical_convergence)

    workflow.return_(
        repeats=repeats,
        raw_run_records=raw_run_records,
        statistical_convergence=statistical_convergence,
        main_run_optimization_convergence=main_run_optimization_convergence,
    )


def _validate_bundle_configuration(
    options: ThreeQGhzWorkflowOptions,
    readout_calibration_result=None,
) -> None:
    if not bool(options.do_analysis) and bool(options.do_convergence_validation):
        raise ValueError("Convergence validation requires do_analysis=True.")
    if (
        bool(options.do_analysis)
        and not bool(options.do_readout_calibration)
        and readout_calibration_result is None
    ):
        raise ValueError(
            "Analysis-capable three_qubit_ghz bundles require readout calibration. "
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
    temporary_parameters: dict[str | tuple[str, str, str], dict | QuantumParameters]
    | None = None,
    options: ThreeQGhzWorkflowOptions | None = None,
    analysis_options: ThreeQGhzAnalysisOptions | None = None,
) -> dict[str, object]:
    """Run split GHZ workflows and assemble a notebook-friendly plain dict."""
    opts = _coerce_workflow_options(options)
    analysis_opts = _coerce_analysis_options(analysis_options)
    _validate_bundle_configuration(opts, readout_calibration_result)

    main_result = experiment_workflow(
        session=session,
        qpu=qpu,
        qubits=qubits,
        bus=bus,
        readout_calibration_result=readout_calibration_result,
        temporary_parameters=temporary_parameters,
        options=opts,
    ).run()
    main_output = _output_to_dict(main_result.output)

    analysis_result = None
    convergence_report = None

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

    return {
        "tomography_result": main_output.get("tomography_result"),
        "readout_calibration_result": main_output.get("readout_calibration_result"),
        "analysis_result": analysis_result,
        "convergence_report": convergence_report,
        "ghz_prep": bool(main_output.get("ghz_prep", False)),
        "target_state_effective": main_output.get("target_state_effective"),
    }


__all__ = [
    "ThreeQGhzExperimentOptions",
    "ThreeQGhzWorkflowOptions",
    "ThreeQGhzAnalysisOptions",
    "create_experiment",
    "experiment_workflow",
    "convergence_validation_workflow",
    "run_bundle",
]

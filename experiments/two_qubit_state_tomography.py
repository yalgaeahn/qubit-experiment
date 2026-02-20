"""This module defines the experiments for 2-qubit state tomography with RIP state preparation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from laboneq import workflow
from laboneq.dsl.enums import AcquisitionType, AveragingMode
from laboneq.simple import Experiment, SectionAlignment, dsl
from laboneq.workflow.tasks import compile_experiment, run_experiment

from analysis.two_qubit_state_tomography import (
    TwoQStateTomographyAnalysisOptions,
    analysis_workflow,
    summarize_statistical_convergence,
)
from experiments.two_qubit_readout_calibration import (
    create_experiment as create_readout_calibration_experiment,
)
from experiments.two_qubit_tomography_common import (
    TOMOGRAPHY_SETTINGS,
    tomography_handle,
)
from laboneq_applications.core import validation
from laboneq_applications.experiments.options import BaseExperimentOptions
from laboneq_applications.tasks.parameter_updating import (
    temporary_qpu,
    temporary_quantum_elements_from_qpu,
)

if TYPE_CHECKING:
    from laboneq.dsl.quantum import QuantumParameters
    from laboneq.dsl.quantum.qpu import QPU
    from laboneq.dsl.session import Session

    from laboneq_applications.typing import QuantumElements


@workflow.task_options(base_class=BaseExperimentOptions)
class TwoQStateTomographyExperimentOptions:
    """Options for 2Q state tomography experiment."""

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
class TwoQStateTomographyWorkflowOptions:
    """Workflow options for 2Q state tomography."""

    do_analysis: bool = workflow.option_field(
        True,
        description="Whether to run tomography analysis workflow.",
    )
    do_readout_calibration: bool = workflow.option_field(
        True,
        description="Whether to run readout calibration before tomography.",
    )
    do_convergence_validation: bool = workflow.option_field(
        False,
        description="Whether to run repeated state-suite convergence validation runs.",
    )
    convergence_repeats_per_state: int = workflow.option_field(
        3,
        description="Number of repeated runs per state in convergence validation.",
    )
    convergence_suite_states: tuple[str, ...] = workflow.option_field(
        ("00", "01", "10", "11", "++"),
        description=(
            "State labels used for statistical convergence suite. "
            "Default minimal product suite: ('00','01','10','11','++')."
        ),
    )
    convergence_do_plotting: bool = workflow.option_field(
        False,
        description="Whether to keep plotting enabled for internal convergence runs.",
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
        "++",
        description=(
            "Initial 2-qubit product state for tomography experiment. "
            "Supported labels: '++', '00', '01', '10', '11', 'gg', 'ge', 'eg', 'ee'."
        ),
    )
    enforce_target_match: bool = workflow.option_field(
        True,
        description=(
            "If validation_mode=True, enforce target_state to match initial_state."
        ),
    )


@workflow.task(save=False)
def _append_item(items: list, item) -> None:
    items.append(item)


@workflow.task(save=False)
def _materialize_list(items: list) -> list:
    return list(items)


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


@workflow.workflow(name="two_qubit_state_tomography")
def experiment_workflow(
    session: Session,
    qpu: QPU,
    ctrl: QuantumElements,
    targ: QuantumElements,
    bus: QuantumElements,
    bus_frequency: float,
    rip_amplitude: float,
    rip_length: float,
    rip_phase: float = np.pi / 2,
    readout_calibration_result=None,
    target_state=None,
    temporary_parameters: dict[str | tuple[str, str, str], dict | QuantumParameters]
    | None = None,
    options: TwoQStateTomographyWorkflowOptions | None = None,
) -> None:
    """Run 2Q tomography with RIP preparation and optional readout calibration."""
    options = (
        TwoQStateTomographyWorkflowOptions() if options is None else options
    )

    temp_qpu = temporary_qpu(qpu, temporary_parameters)
    ctrl = temporary_quantum_elements_from_qpu(temp_qpu, ctrl)
    targ = temporary_quantum_elements_from_qpu(temp_qpu, targ)
    bus = temporary_quantum_elements_from_qpu(temp_qpu, bus)

    calibration_result = readout_calibration_result
    with workflow.if_(
        options.do_readout_calibration and readout_calibration_result is None
    ):
        readout_cal_exp = create_readout_calibration_experiment(temp_qpu, ctrl, targ)
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
        ctrl,
        targ,
        bus,
        bus_frequency=bus_frequency,
        rip_amplitude=rip_amplitude,
        rip_length=rip_length,
        rip_phase=rip_phase,
        use_rip=resolved_config["used_rip"],
        initial_state=resolved_config["initial_state"],
    )
    compiled_exp = compile_experiment(session, exp)
    tomography_result = run_experiment(session, compiled_exp)

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
            repeats_per_state=options.convergence_repeats_per_state
        )
        repeats_per_state = resolve_convergence_repeats_per_state(
            repeats_per_state=options.convergence_repeats_per_state
        )
        conv_analysis_options = TwoQStateTomographyAnalysisOptions()
        conv_analysis_options.do_plotting = bool(options.convergence_do_plotting)
        raw_run_records = []

        with workflow.for_(suite_states, lambda state: state) as state_label:
            with workflow.for_(repeat_indices, lambda idx: idx) as repeat_index:
                suite_resolved_config = resolve_validation_configuration(
                    validation_mode=True,
                    use_rip=False,
                    initial_state=state_label,
                    target_state=state_label,
                    enforce_target_match=True,
                )
                suite_exp = create_experiment(
                    temp_qpu,
                    ctrl,
                    targ,
                    bus,
                    bus_frequency=bus_frequency,
                    rip_amplitude=rip_amplitude,
                    rip_length=rip_length,
                    rip_phase=rip_phase,
                    use_rip=suite_resolved_config["used_rip"],
                    initial_state=suite_resolved_config["initial_state"],
                )
                compiled_suite_exp = compile_experiment(session, suite_exp)
                suite_tomography_result = run_experiment(session, compiled_suite_exp)
                suite_analysis_result = analysis_workflow(
                    tomography_result=suite_tomography_result,
                    ctrl=ctrl,
                    targ=targ,
                    readout_calibration_result=calibration_result,
                    target_state=suite_resolved_config["target_state_effective"],
                    options=conv_analysis_options,
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
        convergence_report = {
            "suite_states": suite_states,
            "repeats_per_state": repeats_per_state,
            "main_run_optimization_convergence": main_run_convergence,
            "statistical_convergence": statistical_convergence,
            "raw_run_records": raw_run_records,
        }

    workflow.return_(
        {
            "tomography_result": tomography_result,
            "readout_calibration_result": calibration_result,
            "analysis_result": analysis_result,
            "convergence_report": convergence_report,
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


@workflow.task
def resolve_convergence_suite_states(
    suite_states: tuple[str, ...] | list[str],
    repeats_per_state: int,
) -> tuple[str, ...]:
    """Validate and normalize state-suite labels for convergence checks."""
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


def _unwrap_workflow_output_like(result_like):
    """Unwrap workflow/task wrappers to a concrete output payload."""
    current = result_like
    for _ in range(8):
        if current is None:
            return None
        if hasattr(current, "output"):
            current = current.output
            continue
        return current
    return current


def _is_reference_like(obj) -> bool:
    return obj is not None and obj.__class__.__name__ == "Reference"


def _contains_reference(obj, depth=0, max_depth=12) -> bool:
    if depth > max_depth:
        return False
    cur = _unwrap_workflow_output_like(obj)
    if _is_reference_like(cur):
        return True
    if isinstance(cur, dict):
        return any(_contains_reference(v, depth + 1, max_depth) for v in cur.values())
    if isinstance(cur, (list, tuple)):
        return any(_contains_reference(v, depth + 1, max_depth) for v in cur)
    return False


def _iter_tasks(node):
    tasks = getattr(node, "tasks", None)
    if tasks is None:
        return []
    try:
        return list(tasks)
    except Exception:
        return []


def _task_output(tasks, key):
    try:
        out = _unwrap_workflow_output_like(tasks[key].output)
    except Exception:
        return None
    if _is_reference_like(out):
        return None
    return out


def _assemble_analysis_output_from_tasks(analysis_node):
    tasks = getattr(analysis_node, "tasks", None)
    if tasks is None:
        return None

    mle = _task_output(tasks, "maximum_likelihood_reconstruct")
    state_metrics = _task_output(tasks, "calculate_state_metrics")
    optimization = _task_output(tasks, "evaluate_optimization_convergence")

    if not isinstance(mle, dict):
        return None
    if not isinstance(state_metrics, dict):
        state_metrics = {}
    if not isinstance(optimization, dict):
        optimization = {}

    return {
        "optimizer_success": mle.get("optimizer_success"),
        "negative_log_likelihood": mle.get("negative_log_likelihood"),
        "metrics": state_metrics,
        "optimization_convergence": optimization,
    }


def _find_analysis_node(root, depth=0, max_depth=12):
    if root is None or depth > max_depth:
        return None
    tasks = _iter_tasks(root)
    if tasks:
        names = {getattr(t, "name", "") for t in tasks}
        if "maximum_likelihood_reconstruct" in names and "extract_assignment_matrix" in names:
            return root
        for t in tasks:
            found = _find_analysis_node(t, depth + 1, max_depth)
            if found is not None:
                return found
            found = _find_analysis_node(getattr(t, "output", None), depth + 1, max_depth)
            if found is not None:
                return found
    out = getattr(root, "output", None)
    if out is not None and out is not root:
        return _find_analysis_node(out, depth + 1, max_depth)
    return None


def _materialize_analysis_output(result_like):
    out = _unwrap_workflow_output_like(result_like)
    if isinstance(out, dict):
        if isinstance(out.get("analysis_result"), dict):
            nested = out["analysis_result"]
            if not _contains_reference(nested):
                return nested
        if not _contains_reference(out):
            return out

    analysis_node = _find_analysis_node(result_like)
    if analysis_node is None:
        return out if isinstance(out, dict) else None

    node_out = _unwrap_workflow_output_like(getattr(analysis_node, "output", None))
    if isinstance(node_out, dict) and not _contains_reference(node_out):
        return node_out

    assembled = _assemble_analysis_output_from_tasks(analysis_node)
    if isinstance(assembled, dict):
        return assembled

    return out if isinstance(out, dict) else None


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


@workflow.task
def collect_convergence_run_record(
    state_label: str,
    repeat_index: int,
    analysis_result,
) -> dict[str, object]:
    """Extract compact per-run convergence record from analysis output."""
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
    """Extract optimization convergence payload from main analysis run."""
    out = _materialize_analysis_output(analysis_result)
    if not isinstance(out, dict):
        return None
    convergence = out.get("optimization_convergence")
    return convergence if isinstance(convergence, dict) else None


def _canonical_initial_state_label(state: str) -> str:
    """Normalize 2Q product-state label."""
    if not isinstance(state, str):
        raise ValueError(f"initial_state must be a string, got {type(state)!r}.")
    s = state.strip().lower().replace(" ", "")
    aliases = {
        "++": "++",
        "00": "00",
        "01": "01",
        "10": "10",
        "11": "11",
        "gg": "00",
        "ge": "01",
        "eg": "10",
        "ee": "11",
    }
    if s in aliases:
        return aliases[s]
    raise ValueError(
        "Unsupported initial_state. Use one of "
        "'++', '00', '01', '10', '11', 'gg', 'ge', 'eg', 'ee'."
    )


def _canonical_target_state_label(target_state) -> str:
    """Normalize target-state string labels used for validation matching."""
    if not isinstance(target_state, str):
        return str(target_state)
    s = target_state.strip().lower().replace(" ", "")
    aliases = {
        "++": "++",
        "plus_plus": "++",
        "00": "00",
        "01": "01",
        "10": "10",
        "11": "11",
        "gg": "00",
        "ge": "01",
        "eg": "10",
        "ee": "11",
    }
    return aliases.get(s, s)


def _single_qubit_state_token(label: str, *, qubit_role: str) -> str:
    """Extract and map a single-qubit token from canonical 2Q label."""
    idx = 0 if qubit_role == "ctrl" else 1
    token = label[idx]
    if token == "+":
        return "+"
    if token == "0":
        return "g"
    if token == "1":
        return "e"
    raise ValueError(f"Unsupported token {token!r} in initial_state {label!r}.")


def _prepare_single_qubit_state(qop, qubit, token: str) -> None:
    """Prepare one qubit in g/e/+ state."""
    if token == "g":
        qop.prepare_state(qubit, state="g")
        return
    if token == "e":
        qop.prepare_state(qubit, state="e")
        return
    if token == "+":
        qop.prepare_state(qubit, state="g")
        qop.y90(qubit)
        return
    raise ValueError(f"Unsupported single-qubit initial-state token: {token!r}.")


def _state_token_for_section_name(token: str) -> str:
    """Map state tokens to section-name friendly labels."""
    return {"g": "g", "e": "e", "+": "plus"}[token]


def _apply_measurement_prerotation(qop, qubit, axis: str):
    """Apply pre-rotation so final Z-basis measurement corresponds to axis."""
    if axis == "X":
        return qop.ry(qubit, angle=-np.pi / 2)
    elif axis == "Y":
        return qop.rx(qubit, angle=np.pi / 2)
    elif axis == "Z":
        return None
    else:
        raise ValueError(f"Unsupported tomography axis: {axis!r}.")


@workflow.task
@dsl.qubit_experiment
def create_experiment(
    qpu: QPU,
    ctrl: QuantumElements,
    targ: QuantumElements,
    bus: QuantumElements,
    bus_frequency: float,
    rip_amplitude: float,
    rip_length: float,
    rip_phase: float = np.pi / 2,
    use_rip: bool = True,
    initial_state: str = "++",
    options: TwoQStateTomographyExperimentOptions | None = None,
) -> Experiment:
    """Create 2Q tomography experiment with RIP state preparation."""
    opts = TwoQStateTomographyExperimentOptions() if options is None else options
    if AcquisitionType(opts.acquisition_type) != AcquisitionType.INTEGRATION:
        raise ValueError(
            "two_qubit_state_tomography only supports AcquisitionType.INTEGRATION."
        )
    if AveragingMode(opts.averaging_mode) != AveragingMode.SINGLE_SHOT:
        raise ValueError(
            "two_qubit_state_tomography only supports AveragingMode.SINGLE_SHOT."
        )
    ctrl = validation.validate_and_convert_single_qubit_sweeps(ctrl)
    targ = validation.validate_and_convert_single_qubit_sweeps(targ)
    bus = validation.validate_and_convert_single_qubit_sweeps(bus)
    canonical_initial_state = _canonical_initial_state_label(initial_state)
    ctrl_token = _single_qubit_state_token(canonical_initial_state, qubit_role="ctrl")
    targ_token = _single_qubit_state_token(canonical_initial_state, qubit_role="targ")
    ctrl_token_name = _state_token_for_section_name(ctrl_token)
    targ_token_name = _state_token_for_section_name(targ_token)

    qop = qpu.quantum_operations
    max_measure_section_length = qop.measure_section_length([ctrl, targ])

    with dsl.acquire_loop_rt(
        count=opts.count,
        averaging_mode=opts.averaging_mode,
        acquisition_type=opts.acquisition_type,
        repetition_mode=opts.repetition_mode,
        repetition_time=opts.repetition_time,
        reset_oscillator_phase=opts.reset_oscillator_phase,
    ):
        # set_bus_frequency can only be called once per signal in one experiment
        qop.set_bus_frequency(bus, frequency=bus_frequency)

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

                with dsl.section(
                    **prep_section_kwargs,
                ) as prep_sec:
                    with dsl.section(
                        name=f"prep_ctrl_{ctrl_token_name}_{setting_label}",
                        alignment=SectionAlignment.LEFT,
                    ) as prep_ctrl_g:
                        _prepare_single_qubit_state(qop, ctrl, ctrl_token)

                    with dsl.section(
                        name=f"prep_targ_{targ_token_name}_{setting_label}",
                        alignment=SectionAlignment.LEFT,
                        play_after=prep_ctrl_g.uid,
                    ):
                        _prepare_single_qubit_state(qop, targ, targ_token)

                basis_play_after = prep_sec.uid
                if use_rip:
                    with dsl.section(
                        name=f"rip_{setting_label}",
                        alignment=SectionAlignment.LEFT,
                        play_after=prep_sec.uid,
                    ) as rip_sec:
                        qop.rip(
                            bus,
                            amplitude=rip_amplitude,
                            phase=rip_phase,
                            length=rip_length,
                        )
                    basis_play_after = rip_sec.uid

                with dsl.section(
                    name=f"basis_{setting_label}",
                    alignment=SectionAlignment.LEFT,
                    play_after=basis_play_after,
                ) as basis_sec:
                    with dsl.section(
                        name=f"basis_ctrl_{setting_label}",
                        alignment=SectionAlignment.LEFT,
                    ) as basis_ctrl:
                        _apply_measurement_prerotation(qop, ctrl, ctrl_axis)

                    with dsl.section(
                        name=f"basis_targ_{setting_label}",
                        alignment=SectionAlignment.LEFT,
                        play_after=basis_ctrl.uid,
                    ):
                        _apply_measurement_prerotation(qop, targ, targ_axis)

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

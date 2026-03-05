"""Experiment workflow for readout pulse-length optimization."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
from laboneq import workflow

from analysis import iq_time_trace as analysis_iq_time_trace
from analysis.readout_length_sweep import calculate_metrics, plot_metrics
from analysis.readout_sweep_common import (
    calibration_shots_by_state,
    evaluate_iq_binary,
    unwrap_result_like,
)
from experiments import iq_cloud, iq_time_trace
from laboneq_applications.core import validation
from laboneq_applications.tasks.parameter_updating import (
    temporary_qpu,
    temporary_quantum_elements_from_qpu,
    update_qpu,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from laboneq.dsl.quantum import QuantumParameters
    from laboneq.dsl.quantum.qpu import QPU
    from laboneq.dsl.quantum.quantum_element import QuantumElement
    from laboneq.dsl.session import Session
    from laboneq.simple import Experiment
    from numpy.typing import ArrayLike


_LENGTH_WINDOW_MODE_ADAPTIVE = "adaptive_2pass"
_LENGTH_WINDOW_MODE_FIXED = "fixed"
_FLAT_WINDOW_FAILURE_DROP = "drop"
_FLAT_WINDOW_CAPTURE_INTEGRATION_DELAY_S = 0.0
_FLAT_WINDOW_CAPTURE_INTEGRATION_LENGTH_S = 2.0e-6


@workflow.workflow_options
class ReadoutLengthSweepWorkflowOptions:
    """Workflow options for readout length optimization."""

    do_analysis: bool = workflow.option_field(
        True, description="Whether to run analysis."
    )
    update: bool = workflow.option_field(
        False, description="Whether to apply optimized parameters to the input qpu."
    )
    do_plotting: bool = workflow.option_field(
        True, description="Whether to plot readout-length metric curves."
    )
    do_plotting_error_bars: bool = workflow.option_field(
        True, description="Whether to include bootstrap error bars in metric plots."
    )
    count: int = workflow.option_field(
        4096,
        description="Single-shot count for iq_time_trace and iq_cloud at each point.",
    )
    max_readout_length: float | None = workflow.option_field(
        None,
        description=(
            "Optional hard upper bound for readout_length sweep points in seconds. "
            "Use this to enforce hardware waveform limits before run submission."
        ),
    )
    length_window_mode: str = workflow.option_field(
        _LENGTH_WINDOW_MODE_ADAPTIVE,
        description=(
            "Window update mode for each readout length point. "
            f"Use '{_LENGTH_WINDOW_MODE_ADAPTIVE}' for iq_time_trace -> iq_cloud "
            "adaptive chain, or "
            f"'{_LENGTH_WINDOW_MODE_FIXED}' for fixed-window iq_cloud-only behavior."
        ),
    )
    flat_window_failure_policy: str = workflow.option_field(
        _FLAT_WINDOW_FAILURE_DROP,
        description="Failure policy for flat-window detection. Only 'drop' is supported.",
    )
    flat_window_sample_dt_ns: float = workflow.option_field(
        0.5,
        description="Sampling period in ns used for flat-window detection traces.",
    )
    flat_window_apply_software_demodulation: bool = workflow.option_field(
        True,
        description=(
            "Apply software demodulation before flat-window detection trace analysis."
        ),
    )
    flat_window_apply_lpf_after_demodulation: bool = workflow.option_field(
        True,
        description=(
            "Apply low-pass filtering after software demodulation for flat-window "
            "detection traces."
        ),
    )
    flat_window_lpf_cutoff_frequency_hz: float | None = workflow.option_field(
        25e6,
        description=(
            "LPF cutoff frequency in Hz for flat-window detection traces when "
            "flat_window_apply_lpf_after_demodulation=True."
        ),
    )
    flat_window_lpf_order: int = workflow.option_field(
        5,
        description="LPF order for flat-window detection traces.",
    )
    flat_window_phase_mask_relative_threshold: float = workflow.option_field(
        0.15,
        description=(
            "Relative magnitude threshold used to mask unreliable phase samples "
            "during flat-window detection."
        ),
    )
    flat_window_smoothing_window_samples: int = workflow.option_field(
        9,
        description="Smoothing window size in samples for flat-window edge features.",
    )
    flat_window_soft_tolerance_ratio: float = workflow.option_field(
        0.10,
        description="Soft tolerance ratio in the flat-window edge-pair objective.",
    )
    flat_window_phase_weight: float = workflow.option_field(
        0.4,
        description="Relative weight of phase derivative in flat-window edge scoring.",
    )
    flat_window_min_peak_z: float = workflow.option_field(
        2.0,
        description="Minimum robust z-score required for detected flat-window edges.",
    )


def _coerce_single_qubit_runtime(qubit_like) -> QuantumElement:
    qubits = validation.validate_and_convert_qubits_sweeps(qubit_like)
    if len(qubits) != 1:
        raise ValueError(
            "readout_length_sweep supports exactly one qubit at a time. "
            f"Got {len(qubits)}."
        )
    return qubits[0]


@workflow.task(save=False)
def _normalize_single_qubit(qubit_like):
    return _coerce_single_qubit_runtime(qubit_like)


@workflow.task(save=False)
def _resolve_readout_lengths(
    readout_lengths: ArrayLike,
    max_readout_length: float | None = None,
) -> list[float]:
    points = np.asarray(readout_lengths, dtype=float).reshape(-1)
    if points.size < 1:
        raise ValueError("readout_lengths must contain at least one value.")
    if not np.isfinite(points).all():
        raise ValueError("readout_lengths must contain only finite values.")
    if np.any(points <= 0.0):
        raise ValueError("readout_lengths must contain only positive values.")
    if max_readout_length is not None:
        max_len = float(max_readout_length)
        if max_len <= 0:
            raise ValueError("max_readout_length must be positive when specified.")
        if np.any(points > max_len):
            too_long = float(np.max(points))
            raise ValueError(
                "readout_lengths contains values exceeding max_readout_length: "
                f"max={too_long:.3e}s > limit={max_len:.3e}s."
            )
    return [float(x) for x in points]


@workflow.task(save=False)
def _resolve_fixed_integration_delay(qubit: QuantumElement) -> float:
    return float(qubit.parameters.readout_integration_delay or 0.0)


@workflow.task(save=False)
def _resolve_length_window_mode(length_window_mode: str) -> str:
    mode = str(length_window_mode).strip().lower()
    if mode not in (_LENGTH_WINDOW_MODE_ADAPTIVE, _LENGTH_WINDOW_MODE_FIXED):
        raise ValueError(
            "Unsupported length_window_mode. "
            f"Use '{_LENGTH_WINDOW_MODE_ADAPTIVE}' or '{_LENGTH_WINDOW_MODE_FIXED}'. "
            f"Got: {length_window_mode!r}."
        )
    return mode


@workflow.task(save=False)
def _resolve_flat_window_failure_policy(flat_window_failure_policy: str) -> str:
    policy = str(flat_window_failure_policy).strip().lower()
    if policy != _FLAT_WINDOW_FAILURE_DROP:
        raise ValueError(
            "Unsupported flat_window_failure_policy. "
            f"Only '{_FLAT_WINDOW_FAILURE_DROP}' is supported. "
            f"Got: {flat_window_failure_policy!r}."
        )
    return policy


@workflow.task(save=False)
def _is_adaptive_window_mode(length_window_mode: str) -> bool:
    return str(length_window_mode).strip().lower() == _LENGTH_WINDOW_MODE_ADAPTIVE


@workflow.task(save=False)
def _build_temp_params_for_point(
    base: dict[str | tuple[str, str, str], dict | QuantumParameters] | None,
    qubit: QuantumElement,
    readout_length: float,
    readout_integration_length: float,
    readout_integration_delay: float,
) -> dict[str | tuple[str, str, str], dict | QuantumParameters]:
    merged: dict[str | tuple[str, str, str], dict | QuantumParameters] = {}
    if base is not None:
        for k, v in base.items():
            merged[k] = dict(v) if isinstance(v, dict) else deepcopy(v)

    temp = deepcopy(qubit.parameters)
    temp.readout_length = float(readout_length)
    temp.readout_integration_length = float(readout_integration_length)
    temp.readout_integration_delay = float(readout_integration_delay)
    merged[qubit.uid] = temp
    return merged


@workflow.task(save=False)
def _extract_flat_window_from_detection_updates(
    flat_window_detection: dict,
    flat_window_updates: dict,
    qubit_uid: str,
) -> dict[str, float | bool | str | None]:
    detection = flat_window_detection.get(qubit_uid, {})
    if not bool(detection.get("success", False)):
        return {
            "success": False,
            "readout_integration_delay": None,
            "readout_integration_length": None,
            "reason": str(detection.get("reason") or "flat_window_detection_failed"),
        }

    new_values = flat_window_updates.get("new_parameter_values", {}).get(qubit_uid, {})
    try:
        delay = float(new_values["readout_integration_delay"])
        length = float(new_values["readout_integration_length"])
    except (KeyError, TypeError, ValueError):
        return {
            "success": False,
            "readout_integration_delay": None,
            "readout_integration_length": None,
            "reason": "flat_window_update_missing_or_invalid",
        }

    if not np.isfinite(delay) or not np.isfinite(length) or length <= 0.0:
        return {
            "success": False,
            "readout_integration_delay": None,
            "readout_integration_length": None,
            "reason": "flat_window_update_non_finite_or_non_positive",
        }

    return {
        "success": True,
        "readout_integration_delay": delay,
        "readout_integration_length": length,
        "reason": None,
    }


@workflow.task(save=False)
def _select_window_for_iq_cloud(
    flat_window: dict[str, float | bool | str | None],
    readout_length: float,
    fallback_integration_delay: float,
) -> dict[str, float]:
    success = bool(flat_window.get("success", False))
    if success:
        delay = flat_window.get("readout_integration_delay")
        length = flat_window.get("readout_integration_length")
    else:
        delay = fallback_integration_delay
        length = readout_length

    try:
        delay_f = float(delay)
    except (TypeError, ValueError):
        delay_f = float(fallback_integration_delay)
    if not np.isfinite(delay_f):
        delay_f = float(fallback_integration_delay)

    try:
        length_f = float(length)
    except (TypeError, ValueError):
        length_f = float(readout_length)
    if not np.isfinite(length_f) or length_f <= 0.0:
        length_f = float(readout_length)

    return {
        "readout_integration_delay": delay_f,
        "readout_integration_length": length_f,
    }


@workflow.task(save=False)
def _record_adaptive_point(
    point_metrics: list[dict[str, float]],
    successful_lengths: list[float],
    dropped_points: list[float],
    dropped_reasons: list[str],
    point_metric: dict[str, float],
    readout_length: float,
    flat_window: dict[str, float | bool | str | None],
) -> None:
    if bool(flat_window.get("success", False)):
        point_metrics.append(point_metric)
        successful_lengths.append(float(readout_length))
        return
    dropped_points.append(float(readout_length))
    dropped_reasons.append(
        str(flat_window.get("reason") or "flat_window_detection_failed")
    )


@workflow.task(save=False)
def _append_item(items: list, item) -> None:
    items.append(item)


@workflow.task(save=False)
def _build_iq_time_trace_experiment_options(count: int):
    return iq_time_trace.IQTimeTraceExperimentOptions(count=int(count))


@workflow.task(save=False)
def _build_iq_cloud_options(count: int):
    return iq_cloud.IQCloudExperimentOptions(count=int(count))


@workflow.task(save=False)
def _compile_experiment_no_log(session: Session, experiment: Experiment):
    return session.compile(experiment=experiment)


@workflow.task(save=False)
def _run_experiment_no_log(session: Session, compiled_experiment):
    return session.run(compiled_experiment)


@workflow.task(save=False)
def _materialize_list(items: list) -> list:
    return list(items)


@workflow.task(save=False)
def _extract_point_metric_from_iq_cloud_result(
    result_like,
    qubit_uid: str,
    readout_length: float,
    readout_integration_delay: float,
    readout_integration_length: float,
    ridge_target_condition: float = 1e6,
) -> dict[str, float]:
    result = unwrap_result_like(result_like)
    shots = calibration_shots_by_state(
        result=result,
        qubit_uid=qubit_uid,
        states=("g", "e"),
    )
    metric = evaluate_iq_binary(
        shots_g=shots["g"],
        shots_e=shots["e"],
        target_condition=float(ridge_target_condition),
    )
    return {
        "readout_length": float(readout_length),
        "readout_integration_delay": float(readout_integration_delay),
        "readout_integration_length": float(readout_integration_length),
        "assignment_fidelity": float(metric["assignment_fidelity"]),
        "delta_mu_over_sigma": float(metric["delta_mu_over_sigma"]),
    }


@workflow.task(save=False)
def _ensure_non_empty_successful_points(
    length_points: list[float],
    length_window_mode: str,
) -> None:
    if len(length_points) < 1:
        raise ValueError(
            "All readout_length points were dropped by flat-window detection "
            f"(mode={length_window_mode!r})."
        )


@workflow.workflow(name="readout_length_sweep")
def experiment_workflow(
    session: Session,
    qpu: QPU,
    qubit: QuantumElement,
    readout_lengths: ArrayLike,
    temporary_parameters: dict[str | tuple[str, str, str], dict | QuantumParameters]
    | None = None,
    options: ReadoutLengthSweepWorkflowOptions | None = None,
) -> None:
    """Run near-time readout length sweep using iq_time_trace -> iq_cloud chain."""
    options = ReadoutLengthSweepWorkflowOptions() if options is None else options
    length_points = _resolve_readout_lengths(
        readout_lengths,
        max_readout_length=options.max_readout_length,
    )
    length_window_mode = _resolve_length_window_mode(options.length_window_mode)
    _resolve_flat_window_failure_policy(options.flat_window_failure_policy)
    is_adaptive_mode = _is_adaptive_window_mode(length_window_mode)

    base_temp_qpu = temporary_qpu(qpu, temporary_parameters)
    base_qubit = _normalize_single_qubit(
        temporary_quantum_elements_from_qpu(base_temp_qpu, qubit)
    )
    fixed_integration_delay = _resolve_fixed_integration_delay(base_qubit)

    iqt_experiment_options = _build_iq_time_trace_experiment_options(count=options.count)
    iqc_options = _build_iq_cloud_options(count=options.count)

    point_metrics = []
    successful_lengths = []
    dropped_points = []
    dropped_reasons = []

    with workflow.if_(is_adaptive_mode):
        with workflow.for_(length_points, lambda x: x) as readout_length:
            flat_window_temp_parameters = _build_temp_params_for_point(
                base=temporary_parameters,
                qubit=base_qubit,
                readout_length=readout_length,
                # Match iq_time_trace.find_flat_window=True capture policy.
                readout_integration_length=_FLAT_WINDOW_CAPTURE_INTEGRATION_LENGTH_S,
                readout_integration_delay=_FLAT_WINDOW_CAPTURE_INTEGRATION_DELAY_S,
            )
            raw_capture_qpu = temporary_qpu(qpu, flat_window_temp_parameters)
            raw_capture_qubit = _normalize_single_qubit(
                temporary_quantum_elements_from_qpu(raw_capture_qpu, qubit)
            )
            iqt_exp = iq_time_trace.create_experiment(
                qpu=raw_capture_qpu,
                qubits=raw_capture_qubit,
                states=("g", "e"),
                options=iqt_experiment_options,
            )
            iqt_compiled = _compile_experiment_no_log(session, iqt_exp)
            iqt_result = _run_experiment_no_log(session, iqt_compiled)
            iqt_processed = analysis_iq_time_trace.collect_time_traces(
                qubits=raw_capture_qubit,
                result=iqt_result,
                states=("g", "e"),
                sample_dt_ns=options.flat_window_sample_dt_ns,
                apply_software_demodulation=options.flat_window_apply_software_demodulation,
                apply_lpf_after_demodulation=options.flat_window_apply_lpf_after_demodulation,
                lpf_cutoff_frequency_hz=options.flat_window_lpf_cutoff_frequency_hz,
                lpf_order=options.flat_window_lpf_order,
            )
            flat_window_detection = analysis_iq_time_trace.detect_flat_window(
                qubits=raw_capture_qubit,
                states=("g", "e"),
                processed_data_dict=iqt_processed,
                sample_dt_ns=options.flat_window_sample_dt_ns,
                phase_mask_relative_threshold=options.flat_window_phase_mask_relative_threshold,
                smoothing_window_samples=options.flat_window_smoothing_window_samples,
                soft_tolerance_ratio=options.flat_window_soft_tolerance_ratio,
                phase_weight=options.flat_window_phase_weight,
                min_peak_z=options.flat_window_min_peak_z,
            )
            flat_window_updates = analysis_iq_time_trace.build_flat_window_parameter_updates(
                qubits=raw_capture_qubit,
                flat_window_detection=flat_window_detection,
            )
            flat_window = _extract_flat_window_from_detection_updates(
                flat_window_detection=flat_window_detection,
                flat_window_updates=flat_window_updates,
                qubit_uid=base_qubit.uid,
            )
            iqc_window = _select_window_for_iq_cloud(
                flat_window=flat_window,
                readout_length=readout_length,
                fallback_integration_delay=fixed_integration_delay,
            )
            adaptive_temp_parameters = _build_temp_params_for_point(
                base=temporary_parameters,
                qubit=base_qubit,
                readout_length=readout_length,
                readout_integration_length=iqc_window[
                    "readout_integration_length"
                ],
                readout_integration_delay=iqc_window[
                    "readout_integration_delay"
                ],
            )
            per_run_qpu = temporary_qpu(qpu, adaptive_temp_parameters)
            per_run_qubit = _normalize_single_qubit(
                temporary_quantum_elements_from_qpu(per_run_qpu, qubit)
            )
            iqc_exp = iq_cloud.create_experiment(
                qpu=per_run_qpu,
                qubits=per_run_qubit,
                options=iqc_options,
            )
            iqc_compiled = _compile_experiment_no_log(session, iqc_exp)
            iqc_result = _run_experiment_no_log(session, iqc_compiled)
            point_metric = _extract_point_metric_from_iq_cloud_result(
                result_like=iqc_result,
                qubit_uid=base_qubit.uid,
                readout_length=readout_length,
                readout_integration_delay=iqc_window["readout_integration_delay"],
                readout_integration_length=iqc_window["readout_integration_length"],
            )
            _record_adaptive_point(
                point_metrics=point_metrics,
                successful_lengths=successful_lengths,
                dropped_points=dropped_points,
                dropped_reasons=dropped_reasons,
                point_metric=point_metric,
                readout_length=readout_length,
                flat_window=flat_window,
            )
    with workflow.else_():
        with workflow.for_(length_points, lambda x: x) as readout_length:
            fixed_temp_parameters = _build_temp_params_for_point(
                base=temporary_parameters,
                qubit=base_qubit,
                readout_length=readout_length,
                readout_integration_length=readout_length,
                readout_integration_delay=fixed_integration_delay,
            )
            per_run_qpu = temporary_qpu(qpu, fixed_temp_parameters)
            per_run_qubit = _normalize_single_qubit(
                temporary_quantum_elements_from_qpu(per_run_qpu, qubit)
            )
            iqc_exp = iq_cloud.create_experiment(
                qpu=per_run_qpu,
                qubits=per_run_qubit,
                options=iqc_options,
            )
            iqc_compiled = _compile_experiment_no_log(session, iqc_exp)
            iqc_result = _run_experiment_no_log(session, iqc_compiled)
            point_metric = _extract_point_metric_from_iq_cloud_result(
                result_like=iqc_result,
                qubit_uid=base_qubit.uid,
                readout_length=readout_length,
                readout_integration_delay=fixed_integration_delay,
                readout_integration_length=readout_length,
            )
            _append_item(point_metrics, point_metric)
            _append_item(successful_lengths, readout_length)

    collected_point_metrics = _materialize_list(point_metrics)
    collected_lengths = _materialize_list(successful_lengths)
    collected_dropped_points = _materialize_list(dropped_points)
    collected_dropped_reasons = _materialize_list(dropped_reasons)
    _ensure_non_empty_successful_points(collected_lengths, length_window_mode)

    analysis_result = None
    with workflow.if_(options.do_analysis):
        analysis_result = calculate_metrics(
            results=[],
            qubits=[],
            readout_lengths=collected_lengths,
            reference_qubit=base_qubit,
            requested_readout_lengths=length_points,
            dropped_points=collected_dropped_points,
            dropped_reasons=collected_dropped_reasons,
            point_metrics=collected_point_metrics,
        )
        with workflow.if_(options.do_plotting):
            plot_metrics(
                metrics=analysis_result,
                include_error_bars=options.do_plotting_error_bars,
            )
        with workflow.if_(options.update):
            update_qpu(qpu, analysis_result["new_parameter_values"])

    workflow.return_({"status": "completed"})

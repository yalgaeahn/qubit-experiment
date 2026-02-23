# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Analysis workflow for IQ time-trace diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from laboneq import workflow
from laboneq.simple import dsl

from laboneq_applications.analysis.options import BasePlottingOptions
from laboneq_applications.analysis.plotting_helpers import timestamped_title
from laboneq_applications.core.validation import (
    validate_and_convert_qubits_sweeps,
    validate_result,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import matplotlib as mpl
    from laboneq.workflow.tasks.run_experiment import RunExperimentResults
    from numpy.typing import ArrayLike

    from laboneq_applications.typing import QuantumElements


_FLAT_WINDOW_DISTANCE_BAND_RATIO = 0.02


@workflow.workflow_options
class IQTimeTraceAnalysisWorkflowOptions:
    """Options for IQ time-trace analysis workflow."""

    do_plotting: bool = workflow.option_field(
        True, description="Whether to create plots."
    )
    do_plotting_iq_time_trace: bool = workflow.option_field(
        True, description="Whether to create IQ time-trace plots."
    )
    sample_dt_ns: float = workflow.option_field(
        0.5, description="Sampling period in ns for raw traces."
    )
    truncate_to_granularity: bool = workflow.option_field(
        False,
        description=(
            "Truncate tail samples to align trace length with acquisition granularity."
        ),
    )
    granularity: int = workflow.option_field(
        16, description="Acquisition granularity for tail truncation."
    )
    apply_software_demodulation: bool = workflow.option_field(
        False,
        description=(
            "Apply extra software demodulation at IF. "
            "Keep False for standard LabOne Q RAW traces."
        ),
    )
    apply_lpf_after_demodulation: bool = workflow.option_field(
        False, description="Apply low-pass filter after software demodulation."
    )
    lpf_cutoff_frequency_hz: float | None = workflow.option_field(
        None, description="LPF cut-off frequency in Hz."
    )
    lpf_order: int = workflow.option_field(
        5, description="Order of low-pass filter used after software demodulation."
    )
    phase_mask_relative_threshold: float = workflow.option_field(
        0.15,
        description=(
            "Hide phase samples where |IQ| is below this fraction of the "
            "per-trace peak amplitude."
        ),
    )
    plot_phase_boundaries: bool = workflow.option_field(
        True,
        description=(
            "When find_flat_window=True, plot detected flat_start/flat_end boundaries."
        ),
    )
    find_flat_window: bool = workflow.option_field(
        False,
        description=(
            "Detect flat_start/flat_end from data using soft length-constrained "
            "two-edge search."
        ),
    )
    flat_window_smoothing_window_samples: int = workflow.option_field(
        9,
        description="Smoothing window for edge-feature extraction in flat-window search.",
    )
    flat_window_soft_tolerance_ratio: float = workflow.option_field(
        0.10,
        description=(
            "Soft tolerance ratio for flat-length prior in pair-search objective."
        ),
    )
    flat_window_phase_weight: float = workflow.option_field(
        0.4,
        description="Relative weight of phase-derivative feature in edge score.",
    )
    flat_window_min_peak_z: float = workflow.option_field(
        2.0,
        description="Minimum robust z-score threshold required for each detected edge.",
    )
    do_dsp_analysis: bool = workflow.option_field(
        True,
        description="Compute RAW DSP diagnostics (Welch PSD, STFT, dphi/dt).",
    )
    do_plotting_iq_dsp: bool = workflow.option_field(
        True,
        description="Create RAW DSP figures.",
    )
    welch_nperseg: int = workflow.option_field(
        256, description="Welch segment length."
    )
    welch_noverlap: int = workflow.option_field(
        128, description="Welch overlap length."
    )
    stft_nperseg: int = workflow.option_field(
        256, description="STFT segment length."
    )
    stft_noverlap: int = workflow.option_field(
        192, description="STFT overlap length."
    )
    stft_window: str = workflow.option_field(
        "hann", description="STFT window function."
    )
    dsp_freq_limit_mhz: float | None = workflow.option_field(
        None,
        description="Optional frequency-axis limit for DSP plots (MHz).",
    )
    dphi_smooth_window: int = workflow.option_field(
        1,
        description="Moving-average smoothing window for dphi/dt. 1 disables smoothing.",
    )


@workflow.workflow
def analysis_workflow(
    result: RunExperimentResults,
    qubits: QuantumElements,
    states: Sequence[str],
    options: IQTimeTraceAnalysisWorkflowOptions | None = None,
    find_flat_window: bool | None = None,
) -> None:
    """Analyze IQ time traces using calibration trace handles."""
    opts = IQTimeTraceAnalysisWorkflowOptions() if options is None else options
    find_flat_window_enabled = (
        opts.find_flat_window if find_flat_window is None else find_flat_window
    )

    pre_demod_traces = collect_time_traces(
        qubits=qubits,
        result=result,
        states=states,
        sample_dt_ns=opts.sample_dt_ns,
        apply_software_demodulation=False,
        apply_lpf_after_demodulation=False,
    )
    demod_pre_lpf_traces = collect_time_traces(
        qubits=qubits,
        result=result,
        states=states,
        sample_dt_ns=opts.sample_dt_ns,
        apply_software_demodulation=opts.apply_software_demodulation,
        apply_lpf_after_demodulation=False,
    )
    analysis_traces = collect_time_traces(
        qubits=qubits,
        result=result,
        states=states,
        sample_dt_ns=opts.sample_dt_ns,
        apply_software_demodulation=opts.apply_software_demodulation,
        apply_lpf_after_demodulation=opts.apply_lpf_after_demodulation,
        lpf_cutoff_frequency_hz=opts.lpf_cutoff_frequency_hz,
        lpf_order=opts.lpf_order,
    )
    compare_pre_demod = opts.apply_software_demodulation
    compare_demod_lpf = opts.apply_lpf_after_demodulation
    with workflow.if_(opts.truncate_to_granularity):
        pre_demod_traces = truncate_time_traces_to_granularity(
            qubits=qubits,
            processed_data_dict=pre_demod_traces,
            states=states,
            granularity=opts.granularity,
        )
        demod_pre_lpf_traces = truncate_time_traces_to_granularity(
            qubits=qubits,
            processed_data_dict=demod_pre_lpf_traces,
            states=states,
            granularity=opts.granularity,
        )
        analysis_traces = truncate_time_traces_to_granularity(
            qubits=qubits,
            processed_data_dict=analysis_traces,
            states=states,
            granularity=opts.granularity,
        )

    diagnostics = compute_diagnostics(
        qubits=qubits,
        states=states,
        processed_data_dict=analysis_traces,
        sample_dt_ns=opts.sample_dt_ns,
        phase_mask_relative_threshold=opts.phase_mask_relative_threshold,
    )
    flat_window_detection = {}
    flat_window_updates = {
        "old_parameter_values": {},
        "new_parameter_values": {},
    }
    with workflow.if_(find_flat_window_enabled):
        flat_window_detection = detect_flat_window(
            qubits=qubits,
            states=states,
            processed_data_dict=analysis_traces,
            sample_dt_ns=opts.sample_dt_ns,
            phase_mask_relative_threshold=opts.phase_mask_relative_threshold,
            smoothing_window_samples=opts.flat_window_smoothing_window_samples,
            soft_tolerance_ratio=opts.flat_window_soft_tolerance_ratio,
            phase_weight=opts.flat_window_phase_weight,
            min_peak_z=opts.flat_window_min_peak_z,
        )
        flat_window_updates = build_flat_window_parameter_updates(
            qubits=qubits,
            flat_window_detection=flat_window_detection,
        )

    figures = {}
    flat_window_figures = {}
    dsp_diagnostics = {}
    dsp_figures = {}

    with workflow.if_(opts.do_dsp_analysis):
        dsp_diagnostics = compute_dsp_diagnostics(
            qubits=qubits,
            states=states,
            raw_data_dict=analysis_traces,
            pre_demod_data_dict=pre_demod_traces,
            compare_pre_demod=compare_pre_demod,
            sample_dt_ns=opts.sample_dt_ns,
            phase_mask_relative_threshold=opts.phase_mask_relative_threshold,
            welch_nperseg=opts.welch_nperseg,
            welch_noverlap=opts.welch_noverlap,
            stft_nperseg=opts.stft_nperseg,
            stft_noverlap=opts.stft_noverlap,
            stft_window=opts.stft_window,
            dphi_smooth_window=opts.dphi_smooth_window,
        )

        with workflow.if_(opts.do_plotting):
            with workflow.if_(opts.do_plotting_iq_dsp):
                dsp_figures = plot_iq_dsp(
                    qubits=qubits,
                    states=states,
                    raw_data_dict=analysis_traces,
                    pre_demod_data_dict=pre_demod_traces,
                    compare_pre_demod=compare_pre_demod,
                    demod_pre_lpf_data_dict=demod_pre_lpf_traces,
                    compare_demod_lpf=compare_demod_lpf,
                    sample_dt_ns=opts.sample_dt_ns,
                    phase_mask_relative_threshold=opts.phase_mask_relative_threshold,
                    welch_nperseg=opts.welch_nperseg,
                    welch_noverlap=opts.welch_noverlap,
                    stft_nperseg=opts.stft_nperseg,
                    stft_noverlap=opts.stft_noverlap,
                    stft_window=opts.stft_window,
                    dsp_freq_limit_mhz=opts.dsp_freq_limit_mhz,
                    dphi_smooth_window=opts.dphi_smooth_window,
                    plot_phase_boundaries=opts.plot_phase_boundaries,
                )

    with workflow.if_(opts.do_plotting):
        with workflow.if_(find_flat_window_enabled):
            flat_window_figures = plot_flat_window_debug(
                qubits=qubits,
                states=states,
                processed_data_dict=analysis_traces,
                flat_window_detection=flat_window_detection,
                sample_dt_ns=opts.sample_dt_ns,
                phase_mask_relative_threshold=opts.phase_mask_relative_threshold,
                smoothing_window_samples=opts.flat_window_smoothing_window_samples,
                phase_weight=opts.flat_window_phase_weight,
            )
        with workflow.if_(opts.do_plotting_iq_time_trace):
            figures = plot_iq_time_traces(
                qubits=qubits,
                states=states,
                processed_data_dict=analysis_traces,
                sample_dt_ns=opts.sample_dt_ns,
                phase_mask_relative_threshold=opts.phase_mask_relative_threshold,
                plot_phase_boundaries=opts.plot_phase_boundaries,
                find_flat_window=find_flat_window_enabled,
                flat_window_detection=flat_window_detection,
            )

    payload = build_iq_time_trace_analysis_payload(
        diagnostics=diagnostics,
        figures=figures,
        flat_window_figures=flat_window_figures,
        dsp_diagnostics=dsp_diagnostics,
        dsp_figures=dsp_figures,
        flat_window_detection=flat_window_detection,
        flat_window_updates=flat_window_updates,
        metadata={
            "states": states,
            "input_contract": "calibration_trace_handle",
            "dsp_input_stage": "analysis_trace",
            "dsp_backend": "scipy",
            "dsp_compare_pre_demod_enabled": compare_pre_demod,
            "dsp_reference_pre_demod_stage": "raw_before_demod",
            "dsp_compare_demod_lpf_enabled": compare_demod_lpf,
            "flat_window_detection_enabled": find_flat_window_enabled,
            "flat_window_debug_figure_enabled": bool(
                find_flat_window_enabled and opts.do_plotting
            ),
        },
    )
    workflow.return_(payload)


@workflow.task
def build_iq_time_trace_analysis_payload(
    diagnostics: dict,
    figures: dict,
    flat_window_figures: dict,
    dsp_diagnostics: dict,
    dsp_figures: dict,
    flat_window_detection: dict,
    flat_window_updates: dict,
    metadata: dict,
) -> dict:
    """Compose final analysis payload with materialized flat-window update dictionaries."""
    old_parameter_values = {}
    new_parameter_values = {}
    if isinstance(flat_window_updates, dict):
        maybe_old = flat_window_updates.get("old_parameter_values")
        maybe_new = flat_window_updates.get("new_parameter_values")
        if isinstance(maybe_old, dict):
            old_parameter_values = maybe_old
        if isinstance(maybe_new, dict):
            new_parameter_values = maybe_new

    return {
        "diagnostics": diagnostics,
        "figures": figures,
        "flat_window_figures": flat_window_figures,
        "dsp_diagnostics": dsp_diagnostics,
        "dsp_figures": dsp_figures,
        "flat_window_detection": flat_window_detection,
        "old_parameter_values": old_parameter_values,
        "new_parameter_values": new_parameter_values,
        "metadata": metadata,
    }


@workflow.task
def collect_time_traces(
    qubits: QuantumElements,
    result: RunExperimentResults,
    states: Sequence[str],
    sample_dt_ns: float = 0.5,
    apply_software_demodulation: bool = False,
    apply_lpf_after_demodulation: bool = False,
    lpf_cutoff_frequency_hz: float | None = None,
    lpf_order: int = 5,
) -> dict[str, dict[str, ArrayLike]]:
    """Collect per-state IQ traces from calibration trace handles."""
    if sample_dt_ns <= 0:
        raise ValueError(f"sample_dt_ns must be > 0, got {sample_dt_ns}.")
    if apply_lpf_after_demodulation and not apply_software_demodulation:
        raise ValueError(
            "apply_lpf_after_demodulation=True requires apply_software_demodulation=True."
        )

    signal = None
    sos = None
    if apply_lpf_after_demodulation:
        if lpf_cutoff_frequency_hz is None or lpf_cutoff_frequency_hz <= 0:
            raise ValueError(
                "lpf_cutoff_frequency_hz must be > 0 when "
                "apply_lpf_after_demodulation=True."
            )
        fs_hz = 1.0 / (float(sample_dt_ns) * 1e-9)
        nyquist_hz = fs_hz / 2.0
        if lpf_cutoff_frequency_hz >= nyquist_hz:
            raise ValueError(
                "lpf_cutoff_frequency_hz must be below Nyquist "
                f"({nyquist_hz:.3e} Hz)."
            )
        signal = _get_scipy_signal()
        filter_order = max(int(lpf_order), 1)
        sos = signal.butter(
            filter_order,
            float(lpf_cutoff_frequency_hz),
            "lowpass",
            fs=fs_hz,
            output="sos",
        )

    qubits = validate_and_convert_qubits_sweeps(qubits)
    validate_result(result)

    processed_data_dict: dict[str, dict[str, np.ndarray]] = {q.uid: {} for q in qubits}
    for q in qubits:
        for state in states:
            handle = dsl.handles.calibration_trace_handle(q.uid, state)
            try:
                trace = np.asarray(result[handle].data, dtype=complex)
            except KeyError as exc:
                raise KeyError(
                    "Missing calibration trace handle for "
                    f"qubit={q.uid!r}, state={state!r}, handle={handle!r}."
                ) from exc

            if apply_software_demodulation:
                if (
                    q.parameters.readout_resonator_frequency is None
                    or q.parameters.readout_lo_frequency is None
                ):
                    raise ValueError(
                        "readout_resonator_frequency/readout_lo_frequency must be set "
                        "when apply_software_demodulation=True."
                    )
                if_freq = (
                    q.parameters.readout_resonator_frequency
                    - q.parameters.readout_lo_frequency
                )
                i_raw = np.real(trace)
                q_raw = np.imag(trace)
                times_s = np.arange(len(i_raw), dtype=float) * sample_dt_ns * 1e-9
                cos = np.cos(2 * np.pi * if_freq * times_s)
                sin = np.sin(2 * np.pi * if_freq * times_s)
                i_demod = i_raw * cos + q_raw * sin
                q_demod = -i_raw * sin + q_raw * cos
                if apply_lpf_after_demodulation and sos is not None and signal is not None:
                    if i_demod.size > 1:
                        i_demod = signal.sosfiltfilt(sos, i_demod, padlen=0)
                        q_demod = signal.sosfiltfilt(sos, q_demod, padlen=0)
                processed_data_dict[q.uid][state] = i_demod + 1j * q_demod
            else:
                processed_data_dict[q.uid][state] = trace

    return processed_data_dict


@workflow.task
def truncate_time_traces_to_granularity(
    qubits: QuantumElements,
    processed_data_dict: dict[str, dict[str, ArrayLike]],
    states: Sequence[str],
    granularity: int = 16,
) -> dict[str, dict[str, ArrayLike]]:
    """Truncate tail samples to match acquisition granularity."""
    qubits = validate_and_convert_qubits_sweeps(qubits)
    if granularity <= 0:
        raise ValueError(f"granularity must be > 0, got {granularity}.")

    truncated_data_dict: dict[str, dict[str, np.ndarray]] = {q.uid: {} for q in qubits}
    for q in qubits:
        for state in states:
            trace = np.asarray(processed_data_dict[q.uid][state], dtype=complex)
            n_samples = len(trace)
            n_aligned = (n_samples // granularity) * granularity
            if n_samples == 0 or n_aligned == 0:
                truncated_data_dict[q.uid][state] = trace
            else:
                truncated_data_dict[q.uid][state] = trace[:n_aligned]

    return truncated_data_dict


def _finite_float_or_none(value) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _readout_integration_delay_ns(qubit) -> float:
    delay_s = _finite_float_or_none(
        getattr(qubit.parameters, "readout_integration_delay", None)
    )
    if delay_s is None:
        return 0.0
    return float(delay_s * 1e9)


def _readout_flat_length_ns(qubit) -> float | None:
    readout_pulse = getattr(qubit.parameters, "readout_pulse", None)
    if not isinstance(readout_pulse, dict):
        return None

    pulse_function = readout_pulse.get("function")
    if pulse_function not in ("GaussianSquare", "GaussianSquareDRAG"):
        return None

    width_s = _finite_float_or_none(readout_pulse.get("width"))
    if width_s is not None and width_s > 0.0:
        return float(width_s * 1e9)

    sigma = _finite_float_or_none(readout_pulse.get("sigma"))
    risefall_sigma_ratio = _finite_float_or_none(
        readout_pulse.get("risefall_sigma_ratio")
    )
    readout_length_s = _finite_float_or_none(
        getattr(qubit.parameters, "readout_length", None)
    )
    if sigma is None or risefall_sigma_ratio is None or readout_length_s is None:
        return None

    flat_length_s = readout_length_s * (1.0 - risefall_sigma_ratio * sigma)
    if not np.isfinite(flat_length_s) or flat_length_s <= 0.0:
        return None
    return float(flat_length_s * 1e9)


def _readout_aligned_boundaries_ns(qubit) -> dict[str, float | None]:
    readout_integration_length_s = getattr(
        qubit.parameters, "readout_integration_length", None
    )
    delay_ns = _readout_integration_delay_ns(qubit)
    int_end_s = _finite_float_or_none(readout_integration_length_s)
    int_end_ns = float(delay_ns + int_end_s * 1e9) if int_end_s is not None else None
    ro_flat_ns = _readout_flat_length_ns(qubit)
    if ro_flat_ns is not None:
        ro_flat_ns = float(delay_ns + ro_flat_ns)
    return {
        "int_start": delay_ns,
        "int_end": int_end_ns,
        "ro_flat": ro_flat_ns,
    }


def _get_scipy_signal():
    try:
        from scipy import signal
    except Exception as exc:
        raise ImportError("SciPy is required for do_dsp_analysis=True") from exc
    return signal


def _sanitize_nperseg(nperseg: int, n_samples: int) -> int:
    if n_samples <= 0:
        return 0
    return max(1, min(int(nperseg), int(n_samples)))


def _sanitize_noverlap(noverlap: int, nperseg: int) -> int:
    if nperseg <= 1:
        return 0
    return max(0, min(int(noverlap), int(nperseg) - 1))


def _sanitize_smooth_window(window: int) -> int:
    w = max(int(window), 1)
    if w % 2 == 0:
        w += 1
    return w


def _nan_moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size == 0:
        return values
    valid = np.isfinite(values).astype(float)
    filled = np.where(np.isfinite(values), values, 0.0)
    kernel = np.ones(window, dtype=float)
    num = np.convolve(filled, kernel, mode="same")
    den = np.convolve(valid, kernel, mode="same")
    out = np.full(values.shape, np.nan, dtype=float)
    nonzero = den > 0
    out[nonzero] = num[nonzero] / den[nonzero]
    return out


def _safe_peak_frequency(freqs_hz: np.ndarray, power: np.ndarray) -> tuple[float | None, float | None]:
    if power.size == 0:
        return None, None
    finite = np.isfinite(power)
    if not np.any(finite):
        return None, None
    idx = int(np.nanargmax(power))
    return float(freqs_hz[idx]), float(power[idx])


def _distance_to_boundary(
    event_time_ns: float | None, boundaries_ns: dict[str, float | None], key: str
) -> float | None:
    boundary = boundaries_ns.get(key)
    if boundary is None or event_time_ns is None:
        return None
    return float(event_time_ns - boundary)


def _required_flat_length_prior_ns(qubit) -> float:
    expected_flat_ns = _readout_flat_length_ns(qubit)
    if expected_flat_ns is None:
        pulse = getattr(qubit.parameters, "readout_pulse", None)
        raise ValueError(
            "Cannot determine expected flat length for flat-window detection on "
            f"qubit={qubit.uid!r}. Required prior: readout_pulse.width or "
            "readout_pulse.{sigma, risefall_sigma_ratio} with GaussianSquare/"
            f"GaussianSquareDRAG pulse. Got readout_pulse={pulse!r}."
        )
    return float(expected_flat_ns)


def _robust_zscore(values: np.ndarray) -> np.ndarray:
    """Return robust z-score (median/MAD based) with finite-sample guards."""
    out = np.full(values.shape, np.nan, dtype=float)
    finite = np.isfinite(values)
    if not np.any(finite):
        return out

    x = values[finite]
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    if mad <= np.finfo(float).eps:
        std = float(np.std(x))
        if std <= np.finfo(float).eps:
            out[finite] = 0.0
            return out
        out[finite] = np.abs((x - med) / std)
        return out

    out[finite] = np.abs(0.6744897501960817 * (x - med) / mad)
    return out


def _finite_phase_gradient(
    phase_unwrapped: np.ndarray, amplitude: np.ndarray, mask_abs: float, smooth_window: int
) -> np.ndarray:
    phase_masked = np.array(phase_unwrapped, copy=True, dtype=float)
    if phase_masked.size > 0:
        phase_masked[amplitude < mask_abs] = np.nan
    phase_smoothed = _nan_moving_average(phase_masked, smooth_window)
    phase_grad = np.abs(np.gradient(phase_smoothed)) if phase_smoothed.size > 0 else np.array([], dtype=float)
    return phase_grad


def _edge_feature_components_for_trace(
    trace: np.ndarray,
    phase_mask_relative_threshold: float,
    smooth_window: int,
    phase_weight: float,
) -> dict[str, np.ndarray]:
    """Return per-trace edge features and the combined state score."""
    i_trace = np.real(trace)
    q_trace = np.imag(trace)
    amplitude = np.abs(trace)
    amplitude_peak = float(np.max(amplitude)) if amplitude.size > 0 else 0.0
    mask_abs = float(phase_mask_relative_threshold * amplitude_peak)
    phase = np.unwrap(np.angle(trace)) if trace.size > 0 else np.array([], dtype=float)

    i_smooth = _nan_moving_average(i_trace.astype(float), smooth_window)
    q_smooth = _nan_moving_average(q_trace.astype(float), smooth_window)
    a_smooth = _nan_moving_average(amplitude.astype(float), smooth_window)
    p_grad = _finite_phase_gradient(phase, amplitude, mask_abs, smooth_window)

    i_grad = np.abs(np.gradient(i_smooth)) if i_smooth.size > 0 else np.array([], dtype=float)
    q_grad = np.abs(np.gradient(q_smooth)) if q_smooth.size > 0 else np.array([], dtype=float)
    a_grad = np.abs(np.gradient(a_smooth)) if a_smooth.size > 0 else np.array([], dtype=float)

    i_grad_z = _robust_zscore(i_grad)
    q_grad_z = _robust_zscore(q_grad)
    a_grad_z = _robust_zscore(a_grad)
    p_grad_z = _robust_zscore(p_grad)
    state_score = (
        i_grad_z
        + q_grad_z
        + a_grad_z
        + max(float(phase_weight), 0.0) * p_grad_z
    )
    return {
        "dI_abs": i_grad,
        "dQ_abs": q_grad,
        "dA_abs": a_grad,
        "dphase_abs": p_grad,
        "state_score_z": state_score,
    }


def _find_best_edge_pair(
    score_z: np.ndarray,
    expected_length_samples: int,
    tolerance_ratio: float,
) -> tuple[int | None, int | None, float | None, int, int]:
    l0 = max(int(expected_length_samples), 1)
    distance_band_ratio = _FLAT_WINDOW_DISTANCE_BAND_RATIO
    min_sep = max(1, int(np.ceil(l0 * (1.0 - distance_band_ratio))))
    max_sep = max(min_sep, int(np.floor(l0 * (1.0 + distance_band_ratio))))

    finite_idx = np.where(np.isfinite(score_z))[0]
    if finite_idx.size < 2:
        return None, None, None, min_sep, max_sep

    sigma_l = float(max(2, int(round(l0 * max(float(tolerance_ratio), 0.0)))))
    denom = 2.0 * sigma_l * sigma_l

    best_i = None
    best_j = None
    best_score = None
    for i in finite_idx[:-1]:
        seps = finite_idx - i
        valid = (seps >= min_sep) & (seps <= max_sep)
        js = finite_idx[valid]
        if js.size == 0:
            continue
        delta = js.astype(float) - float(i) - float(l0)
        objective = score_z[i] + score_z[js] - (delta * delta) / denom
        local_k = int(np.argmax(objective))
        candidate_score = float(objective[local_k])
        if best_score is None or candidate_score > best_score:
            best_i = int(i)
            best_j = int(js[local_k])
            best_score = candidate_score

    return best_i, best_j, best_score, min_sep, max_sep


@workflow.task
def detect_flat_window(
    qubits: QuantumElements,
    states: Sequence[str],
    processed_data_dict: dict[str, dict[str, ArrayLike]],
    sample_dt_ns: float = 0.5,
    phase_mask_relative_threshold: float = 0.15,
    smoothing_window_samples: int = 9,
    soft_tolerance_ratio: float = 0.10,
    phase_weight: float = 0.4,
    min_peak_z: float = 2.0,
) -> dict[str, dict]:
    """Detect flat_start/flat_end using soft length-constrained two-edge search."""
    if sample_dt_ns <= 0:
        raise ValueError(f"sample_dt_ns must be > 0, got {sample_dt_ns}.")

    qubits = validate_and_convert_qubits_sweeps(qubits)
    phase_mask_relative_threshold = max(float(phase_mask_relative_threshold), 0.0)
    smooth_window = _sanitize_smooth_window(smoothing_window_samples)
    soft_tolerance_ratio = max(float(soft_tolerance_ratio), 0.0)
    phase_weight = max(float(phase_weight), 0.0)
    min_peak_z = float(min_peak_z)

    out: dict[str, dict] = {}
    for q in qubits:
        expected_flat_ns = _required_flat_length_prior_ns(q)
        expected_samples = max(int(round(expected_flat_ns / float(sample_dt_ns))), 1)
        band = _FLAT_WINDOW_DISTANCE_BAND_RATIO
        allowed_min_sep_samples = max(1, int(np.ceil(expected_samples * (1.0 - band))))
        allowed_max_sep_samples = max(
            allowed_min_sep_samples, int(np.floor(expected_samples * (1.0 + band)))
        )
        delay_ns = _readout_integration_delay_ns(q)

        state_scores: list[np.ndarray] = []
        min_n = None
        for state in states:
            trace = np.asarray(processed_data_dict[q.uid][state], dtype=complex)
            feature_components = _edge_feature_components_for_trace(
                trace=trace,
                phase_mask_relative_threshold=phase_mask_relative_threshold,
                smooth_window=smooth_window,
                phase_weight=phase_weight,
            )
            score = feature_components["state_score_z"]
            if min_n is None:
                min_n = int(score.size)
            else:
                min_n = min(min_n, int(score.size))
            state_scores.append(score)

        if min_n is None or min_n < 3:
            out[q.uid] = {
                "success": False,
                "flat_start_index": None,
                "flat_end_index": None,
                "flat_start_ns": None,
                "flat_end_ns": None,
                "flat_length_ns_detected": None,
                "flat_length_ns_expected": float(expected_flat_ns),
                "expected_length_samples": int(expected_samples),
                "allowed_min_sep_samples": int(allowed_min_sep_samples),
                "allowed_max_sep_samples": int(allowed_max_sep_samples),
                "detected_sep_samples": None,
                "pair_score": None,
                "reason": "insufficient_samples",
            }
            continue

        stacked = np.vstack([score[:min_n] for score in state_scores])
        combined_score = np.nanmedian(stacked, axis=0)
        combined_score_z = _robust_zscore(combined_score)

        (
            flat_start_idx,
            flat_end_idx,
            pair_score,
            allowed_min_sep_samples,
            allowed_max_sep_samples,
        ) = _find_best_edge_pair(
            score_z=combined_score_z,
            expected_length_samples=expected_samples,
            tolerance_ratio=soft_tolerance_ratio,
        )

        if flat_start_idx is None or flat_end_idx is None:
            out[q.uid] = {
                "success": False,
                "flat_start_index": None,
                "flat_end_index": None,
                "flat_start_ns": None,
                "flat_end_ns": None,
                "flat_length_ns_detected": None,
                "flat_length_ns_expected": float(expected_flat_ns),
                "expected_length_samples": int(expected_samples),
                "allowed_min_sep_samples": int(allowed_min_sep_samples),
                "allowed_max_sep_samples": int(allowed_max_sep_samples),
                "detected_sep_samples": None,
                "pair_score": None,
                "reason": "no_candidate_in_distance_band",
            }
            continue

        start_z = combined_score_z[flat_start_idx]
        end_z = combined_score_z[flat_end_idx]
        if (
            not np.isfinite(start_z)
            or not np.isfinite(end_z)
            or float(start_z) < min_peak_z
            or float(end_z) < min_peak_z
        ):
            out[q.uid] = {
                "success": False,
                "flat_start_index": int(flat_start_idx),
                "flat_end_index": int(flat_end_idx),
                "flat_start_ns": float(delay_ns + flat_start_idx * sample_dt_ns),
                "flat_end_ns": float(delay_ns + flat_end_idx * sample_dt_ns),
                "flat_length_ns_detected": float((flat_end_idx - flat_start_idx) * sample_dt_ns),
                "flat_length_ns_expected": float(expected_flat_ns),
                "expected_length_samples": int(expected_samples),
                "allowed_min_sep_samples": int(allowed_min_sep_samples),
                "allowed_max_sep_samples": int(allowed_max_sep_samples),
                "detected_sep_samples": int(flat_end_idx - flat_start_idx),
                "pair_score": float(pair_score) if pair_score is not None else None,
                "reason": (
                    "peak_below_threshold: "
                    f"start_z={float(start_z):.2f}, end_z={float(end_z):.2f}, "
                    f"min_peak_z={min_peak_z:.2f}"
                ),
            }
            continue

        flat_start_ns = float(delay_ns + flat_start_idx * sample_dt_ns)
        flat_end_ns = float(delay_ns + flat_end_idx * sample_dt_ns)
        out[q.uid] = {
            "success": True,
            "flat_start_index": int(flat_start_idx),
            "flat_end_index": int(flat_end_idx),
            "flat_start_ns": flat_start_ns,
            "flat_end_ns": flat_end_ns,
            "flat_length_ns_detected": float(flat_end_ns - flat_start_ns),
            "flat_length_ns_expected": float(expected_flat_ns),
            "expected_length_samples": int(expected_samples),
            "allowed_min_sep_samples": int(allowed_min_sep_samples),
            "allowed_max_sep_samples": int(allowed_max_sep_samples),
            "detected_sep_samples": int(flat_end_idx - flat_start_idx),
            "pair_score": float(pair_score) if pair_score is not None else None,
            "reason": None,
        }

    return out


@workflow.task
def build_flat_window_parameter_updates(
    qubits: QuantumElements,
    flat_window_detection: dict[str, dict],
) -> dict[str, dict]:
    """Build old/new readout-integration parameters from flat-window detection."""
    qubits = validate_and_convert_qubits_sweeps(qubits)

    out = {
        "old_parameter_values": {},
        "new_parameter_values": {},
    }
    for q in qubits:
        old_delay = _finite_float_or_none(
            getattr(q.parameters, "readout_integration_delay", None)
        )
        old_length = _finite_float_or_none(
            getattr(q.parameters, "readout_integration_length", None)
        )
        out["old_parameter_values"][q.uid] = {
            "readout_integration_delay": old_delay,
            "readout_integration_length": old_length,
        }

        detection = flat_window_detection.get(q.uid, {})
        if not detection.get("success"):
            continue
        start_ns = _finite_float_or_none(detection.get("flat_start_ns"))
        end_ns = _finite_float_or_none(detection.get("flat_end_ns"))
        if start_ns is None or end_ns is None or end_ns <= start_ns:
            continue
        out["new_parameter_values"][q.uid] = {
            "readout_integration_delay": float(start_ns * 1e-9),
            "readout_integration_length": float((end_ns - start_ns) * 1e-9),
        }

    return out


@workflow.task
def compute_diagnostics(
    qubits: QuantumElements,
    states: Sequence[str],
    processed_data_dict: dict[str, dict[str, ArrayLike]],
    sample_dt_ns: float = 0.5,
    phase_mask_relative_threshold: float = 0.15,
) -> dict[str, dict]:
    """Compute compact diagnostics for IQ time traces."""
    qubits = validate_and_convert_qubits_sweeps(qubits)
    phase_mask_relative_threshold = max(float(phase_mask_relative_threshold), 0.0)

    diagnostics: dict[str, dict] = {}
    for q in qubits:
        boundaries_ns = _readout_aligned_boundaries_ns(q)
        delay_ns = _readout_integration_delay_ns(q)
        q_diag: dict[str, object] = {
            "axis_reference": "readout_start",
            "sample_dt_ns": float(sample_dt_ns),
            "states": {},
        }

        for state in states:
            trace = np.asarray(processed_data_dict[q.uid][state], dtype=complex)
            n_samples = int(trace.size)
            axis = delay_ns + np.arange(n_samples, dtype=float) * sample_dt_ns
            time_end_ns = float(axis[-1]) if n_samples > 0 else 0.0

            amplitude = np.abs(trace)
            phase = np.unwrap(np.angle(trace)) if n_samples > 0 else np.array([])

            amplitude_max = float(np.max(amplitude)) if n_samples > 0 else 0.0
            mask_absolute_threshold = phase_mask_relative_threshold * amplitude_max

            if n_samples > 0:
                phase_masked = np.array(phase, copy=True)
                phase_masked[amplitude < mask_absolute_threshold] = np.nan
                valid_phase = np.isfinite(phase_masked)
                valid_phase_fraction = float(np.mean(valid_phase))
                phase_span_masked_rad = (
                    float(np.nanmax(phase_masked) - np.nanmin(phase_masked))
                    if np.any(valid_phase)
                    else None
                )
                amplitude_min = float(np.min(amplitude))
                amplitude_median = float(np.median(amplitude))
                amplitude_argmin_index = int(np.argmin(amplitude))
                amplitude_argmin_time_ns = float(axis[amplitude_argmin_index])
            else:
                valid_phase_fraction = 0.0
                phase_span_masked_rad = None
                amplitude_min = float("nan")
                amplitude_median = float("nan")
                amplitude_argmin_index = -1
                amplitude_argmin_time_ns = 0.0

            q_diag["states"][state] = {
                "n_samples": n_samples,
                "time_end_ns": time_end_ns,
                "mask_relative_threshold": phase_mask_relative_threshold,
                "mask_absolute_threshold": float(mask_absolute_threshold),
                "valid_phase_fraction": valid_phase_fraction,
                "amplitude_min": amplitude_min,
                "amplitude_median": amplitude_median,
                "amplitude_max": amplitude_max,
                "amplitude_argmin_index": amplitude_argmin_index,
                "amplitude_argmin_time_ns": amplitude_argmin_time_ns,
                "phase_span_masked_rad": phase_span_masked_rad,
                "boundaries_ns": dict(boundaries_ns),
            }

        diagnostics[q.uid] = q_diag

    return diagnostics


@workflow.task
def compute_dsp_diagnostics(
    qubits: QuantumElements,
    states: Sequence[str],
    raw_data_dict: dict[str, dict[str, ArrayLike]],
    pre_demod_data_dict: dict[str, dict[str, ArrayLike]] | None = None,
    compare_pre_demod: bool = False,
    sample_dt_ns: float = 0.5,
    phase_mask_relative_threshold: float = 0.15,
    welch_nperseg: int = 256,
    welch_noverlap: int = 128,
    stft_nperseg: int = 256,
    stft_noverlap: int = 192,
    stft_window: str = "hann",
    dphi_smooth_window: int = 1,
) -> dict[str, dict]:
    """Compute DSP diagnostics for analysis trace and optional pre-demod RAW trace."""
    if sample_dt_ns <= 0:
        raise ValueError(f"sample_dt_ns must be > 0, got {sample_dt_ns}.")

    signal = _get_scipy_signal()
    qubits = validate_and_convert_qubits_sweeps(qubits)
    phase_mask_relative_threshold = max(float(phase_mask_relative_threshold), 0.0)
    dt_s = float(sample_dt_ns) * 1e-9
    fs_hz = 1.0 / dt_s
    dphi_smooth_window = _sanitize_smooth_window(dphi_smooth_window)
    compare_enabled = bool(compare_pre_demod and pre_demod_data_dict is not None)

    dsp_diag: dict[str, dict] = {}
    for q in qubits:
        boundaries_ns = _readout_aligned_boundaries_ns(q)
        delay_ns = _readout_integration_delay_ns(q)
        q_diag: dict[str, object] = {
            "axis_reference": "readout_start",
            "sample_dt_ns": float(sample_dt_ns),
            "fs_hz": float(fs_hz),
            "states": {},
        }

        for state in states:
            trace = np.asarray(raw_data_dict[q.uid][state], dtype=complex)
            n_samples = int(trace.size)
            axis_ns = delay_ns + np.arange(n_samples, dtype=float) * float(sample_dt_ns)
            amplitude = np.abs(trace)
            amplitude_peak = float(np.max(amplitude)) if n_samples > 0 else 0.0
            mask_abs = phase_mask_relative_threshold * amplitude_peak

            phase = np.unwrap(np.angle(trace)) if n_samples > 0 else np.array([], dtype=float)
            phase_masked = np.array(phase, copy=True)
            if n_samples > 0:
                phase_masked[amplitude < mask_abs] = np.nan
            valid_phase = np.isfinite(phase_masked)
            valid_phase_fraction = float(np.mean(valid_phase)) if n_samples > 0 else 0.0

            if n_samples > 1:
                dphi = np.diff(phase_masked)
                dphi_abs = np.abs(dphi)
                if np.any(np.isfinite(dphi_abs)):
                    idx_dphi = int(np.nanargmax(dphi_abs))
                    max_abs_dphi_rad = float(dphi_abs[idx_dphi])
                    max_abs_dphi_time_ns = float(axis_ns[idx_dphi + 1])
                else:
                    max_abs_dphi_rad = None
                    max_abs_dphi_time_ns = None
            else:
                max_abs_dphi_rad = None
                max_abs_dphi_time_ns = None

            if n_samples > 0:
                dphi_dt = np.gradient(phase_masked, dt_s)
                dphi_dt = _nan_moving_average(dphi_dt, dphi_smooth_window)
                inst_freq_hz = dphi_dt / (2 * np.pi)
            else:
                inst_freq_hz = np.array([], dtype=float)

            if inst_freq_hz.size > 0 and np.any(np.isfinite(inst_freq_hz)):
                idx_if = int(np.nanargmax(np.abs(inst_freq_hz)))
                max_abs_inst_freq_hz = float(np.abs(inst_freq_hz[idx_if]))
                max_abs_inst_freq_time_ns = float(axis_ns[idx_if])
                amplitude_at_max_jump = float(amplitude[idx_if]) if n_samples > 0 else None
            else:
                max_abs_inst_freq_hz = None
                max_abs_inst_freq_time_ns = None
                amplitude_at_max_jump = None

            if n_samples > 1:
                trace_dc = trace - np.mean(trace)

                wn = _sanitize_nperseg(welch_nperseg, n_samples)
                wo = _sanitize_noverlap(welch_noverlap, wn)
                if wn > 1:
                    freq_welch, psd = signal.welch(
                        trace_dc,
                        fs=fs_hz,
                        nperseg=wn,
                        noverlap=wo,
                        return_onesided=False,
                        scaling="density",
                    )
                    freq_welch = np.fft.fftshift(freq_welch)
                    psd = np.fft.fftshift(np.real(psd))
                    welch_peak_freq_hz, welch_peak_psd = _safe_peak_frequency(
                        freq_welch, psd
                    )
                    welch_total_power = float(np.trapezoid(psd, freq_welch))
                else:
                    welch_peak_freq_hz = None
                    welch_peak_psd = None
                    welch_total_power = 0.0

                sn = _sanitize_nperseg(stft_nperseg, n_samples)
                so = _sanitize_noverlap(stft_noverlap, sn)
                if sn > 1:
                    stft_freq_hz, stft_time_s, stft_vals = signal.stft(
                        trace_dc,
                        fs=fs_hz,
                        window=stft_window,
                        nperseg=sn,
                        noverlap=so,
                        return_onesided=False,
                        boundary=None,
                        padded=False,
                    )
                    stft_freq_hz = np.fft.fftshift(stft_freq_hz)
                    stft_mag = np.abs(np.fft.fftshift(stft_vals, axes=0))

                    if stft_mag.size > 0 and np.any(np.isfinite(stft_mag)):
                        peak_flat = int(np.nanargmax(stft_mag))
                        peak_fi, peak_ti = np.unravel_index(peak_flat, stft_mag.shape)
                        stft_peak_freq_hz = float(stft_freq_hz[peak_fi])
                        stft_peak_time_ns = float(stft_time_s[peak_ti] * 1e9 + delay_ns)
                    else:
                        stft_peak_freq_hz = None
                        stft_peak_time_ns = None
                    stft_time_bins = int(len(stft_time_s))
                    stft_freq_bins = int(len(stft_freq_hz))
                else:
                    stft_peak_freq_hz = None
                    stft_peak_time_ns = None
                    stft_time_bins = 0
                    stft_freq_bins = 0
            else:
                welch_peak_freq_hz = None
                welch_peak_psd = None
                welch_total_power = 0.0
                stft_peak_freq_hz = None
                stft_peak_time_ns = None
                stft_time_bins = 0
                stft_freq_bins = 0

            raw_welch_peak_freq_hz = None
            raw_welch_peak_psd = None
            raw_welch_total_power = None
            raw_stft_peak_freq_hz = None
            raw_stft_peak_time_ns = None
            raw_stft_time_bins = None
            raw_stft_freq_bins = None
            compare_welch_peak_freq_shift_hz = None
            compare_stft_peak_freq_shift_hz = None

            if compare_enabled:
                raw_trace = np.asarray(pre_demod_data_dict[q.uid][state], dtype=complex)  # type: ignore[index]
                raw_n_samples = int(raw_trace.size)
                if raw_n_samples > 1:
                    raw_trace_dc = raw_trace - np.mean(raw_trace)

                    raw_wn = _sanitize_nperseg(welch_nperseg, raw_n_samples)
                    raw_wo = _sanitize_noverlap(welch_noverlap, raw_wn)
                    if raw_wn > 1:
                        raw_freq_welch, raw_psd = signal.welch(
                            raw_trace_dc,
                            fs=fs_hz,
                            nperseg=raw_wn,
                            noverlap=raw_wo,
                            return_onesided=False,
                            scaling="density",
                        )
                        raw_freq_welch = np.fft.fftshift(raw_freq_welch)
                        raw_psd = np.fft.fftshift(np.real(raw_psd))
                        (
                            raw_welch_peak_freq_hz,
                            raw_welch_peak_psd,
                        ) = _safe_peak_frequency(raw_freq_welch, raw_psd)
                        raw_welch_total_power = float(np.trapezoid(raw_psd, raw_freq_welch))
                    else:
                        raw_welch_total_power = 0.0

                    raw_sn = _sanitize_nperseg(stft_nperseg, raw_n_samples)
                    raw_so = _sanitize_noverlap(stft_noverlap, raw_sn)
                    if raw_sn > 1:
                        raw_stft_freq_hz, raw_stft_time_s, raw_stft_vals = signal.stft(
                            raw_trace_dc,
                            fs=fs_hz,
                            window=stft_window,
                            nperseg=raw_sn,
                            noverlap=raw_so,
                            return_onesided=False,
                            boundary=None,
                            padded=False,
                        )
                        raw_stft_freq_hz = np.fft.fftshift(raw_stft_freq_hz)
                        raw_stft_mag = np.abs(np.fft.fftshift(raw_stft_vals, axes=0))
                        if raw_stft_mag.size > 0 and np.any(np.isfinite(raw_stft_mag)):
                            raw_peak_flat = int(np.nanargmax(raw_stft_mag))
                            raw_peak_fi, raw_peak_ti = np.unravel_index(
                                raw_peak_flat, raw_stft_mag.shape
                            )
                            raw_stft_peak_freq_hz = float(raw_stft_freq_hz[raw_peak_fi])
                            raw_stft_peak_time_ns = float(
                                raw_stft_time_s[raw_peak_ti] * 1e9 + delay_ns
                            )
                        raw_stft_time_bins = int(len(raw_stft_time_s))
                        raw_stft_freq_bins = int(len(raw_stft_freq_hz))
                    else:
                        raw_stft_time_bins = 0
                        raw_stft_freq_bins = 0
                elif raw_n_samples >= 0:
                    raw_welch_total_power = 0.0
                    raw_stft_time_bins = 0
                    raw_stft_freq_bins = 0

                if welch_peak_freq_hz is not None and raw_welch_peak_freq_hz is not None:
                    compare_welch_peak_freq_shift_hz = float(
                        welch_peak_freq_hz - raw_welch_peak_freq_hz
                    )
                if stft_peak_freq_hz is not None and raw_stft_peak_freq_hz is not None:
                    compare_stft_peak_freq_shift_hz = float(
                        stft_peak_freq_hz - raw_stft_peak_freq_hz
                    )

            q_diag["states"][state] = {
                "n_samples_raw": n_samples,
                "fs_hz": float(fs_hz),
                "welch_peak_freq_hz": welch_peak_freq_hz,
                "welch_peak_psd": welch_peak_psd,
                "welch_total_power": welch_total_power,
                "raw_welch_peak_freq_hz": raw_welch_peak_freq_hz,
                "raw_welch_peak_psd": raw_welch_peak_psd,
                "raw_welch_total_power": raw_welch_total_power,
                "compare_welch_peak_freq_shift_hz": compare_welch_peak_freq_shift_hz,
                "stft_time_bins": stft_time_bins,
                "stft_freq_bins": stft_freq_bins,
                "stft_peak_freq_hz": stft_peak_freq_hz,
                "stft_peak_time_ns": stft_peak_time_ns,
                "raw_stft_time_bins": raw_stft_time_bins,
                "raw_stft_freq_bins": raw_stft_freq_bins,
                "raw_stft_peak_freq_hz": raw_stft_peak_freq_hz,
                "raw_stft_peak_time_ns": raw_stft_peak_time_ns,
                "compare_stft_peak_freq_shift_hz": compare_stft_peak_freq_shift_hz,
                "valid_phase_fraction": valid_phase_fraction,
                "max_abs_dphi_rad": max_abs_dphi_rad,
                "max_abs_dphi_time_ns": max_abs_dphi_time_ns,
                "max_abs_inst_freq_hz": max_abs_inst_freq_hz,
                "max_abs_inst_freq_time_ns": max_abs_inst_freq_time_ns,
                "amplitude_at_max_jump": amplitude_at_max_jump,
                "distance_to_int_end_ns": _distance_to_boundary(
                    max_abs_inst_freq_time_ns, boundaries_ns, "int_end"
                ),
                "distance_to_ro_flat_ns": _distance_to_boundary(
                    max_abs_inst_freq_time_ns, boundaries_ns, "ro_flat"
                ),
                "boundaries_ns": dict(boundaries_ns),
            }

        dsp_diag[q.uid] = q_diag

    return dsp_diag


@workflow.task
def plot_flat_window_debug(
    qubits: QuantumElements,
    states: Sequence[str],
    processed_data_dict: dict[str, dict[str, ArrayLike]],
    flat_window_detection: dict[str, dict] | None = None,
    options: BasePlottingOptions | None = None,
    sample_dt_ns: float = 0.5,
    phase_mask_relative_threshold: float = 0.15,
    smoothing_window_samples: int = 9,
    phase_weight: float = 0.4,
) -> dict[str, mpl.figure.Figure]:
    """Plot flat-window edge-feature diagnostics used by the detector."""
    if sample_dt_ns <= 0:
        raise ValueError(f"sample_dt_ns must be > 0, got {sample_dt_ns}.")

    opts = BasePlottingOptions() if options is None else options
    qubits = validate_and_convert_qubits_sweeps(qubits)
    phase_mask_relative_threshold = max(float(phase_mask_relative_threshold), 0.0)
    smooth_window = _sanitize_smooth_window(smoothing_window_samples)
    phase_weight = max(float(phase_weight), 0.0)

    figures: dict[str, mpl.figure.Figure] = {}
    color_map = {0: "b", 1: "r", 2: "g"}
    for q in qubits:
        fig = plt.figure(figsize=(21, 11), constrained_layout=True)
        grid = fig.add_gridspec(2, 3, hspace=0.22, wspace=0.22)
        ax_di = fig.add_subplot(grid[0, 0])
        ax_dq = fig.add_subplot(grid[0, 1], sharex=ax_di)
        ax_da = fig.add_subplot(grid[0, 2], sharex=ax_di)
        ax_dp = fig.add_subplot(grid[1, 0], sharex=ax_di)
        ax_score = fig.add_subplot(grid[1, 1], sharex=ax_di)
        ax_info = fig.add_subplot(grid[1, 2])
        ax_info.axis("off")

        fig.suptitle(timestamped_title(f"IQ Flat-Window Debug {q.uid}"), fontsize=14)
        ax_di.set_title("Smoothed |dI|")
        ax_di.set_ylabel("Magnitude (a.u./sample)")
        ax_dq.set_title("Smoothed |dQ|")
        ax_dq.set_ylabel("Magnitude (a.u./sample)")
        ax_da.set_title("Smoothed |dA|")
        ax_da.set_ylabel("Magnitude (a.u./sample)")
        ax_dp.set_title("Smoothed |dphase|")
        ax_dp.set_xlabel("Time (ns, readout-start frame)")
        ax_dp.set_ylabel("Magnitude (rad/sample)")
        ax_score.set_title("Edge Score (Combined)")
        ax_score.set_xlabel("Time (ns, readout-start frame)")
        ax_score.set_ylabel("Robust z-score")

        detection_info = (flat_window_detection or {}).get(q.uid, {})
        delay_ns = _readout_integration_delay_ns(q)

        state_scores: list[np.ndarray] = []
        min_n = None
        for idx, state in enumerate(states):
            color = color_map.get(idx, f"C{idx}")
            trace = np.asarray(processed_data_dict[q.uid][state], dtype=complex)
            feature_components = _edge_feature_components_for_trace(
                trace=trace,
                phase_mask_relative_threshold=phase_mask_relative_threshold,
                smooth_window=smooth_window,
                phase_weight=phase_weight,
            )
            score = feature_components["state_score_z"]
            n = int(score.size)
            axis = delay_ns + np.arange(n, dtype=float) * sample_dt_ns

            ax_di.plot(axis, feature_components["dI_abs"], color=color, linewidth=1.0, alpha=0.9, label=state)
            ax_dq.plot(axis, feature_components["dQ_abs"], color=color, linewidth=1.0, alpha=0.9, label=state)
            ax_da.plot(axis, feature_components["dA_abs"], color=color, linewidth=1.0, alpha=0.9, label=state)
            ax_dp.plot(axis, feature_components["dphase_abs"], color=color, linewidth=1.0, alpha=0.9, label=state)

            if min_n is None:
                min_n = n
            else:
                min_n = min(min_n, n)
            state_scores.append(score)

        if min_n is not None and min_n > 0 and state_scores:
            stacked = np.vstack([score[:min_n] for score in state_scores])
            combined_score = np.nanmedian(stacked, axis=0)
            combined_score_z = _robust_zscore(combined_score)
            score_axis = delay_ns + np.arange(min_n, dtype=float) * sample_dt_ns
            ax_score.plot(
                score_axis,
                combined_score_z,
                color="k",
                linewidth=1.2,
                alpha=0.95,
                label="combined score",
            )
        else:
            ax_score.text(
                0.02,
                0.92,
                "insufficient samples for score",
                transform=ax_score.transAxes,
                fontsize=9,
                va="top",
            )

        flat_start_ns = None
        flat_end_ns = None
        if isinstance(detection_info, dict) and detection_info.get("success"):
            flat_start_ns = _finite_float_or_none(detection_info.get("flat_start_ns"))
            flat_end_ns = _finite_float_or_none(detection_info.get("flat_end_ns"))

        if flat_start_ns is not None:
            for axis_obj in (ax_di, ax_dq, ax_da, ax_dp, ax_score):
                axis_obj.axvline(
                    flat_start_ns,
                    color="k",
                    linestyle=":",
                    linewidth=1.0,
                    alpha=0.35,
                )
        if flat_end_ns is not None:
            for axis_obj in (ax_di, ax_dq, ax_da, ax_dp, ax_score):
                axis_obj.axvline(
                    flat_end_ns,
                    color="k",
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.35,
                )

        if isinstance(detection_info, dict):
            expected_length_samples = detection_info.get("expected_length_samples")
            allowed_min_sep = detection_info.get("allowed_min_sep_samples")
            allowed_max_sep = detection_info.get("allowed_max_sep_samples")
            detected_sep = detection_info.get("detected_sep_samples")
            ax_score.text(
                0.01,
                0.90,
                (
                    f"L0={expected_length_samples}, "
                    f"allowed_sep=[{allowed_min_sep}, {allowed_max_sep}], "
                    f"detected_sep={detected_sep}"
                ),
                transform=ax_score.transAxes,
                va="top",
                fontsize=8,
                alpha=0.82,
            )
            if detection_info.get("success"):
                ax_score.text(
                    0.01,
                    0.97,
                    ": flat_start  |  --: flat_end",
                    transform=ax_score.transAxes,
                    va="top",
                    fontsize=8,
                    alpha=0.8,
                )
            else:
                reason = detection_info.get("reason")
                fail_message = (
                    f"flat-window detection failed: {reason}"
                    if reason
                    else "flat-window detection failed"
                )
                ax_score.text(
                    0.01,
                    0.97,
                    fail_message,
                    transform=ax_score.transAxes,
                    va="top",
                    fontsize=8,
                    alpha=0.85,
                )

        ax_info.text(
            0.02,
            0.95,
            (
                f"smoothing_window={smooth_window}\n"
                f"phase_weight={phase_weight:.3f}\n"
                f"phase_mask_threshold={phase_mask_relative_threshold:.3f}"
            ),
            transform=ax_info.transAxes,
            va="top",
            fontsize=9,
        )

        for axis_obj in (ax_di, ax_dq, ax_da, ax_dp, ax_score):
            axis_obj.grid(True, linestyle=":", alpha=0.35)

        handles, labels = ax_di.get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper right", frameon=False, title="State")

        if opts.save_figures:
            workflow.save_artifact(f"IQ_flat_window_debug_{q.uid}", fig)
        if opts.close_figures:
            plt.close(fig)

        figures[q.uid] = fig

    return figures


@workflow.task
def plot_iq_time_traces(
    qubits: QuantumElements,
    states: Sequence[str],
    processed_data_dict: dict[str, dict[str, ArrayLike]],
    options: BasePlottingOptions | None = None,
    sample_dt_ns: float = 0.5,
    phase_mask_relative_threshold: float = 0.15,
    plot_phase_boundaries: bool = True,
    find_flat_window: bool = False,
    flat_window_detection: dict[str, dict] | None = None,
) -> dict[str, mpl.figure.Figure]:
    """Create 5-panel IQ time-trace plots."""
    opts = BasePlottingOptions() if options is None else options
    qubits = validate_and_convert_qubits_sweeps(qubits)
    phase_mask_relative_threshold = max(float(phase_mask_relative_threshold), 0.0)

    figures = {}
    color_map = {0: "b", 1: "r", 2: "g"}
    for q in qubits:
        fig = plt.figure(figsize=(21, 11), constrained_layout=True)
        grid = fig.add_gridspec(2, 3, hspace=0.22, wspace=0.22)
        ax_i = fig.add_subplot(grid[0, 0])
        ax_q = fig.add_subplot(grid[0, 1], sharex=ax_i)
        ax_phase = fig.add_subplot(grid[0, 2], sharex=ax_i)
        ax_amp = fig.add_subplot(grid[1, 0], sharex=ax_i)
        ax_iq = fig.add_subplot(grid[1, 1])
        ax_iq_time = fig.add_subplot(grid[1, 2])

        fig.suptitle(timestamped_title(f"IQ Time Trace {q.uid}"), fontsize=14)

        ax_i.set_title("I(t)")
        ax_i.set_xlabel("Time (ns, readout-start frame)")
        ax_i.set_ylabel("Real Signal Component, $V_{\\mathrm{I}}$ (a.u.)")

        ax_q.set_title("Q(t)")
        ax_q.set_xlabel("Time (ns, readout-start frame)")
        ax_q.set_ylabel("Imaginary Signal Component, $V_{\\mathrm{Q}}$ (a.u.)")

        ax_phase.set_title("Phase(t)")
        ax_phase.set_xlabel("Time (ns, readout-start frame)")
        ax_phase.set_ylabel("Phase (rad)")
        ax_phase.text(
            0.01,
            0.98,
            f"Masked if |IQ| < {phase_mask_relative_threshold:.0%} of peak",
            transform=ax_phase.transAxes,
            va="top",
            fontsize=9,
        )

        ax_iq.set_title("IQ Plane")
        ax_iq.set_xlabel("Real Signal Component, $V_{\\mathrm{I}}$ (a.u.)")
        ax_iq.set_ylabel("Imaginary Signal Component, $V_{\\mathrm{Q}}$ (a.u.)")

        ax_iq_time.set_title("IQ Plane (Time-Colored)")
        ax_iq_time.set_xlabel("Real Signal Component, $V_{\\mathrm{I}}$ (a.u.)")
        ax_iq_time.set_ylabel("Imaginary Signal Component, $V_{\\mathrm{Q}}$ (a.u.)")

        ax_amp.set_title("|IQ|(t)")
        ax_amp.set_xlabel("Time (ns, readout-start frame)")
        ax_amp.set_ylabel("Magnitude, $|IQ|$ (a.u.)")

        detection_info = (
            (flat_window_detection or {}).get(q.uid, {})
            if find_flat_window
            else {}
        )
        if find_flat_window and plot_phase_boundaries and detection_info.get("success"):
            flat_start_ns = _finite_float_or_none(detection_info.get("flat_start_ns"))
            flat_end_ns = _finite_float_or_none(detection_info.get("flat_end_ns"))
            if flat_start_ns is not None:
                for axis_obj in (ax_i, ax_q, ax_phase, ax_amp):
                    axis_obj.axvline(
                        flat_start_ns,
                        color="k",
                        linestyle=":",
                        linewidth=1.0,
                        alpha=0.35,
                    )
            if flat_end_ns is not None:
                for axis_obj in (ax_i, ax_q, ax_phase, ax_amp):
                    axis_obj.axvline(
                        flat_end_ns,
                        color="k",
                        linestyle="--",
                        linewidth=1.0,
                        alpha=0.35,
                    )
            ax_phase.text(
                0.01,
                0.90,
                ": flat_start  |  --: flat_end",
                transform=ax_phase.transAxes,
                va="top",
                fontsize=8,
                alpha=0.8,
            )
        elif find_flat_window:
            reason = detection_info.get("reason") if isinstance(detection_info, dict) else None
            fail_message = (
                f"flat-window detection failed: {reason}"
                if reason
                else "flat-window detection failed"
            )
            ax_phase.text(
                0.01,
                0.90,
                fail_message,
                transform=ax_phase.transAxes,
                va="top",
                fontsize=8,
                alpha=0.85,
            )

        delay_ns = _readout_integration_delay_ns(q)
        time_color_mappable = None
        for i, state in enumerate(states):
            color = color_map.get(i, f"C{i}")
            trace = np.asarray(processed_data_dict[q.uid][state], dtype=complex)
            i_trace = np.real(trace)
            q_trace = np.imag(trace)
            amplitude = np.abs(trace)
            phase = np.unwrap(np.angle(trace))

            amplitude_peak = float(np.max(amplitude)) if amplitude.size else 0.0
            mask_abs = phase_mask_relative_threshold * amplitude_peak
            phase_masked = np.array(phase, copy=True)
            phase_masked[amplitude < mask_abs] = np.nan

            axis = delay_ns + np.arange(len(i_trace), dtype=float) * sample_dt_ns
            ax_i.plot(axis, i_trace, "-", color=color, linewidth=1.0, alpha=0.9, label=state)
            ax_i.plot(
                axis,
                i_trace,
                linestyle="None",
                marker="o",
                color=color,
                markersize=1.2,
                markeredgewidth=0.0,
                alpha=0.45,
            )

            ax_q.plot(axis, q_trace, "-", color=color, linewidth=1.0, alpha=0.9, label=state)
            ax_q.plot(
                axis,
                q_trace,
                linestyle="None",
                marker="o",
                color=color,
                markersize=1.2,
                markeredgewidth=0.0,
                alpha=0.45,
            )

            ax_iq.plot(i_trace, q_trace, "-", color=color, linewidth=1.0, alpha=0.85, label=state)
            ax_iq.scatter(i_trace, q_trace, c=color, s=6, alpha=0.4, edgecolors="none")

            ax_iq_time.plot(i_trace, q_trace, "-", color="k", linewidth=0.5, alpha=0.2)
            state_scatter = ax_iq_time.scatter(
                i_trace,
                q_trace,
                c=axis,
                cmap="inferno",
                s=8,
                marker="o",
                linewidths=0.0,
                alpha=0.78,
                label=state,
            )
            if time_color_mappable is None:
                time_color_mappable = state_scatter

            ax_phase.plot(
                axis, phase_masked, "-", color=color, linewidth=1.0, alpha=0.9, label=state
            )
            valid_phase = np.isfinite(phase_masked)
            ax_phase.plot(
                axis[valid_phase],
                phase_masked[valid_phase],
                linestyle="None",
                marker="o",
                color=color,
                markersize=1.2,
                markeredgewidth=0.0,
                alpha=0.45,
            )

            ax_amp.plot(
                axis, amplitude, "-", color=color, linewidth=1.0, alpha=0.9, label=state
            )
            ax_amp.plot(
                axis,
                amplitude,
                linestyle="None",
                marker="o",
                color=color,
                markersize=1.2,
                markeredgewidth=0.0,
                alpha=0.45,
            )

        ax_iq.set_aspect("equal", adjustable="box")
        ax_iq.set_box_aspect(1)
        ax_iq_time.set_aspect("equal", adjustable="box")
        ax_iq_time.set_box_aspect(1)
        for axis_obj in (ax_i, ax_q, ax_phase, ax_amp):
            axis_obj.grid(True, linestyle=":", alpha=0.35)
        ax_iq.grid(True, linestyle=":", alpha=0.25)
        ax_iq_time.grid(True, linestyle=":", alpha=0.25)

        if time_color_mappable is not None:
            fig.colorbar(
                time_color_mappable,
                ax=ax_iq_time,
                label="Time (ns, readout-start frame)",
            )

        handles, labels = ax_i.get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper right", frameon=False, title="State")

        if opts.save_figures:
            workflow.save_artifact(f"IQ_time_trace_{q.uid}", fig)

        if opts.close_figures:
            plt.close(fig)

        figures[q.uid] = fig

    return figures


@workflow.task
def plot_iq_dsp(
    qubits: QuantumElements,
    states: Sequence[str],
    raw_data_dict: dict[str, dict[str, ArrayLike]],
    pre_demod_data_dict: dict[str, dict[str, ArrayLike]] | None = None,
    compare_pre_demod: bool = False,
    demod_pre_lpf_data_dict: dict[str, dict[str, ArrayLike]] | None = None,
    compare_demod_lpf: bool = False,
    options: BasePlottingOptions | None = None,
    sample_dt_ns: float = 0.5,
    phase_mask_relative_threshold: float = 0.15,
    welch_nperseg: int = 256,
    welch_noverlap: int = 128,
    stft_nperseg: int = 256,
    stft_noverlap: int = 192,
    stft_window: str = "hann",
    dsp_freq_limit_mhz: float | None = None,
    dphi_smooth_window: int = 1,
    plot_phase_boundaries: bool = True,
) -> dict[str, mpl.figure.Figure]:
    """Create DSP figures with optional pre-demod RAW comparison."""
    if sample_dt_ns <= 0:
        raise ValueError(f"sample_dt_ns must be > 0, got {sample_dt_ns}.")

    signal = _get_scipy_signal()
    opts = BasePlottingOptions() if options is None else options
    qubits = validate_and_convert_qubits_sweeps(qubits)
    fs_hz = 1.0 / (float(sample_dt_ns) * 1e-9)
    color_map = {0: "b", 1: "r", 2: "g"}
    freq_limit_mhz = (
        None
        if dsp_freq_limit_mhz is None or dsp_freq_limit_mhz <= 0
        else float(dsp_freq_limit_mhz)
    )
    compare_enabled = bool(compare_pre_demod and pre_demod_data_dict is not None)
    compare_demod_lpf_enabled = bool(
        compare_demod_lpf and demod_pre_lpf_data_dict is not None
    )
    demod_axis_enabled = bool(compare_pre_demod)
    if compare_demod_lpf_enabled and not demod_axis_enabled:
        raise ValueError(
            "compare_demod_lpf=True requires demodulation axis mode "
            "(compare_pre_demod=True)."
        )

    figures: dict[str, mpl.figure.Figure] = {}
    for q in qubits:
        delay_ns = _readout_integration_delay_ns(q)
        lo_frequency_hz = getattr(q.parameters, "readout_lo_frequency", None)
        raw_freq_offset_hz = float(lo_frequency_hz) if lo_frequency_hz is not None else 0.0
        analysis_freq_offset_hz = 0.0 if demod_axis_enabled else raw_freq_offset_hz
        raw_frequency_axis_label = (
            "RF-Frame Frequency (GHz)"
            if lo_frequency_hz is not None
            else "Frequency (GHz)"
        )
        analysis_frequency_axis_label = (
            "IF-Frame Frequency (MHz)" if demod_axis_enabled else raw_frequency_axis_label
        )

        fig = plt.figure(figsize=(22, 9), constrained_layout=True)
        grid = fig.add_gridspec(1, 2, hspace=0.24, wspace=0.24)
        psd_grid = grid[0, 0].subgridspec(2, 1, hspace=0.12)
        ax_psd_raw = fig.add_subplot(psd_grid[0, 0])
        ax_psd = fig.add_subplot(psd_grid[1, 0])
        stft_grid = grid[0, 1].subgridspec(2, 1, hspace=0.12)
        ax_stft_raw = fig.add_subplot(stft_grid[0, 0])
        ax_stft = fig.add_subplot(stft_grid[1, 0], sharex=ax_stft_raw)
        fig.suptitle(timestamped_title(f"IQ RAW DSP {q.uid}"), fontsize=14)

        ax_psd_raw.set_title("Welch PSD (RAW Trace)")
        ax_psd_raw.set_xlabel(raw_frequency_axis_label)
        ax_psd_raw.set_ylabel("PSD (dB)")
        ax_psd_raw.grid(True, linestyle=":", alpha=0.3)

        ax_psd.set_title("Welch PSD (Demodulated Trace)")
        ax_psd.set_xlabel(analysis_frequency_axis_label)
        ax_psd.set_ylabel("PSD (dB)")
        ax_psd.grid(True, linestyle=":", alpha=0.3)

        ax_stft_raw.set_title("STFT Spectrogram (RAW Trace)")
        ax_stft_raw.set_ylabel(raw_frequency_axis_label)

        ax_stft.set_title("STFT Spectrogram (Demodulated Trace)")
        ax_stft.set_xlabel("Time (ns, readout-start frame)")
        ax_stft.set_ylabel(analysis_frequency_axis_label)

        stft_freq_mhz_ref = None
        stft_time_ns_ref = None
        stft_mag_db_accum = None
        stft_count = 0
        raw_stft_freq_mhz_ref = None
        raw_stft_time_ns_ref = None
        raw_stft_mag_db_accum = None
        raw_stft_count = 0
        raw_psd_has_data = False
        demod_pre_lpf_psd_has_data = False
        demod_peak_summary: list[str] = []
        raw_peak_summary: list[str] = []

        for i, state in enumerate(states):
            color = color_map.get(i, f"C{i}")
            trace = np.asarray(raw_data_dict[q.uid][state], dtype=complex)
            n_samples = int(trace.size)

            if n_samples > 1:
                trace_dc = trace - np.mean(trace)
                wn = _sanitize_nperseg(welch_nperseg, n_samples)
                wo = _sanitize_noverlap(welch_noverlap, wn)
                if wn > 1:
                    freq_hz, psd = signal.welch(
                        trace_dc,
                        fs=fs_hz,
                        nperseg=wn,
                        noverlap=wo,
                        return_onesided=False,
                        scaling="density",
                    )
                    freq_hz = np.fft.fftshift(freq_hz)
                    psd = np.fft.fftshift(np.real(psd))
                    psd_db = 10 * np.log10(np.maximum(psd, np.finfo(float).tiny))
                    ax_psd.plot(
                        (freq_hz + analysis_freq_offset_hz) / 1e6,
                        psd_db,
                        color=color,
                        linewidth=1.3,
                        label=(
                            f"{state} (post-LPF)"
                            if compare_demod_lpf_enabled
                            else state
                        ),
                    )
                    if np.any(np.isfinite(psd)):
                        peak_idx = int(np.nanargmax(psd))
                        peak_freq_mhz = float(
                            (freq_hz[peak_idx] + analysis_freq_offset_hz) / 1e6
                        )
                        peak_psd_db = float(psd_db[peak_idx])
                        ax_psd.scatter(
                            peak_freq_mhz,
                            peak_psd_db,
                            color=color,
                            marker="x",
                            s=30,
                            linewidths=1.3,
                            zorder=5,
                        )
                        ax_psd.axvline(
                            peak_freq_mhz,
                            color=color,
                            linestyle=":",
                            linewidth=0.8,
                            alpha=0.35,
                        )
                        demod_peak_summary.append(f"{state} post: {peak_freq_mhz:.2f} MHz")

                if compare_demod_lpf_enabled:
                    demod_pre_lpf_trace = np.asarray(
                        demod_pre_lpf_data_dict[q.uid][state], dtype=complex  # type: ignore[index]
                    )
                    demod_pre_lpf_n_samples = int(demod_pre_lpf_trace.size)
                    if demod_pre_lpf_n_samples > 1:
                        demod_pre_lpf_dc = demod_pre_lpf_trace - np.mean(demod_pre_lpf_trace)
                        pre_wn = _sanitize_nperseg(welch_nperseg, demod_pre_lpf_n_samples)
                        pre_wo = _sanitize_noverlap(welch_noverlap, pre_wn)
                        if pre_wn > 1:
                            pre_freq_hz, pre_psd = signal.welch(
                                demod_pre_lpf_dc,
                                fs=fs_hz,
                                nperseg=pre_wn,
                                noverlap=pre_wo,
                                return_onesided=False,
                                scaling="density",
                            )
                            pre_freq_hz = np.fft.fftshift(pre_freq_hz)
                            pre_psd = np.fft.fftshift(np.real(pre_psd))
                            pre_psd_db = 10 * np.log10(
                                np.maximum(pre_psd, np.finfo(float).tiny)
                            )
                            ax_psd.plot(
                                (pre_freq_hz + analysis_freq_offset_hz) / 1e6,
                                pre_psd_db,
                                color=color,
                                linewidth=1.2,
                                linestyle="--",
                                alpha=0.95,
                                label=f"{state} (pre-LPF)",
                            )
                            if np.any(np.isfinite(pre_psd)):
                                pre_peak_idx = int(np.nanargmax(pre_psd))
                                pre_peak_freq_mhz = float(
                                    (pre_freq_hz[pre_peak_idx] + analysis_freq_offset_hz)
                                    / 1e6
                                )
                                pre_peak_psd_db = float(pre_psd_db[pre_peak_idx])
                                ax_psd.scatter(
                                    pre_peak_freq_mhz,
                                    pre_peak_psd_db,
                                    color=color,
                                    marker="+",
                                    s=44,
                                    linewidths=1.1,
                                    zorder=5,
                                )
                                ax_psd.axvline(
                                    pre_peak_freq_mhz,
                                    color=color,
                                    linestyle="--",
                                    linewidth=0.7,
                                    alpha=0.25,
                                )
                                demod_peak_summary.append(
                                    f"{state} pre: {pre_peak_freq_mhz:.2f} MHz"
                                )
                            demod_pre_lpf_psd_has_data = True

                sn = _sanitize_nperseg(stft_nperseg, n_samples)
                so = _sanitize_noverlap(stft_noverlap, sn)
                if sn > 1:
                    stft_freq_hz, stft_time_s, stft_vals = signal.stft(
                        trace_dc,
                        fs=fs_hz,
                        window=stft_window,
                        nperseg=sn,
                        noverlap=so,
                        return_onesided=False,
                        boundary=None,
                        padded=False,
                    )
                    stft_freq_hz = np.fft.fftshift(stft_freq_hz)
                    stft_mag = np.abs(np.fft.fftshift(stft_vals, axes=0))
                    stft_mag_db = 20 * np.log10(
                        np.maximum(stft_mag, np.finfo(float).tiny)
                    )

                    if stft_freq_mhz_ref is None:
                        stft_freq_mhz_ref = (stft_freq_hz + analysis_freq_offset_hz) / 1e6
                        stft_time_ns_ref = stft_time_s * 1e9 + delay_ns
                        stft_mag_db_accum = np.array(stft_mag_db, copy=True)
                        stft_count = 1
                    elif (
                        stft_mag_db_accum is not None
                        and stft_freq_mhz_ref.shape == (len(stft_freq_hz),)
                        and stft_time_ns_ref is not None
                        and stft_time_ns_ref.shape == (len(stft_time_s),)
                    ):
                        stft_mag_db_accum += stft_mag_db
                        stft_count += 1

            if compare_enabled:
                raw_trace = np.asarray(pre_demod_data_dict[q.uid][state], dtype=complex)  # type: ignore[index]
                raw_n_samples = int(raw_trace.size)
                if raw_n_samples > 1:
                    raw_trace_dc = raw_trace - np.mean(raw_trace)
                    raw_wn = _sanitize_nperseg(welch_nperseg, raw_n_samples)
                    raw_wo = _sanitize_noverlap(welch_noverlap, raw_wn)
                    if raw_wn > 1:
                        raw_freq_hz, raw_psd = signal.welch(
                            raw_trace_dc,
                            fs=fs_hz,
                            nperseg=raw_wn,
                            noverlap=raw_wo,
                            return_onesided=False,
                            scaling="density",
                        )
                        raw_freq_hz = np.fft.fftshift(raw_freq_hz)
                        raw_psd = np.fft.fftshift(np.real(raw_psd))
                        raw_psd_db = 10 * np.log10(
                            np.maximum(raw_psd, np.finfo(float).tiny)
                        )
                        ax_psd_raw.plot(
                            (raw_freq_hz + raw_freq_offset_hz) / 1e9,
                            raw_psd_db,
                            color=color,
                            linewidth=1.3,
                            linestyle="-",
                            label=state,
                        )
                        if np.any(np.isfinite(raw_psd)):
                            raw_peak_idx = int(np.nanargmax(raw_psd))
                            raw_peak_freq_ghz = float(
                                (raw_freq_hz[raw_peak_idx] + raw_freq_offset_hz) / 1e9
                            )
                            raw_peak_psd_db = float(raw_psd_db[raw_peak_idx])
                            ax_psd_raw.scatter(
                                raw_peak_freq_ghz,
                                raw_peak_psd_db,
                                color=color,
                                marker="x",
                                s=30,
                                linewidths=1.3,
                                zorder=5,
                            )
                            ax_psd_raw.axvline(
                                raw_peak_freq_ghz,
                                color=color,
                                linestyle=":",
                                linewidth=0.8,
                                alpha=0.35,
                            )
                            raw_peak_summary.append(f"{state}: {raw_peak_freq_ghz:.4f} GHz")
                        raw_psd_has_data = True

                    raw_sn = _sanitize_nperseg(stft_nperseg, raw_n_samples)
                    raw_so = _sanitize_noverlap(stft_noverlap, raw_sn)
                    if raw_sn > 1:
                        raw_stft_freq_hz, raw_stft_time_s, raw_stft_vals = signal.stft(
                            raw_trace_dc,
                            fs=fs_hz,
                            window=stft_window,
                            nperseg=raw_sn,
                            noverlap=raw_so,
                            return_onesided=False,
                            boundary=None,
                            padded=False,
                        )
                        raw_stft_freq_hz = np.fft.fftshift(raw_stft_freq_hz)
                        raw_stft_mag = np.abs(np.fft.fftshift(raw_stft_vals, axes=0))
                        raw_stft_mag_db = 20 * np.log10(
                            np.maximum(raw_stft_mag, np.finfo(float).tiny)
                        )

                        if raw_stft_freq_mhz_ref is None:
                            raw_stft_freq_mhz_ref = (
                                raw_stft_freq_hz + raw_freq_offset_hz
                            ) / 1e6
                            raw_stft_time_ns_ref = raw_stft_time_s * 1e9 + delay_ns
                            raw_stft_mag_db_accum = np.array(raw_stft_mag_db, copy=True)
                            raw_stft_count = 1
                        elif (
                            raw_stft_mag_db_accum is not None
                            and raw_stft_freq_mhz_ref.shape == (len(raw_stft_freq_hz),)
                            and raw_stft_time_ns_ref is not None
                            and raw_stft_time_ns_ref.shape == (len(raw_stft_time_s),)
                        ):
                            raw_stft_mag_db_accum += raw_stft_mag_db
                            raw_stft_count += 1

        if stft_mag_db_accum is not None and stft_count > 0 and stft_time_ns_ref is not None:
            stft_plot = stft_mag_db_accum / float(stft_count)
            mesh = ax_stft.pcolormesh(
                stft_time_ns_ref,
                stft_freq_mhz_ref,
                stft_plot,
                shading="auto",
                cmap="inferno",
            )
            fig.colorbar(mesh, ax=ax_stft, label="|STFT| (dB)")
        else:
            ax_stft.text(
                0.5,
                0.5,
                "Not enough samples for STFT",
                ha="center",
                va="center",
                transform=ax_stft.transAxes,
            )

        if compare_enabled:
            if (
                raw_stft_mag_db_accum is not None
                and raw_stft_count > 0
                and raw_stft_time_ns_ref is not None
            ):
                raw_stft_plot = raw_stft_mag_db_accum / float(raw_stft_count)
                mesh_raw = ax_stft_raw.pcolormesh(
                    raw_stft_time_ns_ref,
                    raw_stft_freq_mhz_ref,
                    raw_stft_plot,
                    shading="auto",
                    cmap="inferno",
                )
                fig.colorbar(mesh_raw, ax=ax_stft_raw, label="RAW |STFT| (dB)")
            else:
                ax_stft_raw.text(
                    0.5,
                    0.5,
                    "Not enough samples for RAW STFT",
                    ha="center",
                    va="center",
                    transform=ax_stft_raw.transAxes,
                )
            if not raw_psd_has_data:
                ax_psd_raw.text(
                    0.5,
                    0.5,
                    "Not enough samples for RAW Welch PSD",
                    ha="center",
                    va="center",
                    transform=ax_psd_raw.transAxes,
                )
        else:
            ax_psd_raw.text(
                0.5,
                0.5,
                "RAW compare disabled\n(apply_software_demodulation=False)",
                ha="center",
                va="center",
                transform=ax_psd_raw.transAxes,
            )
            ax_psd_raw.set_xticks([])
            ax_psd_raw.set_yticks([])
            ax_stft_raw.text(
                0.5,
                0.5,
                "RAW compare disabled\n(apply_software_demodulation=False)",
                ha="center",
                va="center",
                transform=ax_stft_raw.transAxes,
            )
            ax_stft_raw.set_xticks([])
            ax_stft_raw.set_yticks([])

        if compare_demod_lpf_enabled and not demod_pre_lpf_psd_has_data:
            ax_psd.text(
                0.5,
                0.08,
                "Pre-LPF demod PSD unavailable (insufficient samples)",
                ha="center",
                va="bottom",
                transform=ax_psd.transAxes,
                fontsize=9,
                alpha=0.8,
            )

        if demod_peak_summary:
            ax_psd.text(
                0.01,
                0.98,
                "Welch peak\n" + "\n".join(demod_peak_summary),
                transform=ax_psd.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "none", "alpha": 0.65},
            )
        if raw_peak_summary:
            ax_psd_raw.text(
                0.01,
                0.98,
                "Welch peak\n" + "\n".join(raw_peak_summary),
                transform=ax_psd_raw.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "none", "alpha": 0.65},
            )

        if freq_limit_mhz is not None:
            analysis_center_mhz = analysis_freq_offset_hz / 1e6
            raw_center_mhz = raw_freq_offset_hz / 1e6
            ax_psd.set_xlim(
                analysis_center_mhz - freq_limit_mhz,
                analysis_center_mhz + freq_limit_mhz,
            )
            ax_stft.set_ylim(
                analysis_center_mhz - freq_limit_mhz,
                analysis_center_mhz + freq_limit_mhz,
            )
            if compare_enabled:
                ax_stft_raw.set_ylim(
                    raw_center_mhz - freq_limit_mhz,
                    raw_center_mhz + freq_limit_mhz,
                )
                ax_psd_raw.set_xlim(
                    (raw_center_mhz - freq_limit_mhz) / 1e3,
                    (raw_center_mhz + freq_limit_mhz) / 1e3,
                )

        # Keep demodulated PSD y-range identical to RAW PSD y-range for direct visual comparison.
        if compare_enabled and raw_psd_has_data:
            raw_psd_ylim = ax_psd_raw.get_ylim()
        else:
            raw_psd_ylim = ax_psd.get_ylim()
            ax_psd_raw.set_ylim(raw_psd_ylim)
        ax_psd.set_ylim(raw_psd_ylim)

        handles, labels = ax_psd.get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper right", frameon=False, title="State")

        if opts.save_figures:
            workflow.save_artifact(f"IQ_dsp_trace_{q.uid}", fig)

        if opts.close_figures:
            plt.close(fig)

        figures[q.uid] = fig

    return figures

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
        0.08,
        description=(
            "Hide phase samples where |IQ| is below this fraction of the "
            "per-trace peak amplitude."
        ),
    )
    plot_phase_boundaries: bool = workflow.option_field(
        True,
        description=(
            "Plot readout/integration timing boundaries on time-domain and phase panels."
        ),
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
) -> None:
    """Analyze IQ time traces using calibration trace handles."""
    opts = IQTimeTraceAnalysisWorkflowOptions() if options is None else options

    pre_demod_traces = collect_time_traces(
        qubits=qubits,
        result=result,
        states=states,
        sample_dt_ns=opts.sample_dt_ns,
        apply_software_demodulation=False,
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
    with workflow.if_(opts.truncate_to_granularity):
        pre_demod_traces = truncate_time_traces_to_granularity(
            qubits=qubits,
            processed_data_dict=pre_demod_traces,
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

    figures = {}
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
        with workflow.if_(opts.do_plotting_iq_time_trace):
            figures = plot_iq_time_traces(
                qubits=qubits,
                states=states,
                processed_data_dict=analysis_traces,
                sample_dt_ns=opts.sample_dt_ns,
                phase_mask_relative_threshold=opts.phase_mask_relative_threshold,
                plot_phase_boundaries=opts.plot_phase_boundaries,
            )

    workflow.return_(
        {
            "diagnostics": diagnostics,
            "figures": figures,
            "dsp_diagnostics": dsp_diagnostics,
            "dsp_figures": dsp_figures,
            "metadata": {
                "states": states,
                "input_contract": "calibration_trace_handle",
                "dsp_input_stage": "analysis_trace",
                "dsp_backend": "scipy",
                "dsp_compare_pre_demod_enabled": compare_pre_demod,
                "dsp_reference_pre_demod_stage": "raw_before_demod",
            },
        }
    )


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


def _integration_aligned_boundaries_ns(qubit) -> dict[str, float | None]:
    readout_length_s = getattr(qubit.parameters, "readout_length", None)
    readout_integration_delay_s = getattr(
        qubit.parameters, "readout_integration_delay", None
    )
    readout_integration_length_s = getattr(
        qubit.parameters, "readout_integration_length", None
    )

    int_delay_ns = (
        float(readout_integration_delay_s) * 1e9
        if readout_integration_delay_s is not None
        else 0.0
    )
    int_end_ns = (
        float(readout_integration_length_s) * 1e9
        if readout_integration_length_s is not None
        else None
    )
    ro_end_ns = (
        float(readout_length_s) * 1e9 - int_delay_ns
        if readout_length_s is not None
        else None
    )
    return {
        "int_start": 0.0,
        "int_end": int_end_ns,
        "ro_end": ro_end_ns,
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


@workflow.task
def compute_diagnostics(
    qubits: QuantumElements,
    states: Sequence[str],
    processed_data_dict: dict[str, dict[str, ArrayLike]],
    sample_dt_ns: float = 0.5,
    phase_mask_relative_threshold: float = 0.08,
) -> dict[str, dict]:
    """Compute compact diagnostics for IQ time traces."""
    qubits = validate_and_convert_qubits_sweeps(qubits)
    phase_mask_relative_threshold = max(float(phase_mask_relative_threshold), 0.0)

    diagnostics: dict[str, dict] = {}
    for q in qubits:
        boundaries_ns = _integration_aligned_boundaries_ns(q)
        q_diag: dict[str, object] = {
            "axis_reference": "integration_start",
            "sample_dt_ns": float(sample_dt_ns),
            "states": {},
        }

        for state in states:
            trace = np.asarray(processed_data_dict[q.uid][state], dtype=complex)
            n_samples = int(trace.size)
            axis = np.arange(n_samples, dtype=float) * sample_dt_ns
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
    phase_mask_relative_threshold: float = 0.08,
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
        boundaries_ns = _integration_aligned_boundaries_ns(q)
        q_diag: dict[str, object] = {
            "axis_reference": "integration_start",
            "sample_dt_ns": float(sample_dt_ns),
            "fs_hz": float(fs_hz),
            "states": {},
        }

        for state in states:
            trace = np.asarray(raw_data_dict[q.uid][state], dtype=complex)
            n_samples = int(trace.size)
            axis_ns = np.arange(n_samples, dtype=float) * float(sample_dt_ns)
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
                        stft_peak_time_ns = float(stft_time_s[peak_ti] * 1e9)
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
                            raw_stft_peak_time_ns = float(raw_stft_time_s[raw_peak_ti] * 1e9)
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
                "distance_to_ro_end_ns": _distance_to_boundary(
                    max_abs_inst_freq_time_ns, boundaries_ns, "ro_end"
                ),
                "boundaries_ns": dict(boundaries_ns),
            }

        dsp_diag[q.uid] = q_diag

    return dsp_diag


@workflow.task
def plot_iq_time_traces(
    qubits: QuantumElements,
    states: Sequence[str],
    processed_data_dict: dict[str, dict[str, ArrayLike]],
    options: BasePlottingOptions | None = None,
    sample_dt_ns: float = 0.5,
    phase_mask_relative_threshold: float = 0.08,
    plot_phase_boundaries: bool = True,
) -> dict[str, mpl.figure.Figure]:
    """Create 5-panel IQ time-trace plots."""
    opts = BasePlottingOptions() if options is None else options
    qubits = validate_and_convert_qubits_sweeps(qubits)
    phase_mask_relative_threshold = max(float(phase_mask_relative_threshold), 0.0)

    figures = {}
    color_map = {0: "b", 1: "r", 2: "g"}
    for q in qubits:
        fig = plt.figure(figsize=(24, 12), constrained_layout=True)
        grid = fig.add_gridspec(2, 3, hspace=0.22, wspace=0.22)
        ax_i = fig.add_subplot(grid[0, 0])
        ax_q = fig.add_subplot(grid[0, 1], sharex=ax_i)
        ax_phase = fig.add_subplot(grid[0, 2], sharex=ax_i)
        ax_iq = fig.add_subplot(grid[1, 0:2])
        ax_amp = fig.add_subplot(grid[1, 2], sharex=ax_i)

        fig.suptitle(timestamped_title(f"IQ Time Trace {q.uid}"), fontsize=14)

        ax_i.set_title("I(t)")
        ax_i.set_xlabel("Time (ns)")
        ax_i.set_ylabel("Real Signal Component, $V_{\\mathrm{I}}$ (a.u.)")

        ax_q.set_title("Q(t)")
        ax_q.set_xlabel("Time (ns)")
        ax_q.set_ylabel("Imaginary Signal Component, $V_{\\mathrm{Q}}$ (a.u.)")

        ax_phase.set_title("Phase(t)")
        ax_phase.set_xlabel("Time (ns)")
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

        ax_amp.set_title("|IQ|(t)")
        ax_amp.set_xlabel("Time (ns)")
        ax_amp.set_ylabel("Magnitude, $|IQ|$ (a.u.)")

        if plot_phase_boundaries:
            boundaries = _integration_aligned_boundaries_ns(q)
            for key, style in (
                ("int_start", ":"),
                ("int_end", "-."),
                ("ro_end", "--"),
            ):
                boundary_ns = boundaries.get(key)
                if boundary_ns is None:
                    continue
                ax_i.axvline(
                    boundary_ns, color="k", linestyle=style, linewidth=1.0, alpha=0.3
                )
                ax_q.axvline(
                    boundary_ns, color="k", linestyle=style, linewidth=1.0, alpha=0.3
                )
                ax_phase.axvline(
                    boundary_ns, color="k", linestyle=style, linewidth=1.0, alpha=0.3
                )
                ax_amp.axvline(
                    boundary_ns, color="k", linestyle=style, linewidth=1.0, alpha=0.3
                )

            ax_phase.text(
                0.01,
                0.90,
                "t=0: int_start  |  --: ro_end  |  -.: int_end",
                transform=ax_phase.transAxes,
                va="top",
                fontsize=8,
                alpha=0.8,
            )

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

            axis = np.arange(len(i_trace), dtype=float) * sample_dt_ns
            marker_every = max(len(axis) // 120, 1)
            ax_i.plot(axis, i_trace, "-", color=color, linewidth=1.3, label=state)
            ax_i.plot(
                axis,
                i_trace,
                "o",
                color=color,
                markersize=2.0,
                alpha=0.55,
                markevery=marker_every,
            )

            ax_q.plot(axis, q_trace, "-", color=color, linewidth=1.3, label=state)
            ax_q.plot(
                axis,
                q_trace,
                "o",
                color=color,
                markersize=2.0,
                alpha=0.55,
                markevery=marker_every,
            )

            ax_iq.plot(i_trace, q_trace, "-", color=color, linewidth=1.1, label=state)
            ax_iq.scatter(
                i_trace[::marker_every],
                q_trace[::marker_every],
                c=color,
                s=10,
                alpha=0.6,
            )

            valid_phase = np.isfinite(phase_masked)
            ax_phase.scatter(
                axis[valid_phase],
                phase_masked[valid_phase],
                color=color,
                s=10,
                alpha=0.7,
            )
            ax_phase.plot(axis, phase_masked, "-", color=color, linewidth=1.3, label=state)

            ax_amp.plot(axis, amplitude, "-", color=color, linewidth=1.3, label=state)

            if len(i_trace) >= 2:
                d_i = np.gradient(i_trace)
                d_q = np.gradient(q_trace)
                step = max(len(i_trace) // 90, 2)
                ax_iq.quiver(
                    i_trace[::step],
                    q_trace[::step],
                    d_i[::step],
                    d_q[::step],
                    angles="xy",
                    scale_units="xy",
                    scale=1,
                    color=color,
                    alpha=0.6,
                    width=0.002,
                )

        ax_iq.set_aspect("equal", adjustable="box")
        for axis_obj in (ax_i, ax_q, ax_phase, ax_amp):
            axis_obj.grid(True, linestyle=":", alpha=0.35)
        ax_iq.grid(True, linestyle=":", alpha=0.25)

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
    options: BasePlottingOptions | None = None,
    sample_dt_ns: float = 0.5,
    phase_mask_relative_threshold: float = 0.08,
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
    phase_mask_relative_threshold = max(float(phase_mask_relative_threshold), 0.0)
    dt_s = float(sample_dt_ns) * 1e-9
    fs_hz = 1.0 / dt_s
    dphi_smooth_window = _sanitize_smooth_window(dphi_smooth_window)
    color_map = {0: "b", 1: "r", 2: "g"}
    freq_limit_mhz = (
        None
        if dsp_freq_limit_mhz is None or dsp_freq_limit_mhz <= 0
        else float(dsp_freq_limit_mhz)
    )
    compare_enabled = bool(compare_pre_demod and pre_demod_data_dict is not None)

    figures: dict[str, mpl.figure.Figure] = {}
    for q in qubits:
        lo_frequency_hz = getattr(q.parameters, "readout_lo_frequency", None)
        raw_freq_offset_hz = float(lo_frequency_hz) if lo_frequency_hz is not None else 0.0
        analysis_freq_offset_hz = 0.0 if compare_enabled else raw_freq_offset_hz
        raw_frequency_axis_label = (
            "RF Frequency (MHz)" if lo_frequency_hz is not None else "Frequency (MHz)"
        )
        analysis_frequency_axis_label = (
            "IF-Frame Frequency (MHz)" if compare_enabled else raw_frequency_axis_label
        )

        fig = plt.figure(figsize=(22, 12), constrained_layout=True)
        grid = fig.add_gridspec(2, 2, hspace=0.24, wspace=0.24)
        psd_grid = grid[0, 0].subgridspec(2, 1, hspace=0.12)
        ax_psd_raw = fig.add_subplot(psd_grid[0, 0])
        ax_psd = fig.add_subplot(psd_grid[1, 0])
        stft_grid = grid[0, 1].subgridspec(2, 1, hspace=0.12)
        ax_stft_raw = fig.add_subplot(stft_grid[0, 0])
        ax_stft = fig.add_subplot(stft_grid[1, 0], sharex=ax_stft_raw)
        ax_if = fig.add_subplot(grid[1, 0])
        ax_amp = fig.add_subplot(grid[1, 1], sharex=ax_if)
        fig.suptitle(timestamped_title(f"IQ RAW DSP {q.uid}"), fontsize=14)

        ax_psd_raw.set_title("Welch PSD (RAW Trace)")
        ax_psd_raw.set_ylabel("PSD (dB)")
        ax_psd_raw.grid(True, linestyle=":", alpha=0.3)
        ax_psd_raw.tick_params(axis="x", labelbottom=False)

        ax_psd.set_title("Welch PSD (Demodulated Trace)")
        ax_psd.set_xlabel(analysis_frequency_axis_label)
        ax_psd.set_ylabel("PSD (dB)")
        ax_psd.grid(True, linestyle=":", alpha=0.3)

        ax_stft_raw.set_title("STFT Spectrogram (RAW Trace)")
        ax_stft_raw.set_ylabel(raw_frequency_axis_label)

        ax_stft.set_title("STFT Spectrogram (Demodulated Trace)")
        ax_stft.set_xlabel("Time (ns)")
        ax_stft.set_ylabel(analysis_frequency_axis_label)

        ax_if.set_title("Instantaneous Frequency")
        ax_if.set_xlabel("Time (ns)")
        ax_if.set_ylabel("Frequency (Hz)")
        ax_if.grid(True, linestyle=":", alpha=0.3)

        ax_amp.set_title("|IQ|(t) + Max-Jump Marker")
        ax_amp.set_xlabel("Time (ns)")
        ax_amp.set_ylabel("Magnitude, |IQ| (a.u.)")
        ax_amp.grid(True, linestyle=":", alpha=0.3)

        boundaries = _integration_aligned_boundaries_ns(q)
        if plot_phase_boundaries:
            for _, style, key in (
                ("int_start", ":", "int_start"),
                ("int_end", "-.", "int_end"),
                ("ro_end", "--", "ro_end"),
            ):
                boundary_ns = boundaries.get(key)
                if boundary_ns is None:
                    continue
                ax_if.axvline(
                    boundary_ns, color="k", linestyle=style, linewidth=1.0, alpha=0.3
                )
                ax_amp.axvline(
                    boundary_ns, color="k", linestyle=style, linewidth=1.0, alpha=0.3
                )

        stft_freq_mhz_ref = None
        stft_time_ns_ref = None
        stft_mag_db_accum = None
        stft_count = 0
        raw_stft_freq_mhz_ref = None
        raw_stft_time_ns_ref = None
        raw_stft_mag_db_accum = None
        raw_stft_count = 0
        raw_psd_has_data = False

        for i, state in enumerate(states):
            color = color_map.get(i, f"C{i}")
            trace = np.asarray(raw_data_dict[q.uid][state], dtype=complex)
            n_samples = int(trace.size)
            axis_ns = np.arange(n_samples, dtype=float) * float(sample_dt_ns)
            amp = np.abs(trace)
            amp_peak = float(np.max(amp)) if n_samples > 0 else 0.0
            mask_abs = phase_mask_relative_threshold * amp_peak

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
                        label=state,
                    )

                phase = np.unwrap(np.angle(trace))
                phase_masked = np.array(phase, copy=True)
                phase_masked[amp < mask_abs] = np.nan
                dphi_dt = np.gradient(phase_masked, dt_s)
                dphi_dt = _nan_moving_average(dphi_dt, dphi_smooth_window)
                inst_freq_hz = dphi_dt / (2 * np.pi)
                ax_if.plot(axis_ns, inst_freq_hz, color=color, linewidth=1.2, label=state)
                ax_amp.plot(axis_ns, amp, color=color, linewidth=1.2, label=state)

                if np.any(np.isfinite(inst_freq_hz)):
                    jump_idx = int(np.nanargmax(np.abs(inst_freq_hz)))
                    ax_amp.scatter(
                        axis_ns[jump_idx],
                        amp[jump_idx],
                        color=color,
                        marker="x",
                        s=50,
                        linewidths=1.5,
                    )

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
                        stft_time_ns_ref = stft_time_s * 1e9
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

            elif n_samples == 1:
                ax_amp.plot(axis_ns, amp, color=color, linewidth=1.2, label=state)

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
                            (raw_freq_hz + raw_freq_offset_hz) / 1e6,
                            raw_psd_db,
                            color=color,
                            linewidth=1.3,
                            linestyle="-",
                            label=state,
                        )
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
                            raw_stft_time_ns_ref = raw_stft_time_s * 1e9
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
                ax_psd_raw.set_xlim(
                    raw_center_mhz - freq_limit_mhz,
                    raw_center_mhz + freq_limit_mhz,
                )
                ax_stft_raw.set_ylim(
                    raw_center_mhz - freq_limit_mhz,
                    raw_center_mhz + freq_limit_mhz,
                )

        handles, labels = ax_psd.get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper right", frameon=False, title="State")

        if opts.save_figures:
            workflow.save_artifact(f"IQ_dsp_trace_{q.uid}", fig)

        if opts.close_figures:
            plt.close(fig)

        figures[q.uid] = fig

    return figures

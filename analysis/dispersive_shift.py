# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Analysis workflow for dispersive-shift calibration.

The primary target is extracting the dispersive shift (2chi) between prepared
|g> and |e> resonator responses. A secondary output is a recommended frequency
window for a subsequent fine readout optimization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from laboneq import workflow
from laboneq.simple import dsl

from laboneq_applications.analysis.options import BasePlottingOptions
from laboneq_applications.analysis.plotting_helpers import timestamped_title
from laboneq_applications.core import validation

if TYPE_CHECKING:
    from collections.abc import Sequence

    import matplotlib as mpl
    from laboneq.dsl.quantum.quantum_element import QuantumElement
    from laboneq.workflow.tasks.run_experiment import RunExperimentResults
    from numpy.typing import ArrayLike


def _validate_ge_states(states: Sequence[str]) -> tuple[str, str]:
    state_list = list(states) if not isinstance(states, str) else list(states)
    if len(state_list) != 2 or set(state_list) != {"g", "e"}:
        raise ValueError(
            "dispersive_shift analysis currently supports only two states: ['g', 'e']."
        )
    return "g", "e"


def _moving_average(x: np.ndarray, window: int = 5) -> np.ndarray:
    if window <= 1 or x.size < window:
        return x.copy()
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(x, kernel, mode="same")


def _phase_delay_params(
    frequencies: np.ndarray,
    phase_delay: dict[str, float] | None,
) -> tuple[float, float, float, bool]:
    tau_s = 0.0
    phi0_rad = 0.0
    f_ref_hz = float(np.mean(frequencies))
    if phase_delay is None:
        return tau_s, phi0_rad, f_ref_hz, False
    tau_s = float(phase_delay.get("tau_s", 0.0))
    phi0_rad = float(phase_delay.get("phi0_rad", 0.0))
    f_ref_hz = float(phase_delay.get("f_ref_hz", f_ref_hz))
    return tau_s, phi0_rad, f_ref_hz, True


def _estimate_resonance_frequency(
    frequencies: np.ndarray,
    phase_corrected: np.ndarray,
    magnitude: np.ndarray,
    *,
    center_idx: int | None = None,
    search_half_width: int | None = None,
) -> tuple[float, int, str]:
    if frequencies.size < 3:
        idx = int(np.argmin(magnitude))
        return float(frequencies[idx]), idx, "magnitude_min"

    phase_smooth = _moving_average(phase_corrected, window=7)
    phase_slope = np.abs(np.gradient(phase_smooth, frequencies))

    if center_idx is not None and search_half_width is not None:
        lo = max(0, int(center_idx) - int(search_half_width))
        hi = min(frequencies.size, int(center_idx) + int(search_half_width) + 1)
        if hi - lo >= 2:
            idx_phase = lo + int(np.argmax(phase_slope[lo:hi]))
        else:
            idx_phase = int(np.argmax(phase_slope))
    else:
        idx_phase = int(np.argmax(phase_slope))
    slope_peak = float(phase_slope[idx_phase])
    slope_med = float(np.median(phase_slope))
    phase_reliable = (
        idx_phase not in (0, frequencies.size - 1)
        and np.isfinite(slope_peak)
        and slope_peak > 1.5 * max(slope_med, 1e-15)
    )
    if phase_reliable:
        return float(frequencies[idx_phase]), idx_phase, "phase_slope"

    if center_idx is not None and search_half_width is not None:
        lo = max(0, int(center_idx) - int(search_half_width))
        hi = min(frequencies.size, int(center_idx) + int(search_half_width) + 1)
        if hi - lo >= 2:
            idx_mag = lo + int(np.argmin(magnitude[lo:hi]))
        else:
            idx_mag = int(np.argmin(magnitude))
    else:
        idx_mag = int(np.argmin(magnitude))
    return float(frequencies[idx_mag]), idx_mag, "magnitude_min"


def _block_bootstrap_indices(
    n: int,
    *,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=int)
    b = max(2, min(int(block_size), n))
    pieces: list[np.ndarray] = []
    while sum(len(p) for p in pieces) < n:
        start = int(rng.integers(0, n))
        idx = (start + np.arange(b)) % n
        pieces.append(idx)
    out = np.concatenate(pieces)[:n]
    return out


def _bootstrap_two_chi_ci(
    frequencies: np.ndarray,
    phase_g: np.ndarray,
    phase_e: np.ndarray,
    mag_g: np.ndarray,
    mag_e: np.ndarray,
    *,
    point_estimate: float,
    samples: int,
    confidence_level: float,
    seed: int,
    ref_idx_g: int,
    ref_idx_e: int,
) -> tuple[float, float]:
    if samples < 2:
        return point_estimate, point_estimate

    rng = np.random.default_rng(seed)
    n = frequencies.size
    if n < 3:
        return point_estimate, point_estimate

    smooth_phase_g = _moving_average(phase_g, window=5)
    smooth_phase_e = _moving_average(phase_e, window=5)
    smooth_mag_g = _moving_average(mag_g, window=5)
    smooth_mag_e = _moving_average(mag_e, window=5)

    resid_phase_g = phase_g - smooth_phase_g
    resid_phase_e = phase_e - smooth_phase_e
    resid_mag_g = mag_g - smooth_mag_g
    resid_mag_e = mag_e - smooth_mag_e

    block_size = max(4, n // 12)
    search_half_width = max(5, n // 10)

    two_chi_samples = np.empty(samples, dtype=float)
    for i in range(samples):
        idx = _block_bootstrap_indices(n, block_size=block_size, rng=rng)
        phase_g_bs = smooth_phase_g + resid_phase_g[idx]
        phase_e_bs = smooth_phase_e + resid_phase_e[idx]
        mag_g_bs = smooth_mag_g + resid_mag_g[idx]
        mag_e_bs = smooth_mag_e + resid_mag_e[idx]
        f_g, _, _ = _estimate_resonance_frequency(
            frequencies,
            phase_g_bs,
            mag_g_bs,
            center_idx=ref_idx_g,
            search_half_width=search_half_width,
        )
        f_e, _, _ = _estimate_resonance_frequency(
            frequencies,
            phase_e_bs,
            mag_e_bs,
            center_idx=ref_idx_e,
            search_half_width=search_half_width,
        )
        two_chi_samples[i] = f_e - f_g

    finite = np.isfinite(two_chi_samples)
    if not np.any(finite):
        return point_estimate, point_estimate
    two_chi_samples = two_chi_samples[finite]
    alpha = max(0.0, min(1.0, 1.0 - confidence_level))
    ci_low = float(np.quantile(two_chi_samples, alpha / 2.0))
    ci_high = float(np.quantile(two_chi_samples, 1.0 - alpha / 2.0))
    return ci_low, ci_high


@workflow.workflow_options
class DispersiveShiftAnalysisWorkflowOptions:
    """Option class for dispersive-shift analysis workflow."""

    do_plotting: bool = workflow.option_field(
        True, description="Whether to create the plots."
    )
    do_plotting_dispersive_shift: bool = workflow.option_field(
        True, description="Whether to create magnitude/phase plot."
    )
    do_plotting_signal_distances: bool = workflow.option_field(
        True, description="Whether to create |delta S21| plot."
    )
    bootstrap_samples: int = workflow.option_field(
        200, description="Bootstrap sample count for two_chi uncertainty."
    )
    bootstrap_seed: int = workflow.option_field(
        9213, description="Bootstrap RNG seed."
    )
    confidence_level: float = workflow.option_field(
        0.95, description="Confidence level for two_chi interval."
    )
    window_k_chi: float = workflow.option_field(
        4.0, description="Window span factor on |two_chi|."
    )
    window_min_span_hz: float = workflow.option_field(
        5e6, description="Minimum recommended sweep window span."
    )


@workflow.workflow
def analysis_workflow(
    result: RunExperimentResults,
    qubit: QuantumElement,
    frequencies: ArrayLike,
    states: Sequence[str],
    phase_delay: dict[str, float] | None = None,
    options: DispersiveShiftAnalysisWorkflowOptions | None = None,
) -> None:
    """Run dispersive-shift analysis for |g>,|e> and return two_chi + sweep window."""
    opts = (
        DispersiveShiftAnalysisWorkflowOptions() if options is None else options
    )
    metrics = calculate_dispersive_shift_metrics(
        qubit=qubit,
        result=result,
        frequencies=frequencies,
        states=states,
        phase_delay=phase_delay,
        bootstrap_samples=opts.bootstrap_samples,
        bootstrap_seed=opts.bootstrap_seed,
        confidence_level=opts.confidence_level,
        window_k_chi=opts.window_k_chi,
        window_min_span_hz=opts.window_min_span_hz,
    )
    with workflow.if_(opts.do_plotting):
        with workflow.if_(opts.do_plotting_dispersive_shift):
            plot_dispersive_shift(qubit=qubit, metrics=metrics)
        with workflow.if_(opts.do_plotting_signal_distances):
            plot_signal_distances(qubit=qubit, metrics=metrics)
    workflow.return_(metrics)


@workflow.task
def calculate_dispersive_shift_metrics(
    qubit: QuantumElement,
    result: RunExperimentResults,
    frequencies: ArrayLike,
    states: Sequence[str],
    phase_delay: dict[str, float] | None,
    bootstrap_samples: int,
    bootstrap_seed: int,
    confidence_level: float,
    window_k_chi: float,
    window_min_span_hz: float,
) -> dict:
    """Extract two_chi and a recommended frequency window."""
    qubit, frequencies = validation.validate_and_convert_single_qubit_sweeps(
        qubit, frequencies
    )
    validation.validate_result(result)
    _validate_ge_states(states)

    freqs = np.asarray(frequencies, dtype=float)
    s21_g = np.asarray(result[dsl.handles.result_handle(qubit.uid, suffix="g")].data)
    s21_e = np.asarray(result[dsl.handles.result_handle(qubit.uid, suffix="e")].data)

    mag_g = np.abs(s21_g)
    mag_e = np.abs(s21_e)
    phase_g = np.unwrap(np.angle(s21_g))
    phase_e = np.unwrap(np.angle(s21_e))

    tau_s, phi0_rad, f_ref_hz, delay_applied = _phase_delay_params(freqs, phase_delay)
    phase_correction = 2.0 * np.pi * (freqs - f_ref_hz) * tau_s + phi0_rad
    phase_g_corr = phase_g - phase_correction
    phase_e_corr = phase_e - phase_correction

    f_r_g_hz, idx_g, method_g = _estimate_resonance_frequency(freqs, phase_g_corr, mag_g)
    f_r_e_hz, idx_e, method_e = _estimate_resonance_frequency(freqs, phase_e_corr, mag_e)

    two_chi_hz = float(f_r_e_hz - f_r_g_hz)
    ci_low, ci_high = _bootstrap_two_chi_ci(
        frequencies=freqs,
        phase_g=phase_g_corr,
        phase_e=phase_e_corr,
        mag_g=mag_g,
        mag_e=mag_e,
        point_estimate=two_chi_hz,
        samples=int(bootstrap_samples),
        confidence_level=float(confidence_level),
        seed=int(bootstrap_seed),
        ref_idx_g=idx_g,
        ref_idx_e=idx_e,
    )

    span_hz = max(float(window_k_chi) * abs(two_chi_hz), float(window_min_span_hz))
    f_center_hz = 0.5 * (f_r_g_hz + f_r_e_hz)
    f_low_hz = max(float(freqs[0]), float(f_center_hz - 0.5 * span_hz))
    f_high_hz = min(float(freqs[-1]), float(f_center_hz + 0.5 * span_hz))

    freq_step_hz = float(np.median(np.diff(freqs))) if freqs.size > 1 else 0.0
    ci_zero_tolerance_hz = max(freq_step_hz, 0.1 * abs(two_chi_hz))
    ci_contains_zero_strong = (
        ci_low < -ci_zero_tolerance_hz and ci_high > ci_zero_tolerance_hz
    )
    ci_width_hz = ci_high - ci_low
    very_wide_ci = ci_width_hz > max(3.0 * abs(two_chi_hz), 4.0 * freq_step_hz)
    edge_hit = idx_g in (0, freqs.size - 1) or idx_e in (0, freqs.size - 1)
    quality_flag = "low_confidence" if (ci_contains_zero_strong or very_wide_ci or edge_hit) else "ok"

    distance_curve = np.abs(s21_e - s21_g)
    distance_max_idx = int(np.argmax(distance_curve))

    return {
        "two_chi_hz": two_chi_hz,
        "two_chi_uncertainty": {
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
        },
        "frequency_window": {
            "f_low_hz": float(f_low_hz),
            "f_center_hz": float(f_center_hz),
            "f_high_hz": float(f_high_hz),
        },
        "quality_flag": quality_flag,
        "resonance_hz": {"g": float(f_r_g_hz), "e": float(f_r_e_hz)},
        "resonance_method": {"g": method_g, "e": method_e},
        "phase_delay": {
            "applied": delay_applied,
            "tau_s": float(tau_s),
            "phi0_rad": float(phi0_rad),
            "f_ref_hz": float(f_ref_hz),
        },
        "quality_diagnostics": {
            "freq_step_hz": float(freq_step_hz),
            "ci_zero_tolerance_hz": float(ci_zero_tolerance_hz),
            "ci_width_hz": float(ci_width_hz),
            "ci_contains_zero_strong": bool(ci_contains_zero_strong),
            "very_wide_ci": bool(very_wide_ci),
            "edge_hit": bool(edge_hit),
        },
        "frequencies_hz": freqs,
        "magnitude": {"g": mag_g, "e": mag_e},
        "phase_unwrapped": {"g": phase_g, "e": phase_e},
        "phase_corrected": {"g": phase_g_corr, "e": phase_e_corr},
        "signal_distance": {
            "curve": distance_curve,
            "max_freq_hz": float(freqs[distance_max_idx]),
            "max_value": float(distance_curve[distance_max_idx]),
        },
        "old_parameter_values": {
            qubit.uid: {
                "readout_resonator_frequency": qubit.parameters.readout_resonator_frequency,
            }
        },
        "new_parameter_values": {
            qubit.uid: {
                "readout_resonator_frequency": float(f_center_hz),
            }
        },
    }


@workflow.task
def plot_dispersive_shift(
    qubit: QuantumElement,
    metrics: dict,
    options: BasePlottingOptions | None = None,
) -> mpl.figure.Figure | None:
    """Plot magnitude + phase with extracted resonances and recommended window."""
    opts = BasePlottingOptions() if options is None else options
    qubit = validation.validate_and_convert_single_qubit_sweeps(qubit)

    f_ghz = np.asarray(metrics["frequencies_hz"], dtype=float) / 1e9
    mag = metrics["magnitude"]
    phase = metrics["phase_corrected"]
    f_res = metrics["resonance_hz"]
    f_win = metrics["frequency_window"]
    two_chi = float(metrics["two_chi_hz"])
    ci = metrics["two_chi_uncertainty"]
    quality = metrics["quality_flag"]

    fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax_mag.plot(f_ghz, mag["g"], label="g")
    ax_mag.plot(f_ghz, mag["e"], label="e")
    ax_mag.axvline(f_res["g"] / 1e9, linestyle="--", linewidth=1.0)
    ax_mag.axvline(f_res["e"] / 1e9, linestyle="--", linewidth=1.0)
    ax_mag.axvspan(f_win["f_low_hz"] / 1e9, f_win["f_high_hz"] / 1e9, alpha=0.15)
    ax_mag.set_ylabel(r"$|S_{21}|$ (a.u.)")
    ax_mag.legend(frameon=False)
    ax_mag.set_title(timestamped_title(f"Dispersive Shift {qubit.uid} | Magnitude"))
    ax_mag.text(
        0.01,
        0.02,
        (
            f"2chi={two_chi/1e6:.3f} MHz, "
            f"95% CI=[{ci['ci_low']/1e6:.3f}, {ci['ci_high']/1e6:.3f}] MHz, "
            f"quality={quality}"
        ),
        transform=ax_mag.transAxes,
        fontsize=9,
    )

    ax_phase.plot(f_ghz, phase["g"], label="g (corrected)")
    ax_phase.plot(f_ghz, phase["e"], label="e (corrected)")
    ax_phase.axvline(f_res["g"] / 1e9, linestyle="--", linewidth=1.0)
    ax_phase.axvline(f_res["e"] / 1e9, linestyle="--", linewidth=1.0)
    ax_phase.axvspan(f_win["f_low_hz"] / 1e9, f_win["f_high_hz"] / 1e9, alpha=0.15)
    ax_phase.set_xlabel(r"Readout Frequency, $f_{\mathrm{RO}}$ (GHz)")
    ax_phase.set_ylabel("Phase (rad)")
    ax_phase.set_title("Phase (corrected)")
    ax_phase.legend(frameon=False)

    fig.tight_layout()

    if opts.save_figures:
        workflow.save_artifact(f"Dispersive_shift_{qubit.uid}", fig)
    if opts.close_figures:
        plt.close(fig)
    return fig


@workflow.task
def plot_signal_distances(
    qubit: QuantumElement,
    metrics: dict,
    options: BasePlottingOptions | None = None,
) -> mpl.figure.Figure | None:
    """Plot |delta S21| with max point and recommended window."""
    opts = BasePlottingOptions() if options is None else options
    qubit = validation.validate_and_convert_single_qubit_sweeps(qubit)

    f_ghz = np.asarray(metrics["frequencies_hz"], dtype=float) / 1e9
    dist = np.asarray(metrics["signal_distance"]["curve"], dtype=float)
    max_freq_hz = float(metrics["signal_distance"]["max_freq_hz"])
    max_val = float(metrics["signal_distance"]["max_value"])
    f_win = metrics["frequency_window"]

    fig, ax = plt.subplots()
    ax.plot(f_ghz, dist, label=r"$|S_{21}^{e} - S_{21}^{g}|$")
    ax.plot(max_freq_hz / 1e9, max_val, "o", label="max distance")
    ax.axvspan(f_win["f_low_hz"] / 1e9, f_win["f_high_hz"] / 1e9, alpha=0.15)
    ax.set_xlabel(r"Readout Frequency, $f_{\mathrm{RO}}$ (GHz)")
    ax.set_ylabel(r"$|\Delta S_{21}|$ (a.u.)")
    ax.set_title(timestamped_title(f"Signal Difference {qubit.uid}"))
    ax.legend(frameon=False)

    if opts.save_figures:
        workflow.save_artifact(f"Signal_distances_{qubit.uid}", fig)
    if opts.close_figures:
        plt.close(fig)
    return fig

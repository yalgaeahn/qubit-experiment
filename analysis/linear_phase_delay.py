# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Analysis workflow for linear readout phase-delay calibration.

The goal is to estimate a single delay constant tau from complex S21 data:

    phase(f) ~= a * f + b,  tau = -a / (2*pi)

The fitted tau is intended to be stored as
`FixedTransmonQubitParameters.readout_phase_delay`.
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
    import matplotlib as mpl
    from laboneq.dsl.quantum.quantum_element import QuantumElement
    from laboneq.workflow.tasks.run_experiment import RunExperimentResults
    from numpy.typing import ArrayLike


def _edge_mask(n: int, edge_fraction: float) -> np.ndarray:
    if n <= 0:
        return np.zeros(0, dtype=bool)
    frac = float(np.clip(edge_fraction, 0.05, 0.49))
    m = max(2, int(round(frac * n)))
    m = min(m, n // 2) if n >= 4 else n
    mask = np.zeros(n, dtype=bool)
    if m > 0:
        mask[:m] = True
        mask[-m:] = True
    if not np.any(mask):
        mask[:] = True
    return mask


def _fit_tau_linear(
    frequencies: np.ndarray,
    phase_unwrapped: np.ndarray,
    *,
    edge_fraction: float,
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    mask = _edge_mask(frequencies.size, edge_fraction)
    x = frequencies[mask]
    y = phase_unwrapped[mask]
    if x.size < 2:
        x = frequencies
        y = phase_unwrapped
        mask = np.ones_like(frequencies, dtype=bool)
    slope, intercept = np.polyfit(x, y, deg=1)
    fit = slope * frequencies + intercept
    tau_s = float(-slope / (2.0 * np.pi))
    return float(tau_s), float(slope), float(intercept), fit, mask


def _bootstrap_tau_ci(
    frequencies: np.ndarray,
    phase_unwrapped: np.ndarray,
    fit: np.ndarray,
    mask: np.ndarray,
    *,
    edge_fraction: float,
    samples: int,
    seed: int,
    confidence_level: float,
) -> tuple[float, float]:
    if samples < 2:
        tau, _, _, _, _ = _fit_tau_linear(
            frequencies,
            phase_unwrapped,
            edge_fraction=edge_fraction,
        )
        return tau, tau

    x = frequencies[mask]
    y = phase_unwrapped[mask]
    y_fit = fit[mask]
    if x.size < 3:
        tau, _, _, _, _ = _fit_tau_linear(
            frequencies,
            phase_unwrapped,
            edge_fraction=edge_fraction,
        )
        return tau, tau

    residual = y - y_fit
    rng = np.random.default_rng(int(seed))
    tau_samples = np.empty(int(samples), dtype=float)
    for i in range(int(samples)):
        idx = rng.integers(0, x.size, size=x.size)
        y_bs = y_fit + residual[idx]
        slope_bs, _ = np.polyfit(x, y_bs, deg=1)
        tau_samples[i] = float(-slope_bs / (2.0 * np.pi))

    alpha = max(0.0, min(1.0, 1.0 - float(confidence_level)))
    ci_low = float(np.quantile(tau_samples, alpha / 2.0))
    ci_high = float(np.quantile(tau_samples, 1.0 - alpha / 2.0))
    return ci_low, ci_high


@workflow.workflow_options
class LinearPhaseDelayAnalysisWorkflowOptions:
    """Option class for linear phase-delay analysis workflow."""

    do_plotting: bool = workflow.option_field(
        True, description="Whether to create the phase-delay plot."
    )
    edge_fraction: float = workflow.option_field(
        0.25,
        description=(
            "Fraction of points used at each edge for linear fitting to avoid "
            "resonance-distorted center points."
        ),
    )
    bootstrap_samples: int = workflow.option_field(
        200, description="Bootstrap sample count for phase-delay uncertainty."
    )
    bootstrap_seed: int = workflow.option_field(
        8301, description="Bootstrap RNG seed."
    )
    confidence_level: float = workflow.option_field(
        0.95, description="Confidence level for phase-delay interval."
    )


@workflow.workflow
def analysis_workflow(
    result: RunExperimentResults,
    qubit: QuantumElement,
    frequencies: ArrayLike,
    state: str = "g",
    options: LinearPhaseDelayAnalysisWorkflowOptions | None = None,
) -> None:
    """Run phase-delay analysis and return extracted parameter updates."""
    opts = LinearPhaseDelayAnalysisWorkflowOptions() if options is None else options
    metrics = calculate_linear_phase_delay_metrics(
        qubit=qubit,
        result=result,
        frequencies=frequencies,
        state=state,
        edge_fraction=opts.edge_fraction,
        bootstrap_samples=opts.bootstrap_samples,
        bootstrap_seed=opts.bootstrap_seed,
        confidence_level=opts.confidence_level,
    )
    with workflow.if_(opts.do_plotting):
        plot_linear_phase_delay(qubit=qubit, metrics=metrics)
    workflow.return_(metrics)


@workflow.task
def calculate_linear_phase_delay_metrics(
    qubit: QuantumElement,
    result: RunExperimentResults,
    frequencies: ArrayLike,
    state: str,
    edge_fraction: float,
    bootstrap_samples: int,
    bootstrap_seed: int,
    confidence_level: float,
) -> dict:
    """Estimate readout phase-delay tau from spectroscopy traces."""
    qubit, frequencies = validation.validate_and_convert_single_qubit_sweeps(
        qubit, frequencies
    )
    validation.validate_result(result)

    if not isinstance(state, str) or len(state) == 0:
        raise ValueError("state must be a non-empty string.")

    freqs = np.asarray(frequencies, dtype=float)
    handle = dsl.handles.result_handle(qubit.uid, suffix=state)
    s21 = np.asarray(result[handle].data)
    if s21.size != freqs.size:
        raise ValueError(
            f"Frequency points ({freqs.size}) and S21 points ({s21.size}) do not match."
        )

    magnitude = np.abs(s21)
    phase_wrapped = np.angle(s21)
    phase_unwrapped = np.unwrap(phase_wrapped)

    tau_s, slope, intercept, fit, fit_mask = _fit_tau_linear(
        freqs,
        phase_unwrapped,
        edge_fraction=edge_fraction,
    )
    ci_low, ci_high = _bootstrap_tau_ci(
        frequencies=freqs,
        phase_unwrapped=phase_unwrapped,
        fit=fit,
        mask=fit_mask,
        edge_fraction=edge_fraction,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
        confidence_level=confidence_level,
    )

    y = phase_unwrapped[fit_mask]
    y_fit = fit[fit_mask]
    ss_res = float(np.sum((y - y_fit) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / max(ss_tot, 1e-30)

    old_tau = getattr(qubit.parameters, "readout_phase_delay", None)
    return {
        "readout_phase_delay": {
            "tau_s": float(tau_s),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "slope_rad_per_hz": float(slope),
            "intercept_rad": float(intercept),
            "r_squared": float(r_squared),
            "state": state,
        },
        "frequencies_hz": freqs,
        "magnitude": magnitude,
        "phase_wrapped": phase_wrapped,
        "phase_unwrapped": phase_unwrapped,
        "phase_fit": fit,
        "phase_fit_mask": fit_mask,
        "old_parameter_values": {
            qubit.uid: {
                "readout_phase_delay": old_tau,
            }
        },
        "new_parameter_values": {
            qubit.uid: {
                "readout_phase_delay": float(tau_s),
            }
        },
    }


@workflow.task
def plot_linear_phase_delay(
    qubit: QuantumElement,
    metrics: dict,
    options: BasePlottingOptions | None = None,
) -> mpl.figure.Figure | None:
    """Plot magnitude and unwrapped phase with linear fit."""
    opts = BasePlottingOptions() if options is None else options

    f_ghz = np.asarray(metrics["frequencies_hz"], dtype=float) / 1e9
    mag = np.asarray(metrics["magnitude"], dtype=float)
    phase_unwrapped = np.asarray(metrics["phase_unwrapped"], dtype=float)
    phase_fit = np.asarray(metrics["phase_fit"], dtype=float)
    fit_mask = np.asarray(metrics["phase_fit_mask"], dtype=bool)
    delay = metrics["readout_phase_delay"]
    tau_ns = 1e9 * float(delay["tau_s"])
    ci_low_ns = 1e9 * float(delay["ci_low"])
    ci_high_ns = 1e9 * float(delay["ci_high"])

    fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax_mag.plot(f_ghz, mag, label=r"$|S_{21}|$")
    ax_mag.set_ylabel(r"$|S_{21}|$ (a.u.)")
    ax_mag.legend(frameon=False)
    ax_mag.set_title(timestamped_title(f"Linear Phase Delay {qubit.uid}"))

    ax_phase.plot(f_ghz, phase_unwrapped, label="phase (unwrap)")
    ax_phase.plot(f_ghz, phase_fit, "--", label="linear fit")
    ax_phase.plot(
        f_ghz[fit_mask],
        phase_unwrapped[fit_mask],
        ".",
        markersize=3,
        label="fit points",
    )
    ax_phase.set_xlabel(r"Readout Frequency, $f_{\mathrm{RO}}$ (GHz)")
    ax_phase.set_ylabel("Phase (rad)")
    ax_phase.legend(frameon=False)
    ax_phase.text(
        0.01,
        0.02,
        (
            f"tau={tau_ns:.3f} ns, "
            f"95% CI=[{ci_low_ns:.3f}, {ci_high_ns:.3f}] ns, "
            f"R^2={delay['r_squared']:.4f}"
        ),
        transform=ax_phase.transAxes,
        fontsize=9,
    )

    fig.tight_layout()
    if opts.save_figures:
        workflow.save_artifact(f"Linear_phase_delay_{qubit.uid}", fig)
    if opts.close_figures:
        plt.close(fig)
    return fig


"""Three-panel resonator amplitude spectroscopy plots with cable-delay correction."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from laboneq import workflow
from laboneq.simple import dsl
from laboneq_applications.analysis.options import BasePlottingOptions
from laboneq_applications.analysis.spectroscopy_two_dimensional_plotting import (
    Spectroscopy2DPlottingWorkflowOptions,
)
from laboneq_applications.core import validation

from .dispersive_shift import correct_readout_phase_delay
from .plot_theme import with_plot_theme
from .plotting_helpers import timestamped_title


@workflow.task_options(base_class=BasePlottingOptions)
class PlotSignalMagnitudeAndPhase2DOptions:
    """Figure size for the three-panel amplitude sweep plot."""

    figure_size_magnitude_phase: tuple | None = workflow.option_field(None)


@workflow.task_options(base_class=BasePlottingOptions)
class PlotRawData2DOptions:
    """Figure size for raw I/Q heatmaps."""

    figure_size_raw_data: tuple | None = workflow.option_field(None)


@workflow.workflow(name="analysis_workflow")
def analysis_workflow(
    result,
    qubits,
    sweep_points_1d,
    sweep_points_2d,
    label_sweep_points_1d,
    label_sweep_points_2d,
    scaling_sweep_points_2d=1.0,
    options: Spectroscopy2DPlottingWorkflowOptions | None = None,
):
    """Keep existing plotting options and add corrected phase to the main figure."""
    with workflow.if_(options.do_raw_data_plotting):
        plot_raw_complex_data_2d(
            qubits=qubits,
            result=result,
            sweep_points_1d=sweep_points_1d,
            sweep_points_2d=sweep_points_2d,
            label_sweep_points_1d=label_sweep_points_1d,
            label_sweep_points_2d=label_sweep_points_2d,
            scaling_sweep_points_1d=1e-9,
            scaling_sweep_points_2d=scaling_sweep_points_2d,
        )
    with workflow.if_(options.do_plotting_magnitude_phase):
        plot_signal_magnitude_and_phase_2d(
            qubits=qubits,
            result=result,
            sweep_points_1d=sweep_points_1d,
            sweep_points_2d=sweep_points_2d,
            label_sweep_points_1d=label_sweep_points_1d,
            label_sweep_points_2d=label_sweep_points_2d,
            scaling_sweep_points_1d=1e-9,
            scaling_sweep_points_2d=scaling_sweep_points_2d,
        )


@workflow.task
@with_plot_theme
def plot_raw_complex_data_2d(
    qubits,
    result,
    sweep_points_1d,
    sweep_points_2d,
    label_sweep_points_1d,
    label_sweep_points_2d,
    scaling_sweep_points_1d=1.0,
    scaling_sweep_points_2d=1.0,
    options: PlotRawData2DOptions | None = None,
):
    """Retain raw I/Q plots using the repository's local plotting convention."""
    opts = PlotRawData2DOptions() if options is None else options
    _, frequencies = validation.validate_and_convert_qubits_sweeps(qubits, sweep_points_1d)
    qubits, amplitudes = validation.validate_and_convert_qubits_sweeps(qubits, sweep_points_2d)
    validation.validate_result(result)
    figures = {}
    for q, freqs, amps in zip(qubits, frequencies, amplitudes, strict=True):
        raw = np.asarray(result[dsl.handles.result_handle(q.uid)].data)
        if raw.shape != (len(amps), len(freqs)):
            raise ValueError("Data must have shape (number of amplitudes, number of frequencies).")
        fig, axes = plt.subplots(
            1,
            2,
            figsize=opts.figure_size_raw_data or (10, 4),
            sharex=True,
            sharey=True,
            layout="constrained",
        )
        fig.suptitle(timestamped_title(f"Raw data {q.uid}"))
        for ax, values, title in zip(
            axes, (raw.real, raw.imag), ("Real", "Imaginary"), strict=True
        ):
            mesh = ax.pcolormesh(
                np.asarray(freqs) * scaling_sweep_points_1d,
                np.asarray(amps) * scaling_sweep_points_2d,
                values,
                shading="auto",
            )
            ax.set_title(title)
            ax.set_xlabel(label_sweep_points_1d)
            fig.colorbar(mesh, ax=ax, label="Signal (a.u.)")
        axes[0].set_ylabel(label_sweep_points_2d)
        if opts.save_figures:
            workflow.save_artifact(f"Raw_data_{q.uid}", fig)
        if opts.close_figures:
            plt.close(fig)
        figures[q.uid] = fig
    return figures


@workflow.task
@with_plot_theme
def plot_signal_magnitude_and_phase_2d(
    qubits,
    result,
    sweep_points_1d,
    sweep_points_2d,
    label_sweep_points_1d,
    label_sweep_points_2d,
    scaling_sweep_points_1d=1.0,
    scaling_sweep_points_2d=1.0,
    options: PlotSignalMagnitudeAndPhase2DOptions | None = None,
):
    """Plot magnitude, raw wrapped phase, and delay-corrected wrapped phase.

    Use the temporary qubit's readout_phase_delay. If calibration is missing,
    label the third panel as unavailable rather than implying a correction.
    """
    opts = PlotSignalMagnitudeAndPhase2DOptions() if options is None else options
    _, frequencies = validation.validate_and_convert_qubits_sweeps(qubits, sweep_points_1d)
    qubits, amplitudes = validation.validate_and_convert_qubits_sweeps(qubits, sweep_points_2d)
    validation.validate_result(result)
    figures = {}
    for q, freqs, amps in zip(qubits, frequencies, amplitudes, strict=True):
        raw = np.asarray(result[dsl.handles.result_handle(q.uid)].data)
        if raw.shape != (len(amps), len(freqs)):
            raise ValueError("Data must have shape (number of amplitudes, number of frequencies).")
        corrected = correct_readout_phase_delay(q, freqs, raw)
        delay = corrected["phase_delay"]
        size = opts.figure_size_magnitude_phase
        fig, axes = plt.subplots(
            1,
            3,
            figsize=(15, 4.5) if size is None else size,
            sharex=True,
            sharey=True,
            layout="constrained",
        )
        fig.suptitle(timestamped_title(f"Magnitude-Phase {q.uid}"))
        panels = (
            (np.abs(raw), "Magnitude", "Magnitude (a.u.)"),
            (np.angle(raw), "Raw phase", "Phase (rad)"),
            (corrected["phase_corrected"], "Delay-corrected phase", "Phase (rad)"),
        )
        for index, (ax, (values, title, label)) in enumerate(zip(axes, panels, strict=True)):
            ax.set_title(title)
            ax.set_xlabel(label_sweep_points_1d)
            if values is None:
                ax.text(
                    0.5,
                    0.5,
                    "Correction unavailable:\nreadout_phase_delay is not set",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                )
                continue
            color_options = {} if index == 0 else dict(cmap="twilight", vmin=-np.pi, vmax=np.pi)
            mesh = ax.pcolormesh(
                np.asarray(freqs) * scaling_sweep_points_1d,
                np.asarray(amps) * scaling_sweep_points_2d,
                values,
                shading="auto",
                **color_options,
            )
            fig.colorbar(mesh, ax=ax, label=label)
        axes[0].set_ylabel(label_sweep_points_2d)
        if delay["applied"]:
            axes[2].set_title(f"Delay-corrected phase\nτ = {delay['tau_s'] * 1e9:.4f} ns")
        if opts.save_figures:
            workflow.save_artifact(f"Magnitude_Phase_{q.uid}", fig)
        if opts.close_figures:
            plt.close(fig)
        figures[q.uid] = fig
    return figures

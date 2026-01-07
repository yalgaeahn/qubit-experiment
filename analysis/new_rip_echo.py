# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Analysis for RIP echo experiments.

Produces population heatmaps with delay on the x-axis and RIP detuning on the y-axis.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from laboneq import workflow

from laboneq_applications.analysis.calibration_traces_rotation import (
    calculate_qubit_population_2d,
)
from laboneq_applications.analysis.options import TuneUpAnalysisWorkflowOptions
from laboneq_applications.analysis.plotting_helpers import timestamped_title
from laboneq_applications.core.validation import validate_and_convert_qubits_sweeps

if TYPE_CHECKING:
    import matplotlib as mpl
    from laboneq.workflow.tasks.run_experiment import RunExperimentResults
    from numpy.typing import ArrayLike

    from laboneq_applications.typing import QuantumElements, QubitSweepPoints


@workflow.workflow
def analysis_workflow(
    result: RunExperimentResults,
    qubits: QuantumElements,
    delays: QubitSweepPoints,
    detunings: QubitSweepPoints,
    raw_iq_save_dir: str | None = None,
    options: TuneUpAnalysisWorkflowOptions | None = None,
) -> None:
    """RIP echo analysis workflow.

    Arguments:
        result:
            The experiment results returned by run_experiment.
        qubits:
            The qubits on which to run the analysis.
        delays:
            The delay sweep points (seconds).
        detunings:
            The RIP detuning sweep points (Hz).
        raw_iq_save_dir:
            Directory to save raw I/Q heatmap data as .npy files.
        options:
            Analysis options controlling plotting behavior.
    """
    opts = TuneUpAnalysisWorkflowOptions() if options is None else options

    processed_data_dict = calculate_qubit_population_2d(
        qubits=qubits,
        result=result,
        sweep_points_1d=delays,
        sweep_points_2d=detunings,
    )
    with workflow.if_(opts.do_plotting):
        with workflow.if_(opts.do_raw_data_plotting):
            plot_raw_iq_heatmap_2d(
                qubits,
                processed_data_dict,
                save_dir=raw_iq_save_dir,
            )
        with workflow.if_(opts.do_qubit_population_plotting):
            plot_population_heatmap_2d(qubits, processed_data_dict)

    workflow.return_(processed_data_dict)


@workflow.task
def plot_population_heatmap_2d(
    qubits: QuantumElements,
    processed_data_dict: dict[str, dict[str, ArrayLike]],
) -> dict[str, mpl.figure.Figure]:
    """Plot population as a 2D heatmap over delay and RIP detuning."""
    qubits = validate_and_convert_qubits_sweeps(qubits)
    figures: dict[str, mpl.figure.Figure] = {}

    for q in qubits:
        data = processed_data_dict[q.uid]["population"]  # shape: (detuning, delay)
        delays = np.array(processed_data_dict[q.uid]["sweep_points_1d"])
        detunings = np.array(processed_data_dict[q.uid]["sweep_points_2d"])

        fig, ax = plt.subplots()
        im = ax.imshow(
            data,
            aspect="auto",
            origin="lower",
            extent=[
                delays[0] * 1e6,
                delays[-1] * 1e6,
                detunings[0] / 1e6,
                detunings[-1] / 1e6,
            ],
            vmin=0,
            vmax=1,
            cmap="viridis",
        )
        ax.set_xlabel("Delay, $\\tau$ ($\\mu$s)")
        ax.set_ylabel("RIP detuning (MHz)")
        ax.set_title(timestamped_title(f"Population {q.uid}"))
        fig.colorbar(im, ax=ax, label="Population |e>")
        figures[q.uid] = fig

    return figures


@workflow.task
def plot_raw_iq_heatmap_2d(
    qubits: QuantumElements,
    processed_data_dict: dict[str, dict[str, ArrayLike]],
    save_dir: str | None = None,
) -> dict[str, mpl.figure.Figure]:
    """Plot raw I/Q heatmaps over delay and RIP detuning."""
    qubits = validate_and_convert_qubits_sweeps(qubits)
    figures: dict[str, mpl.figure.Figure] = {}

    if save_dir is not None:
        from pathlib import Path

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

    for q in qubits:
        raw_data = processed_data_dict[q.uid]["data_raw"]  # shape: (detuning, delay)
        delays = np.array(processed_data_dict[q.uid]["sweep_points_1d"])
        detunings = np.array(processed_data_dict[q.uid]["sweep_points_2d"])

        real_part = np.real(raw_data)
        imag_part = np.imag(raw_data)
        if save_dir is not None:
            np.save(save_path / f"{q.uid}_raw_i.npy", real_part)
            np.save(save_path / f"{q.uid}_raw_q.npy", imag_part)
            np.save(save_path / f"{q.uid}_delays_s.npy", delays)
            np.save(save_path / f"{q.uid}_detunings_hz.npy", detunings)
        color_limit = np.nanmax(
            np.abs(np.concatenate([real_part.ravel(), imag_part.ravel()]))
        )
        if not np.isfinite(color_limit) or color_limit == 0:
            color_limit = 1.0

        fig_width, fig_height = plt.rcParams["figure.figsize"]
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(1.3 * fig_width, fig_height),
            sharex=True,
            sharey=True,
        )
        extent = [
            delays[0] * 1e6,
            delays[-1] * 1e6,
            detunings[0] / 1e6,
            detunings[-1] / 1e6,
        ]

        for ax, data, title in zip(
            axes,
            (real_part, imag_part),
            ("I (Real)", "Q (Imag)"),
        ):
            im = ax.imshow(
                data,
                aspect="auto",
                origin="lower",
                extent=extent,
                vmin=-color_limit,
                vmax=color_limit,
                cmap="RdBu_r",
            )
            ax.set_title(title)
            ax.set_xlabel("Delay, $\\tau$ ($\\mu$s)")
            fig.colorbar(im, ax=ax, label="Amplitude (arb.)")

        axes[0].set_ylabel("RIP detuning (MHz)")
        fig.suptitle(timestamped_title(f"Raw IQ {q.uid}"))
        figures[q.uid] = fig

    return figures

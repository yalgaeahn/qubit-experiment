# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Analysis for the cavity pi/nopi experiment."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import matplotlib.pyplot as plt
import numpy as np
from laboneq import workflow

from laboneq_applications.analysis.plotting_helpers import timestamped_title
from laboneq_applications.core.validation import validate_result

if TYPE_CHECKING:
    from laboneq.workflow.tasks.run_experiment import RunExperimentResults


def _timestamped_filename(title: str) -> str:
    """Return a filesystem-friendly timestamped filename."""
    return (
        timestamped_title(title)
        .replace(" ", "_")
        .replace(":", "-")
    )


@workflow.workflow_options
class CavityPiNoPiAnalysisWorkflowOptions:
    """Workflow options for the cavity pi/nopi analysis."""

    quadrature: str = workflow.option_field(
        "I",
        description="Quadrature to extract: 'I', 'Q', 'abs', or 'complex'.",
    )
    do_plotting: bool = workflow.option_field(
        True, description="Whether to create plots."
    )
    save_figures: bool = workflow.option_field(
        True, description="Whether to save figures with timestamped names."
    )
    close_figures: bool = workflow.option_field(
        True, description="Whether to close figures after saving."
    )


@workflow.workflow
def analysis_workflow(
    result: RunExperimentResults,
    *,
    frequency_offsets: Sequence[float],
    base_frequency: float | None = None,
    options: CavityPiNoPiAnalysisWorkflowOptions | None = None,
):
    """Analysis workflow for the cavity pi/nopi experiment."""
    opts = (
        CavityPiNoPiAnalysisWorkflowOptions() if options is None else options
    )
    processed = collect_data(
        result,
        frequency_offsets=frequency_offsets,
        base_frequency=base_frequency,
        quadrature=opts.quadrature,
    )
    contrasts = calculate_contrasts(processed)
    with workflow.if_(opts.do_plotting):
        plot_results(processed, contrasts, options=opts)
    workflow.return_(contrasts)


@workflow.task
def collect_data(
    result: RunExperimentResults,
    *,
    frequency_offsets: Sequence[float],
    base_frequency: float | None,
    quadrature: str,
) -> dict:
    """Collect raw data and reshape metadata for downstream plotting."""
    validate_result(result)
    if "cavity_pi_nopi" not in result.acquired_results:
        raise KeyError("Handle 'cavity_pi_nopi' not found in the result.")
    raw = result.get_data("cavity_pi_nopi")
    axes = result.acquired_results["cavity_pi_nopi"].axis

    freq_axis = (
        np.asarray(frequency_offsets)
        if frequency_offsets is not None
        else np.asarray(axes[0])
    )
    if base_frequency is not None:
        freq_axis = freq_axis + base_frequency

    cross_axis = np.asarray(axes[1]) if len(axes) > 1 else np.arange(raw.shape[1])
    pi_axis = np.asarray(axes[2]) if len(axes) > 2 else np.arange(raw.shape[2])

    if quadrature.lower() == "i":
        data = np.real(raw)
    elif quadrature.lower() == "q":
        data = np.imag(raw)
    elif quadrature.lower() == "abs":
        data = np.abs(raw)
    else:
        data = raw

    return {
        "raw": raw,
        "data": data,
        "frequency": freq_axis,
        "cross_cases": cross_axis,
        "pi_cases": pi_axis,
        "quadrature": quadrature,
    }


@workflow.task
def calculate_contrasts(processed: dict) -> dict:
    """Calculate differences between pi/nopi and cross-Kerr conditions."""
    data = np.asarray(processed["data"])
    cross_cases = processed["cross_cases"]

    pi_difference = None
    cross_difference = None

    if data.shape[2] >= 2:
        pi_difference = data[:, :, 1] - data[:, :, 0]
    if data.shape[1] >= 2:
        cross_difference = data[:, 1, :] - data[:, 0, :]

    return {
        "pi_difference": pi_difference,
        "cross_difference": cross_difference,
        "cross_cases": cross_cases,
    }


@workflow.task
def plot_results(
    processed: dict,
    contrasts: dict,
    options: CavityPiNoPiAnalysisWorkflowOptions | None = None,
):
    """Plot the raw sweeps and derived contrasts."""
    opts = (
        CavityPiNoPiAnalysisWorkflowOptions() if options is None else options
    )
    freq = np.asarray(processed["frequency"])
    data = np.asarray(processed["data"])
    cross_cases = processed["cross_cases"]
    pi_cases = processed["pi_cases"]

    fig, axes = plt.subplots(
        len(cross_cases), 1, figsize=(12, 5 * len(cross_cases)), sharex=True
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for idx, ax in enumerate(axes):
        if idx >= data.shape[1]:
            continue
        ax.plot(freq / 1e6, data[:, idx, 0], "o-", label=f"pi={pi_cases[0]}")
        if data.shape[2] > 1:
            ax.plot(freq / 1e6, data[:, idx, 1], "o-", label=f"pi={pi_cases[1]}")
        ax.set_title(f"Cross-Kerr case {cross_cases[idx]}")
        ax.set_ylabel(f"{processed['quadrature']}")
        ax.grid(True)
        ax.legend()
    axes[-1].set_xlabel("Cavity frequency (MHz)")
    fig.suptitle(
        timestamped_title("Cavity pi/nopi"), fontsize=14, y=0.98
    )

    if opts.save_figures:
        fig.savefig(_timestamped_filename("cavity_pi_nopi_raw"))
    if opts.close_figures:
        plt.close(fig)

    if contrasts["pi_difference"] is not None:
        fig2, ax2 = plt.subplots(figsize=(12, 5))
        for idx in range(contrasts["pi_difference"].shape[1]):
            label = f"cross={cross_cases[idx]}" if idx < len(cross_cases) else None
            ax2.plot(
                freq / 1e6,
                contrasts["pi_difference"][:, idx],
                "o-",
                label=label,
            )
        ax2.set_title("pi - nopi")
        ax2.set_xlabel("Cavity frequency (MHz)")
        ax2.set_ylabel(f"{processed['quadrature']}")
        ax2.grid(True)
        ax2.legend()
        if opts.save_figures:
            fig2.savefig(_timestamped_filename("cavity_pi_nopi_pi_diff"))
        if opts.close_figures:
            plt.close(fig2)

    if contrasts["cross_difference"] is not None:
        fig3, ax3 = plt.subplots(figsize=(12, 5))
        labels = ["nopi", "pi"] if data.shape[2] >= 2 else [f"pi={pi_cases[0]}"]
        for idx in range(contrasts["cross_difference"].shape[1]):
            lbl = labels[idx] if idx < len(labels) else f"pi index {idx}"
            ax3.plot(
                freq / 1e6,
                contrasts["cross_difference"][:, idx],
                "o-",
                label=lbl,
            )
        ax3.set_title("cross-Kerr on - off")
        ax3.set_xlabel("Cavity frequency (MHz)")
        ax3.set_ylabel(f"{processed['quadrature']}")
        ax3.grid(True)
        ax3.legend()
        if opts.save_figures:
            fig3.savefig(_timestamped_filename("cavity_pi_nopi_cross_diff"))
        if opts.close_figures:
            plt.close(fig3)

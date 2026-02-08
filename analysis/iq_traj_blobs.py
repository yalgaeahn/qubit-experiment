# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Combined analysis for IQ trajectory and integrated IQ blobs.

The workflow reuses a single raw acquisition result to:
1) demodulate and visualize state trajectories in IQ/time domains, and
2) software-integrate the same traces into shot-wise IQ points for blob fitting.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from laboneq import workflow
from laboneq.simple import dsl

from analysis.iq_blobs import (
    calculate_assignment_fidelities,
    calculate_assignment_matrices,
    fit_data,
    plot_assignment_matrices,
    plot_iq_blobs,
)
from analysis.iq_traj import plot_iq_trajectories
from laboneq_applications.core.validation import (
    validate_and_convert_qubits_sweeps,
    validate_result,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from laboneq.workflow.tasks.run_experiment import RunExperimentResults
    from numpy.typing import ArrayLike

    from laboneq_applications.typing import QuantumElements


@workflow.workflow_options
class IQTrajBlobsAnalysisWorkflowOptions:
    """Options for combined trajectory + blob analysis."""

    do_fitting: bool = workflow.option_field(
        True, description="Whether to perform IQ blob fitting."
    )
    do_plotting: bool = workflow.option_field(
        True, description="Whether to create plots."
    )
    do_plotting_iq_traj: bool = workflow.option_field(
        True, description="Whether to create IQ trajectory plots."
    )
    do_plotting_iq_blobs: bool = workflow.option_field(
        True, description="Whether to create integrated IQ blob plots."
    )
    do_plotting_assignment_matrices: bool = workflow.option_field(
        True, description="Whether to create assignment matrix plots."
    )
    fit_method: Literal["lda", "gmm"] = workflow.option_field(
        "gmm",
        description='Classifier to use for the integrated blobs: "lda" or "gmm".',
    )
    chunk_size: int = workflow.option_field(
        8,
        description="Chunk size in samples for trajectory smoothing.",
    )
    sample_dt_ns: float = workflow.option_field(
        0.5,
        description="Sample period in ns for digital demodulation.",
    )
    integration_start: int = workflow.option_field(
        0,
        description="Start sample (inclusive) of software integration window.",
    )
    integration_stop: int | None = workflow.option_field(
        None,
        description="Stop sample (exclusive) of software integration window.",
    )


@workflow.workflow
def analysis_workflow(
    result: RunExperimentResults,
    qubits: QuantumElements,
    states: Sequence[str],
    options: IQTrajBlobsAnalysisWorkflowOptions | None = None,
) -> None:
    """Run combined IQ trajectory and integrated IQ blob analysis."""
    options = IQTrajBlobsAnalysisWorkflowOptions() if options is None else options

    demodulated_data = demodulate_time_traces(
        qubits,
        result,
        states,
        sample_dt_ns=options.sample_dt_ns,
    )
    trajectory_data = average_chunk_time_traces(
        qubits,
        demodulated_data,
        states,
        chunk_size=options.chunk_size,
    )
    integrated_blob_data = integrate_shots_for_blobs(
        qubits,
        demodulated_data,
        states,
        integration_start=options.integration_start,
        integration_stop=options.integration_stop,
    )

    fit_results = None
    assignment_matrices = None
    assignment_fidelities = None
    with workflow.if_(options.do_fitting):
        fit_results = fit_data(qubits, integrated_blob_data, options.fit_method)
        assignment_matrices = calculate_assignment_matrices(
            qubits, integrated_blob_data, fit_results
        )
        assignment_fidelities = calculate_assignment_fidelities(
            qubits, assignment_matrices
        )

    with workflow.if_(options.do_plotting):
        with workflow.if_(options.do_plotting_iq_traj):
            plot_iq_trajectories(
                qubits,
                states,
                trajectory_data,
                chunk_size=options.chunk_size,
            )

        with workflow.if_(options.do_plotting_iq_blobs):
            plot_iq_blobs(
                qubits,
                states,
                integrated_blob_data,
                fit_results,
            )

        with workflow.if_(options.do_plotting_assignment_matrices):
            with workflow.if_(options.do_fitting):
                plot_assignment_matrices(
                    qubits,
                    states,
                    assignment_matrices,
                    assignment_fidelities,
                )

    workflow.return_(assignment_fidelities)


@workflow.task
def demodulate_time_traces(
    qubits: QuantumElements,
    result: RunExperimentResults,
    states: Sequence[str],
    sample_dt_ns: float = 0.5,
) -> dict[str, dict[str, dict[str, ArrayLike]]]:
    """Read raw calibration traces and demodulate them into complex baseband."""
    qubits = validate_and_convert_qubits_sweeps(qubits)
    validate_result(result)

    processed_data = {q.uid: {} for q in qubits}
    for q in qubits:
        if_freq_hz = (
            q.parameters.readout_resonator_frequency - q.parameters.readout_lo_frequency
        )
        for s in states:
            raw = np.asarray(result[dsl.handles.calibration_trace_handle(q.uid, s)].data)
            shots = _to_shots_matrix(raw)

            n_samples = shots.shape[1]
            times_s = np.arange(n_samples, dtype=float) * sample_dt_ns * 1e-9
            phase = np.exp(-1j * 2 * np.pi * if_freq_hz * times_s)
            demod_shots = shots * phase[np.newaxis, :]

            processed_data[q.uid][s] = {
                "shots": demod_shots,
                "mean_trace": np.mean(demod_shots, axis=0),
            }

    return processed_data


@workflow.task
def average_chunk_time_traces(
    qubits: QuantumElements,
    demodulated_data: dict[str, dict[str, dict[str, ArrayLike]]],
    states: Sequence[str],
    chunk_size: int = 8,
) -> dict[str, dict[str, ArrayLike]]:
    """Average demodulated mean traces in fixed-size chunks for trajectory plots."""
    qubits = validate_and_convert_qubits_sweeps(qubits)
    chunk_size = max(1, int(chunk_size))

    trajectory_data: dict[str, dict[str, ArrayLike]] = {q.uid: {} for q in qubits}
    for q in qubits:
        for s in states:
            trace = np.asarray(demodulated_data[q.uid][s]["mean_trace"])
            n_chunks = len(trace) // chunk_size
            if n_chunks == 0:
                chunked = trace
            else:
                chunked = trace[: n_chunks * chunk_size].reshape(n_chunks, chunk_size).mean(
                    axis=1
                )
            trajectory_data[q.uid][s] = chunked

    return trajectory_data


@workflow.task
def integrate_shots_for_blobs(
    qubits: QuantumElements,
    demodulated_data: dict[str, dict[str, dict[str, ArrayLike]]],
    states: Sequence[str],
    integration_start: int = 0,
    integration_stop: int | None = None,
) -> dict[str, dict[str, ArrayLike | dict]]:
    """Software-integrate demodulated shots and build IQ blob analysis payload."""
    qubits = validate_and_convert_qubits_sweeps(qubits)
    state_to_label = {"g": 0, "e": 1, "f": 2}

    processed_data_dict: dict[str, dict[str, ArrayLike | dict]] = {q.uid: {} for q in qubits}
    for q in qubits:
        shots_per_state = {}
        shots_combined = []
        ideal_states = []

        for idx, s in enumerate(states):
            shots_matrix = np.asarray(demodulated_data[q.uid][s]["shots"])
            n_samples = shots_matrix.shape[1]

            start = max(0, int(integration_start))
            stop = n_samples if integration_stop is None else min(int(integration_stop), n_samples)
            if stop <= start:
                start = 0
                stop = n_samples

            integrated = np.mean(shots_matrix[:, start:stop], axis=1)
            shots_per_state[s] = integrated

            shots_combined.append(
                np.column_stack([np.real(integrated), np.imag(integrated)])
            )
            label = state_to_label.get(s, idx)
            ideal_states.append(label * np.ones(len(integrated)))

        processed_data_dict[q.uid] = {
            "shots_per_state": shots_per_state,
            "shots_combined": np.concatenate(shots_combined, axis=0),
            "ideal_states_shots": np.concatenate(ideal_states),
        }

    return processed_data_dict


def _to_shots_matrix(raw: np.ndarray) -> np.ndarray:
    """Convert raw complex trace data to a (shots, samples) matrix."""
    if raw.ndim == 0:
        return raw.reshape(1, 1)
    if raw.ndim == 1:
        return raw.reshape(1, -1)
    return raw.reshape(-1, raw.shape[-1])

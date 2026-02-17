# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""This module defines the analysis for an IQ-blob experiment.

The experiment is defined in laboneq_applications.experiments.

In this analysis, we collect the single shots acquire for each prepared stated and
then use LinearDiscriminantAnalysis from the sklearn library to classify the data into
the prepared states. From this classification, we calculate the correct-state-assignment
matrix and the correct-state-assignment fidelity. Finally, we plot the single shots for
each prepared state and the correct-state-assignment matrix.
"""

from __future__ import annotations

import logging
from itertools import permutations, product
from typing import TYPE_CHECKING, Literal

import matplotlib.colors as mc
import matplotlib.pyplot as plt
import numpy as np
from laboneq import workflow
from laboneq.simple import dsl
from matplotlib.patches import Ellipse
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import confusion_matrix
from sklearn.mixture import GaussianMixture

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
class IQBlobAnalysisWorkflowOptions:
    """Option class for IQ-blob analysis workflows.

    Attributes:
        do_fitting:
            Whether to perform the fit.
            Default: `True`.
        do_plotting:
            Whether to create plots.
            Default: 'True'.
        do_plotting_iq_blobs:
            Whether to create the IQ-blob plots of the single shots.
            Default: 'True'.
        do_plotting_assignment_matrices:
            Whether to create the assignment matrix plots.
            Default: 'True'.
        fit_method:
            Classifier to use for the fit. Supported: "lda" (default) and "gmm" (two
            states only).
        do_threshold_calibration:
            Whether to estimate g/e discrimination thresholds from integrated IQ shots.
            Default: `True`.
        enforce_constant_kernel:
            Whether to enforce the default constant integration kernel when extracting
            qubit parameters for discrimination calibration.
            Default: `True`.
    """

    do_fitting: bool = workflow.option_field(
        True, description="Whether to perform the fit."
    )
    do_plotting: bool = workflow.option_field(
        True, description="Whether to create plots."
    )
    do_plotting_iq_blobs: bool = workflow.option_field(
        True, description="Whether to create the IQ-blob plots of the single shots."
    )
    do_plotting_assignment_matrices: bool = workflow.option_field(
        True, description="Whether to create the assignment matrix plots."
    )
    fit_method: Literal["lda", "gmm"] = workflow.option_field(
        "gmm",
        description='Classifier to use for the fit. Supported: "lda" and "gmm" (two states).',
    )
    do_threshold_calibration: bool = workflow.option_field(
        True,
        description="Whether to estimate g/e discrimination thresholds.",
    )
    enforce_constant_kernel: bool = workflow.option_field(
        True,
        description=(
            "Whether to force default constant integration kernels when extracting "
            "discrimination calibration parameters."
        ),
    )


@workflow.workflow
def analysis_workflow(
    result: RunExperimentResults,
    qubits: QuantumElements,
    states: Sequence[str],
    options: IQBlobAnalysisWorkflowOptions | None = None,
) -> None:
    """The IQ Blobs analysis Workflow.

    The workflow consists of the following steps:

    - [collect_shots]()
    - [fit_data]()
    - [calculate_assignment_matrices]()
    - [calculate_assignment_fidelities]()
    - [plot_iq_blobs]()
    - [plot_assignment_matrices]()

    Arguments:
        result:
            The experiment results returned by the run_experiment task.
        qubits:
            The qubits on which to run the analysis. May be either a single qubit or
            a list of qubits. The UIDs of these qubits must exist in the result.
        states:
            The basis states the qubits should be prepared in. May be either a string,
            e.g. "gef", or a list of letters, e.g. ["g","e","f"].
        options:
            The options for building the workflow as an instance of
            [IQBlobAnalysisWorkflowOptions]. See the docstring of this class for more
            details.

    Returns:
        WorkflowBuilder:
            The builder for the analysis workflow.

    Example:
        ```python
        result = analysis_workflow(
            results=results
            qubits=[q0, q1],
            amplitudes=[
                np.linspace(0, 1, 11),
                np.linspace(0, 0.75, 11),
            ],
            options=analysis_workflow.options(),
        ).run()
        ```
    """
    options = IQBlobAnalysisWorkflowOptions() if options is None else options
    processed_data_dict = collect_shots(qubits, result, states)
    fit_results = None
    assignment_matrices = None
    assignment_fidelities = None
    discrimination_thresholds = {}
    qubit_parameters = {
        "old_parameter_values": {},
        "new_parameter_values": {},
    }
    with workflow.if_(options.do_fitting):
        fit_results = fit_data(qubits, processed_data_dict, options.fit_method)
        assignment_matrices = calculate_assignment_matrices(
            qubits, processed_data_dict, fit_results
        )
        assignment_fidelities = calculate_assignment_fidelities(
            qubits, assignment_matrices
        )

    with workflow.if_(options.do_plotting):
        with workflow.if_(options.do_plotting_iq_blobs):
            plot_iq_blobs(qubits, states, processed_data_dict, fit_results)
        with workflow.if_(options.do_plotting_assignment_matrices):
            with workflow.if_(options.do_fitting):
                plot_assignment_matrices(
                    qubits, states, assignment_matrices, assignment_fidelities
                )
    with workflow.if_(options.do_threshold_calibration):
        discrimination_thresholds = extract_discrimination_thresholds(
            qubits, states, processed_data_dict
        )
        qubit_parameters = extract_qubit_parameters_for_discrimination(
            qubits,
            discrimination_thresholds,
            enforce_constant_kernel=options.enforce_constant_kernel,
        )
    output_payload = assemble_analysis_output(
        assignment_fidelities,
        discrimination_thresholds,
        qubit_parameters,
    )
    workflow.return_(output_payload)


@workflow.task
def collect_shots(
    qubits: QuantumElements,
    result: RunExperimentResults,
    states: Sequence[str],
) -> dict[str, dict[str, ArrayLike | dict]]:
    """Collect the single shots acquired for each preparation state in states.

    Arguments:
        qubits:
            The qubits on which to run the analysis. May be either a single qubit or
            a list of qubits. The UIDs of these qubits must exist in result.
        result:
            The experiment results returned by the run_experiment task.
        states:
            The basis states the qubits should be prepared in. May be either a string,
            e.g. "gef", or a list of letters, e.g. ["g","e","f"].

    Returns:
        dict with qubit UIDs as keys and values as a dict with the following keys:
              shots_per_state - dict with states as keys and raw single shots as values.
              shots_combined - list of the real and imaginary part of the shots in
                shots_per_state, in a form expected by LinearDiscriminantAnalysis.
              ideal_states_shots - list of the same shape as shots_combined with ints
                specifying the state (0, 1, 2) the qubit is expected to be found in
                ideally for each shot.
    """
    qubits = validate_and_convert_qubits_sweeps(qubits)
    validate_result(result)
    states_map = {"g": 0, "e": 1, "f": 2}
    processed_data_dict = {q.uid: {} for q in qubits}
    for q in qubits:
        shots = {}
        shots_combined = []
        ideal_states = []

        for s in states:
            shots[s] = result[dsl.handles.calibration_trace_handle(q.uid, s)].data
            shots_combined += [
                np.concatenate(
                    [
                        np.real(shots[s])[:, np.newaxis],
                        np.imag(shots[s])[:, np.newaxis],
                    ],
                    axis=1,
                )
            ]

            ideal_states += [states_map[s] * np.ones(len(shots[s]))]

        processed_data_dict[q.uid] = {
            "shots_per_state": shots,
            "shots_combined": np.concatenate(shots_combined, axis=0),
            "ideal_states_shots": np.concatenate(ideal_states),
        }

    return processed_data_dict


@workflow.task
def fit_data(
    qubits: QuantumElements,
    processed_data_dict: dict[str, dict[str, ArrayLike | dict]],
    fit_method: Literal["lda", "gmm"] = "lda",
) -> dict[str, dict] | dict[str, None]:
    """Perform a classification of the shots.

    Arguments:
        qubits:
            The qubits on which to run the analysis. May be either a single qubit or
            a list of qubits. The UIDs of these qubits must exist in the
            processed_data_dict.
        processed_data_dict: the processed data dictionary returned by collect_shots
        fit_method: classifier to use: "lda" (default) or "gmm" (two states only).

    Returns:
        dict with qubit UIDs as keys and the classification result for each qubit as
        keys.
    """
    qubits = validate_and_convert_qubits_sweeps(qubits)
    fit_results = {}
    fit_method = fit_method.lower()
    if fit_method not in {"lda", "gmm"}:
        workflow.log(
            logging.WARNING,
            "Unsupported fit_method '%s'. Falling back to 'lda'.",
            fit_method,
        )
        fit_method = "lda"

    for q in qubits:
        shots_combined = processed_data_dict[q.uid]["shots_combined"]
        ideal_states_shots = processed_data_dict[q.uid]["ideal_states_shots"]
        if fit_method == "gmm":
            unique_states = np.unique(ideal_states_shots)
            if len(unique_states) != 2:
                workflow.log(
                    logging.WARNING,
                    "GMM fit for %s requires exactly two states (g/e). Skipping.",
                    q.uid,
                )
                continue

            clf = GaussianMixture(
                n_components=2,
                covariance_type="full",
                init_params="kmeans",
                max_iter=500,
                random_state=0,
            )
            try:
                clf.fit(shots_combined)
                mapping = _build_gmm_label_mapping(
                    clf, shots_combined, ideal_states_shots
                )
            except Exception as err:  # noqa: BLE001
                workflow.log(logging.ERROR, "GMM fit failed for %s: %s.", q.uid, err)
            else:
                fit_results[q.uid] = {"model": clf, "method": "gmm", "mapping": mapping}
        else:
            clf = LinearDiscriminantAnalysis()
            try:
                clf.fit(shots_combined, ideal_states_shots)
            except Exception as err:  # noqa: BLE001
                workflow.log(logging.ERROR, "LDA fit failed for %s: %s.", q.uid, err)
            else:
                fit_results[q.uid] = {"model": clf, "method": "lda", "mapping": None}

    return fit_results


@workflow.task
def calculate_assignment_matrices(
    qubits: QuantumElements,
    processed_data_dict: dict[str, dict[str, ArrayLike | dict]],
    fit_results: dict[str, None] | dict,
) -> dict[str, None]:
    """Calculate the correct assignment matrices from the result of the classification.

    Arguments:
        qubits:
            The qubits on which to run the analysis. May be either a single qubit or
            a list of qubits. The UIDs of these qubits must exist in the
            processed_data_dict.
        processed_data_dict: the processed data dictionary returned by collect_shots
        fit_results: the classification fit results returned by fit_data.

    Returns:
        dict with qubit UIDs as keys and the assignment matrix for each qubit as keys.
    """
    qubits = validate_and_convert_qubits_sweeps(qubits)
    assignment_matrices = {}
    for q in qubits:
        if q.uid not in fit_results:
            continue

        shots_combined = processed_data_dict[q.uid]["shots_combined"]
        ideal_states_shots = processed_data_dict[q.uid]["ideal_states_shots"]
        predicted_states = _predict_state_labels(fit_results[q.uid], shots_combined)
        if predicted_states is None:
            workflow.log(
                logging.WARNING,
                "Prediction unavailable for %s with method %s.",
                q.uid,
                fit_results[q.uid].get("method"),
            )
            continue
        assignment_matrices[q.uid] = confusion_matrix(
            ideal_states_shots, predicted_states, normalize="true"
        )

    return assignment_matrices


def _build_gmm_label_mapping(
    gmm: GaussianMixture, shots_combined: ArrayLike, ideal_states_shots: ArrayLike
) -> dict[int, int] | None:
    """Map GMM component indices to state labels using mean proximity."""
    state_labels = np.unique(ideal_states_shots)
    if len(state_labels) != gmm.n_components:
        return None

    ideal_means = []
    for state in state_labels:
        mask = ideal_states_shots == state
        if not np.any(mask):
            return None
        ideal_means.append(np.mean(shots_combined[mask], axis=0))
    ideal_means = np.stack(ideal_means)

    best_score = np.inf
    best_mapping = None
    for perm in permutations(range(len(state_labels))):
        score = 0.0
        for comp_idx, state_idx in enumerate(perm):
            score += np.linalg.norm(gmm.means_[comp_idx] - ideal_means[state_idx])
        if score < best_score:
            best_score = score
            best_mapping = {
                comp_idx: int(state_labels[state_idx])
                for comp_idx, state_idx in enumerate(perm)
            }
    return best_mapping


def _predict_state_labels(
    fit_result: dict | None, shots_combined: ArrayLike
) -> np.ndarray | None:
    """Predict state labels from a fit_result entry."""
    if fit_result is None or "method" not in fit_result:
        return None

    method = fit_result.get("method")
    model = fit_result.get("model")
    if method == "lda":
        return model.predict(shots_combined)

    if method == "gmm":
        mapping = fit_result.get("mapping")
        if mapping is None:
            return None
        component_labels = model.predict(shots_combined)
        return np.array([mapping.get(int(c), int(c)) for c in component_labels])

    return None


@workflow.task
def calculate_assignment_fidelities(
    qubits: QuantumElements,
    assignment_matrices: dict[str, None],
) -> dict[str, float]:
    """Calculate the correct assignment fidelity from the correct assignment matrices.

    Arguments:
        qubits:
            The qubits on which to run the analysis. May be either a single qubit or
            a list of qubits. The UIDs of these qubits must exist in the
            assignment_matrices.
        assignment_matrices: the dictionary of assignment matrices returned by
            calculate_assignment_matrices.

    Returns:
        dict with qubit UIDs as keys and the assignment fidelity for each qubit as keys.
    """
    qubits = validate_and_convert_qubits_sweeps(qubits)
    assignment_fidelities = {}
    for q in qubits:
        if q.uid not in assignment_matrices:
            continue

        assigm_mtx = assignment_matrices[q.uid]
        assignment_fidelities[q.uid] = np.trace(assigm_mtx) / float(np.sum(assigm_mtx))

    return assignment_fidelities


@workflow.task
def extract_discrimination_thresholds(
    qubits: QuantumElements,
    states: Sequence[str],
    processed_data_dict: dict[str, dict[str, ArrayLike | dict]],
) -> dict[str, dict[str, float | str]]:
    """Estimate binary (g/e) discrimination thresholds from integrated IQ shots.

    The threshold is chosen on the real axis to maximize balanced assignment
    accuracy between prepared |g> and |e> shots.
    """
    qubits = validate_and_convert_qubits_sweeps(qubits)
    thresholds: dict[str, dict[str, float | str]] = {}

    if "g" not in states or "e" not in states:
        workflow.log(
            logging.WARNING,
            "Threshold extraction requires both 'g' and 'e' states. "
            "Skipping discrimination-threshold estimation.",
        )
        return thresholds

    for q in qubits:
        shots_per_state = processed_data_dict[q.uid].get("shots_per_state", {})
        if "g" not in shots_per_state or "e" not in shots_per_state:
            workflow.log(
                logging.WARNING,
                "Missing g/e shots for %s. Skipping threshold extraction.",
                q.uid,
            )
            continue

        shots_g = np.asarray(shots_per_state["g"]).reshape(-1)
        shots_e = np.asarray(shots_per_state["e"]).reshape(-1)
        if shots_g.size == 0 or shots_e.size == 0:
            workflow.log(
                logging.WARNING,
                "Empty g/e shots for %s. Skipping threshold extraction.",
                q.uid,
            )
            continue

        if (
            np.all(np.isfinite(np.real(shots_g)))
            and np.all(np.isfinite(np.real(shots_e)))
            and np.allclose(np.imag(shots_g), 0.0)
            and np.allclose(np.imag(shots_e), 0.0)
            and np.all(np.isin(np.unique(np.real(shots_g)), [0.0, 1.0]))
            and np.all(np.isin(np.unique(np.real(shots_e)), [0.0, 1.0]))
        ):
            workflow.log(
                logging.WARNING,
                "Data for %s appears already discriminated (0/1). "
                "Cannot calibrate an IQ threshold from these shots.",
                q.uid,
            )
            continue

        real_g = np.real(shots_g).astype(float)
        real_e = np.real(shots_e).astype(float)
        threshold, high_state, balanced_accuracy = _best_binary_threshold_real_axis(
            real_g, real_e
        )
        thresholds[q.uid] = {
            "threshold": float(threshold),
            "high_state": high_state,
            "balanced_accuracy": float(balanced_accuracy),
            "mean_g": float(np.mean(real_g)),
            "mean_e": float(np.mean(real_e)),
        }
        workflow.log(
            logging.INFO,
            "Estimated threshold for %s: %.6g (high values -> %s, balanced accuracy %.4f).",
            q.uid,
            threshold,
            high_state,
            balanced_accuracy,
        )
        if high_state != "e":
            workflow.log(
                logging.WARNING,
                "For %s, larger real integrated values map to |g> than |e>. "
                "Hardware 0/1 labels may be inverted unless readout phase/sign is adjusted.",
                q.uid,
            )

    return thresholds


def _best_binary_threshold_real_axis(
    real_g: np.ndarray, real_e: np.ndarray
) -> tuple[float, str, float]:
    """Return threshold and polarity that maximize balanced g/e assignment."""
    values = np.unique(np.concatenate([real_g, real_e]))
    if len(values) == 1:
        threshold = float(values[0])
        score_e_high = 0.5 * (
            np.mean(real_g < threshold) + np.mean(real_e >= threshold)
        )
        score_g_high = 0.5 * (
            np.mean(real_g >= threshold) + np.mean(real_e < threshold)
        )
        if score_e_high >= score_g_high:
            return threshold, "e", float(score_e_high)
        return threshold, "g", float(score_g_high)

    mids = 0.5 * (values[:-1] + values[1:])
    eps = max(1e-12, 1e-12 * float(np.max(np.abs(values))))
    candidates = np.concatenate(([values[0] - eps], mids, [values[-1] + eps]))

    g_ge = real_g[:, np.newaxis] >= candidates[np.newaxis, :]
    e_ge = real_e[:, np.newaxis] >= candidates[np.newaxis, :]
    score_e_high = 0.5 * (np.mean(~g_ge, axis=0) + np.mean(e_ge, axis=0))
    score_g_high = 0.5 * (np.mean(g_ge, axis=0) + np.mean(~e_ge, axis=0))

    idx_e_high = int(np.argmax(score_e_high))
    idx_g_high = int(np.argmax(score_g_high))
    if score_e_high[idx_e_high] >= score_g_high[idx_g_high]:
        return (
            float(candidates[idx_e_high]),
            "e",
            float(score_e_high[idx_e_high]),
        )
    return (
        float(candidates[idx_g_high]),
        "g",
        float(score_g_high[idx_g_high]),
    )


@workflow.task
def extract_qubit_parameters_for_discrimination(
    qubits: QuantumElements,
    discrimination_thresholds: dict[str, dict[str, float | str]],
    enforce_constant_kernel: bool = True,
) -> dict[str, dict[str, dict[str, float | list[float] | str | None]]]:
    """Build qubit-parameter updates for threshold-based DISCRIMINATION readout."""
    qubits = validate_and_convert_qubits_sweeps(qubits)
    qubit_parameters = {
        "old_parameter_values": {q.uid: {} for q in qubits},
        "new_parameter_values": {q.uid: {} for q in qubits},
    }

    for q in qubits:
        qubit_parameters["old_parameter_values"][q.uid] = {
            "readout_integration_discrimination_thresholds": (
                q.parameters.readout_integration_discrimination_thresholds
            ),
            "readout_integration_kernels_type": (
                q.parameters.readout_integration_kernels_type
            ),
            "readout_integration_kernels": q.parameters.readout_integration_kernels,
        }

        threshold_info = discrimination_thresholds.get(q.uid)
        if threshold_info is None:
            continue

        threshold = float(threshold_info["threshold"])
        qubit_parameters["new_parameter_values"][q.uid][
            "readout_integration_discrimination_thresholds"
        ] = [threshold]
        if enforce_constant_kernel:
            qubit_parameters["new_parameter_values"][q.uid][
                "readout_integration_kernels_type"
            ] = "default"
            qubit_parameters["new_parameter_values"][q.uid][
                "readout_integration_kernels"
            ] = None

    return qubit_parameters


@workflow.task
def assemble_analysis_output(
    assignment_fidelities: dict[str, float] | None,
    discrimination_thresholds: dict[str, dict[str, float | str]],
    qubit_parameters: dict,
) -> dict:
    """Assemble final analysis output with concrete values."""
    return {
        "assignment_fidelities": assignment_fidelities,
        "discrimination_thresholds": discrimination_thresholds,
        "qubit_parameters": qubit_parameters,
    }


@workflow.task
def plot_iq_blobs(
    qubits: QuantumElements,
    states: Sequence[str],
    processed_data_dict: dict[str, dict[str, ArrayLike | dict]],
    fit_results: dict[str, None] | None,
    options: BasePlottingOptions | None = None,
) -> dict[str, mpl.figure.Figure]:
    """Create the IQ-blobs plots.

    Arguments:
        qubits:
            The qubits on which to run the analysis. May be either a single qubit or
            a list of qubits. The UIDs of these qubits must exist in
            processed_data_dict and fit_results.
        states:
            The basis states the qubits should be prepared in. May be either a string,
            e.g. "gef", or a list of letters, e.g. ["g","e","f"].
        processed_data_dict: the processed data dictionary returned by collect_shots.
        fit_results: the classification fit results returned by fit_data.
        options:
            The options class for this task as an instance of [BasePlottingOptions]. See
            the docstring of this class for accepted options.

    Returns:
        dict with qubit UIDs as keys and the figures for each qubit as values.

        If a qubit uid is not found in fit_results, the fit is not plotted.
    """
    opts = BasePlottingOptions() if options is None else options
    qubits = validate_and_convert_qubits_sweeps(qubits)
    figures = {}
    for q in qubits:
        shots_per_state = processed_data_dict[q.uid]["shots_per_state"]
        shots_combined = processed_data_dict[q.uid]["shots_combined"]
        state_colors = {"g": "b", "e": "r", "f": "g"}

        fig, (ax, ax_proj) = plt.subplots(
            1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [2, 1]}
        )
        ax.set_title(timestamped_title(f"IQ Blobs {q.uid}"))
        ax.set_xlabel("Real Signal Component, $V_{\\mathrm{I}}$ (a.u.)")
        ax.set_ylabel("Imaginary Signal Component, $V_{\\mathrm{Q}}$ (a.u.)")
        ax.set_aspect("equal", adjustable="box")

        for i, s in enumerate(states):
            state_shots = shots_per_state[s]
            color = state_colors.get(s, f"C{i}")
            # plot shots
            ax.scatter(
                np.real(state_shots),
                np.imag(state_shots),
                c=color,
                alpha=0.25,
                label=s,
            )
            # plot mean point
            mean_state = np.mean(state_shots)
            ax.plot(np.real(mean_state), np.imag(mean_state), "o", mfc=color, mec="k")

        axis_vec = _decision_axis_vector(fit_results.get(q.uid) if fit_results else None)
        proj_min, proj_max = np.inf, -np.inf
        for i, s in enumerate(states):
            state_shots = shots_per_state[s]
            color = state_colors.get(s, f"C{i}")
            proj_data = np.column_stack([np.real(state_shots), np.imag(state_shots)])
            proj = proj_data @ axis_vec
            proj_min = min(proj_min, np.min(proj))
            proj_max = max(proj_max, np.max(proj))
            ax_proj.hist(proj, bins=50, density=True, alpha=0.35, color=color, label=s)
        ax_proj.set_title("Decision-axis projection")
        ax_proj.set_xlabel("Projection (a.u.)")

        if len(states) > 1 and fit_results is not None and q.uid in fit_results:
            clf_result = fit_results[q.uid]
            if clf_result.get("method") == "lda":
                clf = clf_result["model"]
                levels = None if len(states) > 2 else [0.5]  # noqa: PLR2004
                DecisionBoundaryDisplay.from_estimator(
                    clf,
                    shots_combined,
                    grid_resolution=500,
                    plot_method="contour",
                    ax=ax,
                    eps=1e-1,
                    levels=levels,
                    colors="k",
                    linestyles="--",
                    linewidths=1.5,
                )
            elif clf_result.get("method") == "gmm":
                _plot_gmm_boundaries(ax, clf_result["model"], shots_combined, clf_result.get("mapping"))
            _plot_projection_model(
                ax_proj,
                axis_vec,
                clf_result,
                state_colors,
                proj_min,
                proj_max,
            )

        ax.legend(frameon=False)
        ax_proj.legend(frameon=False)
        fig.tight_layout()

        if opts.save_figures:
            workflow.save_artifact(f"IQ_blobs_{q.uid}", fig)

        if opts.close_figures:
            plt.close(fig)

        figures[q.uid] = fig

    return figures


def _plot_gmm_boundaries(
    ax: "mpl.axes.Axes",
    gmm: GaussianMixture,
    shots_combined: ArrayLike,
    mapping: dict[int, int] | None,
) -> None:
    """Plot GMM decision boundary and covariance ellipses."""
    x, y = shots_combined[:, 0], shots_combined[:, 1]
    margin = max(1e-3, 0.2 * max(np.ptp(x), np.ptp(y)))
    grid_x = np.linspace(np.min(x) - margin, np.max(x) + margin, 300)
    grid_y = np.linspace(np.min(y) - margin, np.max(y) + margin, 300)
    xx, yy = np.meshgrid(grid_x, grid_y)
    grid = np.column_stack([xx.ravel(), yy.ravel()])

    probs = gmm.predict_proba(grid)
    g_prob = np.zeros(len(grid))
    e_prob = np.zeros(len(grid))
    for comp_idx in range(probs.shape[1]):
        state_label = mapping.get(comp_idx) if mapping else comp_idx
        if state_label == 0:
            g_prob += probs[:, comp_idx]
        elif state_label == 1:
            e_prob += probs[:, comp_idx]
    delta = g_prob - e_prob
    ax.contour(
        xx,
        yy,
        delta.reshape(xx.shape),
        levels=[0.0],
        colors="k",
        linestyles="--",
        linewidths=1.8,
    )

    color_map = {0: "b", 1: "r", 2: "g"}
    for i, (mean, cov) in enumerate(zip(gmm.means_, gmm.covariances_)):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        color = color_map.get(mapping.get(i) if mapping else i, f"C{i}")
        for scale in (1.0, 2.0):
            width, height = 2 * scale * np.sqrt(vals)
            ell = Ellipse(
                xy=mean,
                width=width,
                height=height,
                angle=angle,
                edgecolor=color,
                facecolor="none",
                linestyle=":",
                linewidth=1.2,
                alpha=0.9,
            )
            ax.add_patch(ell)


def _decision_axis_vector(fit_result: dict | None) -> np.ndarray:
    """Return a 2D unit vector for the decision-axis projection."""
    default_axis = np.array([1.0, 0.0])
    if not fit_result or "method" not in fit_result:
        return default_axis

    method = fit_result.get("method")
    model = fit_result.get("model")

    if method == "lda":
        w = np.array(model.coef_[0])
    elif method == "gmm":
        means = model.means_
        mapping = fit_result.get("mapping") or {}
        g_idx = None
        e_idx = None
        for comp_idx, state_label in mapping.items():
            if state_label == 0:
                g_idx = comp_idx
            elif state_label == 1:
                e_idx = comp_idx
        if g_idx is not None and e_idx is not None:
            w = means[e_idx] - means[g_idx]
        else:
            w = means[1] - means[0]
    else:
        return default_axis

    norm = np.linalg.norm(w)
    return w / norm if norm > 0 else default_axis


def _plot_projection_model(
    ax: "mpl.axes.Axes",
    axis_vec: np.ndarray,
    fit_result: dict | None,
    state_colors: dict[str, str],
    proj_min: float,
    proj_max: float,
) -> None:
    """Overlay 1D model PDFs on the projection histogram."""
    if not fit_result or proj_min == np.inf or proj_max == -np.inf:
        return

    method = fit_result.get("method")
    model = fit_result.get("model")
    if method != "gmm":
        return  # only GMM has a clear Gaussian projection model here

    mapping = fit_result.get("mapping") or {}
    covs = model.covariances_
    means = model.means_
    span = proj_max - proj_min
    grid = np.linspace(proj_min - 0.1 * span, proj_max + 0.1 * span, 400)

    for comp_idx, state_label in mapping.items():
        mean_proj = axis_vec @ means[comp_idx]
        var_proj = axis_vec.T @ covs[comp_idx] @ axis_vec
        std_proj = float(np.sqrt(max(var_proj, 1e-12)))
        pdf = (
            1
            / (np.sqrt(2 * np.pi) * std_proj)
            * np.exp(-0.5 * ((grid - mean_proj) / std_proj) ** 2)
        )
        label_char = {0: "g", 1: "e", 2: "f"}.get(state_label, str(state_label))
        color = state_colors.get(label_char, None)
        ax.plot(grid, pdf, color=color, linewidth=1.8, label=f"{label_char}-fit")


@workflow.task
def plot_assignment_matrices(
    qubits: QuantumElements,
    states: Sequence[str],
    assignment_matrices: dict[str, ArrayLike],
    assignment_fidelities: dict[str, float],
    options: BasePlottingOptions | None = None,
) -> dict[str, mpl.figure.Figure]:
    """Create the correct-assignment-matrices plots.

    Arguments:
        qubits:
            The qubits on which to run the analysis. May be either a single qubit or
            a list of qubits. The UIDs of these qubits must exist in
            assignment_matrices and assignment_fidelities.
        states:
            The basis states the qubits should be prepared in. May be either a string,
            e.g. "gef", or a list of letters, e.g. ["g","e","f"].
        assignment_matrices: the dictionary of assignment matrices returned by
            calculate_assignment_matrices.
        assignment_fidelities: the dictionary of assignment fidelities returned by
            calculate_assignment_matrices.
        options:
            The options class for this task as an instance of [BasePlottingOptions]. See
            the docstring of this class for accepted options.

    Returns:
        dict with qubit UIDs as keys and the figures for each qubit as values.
    """
    opts = BasePlottingOptions() if options is None else options
    qubits = validate_and_convert_qubits_sweeps(qubits)
    figures = {}
    for q in qubits:
        if q.uid not in assignment_matrices:
            figures[q.uid] = None
            continue

        assignm_mtx = assignment_matrices[q.uid]

        fig, ax = plt.subplots()
        ax.set_ylabel("Prepared State")
        ax.set_xlabel(
            f"Assigned State\n$F_{{avg}}$ = {assignment_fidelities[q.uid] * 100:0.2f}%"
            if q.uid in assignment_fidelities
            else ""
        )
        ax.set_title(timestamped_title(f"Assignment matrix {q.uid}"))

        cmap = plt.get_cmap("Reds")
        im = ax.imshow(
            assignm_mtx,
            interpolation="nearest",
            cmap=cmap,
            norm=mc.LogNorm(vmin=5e-3, vmax=1.0),
        )
        cb = fig.colorbar(im)
        cb.set_label("Assignment Probability, $P$")

        target_names = ["$|g\\rangle$", "$|e\\rangle$"]
        if "f" in states:
            target_names += ["$|f\\rangle$"]
        tick_marks = np.arange(len(target_names))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(target_names)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(target_names)

        thresh = assignm_mtx.max() / 1.5
        for i, j in product(range(assignm_mtx.shape[0]), range(assignm_mtx.shape[1])):
            ax.text(
                j,
                i,
                f"{assignm_mtx[i, j]:0.4f}",
                horizontalalignment="center",
                color="white" if assignm_mtx[i, j] > thresh else "black",
                fontsize=plt.rcParams["font.size"] + 2,
            )

        if opts.save_figures:
            workflow.save_artifact(f"Assignment_matrix_{q.uid}", fig)

        if opts.close_figures:
            plt.close(fig)

        figures[q.uid] = fig

    return figures

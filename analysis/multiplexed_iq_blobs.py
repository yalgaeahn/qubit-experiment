"""This module defines the analysis for 2Q multiplexed IQ blob threshold calibration."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from laboneq import workflow
from sklearn.metrics import confusion_matrix

from experiments.multiplexed_iq_blobs_common import (
    OUTCOME_LABELS_2Q,
    PREPARED_STATES_2Q,
    multiplexed_iq_blob_handle,
)
from laboneq_applications.core.validation import validate_result

if TYPE_CHECKING:
    import matplotlib as mpl
    from laboneq.workflow.tasks.run_experiment import RunExperimentResults


@workflow.workflow_options
class MultiplexedIQBlobAnalysisWorkflowOptions:
    """Options for multiplexed IQ blob analysis."""

    do_plotting: bool = workflow.option_field(
        True,
        description="Whether to create threshold and confusion-matrix plots.",
    )


@workflow.workflow(name="analysis_multiplexed_iq_blobs")
def analysis_workflow(
    result: RunExperimentResults,
    ctrl,
    targ,
    options: MultiplexedIQBlobAnalysisWorkflowOptions | None = None,
) -> None:
    """Run threshold extraction and assignment analysis for 2Q multiplexed IQ blobs."""
    options = (
        MultiplexedIQBlobAnalysisWorkflowOptions() if options is None else options
    )
    processed = collect_shots(result=result, ctrl_uid=ctrl.uid, targ_uid=targ.uid)
    thresholds = extract_thresholds(
        processed_data=processed, ctrl_uid=ctrl.uid, targ_uid=targ.uid
    )
    assignment = calculate_assignment_matrices(
        processed_data=processed,
        thresholds=thresholds,
        ctrl_uid=ctrl.uid,
        targ_uid=targ.uid,
    )
    parameters = extract_qubit_parameters_for_discrimination(
        ctrl=ctrl,
        targ=targ,
        thresholds=thresholds,
    )

    with workflow.if_(options.do_plotting):
        plot_projection_histograms(
            processed_data=processed,
            thresholds=thresholds,
            ctrl_uid=ctrl.uid,
            targ_uid=targ.uid,
        )
        plot_assignment_matrices(
            assignment=assignment,
            ctrl_uid=ctrl.uid,
            targ_uid=targ.uid,
        )

    workflow.return_(
        {
            "thresholds": thresholds,
            "assignment": assignment,
            "qubit_parameters": parameters,
        }
    )


def _unwrap_result_like(result_like):
    current = result_like
    for _ in range(8):
        if current is None:
            return None
        if hasattr(current, "output"):
            current = current.output
            continue
        if isinstance(current, dict):
            if "result" in current:
                current = current["result"]
                continue
        return current
    return current


@workflow.task
def collect_shots(
    result: RunExperimentResults,
    ctrl_uid: str,
    targ_uid: str,
) -> dict[str, dict[str, list[float]]]:
    """Collect complex integrated single-shot IQ data per prepared 2Q basis state."""
    result = _unwrap_result_like(result)
    validate_result(result)

    payload: dict[str, dict[str, list[float]]] = {
        ctrl_uid: {},
        targ_uid: {},
    }

    for prepared_label, _states in PREPARED_STATES_2Q:
        ctrl_data = np.asarray(
            result[multiplexed_iq_blob_handle(ctrl_uid, prepared_label)].data
        ).reshape(-1)
        targ_data = np.asarray(
            result[multiplexed_iq_blob_handle(targ_uid, prepared_label)].data
        ).reshape(-1)
        payload[ctrl_uid][prepared_label] = ctrl_data.tolist()
        payload[targ_uid][prepared_label] = targ_data.tolist()

    return payload


def _best_binary_threshold_real_axis(
    real_g: np.ndarray, real_e: np.ndarray
) -> tuple[float, str, float]:
    """Return threshold and polarity maximizing balanced g/e assignment."""
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

    candidates = 0.5 * (values[:-1] + values[1:])
    best_threshold = float(candidates[0])
    best_polarity = "e"
    best_score = -np.inf

    for thr in candidates:
        score_e_high = 0.5 * (np.mean(real_g < thr) + np.mean(real_e >= thr))
        if score_e_high > best_score:
            best_score = float(score_e_high)
            best_threshold = float(thr)
            best_polarity = "e"

        score_g_high = 0.5 * (np.mean(real_g >= thr) + np.mean(real_e < thr))
        if score_g_high > best_score:
            best_score = float(score_g_high)
            best_threshold = float(thr)
            best_polarity = "g"

    return best_threshold, best_polarity, best_score


def _qubit_state_from_prepared_label(prepared_label: str, which: str) -> int:
    if which == "ctrl":
        return int(prepared_label[0])
    return int(prepared_label[1])


def _predict_bits(
    shots: np.ndarray, axis: complex, threshold: float, high_state: str
) -> np.ndarray:
    projection = np.real(shots * np.conjugate(axis))
    if high_state == "e":
        return (projection >= threshold).astype(int)
    return (projection < threshold).astype(int)


@workflow.task
def extract_thresholds(
    processed_data: dict[str, dict[str, list[float]]],
    ctrl_uid: str,
    targ_uid: str,
) -> dict[str, dict[str, float | str]]:
    """Extract per-qubit discrimination thresholds from multiplexed g/e grouped shots."""
    out: dict[str, dict[str, float | str]] = {}

    for uid, which in ((ctrl_uid, "ctrl"), (targ_uid, "targ")):
        g_labels = [l for l, _ in PREPARED_STATES_2Q if _qubit_state_from_prepared_label(l, which) == 0]
        e_labels = [l for l, _ in PREPARED_STATES_2Q if _qubit_state_from_prepared_label(l, which) == 1]

        shots_g = np.concatenate(
            [np.asarray(processed_data[uid][label], dtype=complex) for label in g_labels]
        )
        shots_e = np.concatenate(
            [np.asarray(processed_data[uid][label], dtype=complex) for label in e_labels]
        )

        mean_g = np.mean(shots_g)
        mean_e = np.mean(shots_e)
        axis = mean_e - mean_g
        if np.abs(axis) < 1e-12:
            axis = 1.0 + 0j
        axis_unit = axis / np.abs(axis)

        real_g = np.real(shots_g * np.conjugate(axis_unit))
        real_e = np.real(shots_e * np.conjugate(axis_unit))
        threshold, high_state, balanced_accuracy = _best_binary_threshold_real_axis(
            real_g, real_e
        )

        out[uid] = {
            "threshold": float(threshold),
            "high_state": str(high_state),
            "balanced_accuracy": float(balanced_accuracy),
            "axis_real": float(np.real(axis_unit)),
            "axis_imag": float(np.imag(axis_unit)),
        }

    return out


@workflow.task
def calculate_assignment_matrices(
    processed_data: dict[str, dict[str, list[float]]],
    thresholds: dict[str, dict[str, float | str]],
    ctrl_uid: str,
    targ_uid: str,
) -> dict[str, list]:
    """Calculate per-qubit 2x2 and joint 4x4 assignment matrices."""
    out: dict[str, list] = {}

    per_qubit_true = {ctrl_uid: [], targ_uid: []}
    per_qubit_pred = {ctrl_uid: [], targ_uid: []}
    joint_true = []
    joint_pred = []

    ctrl_axis = complex(
        thresholds[ctrl_uid]["axis_real"], thresholds[ctrl_uid]["axis_imag"]
    )
    targ_axis = complex(
        thresholds[targ_uid]["axis_real"], thresholds[targ_uid]["axis_imag"]
    )
    ctrl_thr = float(thresholds[ctrl_uid]["threshold"])
    targ_thr = float(thresholds[targ_uid]["threshold"])
    ctrl_high = str(thresholds[ctrl_uid]["high_state"])
    targ_high = str(thresholds[targ_uid]["high_state"])

    for prepared_idx, (prepared_label, _states) in enumerate(PREPARED_STATES_2Q):
        ctrl_shots = np.asarray(processed_data[ctrl_uid][prepared_label], dtype=complex)
        targ_shots = np.asarray(processed_data[targ_uid][prepared_label], dtype=complex)
        n = int(min(len(ctrl_shots), len(targ_shots)))
        if n < 1:
            continue

        ctrl_pred_bits = _predict_bits(ctrl_shots[:n], ctrl_axis, ctrl_thr, ctrl_high)
        targ_pred_bits = _predict_bits(targ_shots[:n], targ_axis, targ_thr, targ_high)
        ctrl_true_bit = _qubit_state_from_prepared_label(prepared_label, "ctrl")
        targ_true_bit = _qubit_state_from_prepared_label(prepared_label, "targ")

        per_qubit_true[ctrl_uid].extend([ctrl_true_bit] * n)
        per_qubit_true[targ_uid].extend([targ_true_bit] * n)
        per_qubit_pred[ctrl_uid].extend(ctrl_pred_bits.tolist())
        per_qubit_pred[targ_uid].extend(targ_pred_bits.tolist())

        joint_true.extend([prepared_idx] * n)
        joint_pred.extend((2 * ctrl_pred_bits + targ_pred_bits).tolist())

    ctrl_matrix = confusion_matrix(
        per_qubit_true[ctrl_uid],
        per_qubit_pred[ctrl_uid],
        labels=[0, 1],
        normalize="true",
    )
    targ_matrix = confusion_matrix(
        per_qubit_true[targ_uid],
        per_qubit_pred[targ_uid],
        labels=[0, 1],
        normalize="true",
    )
    joint_matrix = confusion_matrix(
        joint_true,
        joint_pred,
        labels=[0, 1, 2, 3],
        normalize="true",
    )

    out["per_qubit_matrices"] = {
        ctrl_uid: ctrl_matrix.tolist(),
        targ_uid: targ_matrix.tolist(),
    }
    out["joint_matrix"] = joint_matrix.tolist()
    out["per_qubit_fidelity"] = {
        ctrl_uid: float(np.trace(ctrl_matrix) / np.sum(ctrl_matrix)),
        targ_uid: float(np.trace(targ_matrix) / np.sum(targ_matrix)),
    }
    out["joint_fidelity"] = float(np.trace(joint_matrix) / np.sum(joint_matrix))
    return out


@workflow.task
def extract_qubit_parameters_for_discrimination(
    ctrl,
    targ,
    thresholds: dict[str, dict[str, float | str]],
) -> dict[str, dict[str, dict[str, list[float] | None]]]:
    """Build parameter update payload for discrimination thresholds."""
    qubits = [ctrl, targ]
    out = {
        "old_parameter_values": {q.uid: {} for q in qubits},
        "new_parameter_values": {q.uid: {} for q in qubits},
    }
    for q in qubits:
        out["old_parameter_values"][q.uid] = {
            "readout_integration_discrimination_thresholds": (
                q.parameters.readout_integration_discrimination_thresholds
            )
        }
        out["new_parameter_values"][q.uid] = {
            "readout_integration_discrimination_thresholds": [
                float(thresholds[q.uid]["threshold"])
            ]
        }
    return out


@workflow.task
def plot_projection_histograms(
    processed_data: dict[str, dict[str, list[float]]],
    thresholds: dict[str, dict[str, float | str]],
    ctrl_uid: str,
    targ_uid: str,
) -> dict[str, mpl.figure.Figure]:
    """Plot projection histograms and threshold lines for ctrl/targ."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, uid, which in (
        (axes[0], ctrl_uid, "ctrl"),
        (axes[1], targ_uid, "targ"),
    ):
        g_labels = [l for l, _ in PREPARED_STATES_2Q if _qubit_state_from_prepared_label(l, which) == 0]
        e_labels = [l for l, _ in PREPARED_STATES_2Q if _qubit_state_from_prepared_label(l, which) == 1]
        shots_g = np.concatenate(
            [np.asarray(processed_data[uid][label], dtype=complex) for label in g_labels]
        )
        shots_e = np.concatenate(
            [np.asarray(processed_data[uid][label], dtype=complex) for label in e_labels]
        )
        axis = complex(thresholds[uid]["axis_real"], thresholds[uid]["axis_imag"])
        proj_g = np.real(shots_g * np.conjugate(axis))
        proj_e = np.real(shots_e * np.conjugate(axis))
        thr = float(thresholds[uid]["threshold"])
        ax.hist(proj_g, bins=80, alpha=0.4, density=True, label="g-group")
        ax.hist(proj_e, bins=80, alpha=0.4, density=True, label="e-group")
        ax.axvline(thr, color="k", linestyle="--", label=f"thr={thr:.4g}")
        ax.set_title(f"{uid} projection")
        ax.set_xlabel("Projected IQ")
        ax.legend(frameon=False)

    fig.tight_layout()
    workflow.save_artifact("multiplexed_iq_blob_threshold_hist", fig)
    return {"projection_hist": fig}


@workflow.task
def plot_assignment_matrices(
    assignment: dict[str, list],
    ctrl_uid: str,
    targ_uid: str,
) -> dict[str, mpl.figure.Figure]:
    """Plot ctrl/targ 2x2 and joint 4x4 assignment matrices."""
    ctrl_matrix = np.asarray(assignment["per_qubit_matrices"][ctrl_uid], dtype=float)
    targ_matrix = np.asarray(assignment["per_qubit_matrices"][targ_uid], dtype=float)
    joint_matrix = np.asarray(assignment["joint_matrix"], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    panels = [
        (axes[0], ctrl_matrix, [0, 1], f"{ctrl_uid} assignment"),
        (axes[1], targ_matrix, [0, 1], f"{targ_uid} assignment"),
        (axes[2], joint_matrix, OUTCOME_LABELS_2Q, "joint assignment"),
    ]
    for ax, mat, ticks, title in panels:
        im = ax.imshow(mat, vmin=0.0, vmax=1.0, cmap="Reds")
        ax.set_title(title)
        ax.set_xlabel("predicted")
        ax.set_ylabel("prepared")
        ax.set_xticks(range(len(ticks)), ticks)
        ax.set_yticks(range(len(ticks)), ticks)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                ax.text(
                    j,
                    i,
                    f"{val:.3f}",
                    ha="center",
                    va="center",
                    color="white" if val > 0.6 else "black",
                )
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.tight_layout()
    workflow.save_artifact("multiplexed_iq_blob_assignment", fig)
    return {"assignment_matrices": fig}

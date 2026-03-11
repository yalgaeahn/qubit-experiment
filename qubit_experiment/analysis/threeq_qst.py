"""Canonical three-qubit QST analysis under the `threeq_qst` name.

This workflow keeps the INTEGRATION + SINGLE_SHOT contract and returns a
single plain analysis payload for one tomography run. Convergence and shot
sweep helpers are re-exported for the split experiment workflows.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from laboneq import workflow

from .plot_theme import with_plot_theme
from laboneq_applications.core.validation import validate_result

from qubit_experiment.experiments.three_qubit_tomography_common import (
    OUTCOME_LABELS,
    READOUT_CALIBRATION_STATES,
    TOMOGRAPHY_SETTINGS,
    canonical_three_qubit_state_label,
    readout_calibration_handle,
    tomography_handle,
)

if TYPE_CHECKING:
    import matplotlib as mpl
    from laboneq.workflow.tasks.run_experiment import RunExperimentResults


def _complex_to_features(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z).reshape(-1)
    return np.column_stack([np.real(z), np.imag(z)])

def _fit_binary_gaussian_discriminator(
    samples_g: np.ndarray,
    samples_e: np.ndarray,
) -> dict[str, np.ndarray | float]:
    x_g = _complex_to_features(samples_g)
    x_e = _complex_to_features(samples_e)
    if x_g.shape[0] < 2 or x_e.shape[0] < 2:
        raise ValueError("Need at least 2 samples per class to fit discriminator.")

    mu_g = np.mean(x_g, axis=0)
    mu_e = np.mean(x_e, axis=0)
    cov_g = np.cov(x_g, rowvar=False)
    cov_e = np.cov(x_e, rowvar=False)
    pooled = 0.5 * (cov_g + cov_e)
    ridge = 1e-9 * max(float(np.trace(pooled) / 2.0), 1.0)
    sigma = pooled + ridge * np.eye(2)
    sigma_inv = np.linalg.inv(sigma)

    delta = mu_e - mu_g
    w = sigma_inv @ delta
    b = 0.5 * float((mu_e + mu_g) @ w)

    return {
        "w": w,
        "b": b,
        "mu_g": mu_g,
        "mu_e": mu_e,
        "sigma": sigma,
        "ridge": ridge,
    }

def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    pos = x >= 0
    out = np.empty_like(x)
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out

def _predict_p_e(samples: np.ndarray, model: dict) -> np.ndarray:
    x = _complex_to_features(samples)
    logits = x @ np.asarray(model["w"], dtype=float) - float(model["b"])
    return _sigmoid(logits)

def _unwrap_result_like(result_like):
    """Unwrap workflow/task wrappers to reach RunExperimentResults-like payload."""
    current = result_like
    for _ in range(8):
        if current is None:
            return None
        if hasattr(current, "output"):
            current = current.output
            continue
        if isinstance(current, dict):
            if "readout_calibration_result" in current:
                current = current["readout_calibration_result"]
                continue
            if "tomography_result" in current:
                current = current["tomography_result"]
                continue
        return current
    return current

def _collect_calibration_training_sets(
    readout_calibration_result: RunExperimentResults,
    q0_uid: str,
    q1_uid: str,
    q2_uid: str,
) -> dict[str, np.ndarray]:
    q0_by_state = {"g": [], "e": []}
    q1_by_state = {"g": [], "e": []}
    q2_by_state = {"g": [], "e": []}

    for prepared_label, (q0_state, q1_state, q2_state) in READOUT_CALIBRATION_STATES:
        q0_shots = np.asarray(
            readout_calibration_result[
                readout_calibration_handle(q0_uid, prepared_label)
            ].data
        ).reshape(-1)
        q1_shots = np.asarray(
            readout_calibration_result[
                readout_calibration_handle(q1_uid, prepared_label)
            ].data
        ).reshape(-1)
        q2_shots = np.asarray(
            readout_calibration_result[
                readout_calibration_handle(q2_uid, prepared_label)
            ].data
        ).reshape(-1)

        nshots = min(len(q0_shots), len(q1_shots), len(q2_shots))
        if nshots == 0:
            continue

        q0_by_state[q0_state].append(q0_shots[:nshots])
        q1_by_state[q1_state].append(q1_shots[:nshots])
        q2_by_state[q2_state].append(q2_shots[:nshots])

    if not all(q0_by_state[s] for s in ("g", "e")):
        raise ValueError("Insufficient q0 calibration samples for g/e states.")
    if not all(q1_by_state[s] for s in ("g", "e")):
        raise ValueError("Insufficient q1 calibration samples for g/e states.")
    if not all(q2_by_state[s] for s in ("g", "e")):
        raise ValueError("Insufficient q2 calibration samples for g/e states.")

    return {
        "q0_g": np.concatenate(q0_by_state["g"]),
        "q0_e": np.concatenate(q0_by_state["e"]),
        "q1_g": np.concatenate(q1_by_state["g"]),
        "q1_e": np.concatenate(q1_by_state["e"]),
        "q2_g": np.concatenate(q2_by_state["g"]),
        "q2_e": np.concatenate(q2_by_state["e"]),
    }

def _hard_counts_from_posteriors(posteriors: np.ndarray) -> np.ndarray:
    hard_outcomes = np.argmax(posteriors, axis=1)
    return np.bincount(hard_outcomes, minlength=8).astype(int)

def _classification_diagnostics_single_qubit(
    *,
    readout_calibration_result: RunExperimentResults,
    qubit_uid: str,
    model: dict,
    state_index: int,
) -> tuple[list[list[float]], float]:
    state_to_bit = {"g": 0, "e": 1}
    true_labels = []
    pred_labels = []

    for prepared_label, states in READOUT_CALIBRATION_STATES:
        shots = np.asarray(
            readout_calibration_result[
                readout_calibration_handle(qubit_uid, prepared_label)
            ].data
        ).reshape(-1)
        if shots.size == 0:
            continue

        state = states[state_index]
        probs_e = _predict_p_e(shots, model)
        true_labels.append(np.full(shots.size, state_to_bit[state], dtype=int))
        pred_labels.append((probs_e >= 0.5).astype(int))

    true_arr = np.concatenate(true_labels)
    pred_arr = np.concatenate(pred_labels)

    cm = np.zeros((2, 2), dtype=float)
    for t, p in zip(true_arr, pred_arr):
        cm[t, p] += 1
    cm /= np.maximum(np.sum(cm, axis=1, keepdims=True), 1.0)
    accuracy = float(np.mean(true_arr == pred_arr))
    return cm.tolist(), accuracy

@workflow.task
def fit_discriminator_from_readout_calibration(
    readout_calibration_result: RunExperimentResults | None,
    q0_uid: str,
    q1_uid: str,
    q2_uid: str,
) -> dict[str, object]:
    """Fit qubit-wise IQ discriminators from 3Q readout calibration shots."""
    readout_calibration_result = _unwrap_result_like(readout_calibration_result)
    if readout_calibration_result is None:
        raise ValueError(
            "readout_calibration_result is required for INTEGRATION-based tomography analysis."
        )
    validate_result(readout_calibration_result)

    train = _collect_calibration_training_sets(
        readout_calibration_result=readout_calibration_result,
        q0_uid=q0_uid,
        q1_uid=q1_uid,
        q2_uid=q2_uid,
    )
    q0_model = _fit_binary_gaussian_discriminator(train["q0_g"], train["q0_e"])
    q1_model = _fit_binary_gaussian_discriminator(train["q1_g"], train["q1_e"])
    q2_model = _fit_binary_gaussian_discriminator(train["q2_g"], train["q2_e"])

    q0_cm, q0_acc = _classification_diagnostics_single_qubit(
        readout_calibration_result=readout_calibration_result,
        qubit_uid=q0_uid,
        model=q0_model,
        state_index=0,
    )
    q1_cm, q1_acc = _classification_diagnostics_single_qubit(
        readout_calibration_result=readout_calibration_result,
        qubit_uid=q1_uid,
        model=q1_model,
        state_index=1,
    )
    q2_cm, q2_acc = _classification_diagnostics_single_qubit(
        readout_calibration_result=readout_calibration_result,
        qubit_uid=q2_uid,
        model=q2_model,
        state_index=2,
    )

    diagnostics = {
        "q0_confusion_matrix": q0_cm,
        "q1_confusion_matrix": q1_cm,
        "q2_confusion_matrix": q2_cm,
        "q0_accuracy": q0_acc,
        "q1_accuracy": q1_acc,
        "q2_accuracy": q2_acc,
    }

    def _export(model: dict) -> dict[str, list[float] | float | list[list[float]]]:
        return {
            "w": np.asarray(model["w"], dtype=float).tolist(),
            "b": float(model["b"]),
            "mu_g": np.asarray(model["mu_g"], dtype=float).tolist(),
            "mu_e": np.asarray(model["mu_e"], dtype=float).tolist(),
            "sigma": np.asarray(model["sigma"], dtype=float).tolist(),
        }

    model_export = {
        "q0": _export(q0_model),
        "q1": _export(q1_model),
        "q2": _export(q2_model),
    }
    return {
        "model": model_export,
        "internal": {"q0": q0_model, "q1": q1_model, "q2": q2_model},
        "diagnostics": diagnostics,
    }

def _build_joint_posteriors_3q(
    p_q0_e: np.ndarray,
    p_q1_e: np.ndarray,
    p_q2_e: np.ndarray,
) -> np.ndarray:
    p_q0_g = 1.0 - p_q0_e
    p_q1_g = 1.0 - p_q1_e
    p_q2_g = 1.0 - p_q2_e

    cols: list[np.ndarray] = []
    for label in OUTCOME_LABELS:
        f0 = p_q0_g if label[0] == "0" else p_q0_e
        f1 = p_q1_g if label[1] == "0" else p_q1_e
        f2 = p_q2_g if label[2] == "0" else p_q2_e
        cols.append(f0 * f1 * f2)
    return np.column_stack(cols)

@workflow.task
def collect_tomography_counts(
    tomography_result: RunExperimentResults,
    q0_uid: str,
    q1_uid: str,
    q2_uid: str,
    discriminator: dict[str, object],
) -> dict[str, list]:
    """Collect 8-outcome soft counts per tomography setting from IQ posteriors."""
    tomography_result = _unwrap_result_like(tomography_result)
    validate_result(tomography_result)

    internal = discriminator["internal"]
    q0_model = internal["q0"]
    q1_model = internal["q1"]
    q2_model = internal["q2"]

    counts = []
    counts_hard = []
    shots_per_setting = []
    setting_labels = []

    for setting_label, _axes in TOMOGRAPHY_SETTINGS:
        q0_shots = np.asarray(
            tomography_result[tomography_handle(q0_uid, setting_label)].data
        ).reshape(-1)
        q1_shots = np.asarray(
            tomography_result[tomography_handle(q1_uid, setting_label)].data
        ).reshape(-1)
        q2_shots = np.asarray(
            tomography_result[tomography_handle(q2_uid, setting_label)].data
        ).reshape(-1)

        nshots = min(len(q0_shots), len(q1_shots), len(q2_shots))
        q0_shots = q0_shots[:nshots]
        q1_shots = q1_shots[:nshots]
        q2_shots = q2_shots[:nshots]

        p_q0_e = _predict_p_e(q0_shots, q0_model)
        p_q1_e = _predict_p_e(q1_shots, q1_model)
        p_q2_e = _predict_p_e(q2_shots, q2_model)

        posteriors = _build_joint_posteriors_3q(p_q0_e, p_q1_e, p_q2_e)
        setting_counts = np.sum(posteriors, axis=0)
        setting_counts_hard = _hard_counts_from_posteriors(posteriors)

        counts.append(setting_counts.tolist())
        counts_hard.append(setting_counts_hard.tolist())
        shots_per_setting.append(int(nshots))
        setting_labels.append(setting_label)

    return {
        "counts": counts,
        "counts_hard": counts_hard,
        "shots_per_setting": shots_per_setting,
        "setting_labels": setting_labels,
    }

@workflow.task
def extract_assignment_matrix(
    readout_calibration_result: RunExperimentResults | None,
    q0_uid: str,
    q1_uid: str,
    q2_uid: str,
    discriminator: dict[str, object],
) -> dict[str, list]:
    """Extract 8x8 assignment matrix A_{ik} from IQ posterior probabilities."""
    readout_calibration_result = _unwrap_result_like(readout_calibration_result)
    if readout_calibration_result is None:
        raise ValueError(
            "readout_calibration_result is required for INTEGRATION-based assignment extraction."
        )

    validate_result(readout_calibration_result)
    internal = discriminator["internal"]
    q0_model = internal["q0"]
    q1_model = internal["q1"]
    q2_model = internal["q2"]

    counts_matrix_soft = np.zeros((8, 8), dtype=float)
    counts_matrix_hard = np.zeros((8, 8), dtype=int)
    for k, (prepared_label, _states) in enumerate(READOUT_CALIBRATION_STATES):
        q0_shots = np.asarray(
            readout_calibration_result[
                readout_calibration_handle(q0_uid, prepared_label)
            ].data
        ).reshape(-1)
        q1_shots = np.asarray(
            readout_calibration_result[
                readout_calibration_handle(q1_uid, prepared_label)
            ].data
        ).reshape(-1)
        q2_shots = np.asarray(
            readout_calibration_result[
                readout_calibration_handle(q2_uid, prepared_label)
            ].data
        ).reshape(-1)

        nshots = min(len(q0_shots), len(q1_shots), len(q2_shots))
        q0_shots = q0_shots[:nshots]
        q1_shots = q1_shots[:nshots]
        q2_shots = q2_shots[:nshots]

        p_q0_e = _predict_p_e(q0_shots, q0_model)
        p_q1_e = _predict_p_e(q1_shots, q1_model)
        p_q2_e = _predict_p_e(q2_shots, q2_model)

        posteriors = _build_joint_posteriors_3q(p_q0_e, p_q1_e, p_q2_e)
        counts_matrix_soft[:, k] = np.sum(posteriors, axis=0)
        counts_matrix_hard[:, k] = _hard_counts_from_posteriors(posteriors)

    with np.errstate(invalid="ignore", divide="ignore"):
        assignment_matrix = counts_matrix_soft / np.maximum(
            np.sum(counts_matrix_soft, axis=0, keepdims=True),
            1.0,
        )

    return {
        "assignment_matrix": assignment_matrix.tolist(),
        "counts_matrix_soft": counts_matrix_soft.tolist(),
        "counts_matrix_hard": counts_matrix_hard.tolist(),
    }

def _rotation_x(theta: float) -> np.ndarray:
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)

def _rotation_y(theta: float) -> np.ndarray:
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=complex)

_SINGLE_QUBIT_PREROTATION_UNITARIES = {
    "X": _rotation_y(-np.pi / 2),
    "Y": _rotation_x(np.pi / 2),
    "Z": np.eye(2, dtype=complex),
}

def _single_qubit_prerotation_unitary(axis: str) -> np.ndarray:
    """Return prerotation unitaries consistent with experiment qop mapping."""
    try:
        return _SINGLE_QUBIT_PREROTATION_UNITARIES[axis]
    except KeyError as exc:
        raise ValueError(f"Unsupported axis: {axis!r}.") from exc

def _computational_projectors_3q() -> list[np.ndarray]:
    projectors = []
    for i in range(8):
        ket = np.zeros((8, 1), dtype=complex)
        ket[i, 0] = 1.0
        projectors.append(ket @ ket.conj().T)
    return projectors

def _build_noisy_povm(assignment_matrix: np.ndarray) -> np.ndarray:
    ideal_projectors = _computational_projectors_3q()
    noisy_povm = np.zeros((len(TOMOGRAPHY_SETTINGS), 8, 8, 8), dtype=complex)

    for s_idx, (_label, (q0_axis, q1_axis, q2_axis)) in enumerate(TOMOGRAPHY_SETTINGS):
        u_q0 = _single_qubit_prerotation_unitary(q0_axis)
        u_q1 = _single_qubit_prerotation_unitary(q1_axis)
        u_q2 = _single_qubit_prerotation_unitary(q2_axis)
        u_setting = np.kron(np.kron(u_q0, u_q1), u_q2)
        ideal_povm = [u_setting.conj().T @ p @ u_setting for p in ideal_projectors]

        for i in range(8):
            e_meas = np.zeros((8, 8), dtype=complex)
            for k in range(8):
                e_meas += assignment_matrix[i, k] * ideal_povm[k]
            noisy_povm[s_idx, i] = e_meas

    return noisy_povm

def _theta_to_density_matrix(theta: np.ndarray, dim: int = 8) -> np.ndarray:
    t_mat = np.zeros((dim, dim), dtype=complex)
    idx = 0

    for i in range(dim):
        t_mat[i, i] = np.exp(theta[idx])
        idx += 1

    for i in range(1, dim):
        for j in range(i):
            t_mat[i, j] = theta[idx] + 1j * theta[idx + 1]
            idx += 2

    rho = t_mat.conj().T @ t_mat
    rho /= np.trace(rho)
    return rho

def _predict_probabilities(noisy_povm: np.ndarray, rho: np.ndarray) -> np.ndarray:
    probabilities = np.zeros((noisy_povm.shape[0], noisy_povm.shape[1]), dtype=float)
    for s_idx in range(noisy_povm.shape[0]):
        for i in range(noisy_povm.shape[1]):
            probabilities[s_idx, i] = np.real(np.trace(noisy_povm[s_idx, i] @ rho))
    return np.clip(probabilities, 1e-12, 1.0)

@workflow.task
def maximum_likelihood_reconstruct(
    tomography_counts: dict[str, list],
    assignment: dict[str, list],
    max_iterations: int = 2000,
) -> dict[str, object]:
    """Reconstruct density matrix with readout-mitigated MLE."""
    from scipy.optimize import minimize

    counts = np.asarray(tomography_counts["counts"], dtype=float)
    shots_per_setting = np.asarray(tomography_counts["shots_per_setting"], dtype=float)
    assignment_matrix = np.asarray(assignment["assignment_matrix"], dtype=float)
    noisy_povm = _build_noisy_povm(assignment_matrix)

    def objective(theta: np.ndarray) -> float:
        rho = _theta_to_density_matrix(theta, dim=8)
        probs = _predict_probabilities(noisy_povm, rho)
        return float(-np.sum(counts * np.log(probs)))

    initial_theta = np.zeros(64, dtype=float)
    result = minimize(
        objective,
        initial_theta,
        method="L-BFGS-B",
        options={"maxiter": int(max_iterations)},
    )

    rho_hat = _theta_to_density_matrix(result.x, dim=8)
    predicted_probabilities = _predict_probabilities(noisy_povm, rho_hat)
    predicted_counts = predicted_probabilities * shots_per_setting[:, np.newaxis]

    return {
        "rho_hat_real": rho_hat.real.tolist(),
        "rho_hat_imag": rho_hat.imag.tolist(),
        "predicted_probabilities": predicted_probabilities.tolist(),
        "predicted_counts": predicted_counts.tolist(),
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "negative_log_likelihood": float(result.fun),
    }

def _target_to_density_matrix(target_state) -> np.ndarray | None:
    if target_state is None:
        return None

    if isinstance(target_state, str):
        key = target_state.strip().lower()
        plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
        minus = np.array([1, -1], dtype=complex) / np.sqrt(2)
        single_qubit_statevectors = {
            "0": np.array([1, 0], dtype=complex),
            "1": np.array([0, 1], dtype=complex),
            "+": plus,
            "-": minus,
        }

        try:
            canonical = canonical_three_qubit_state_label(key)
        except ValueError:
            canonical = None

        if canonical is not None:
            psi = np.kron(
                np.kron(
                    single_qubit_statevectors[canonical[0]],
                    single_qubit_statevectors[canonical[1]],
                ),
                single_qubit_statevectors[canonical[2]],
            )
            return np.outer(psi, psi.conj())

        if key in {"ghz", "ghz_plus", "ghz+"}:
            psi = np.zeros(8, dtype=complex)
            psi[0] = 1.0 / np.sqrt(2)
            psi[7] = 1.0 / np.sqrt(2)
            return np.outer(psi, psi.conj())
        if key in {"ghz_minus", "ghz-"}:
            psi = np.zeros(8, dtype=complex)
            psi[0] = 1.0 / np.sqrt(2)
            psi[7] = -1.0 / np.sqrt(2)
            return np.outer(psi, psi.conj())

        raise ValueError(f"Unsupported target_state string: {target_state!r}.")

    arr = np.asarray(target_state, dtype=complex)
    if arr.shape == (8,):
        psi = arr / np.linalg.norm(arr)
        return np.outer(psi, psi.conj())
    if arr.shape == (8, 8):
        rho_target = (arr + arr.conj().T) / 2.0
        tr = np.trace(rho_target)
        return rho_target / tr if tr != 0 else rho_target

    raise ValueError(
        "target_state must be None, a known string, a length-8 statevector, "
        "or an 8x8 density matrix."
    )

def _fidelity(rho: np.ndarray, rho_target: np.ndarray | None) -> float | None:
    if rho_target is None:
        return None

    eigvals, eigvecs = np.linalg.eigh(rho_target)
    rank_one = np.sum(np.abs(eigvals) > 1e-12) == 1
    if rank_one:
        vec = eigvecs[:, np.argmax(eigvals)]
        return float(np.real(np.conjugate(vec) @ rho @ vec))

    from scipy.linalg import sqrtm

    sqrt_target = sqrtm(rho_target)
    fidelity_mat = sqrtm(sqrt_target @ rho @ sqrt_target)
    return float(np.real(np.trace(fidelity_mat) ** 2))

@workflow.task
def calculate_state_metrics(
    rho_hat_real: list[list[float]],
    rho_hat_imag: list[list[float]],
    target_state=None,
) -> dict[str, object]:
    """Calculate derived tomography metrics."""
    rho = np.asarray(rho_hat_real, dtype=float) + 1j * np.asarray(
        rho_hat_imag,
        dtype=float,
    )
    rho = (rho + rho.conj().T) / 2.0
    rho /= np.trace(rho)

    purity = float(np.real(np.trace(rho @ rho)))
    trace = float(np.real(np.trace(rho)))
    min_eigenvalue = float(np.min(np.linalg.eigvalsh(rho)).real)

    pauli_i = np.array([[1, 0], [0, 1]], dtype=complex)
    pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
    pauli_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)
    paulis = {"I": pauli_i, "X": pauli_x, "Y": pauli_y, "Z": pauli_z}

    correlators = {}
    for a in "XYZ":
        for b in "XYZ":
            for c in "XYZ":
                op = np.kron(np.kron(paulis[a], paulis[b]), paulis[c])
                correlators[f"{a}{b}{c}"] = float(np.real(np.trace(op @ rho)))

    rho_target = _target_to_density_matrix(target_state)
    fidelity = _fidelity(rho, rho_target)

    return {
        "trace": trace,
        "purity": purity,
        "min_eigenvalue": min_eigenvalue,
        "fidelity_to_target": fidelity,
        "pauli_correlators": correlators,
    }

@workflow.task
def evaluate_optimization_convergence(
    tomography_counts: dict[str, list],
    predicted_counts: list[list[float]],
    optimizer_success: bool,
    optimizer_message: str,
    negative_log_likelihood: float,
    rho_hat_real: list[list[float]],
    rho_hat_imag: list[list[float]],
) -> dict[str, object]:
    """Compute optimizer convergence diagnostics for one tomography run."""
    observed = np.asarray(tomography_counts.get("counts", []), dtype=float)
    predicted = np.asarray(predicted_counts, dtype=float)
    shots = np.asarray(tomography_counts.get("shots_per_setting", []), dtype=float)
    total_shots = float(np.sum(shots)) if shots.size else float(np.sum(observed))
    nll = float(negative_log_likelihood)
    nll_finite = bool(np.isfinite(nll))
    nll_per_shot = float(nll / total_shots) if nll_finite and total_shots > 0 else None

    if observed.size and predicted.size and observed.shape == predicted.shape:
        abs_err = np.abs(observed - predicted)
        mae_counts = float(np.mean(abs_err))
        max_abs_counts_error = float(np.max(abs_err))
    else:
        mae_counts = None
        max_abs_counts_error = None

    avg_shots = float(np.mean(shots)) if shots.size else None
    if mae_counts is not None and avg_shots is not None and avg_shots > 0:
        normalized_mae = float(mae_counts / avg_shots)
    else:
        normalized_mae = None

    rho = np.asarray(rho_hat_real, dtype=float) + 1j * np.asarray(rho_hat_imag, dtype=float)
    rho = (rho + rho.conj().T) / 2.0
    tr = np.trace(rho)
    if np.abs(tr) > 0:
        rho = rho / tr
    trace_real = float(np.real(np.trace(rho)))
    min_eigenvalue = float(np.min(np.linalg.eigvalsh(rho)).real)
    purity = float(np.real(np.trace(rho @ rho)))

    return {
        "optimizer_success": bool(optimizer_success),
        "optimizer_message": str(optimizer_message),
        "nll_finite": nll_finite,
        "negative_log_likelihood": nll,
        "nll_per_shot": nll_per_shot,
        "total_shots": total_shots,
        "mae_counts": mae_counts,
        "max_abs_counts_error": max_abs_counts_error,
        "normalized_mae_counts": normalized_mae,
        "trace": trace_real,
        "min_eigenvalue": min_eigenvalue,
        "purity": purity,
    }

@workflow.task
@with_plot_theme
def plot_density_matrix(
    rho_hat_real: list[list[float]],
    rho_hat_imag: list[list[float]],
) -> dict[str, mpl.figure.Figure]:
    """Plot real/imaginary parts of reconstructed 3Q density matrix."""
    rho_real = np.asarray(rho_hat_real, dtype=float)
    rho_imag = np.asarray(rho_hat_imag, dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels = [f"|{s}>" for s in OUTCOME_LABELS]

    im0 = axes[0].imshow(rho_real, cmap="RdBu_r")
    axes[0].set_title("Re[rho]")
    axes[0].set_xticks(range(8), labels, rotation=45)
    axes[0].set_yticks(range(8), labels)
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(rho_imag, cmap="RdBu_r")
    axes[1].set_title("Im[rho]")
    axes[1].set_xticks(range(8), labels, rotation=45)
    axes[1].set_yticks(range(8), labels)
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    fig.tight_layout()
    workflow.save_artifact("threeq_tomography_density_matrix", fig)
    return {"density_matrix": fig}

@workflow.task
@with_plot_theme
def plot_counts(
    observed_counts: list[list[int]],
    predicted_counts: list[list[float]],
    setting_labels: list[str],
) -> dict[str, mpl.figure.Figure]:
    """Plot observed and MLE-predicted counts for each setting/outcome."""
    observed = np.asarray(observed_counts, dtype=float)
    predicted = np.asarray(predicted_counts, dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    im0 = axes[0].imshow(observed, aspect="auto", cmap="viridis")
    axes[0].set_title("Observed Counts")
    axes[0].set_xlabel("Outcome")
    axes[0].set_ylabel("Setting")
    axes[0].set_xticks(range(8), OUTCOME_LABELS, rotation=45)
    axes[0].set_yticks(range(len(setting_labels)), setting_labels)
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(predicted, aspect="auto", cmap="viridis")
    axes[1].set_title("Predicted Counts (MLE)")
    axes[1].set_xlabel("Outcome")
    axes[1].set_xticks(range(8), OUTCOME_LABELS, rotation=45)
    axes[1].set_yticks(range(len(setting_labels)), setting_labels)
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    fig.tight_layout()
    workflow.save_artifact("threeq_tomography_counts", fig)
    return {"counts": fig}

DEFAULT_PRODUCT_SUITE_STATES: tuple[str, ...] = (
    "000",
    "001",
    "010",
    "011",
    "100",
    "101",
    "110",
    "111",
    "+++",
    "++-",
    "+-+",
    "+--",
    "-++",
    "-+-",
    "--+",
    "---",
)
DEFAULT_SHOT_SWEEP_LOG2_VALUES: tuple[int, ...] = tuple(range(3, 13))
SHOT_SWEEP_EPS: float = 1e-12
SHOT_SWEEP_INFID_TOL: float = 1e-9


@workflow.workflow_options
class ThreeQQstAnalysisOptions:
    """Options for canonical single-run 3Q QST analysis."""

    do_plotting: bool = workflow.option_field(
        True,
        description="Whether to generate density-matrix and counts plots.",
    )
    max_mle_iterations: int = workflow.option_field(
        2000,
        description="Maximum iterations for MLE optimization.",
    )


def _build_analysis_payload_impl(
    tomography_result: RunExperimentResults,
    q0_uid: str,
    q1_uid: str,
    q2_uid: str,
    readout_calibration_result: RunExperimentResults | None = None,
    target_state=None,
    max_iterations: int = 2000,
) -> dict[str, object]:
    discriminator = fit_discriminator_from_readout_calibration.func(
        readout_calibration_result=readout_calibration_result,
        q0_uid=q0_uid,
        q1_uid=q1_uid,
        q2_uid=q2_uid,
    )
    assignment = extract_assignment_matrix.func(
        readout_calibration_result=readout_calibration_result,
        q0_uid=q0_uid,
        q1_uid=q1_uid,
        q2_uid=q2_uid,
        discriminator=discriminator,
    )
    tomography_counts = collect_tomography_counts.func(
        tomography_result=tomography_result,
        q0_uid=q0_uid,
        q1_uid=q1_uid,
        q2_uid=q2_uid,
        discriminator=discriminator,
    )
    mle_result = maximum_likelihood_reconstruct.func(
        tomography_counts=tomography_counts,
        assignment=assignment,
        max_iterations=max_iterations,
    )
    state_metrics = calculate_state_metrics.func(
        rho_hat_real=mle_result["rho_hat_real"],
        rho_hat_imag=mle_result["rho_hat_imag"],
        target_state=target_state,
    )
    optimization_convergence = evaluate_optimization_convergence.func(
        tomography_counts=tomography_counts,
        predicted_counts=mle_result["predicted_counts"],
        optimizer_success=mle_result["optimizer_success"],
        optimizer_message=mle_result["optimizer_message"],
        negative_log_likelihood=mle_result["negative_log_likelihood"],
        rho_hat_real=mle_result["rho_hat_real"],
        rho_hat_imag=mle_result["rho_hat_imag"],
    )

    return {
        "assignment_matrix": assignment["assignment_matrix"],
        "assignment_counts": assignment["counts_matrix_soft"],
        "assignment_counts_soft": assignment["counts_matrix_soft"],
        "assignment_counts_hard": assignment["counts_matrix_hard"],
        "tomography_counts": tomography_counts["counts"],
        "tomography_counts_hard": tomography_counts["counts_hard"],
        "setting_labels": tomography_counts["setting_labels"],
        "shots_per_setting": tomography_counts["shots_per_setting"],
        "rho_hat_real": mle_result["rho_hat_real"],
        "rho_hat_imag": mle_result["rho_hat_imag"],
        "predicted_probabilities": mle_result["predicted_probabilities"],
        "predicted_counts": mle_result["predicted_counts"],
        "optimizer_success": mle_result["optimizer_success"],
        "optimizer_message": mle_result["optimizer_message"],
        "negative_log_likelihood": mle_result["negative_log_likelihood"],
        "metrics": state_metrics,
        "discriminator_model": discriminator["model"],
        "classification_diagnostics": discriminator["diagnostics"],
        "optimization_convergence": optimization_convergence,
    }


@workflow.task
def analyze_tomography_run(
    tomography_result: RunExperimentResults,
    q0_uid: str,
    q1_uid: str,
    q2_uid: str,
    readout_calibration_result: RunExperimentResults | None = None,
    target_state=None,
    max_iterations: int = 2000,
) -> dict[str, object]:
    """Build the full single-run analysis payload without plotting."""
    return _build_analysis_payload_impl(
        tomography_result=tomography_result,
        q0_uid=q0_uid,
        q1_uid=q1_uid,
        q2_uid=q2_uid,
        readout_calibration_result=readout_calibration_result,
        target_state=target_state,
        max_iterations=max_iterations,
    )


@workflow.workflow(name="analysis_threeq_qst")
def analysis_workflow(
    tomography_result: RunExperimentResults,
    q0,
    q1,
    q2,
    readout_calibration_result: RunExperimentResults | None = None,
    target_state=None,
    options: ThreeQQstAnalysisOptions | None = None,
) -> None:
    """Run readout-mitigated MLE analysis for one 3Q QST dataset."""
    opts = ThreeQQstAnalysisOptions() if options is None else options

    analysis_payload = analyze_tomography_run(
        tomography_result=tomography_result,
        q0_uid=q0.uid,
        q1_uid=q1.uid,
        q2_uid=q2.uid,
        readout_calibration_result=readout_calibration_result,
        target_state=target_state,
        max_iterations=opts.max_mle_iterations,
    )

    with workflow.if_(opts.do_plotting):
        plot_density_matrix(
            rho_hat_real=analysis_payload["rho_hat_real"],
            rho_hat_imag=analysis_payload["rho_hat_imag"],
        )
        plot_counts(
            observed_counts=analysis_payload["tomography_counts"],
            predicted_counts=analysis_payload["predicted_counts"],
            setting_labels=analysis_payload["setting_labels"],
        )

    workflow.return_(analysis_payload)


def _unwrap_analysis_output(result_like):
    current = result_like
    for _ in range(8):
        if current is None:
            return None
        if hasattr(current, "output"):
            current = current.output
            continue
        return current
    return current


def _materialize_analysis_output(result_like) -> dict[str, object] | None:
    current = _unwrap_analysis_output(result_like)
    return dict(current) if isinstance(current, dict) else None


def _safe_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_bool(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return bool(value)
    return False


def _to_float_or_nan(value) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return numeric if np.isfinite(numeric) else float("nan")


@workflow.task
def collect_convergence_run_record(
    state_label: str,
    repeat_index: int,
    analysis_result,
) -> dict[str, object]:
    """Extract compact per-run convergence record from a 3Q analysis output."""
    out = _materialize_analysis_output(analysis_result)
    if not isinstance(out, dict):
        return {
            "state_label": str(state_label),
            "repeat_index": int(repeat_index),
            "fidelity_to_target": None,
            "optimizer_success": False,
            "negative_log_likelihood": None,
            "rho_min_eigenvalue": None,
            "nll_finite": False,
            "nll_per_shot": None,
            "mae_counts": None,
            "max_abs_counts_error": None,
            "normalized_mae_counts": None,
        }

    metrics = out.get("metrics", {}) if isinstance(out.get("metrics"), dict) else {}
    opt = (
        out.get("optimization_convergence", {})
        if isinstance(out.get("optimization_convergence"), dict)
        else {}
    )
    negative_log_likelihood = out.get(
        "negative_log_likelihood",
        opt.get("negative_log_likelihood"),
    )
    rho_min_eigenvalue = metrics.get("min_eigenvalue", opt.get("min_eigenvalue"))
    optimizer_success = out.get("optimizer_success", opt.get("optimizer_success", False))

    return {
        "state_label": str(state_label),
        "repeat_index": int(repeat_index),
        "fidelity_to_target": _safe_float(metrics.get("fidelity_to_target")),
        "optimizer_success": _safe_bool(optimizer_success),
        "negative_log_likelihood": _safe_float(negative_log_likelihood),
        "rho_min_eigenvalue": _safe_float(rho_min_eigenvalue),
        "nll_finite": _safe_bool(opt.get("nll_finite", False)),
        "nll_per_shot": _safe_float(opt.get("nll_per_shot")),
        "mae_counts": _safe_float(opt.get("mae_counts")),
        "max_abs_counts_error": _safe_float(opt.get("max_abs_counts_error")),
        "normalized_mae_counts": _safe_float(opt.get("normalized_mae_counts")),
    }


@workflow.task
def extract_main_run_optimization_convergence(
    analysis_result,
) -> dict[str, object] | None:
    """Extract optimization convergence payload from the main analysis run."""
    out = _materialize_analysis_output(analysis_result)
    if not isinstance(out, dict):
        return None
    convergence = out.get("optimization_convergence")
    return convergence if isinstance(convergence, dict) else None


def _finite_stats(values) -> dict[str, float | int | None]:
    finite_values = []
    for value in values:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric):
            finite_values.append(numeric)
    arr = np.asarray(finite_values, dtype=float)
    n = int(arr.size)
    if n == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "sem": None,
            "ci95": None,
        }
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    sem = float(std / np.sqrt(n)) if n > 1 else 0.0
    ci95 = float(1.96 * sem) if n > 1 else 0.0
    return {
        "count": n,
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci95": ci95,
    }


def _summarize_statistical_convergence_impl(
    run_records: list[dict[str, object]],
) -> dict[str, object]:
    records = run_records or []
    grouped: dict[str, list[dict[str, object]]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        state_label = str(record.get("state_label", "unknown"))
        grouped.setdefault(state_label, []).append(record)

    per_state: dict[str, dict[str, object]] = {}
    all_fidelities: list[float] = []
    success_flags: list[float] = []
    min_eigs: list[float] = []

    for state, items in grouped.items():
        fidelities: list[float] = []
        nlls: list[float] = []
        state_min_eigs: list[float] = []
        state_success: list[float] = []

        for item in items:
            fidelity = item.get("fidelity_to_target")
            if fidelity is not None and np.isfinite(float(fidelity)):
                fidelities.append(float(fidelity))
                all_fidelities.append(float(fidelity))

            nll = item.get("negative_log_likelihood")
            if nll is not None and np.isfinite(float(nll)):
                nlls.append(float(nll))

            eig = item.get("rho_min_eigenvalue")
            if eig is not None and np.isfinite(float(eig)):
                state_min_eigs.append(float(eig))
                min_eigs.append(float(eig))

            success_value = 1.0 if bool(item.get("optimizer_success", False)) else 0.0
            state_success.append(success_value)
            success_flags.append(success_value)

        fidelity_stats = _finite_stats(fidelities)
        nll_stats = _finite_stats(nlls)
        eig_stats = _finite_stats(state_min_eigs)
        per_state[state] = {
            "num_runs": len(items),
            "num_valid_fidelity_runs": int(fidelity_stats["count"]),
            "optimizer_success_rate": (
                float(np.mean(state_success)) if state_success else 0.0
            ),
            "fidelity_mean": fidelity_stats["mean"],
            "fidelity_std": fidelity_stats["std"],
            "fidelity_sem": fidelity_stats["sem"],
            "fidelity_ci95": fidelity_stats["ci95"],
            "nll_mean": nll_stats["mean"],
            "nll_std": nll_stats["std"],
            "rho_min_eigenvalue_mean": eig_stats["mean"],
            "rho_min_eigenvalue_min": (
                float(np.min(state_min_eigs)) if state_min_eigs else None
            ),
        }

    all_fidelity_stats = _finite_stats(all_fidelities)
    aggregate = {
        "num_total_runs": len(records),
        "overall_optimizer_success_rate": (
            float(np.mean(success_flags)) if success_flags else 0.0
        ),
        "pooled_fidelity_mean": all_fidelity_stats["mean"],
        "pooled_fidelity_std": all_fidelity_stats["std"],
        "pooled_fidelity_sem": all_fidelity_stats["sem"],
        "pooled_fidelity_ci95": all_fidelity_stats["ci95"],
        "worst_rho_min_eigenvalue": float(np.min(min_eigs)) if min_eigs else None,
    }
    return {
        "per_state": per_state,
        "aggregate": aggregate,
    }


@workflow.task
def summarize_statistical_convergence(
    run_records: list[dict[str, object]],
) -> dict[str, object]:
    """Aggregate repeated-run convergence statistics across a state suite."""
    return _summarize_statistical_convergence_impl(run_records)


@workflow.task
def collect_shot_sweep_run_record(
    state_label: str,
    log2_shots: int,
    shots: int,
    repeat: int,
    tomography_result: RunExperimentResults,
    q0_uid: str,
    q1_uid: str,
    q2_uid: str,
    readout_calibration_result: RunExperimentResults | None,
    target_state=None,
    max_iterations: int = 2000,
    eps: float = SHOT_SWEEP_EPS,
) -> dict[str, object]:
    """Analyze one 3Q shot-sweep run and return a row plus failure metadata."""
    failure = None
    try:
        payload = analyze_tomography_run.func(
            tomography_result=tomography_result,
            q0_uid=q0_uid,
            q1_uid=q1_uid,
            q2_uid=q2_uid,
            readout_calibration_result=readout_calibration_result,
            target_state=target_state,
            max_iterations=max_iterations,
        )
        metrics = payload.get("metrics", {}) if isinstance(payload, dict) else {}
        fid_raw = metrics.get("fidelity_to_target")
        nll_raw = payload.get("negative_log_likelihood")
        min_eig_raw = metrics.get("min_eigenvalue")

        fid = _to_float_or_nan(fid_raw)
        if np.isfinite(fid):
            infid = max(float(eps), 1.0 - fid)
            log10_infid = float(np.log10(infid))
        else:
            infid = float("nan")
            log10_infid = float("nan")
            failure = {
                "state": str(state_label),
                "log2_shots": int(log2_shots),
                "shots": int(shots),
                "repeat": int(repeat),
                "reason": f"invalid fidelity: {fid_raw!r}",
            }

        row = {
            "state": str(state_label),
            "log2_shots": int(log2_shots),
            "shots": int(shots),
            "repeat": int(repeat),
            "fidelity": fid,
            "infidelity": infid,
            "log10_infidelity": log10_infid,
            "nll": _to_float_or_nan(nll_raw),
            "min_eig": _to_float_or_nan(min_eig_raw),
        }
    except Exception as exc:
        row = {
            "state": str(state_label),
            "log2_shots": int(log2_shots),
            "shots": int(shots),
            "repeat": int(repeat),
            "fidelity": float("nan"),
            "infidelity": float("nan"),
            "log10_infidelity": float("nan"),
            "nll": float("nan"),
            "min_eig": float("nan"),
        }
        failure = {
            "state": str(state_label),
            "log2_shots": int(log2_shots),
            "shots": int(shots),
            "repeat": int(repeat),
            "reason": repr(exc),
        }

    return {"record": row, "failure": failure}


def _validate_shot_sweep_run_records_impl(
    run_records: list[dict[str, object]],
    suite_states: tuple[str, ...] | list[str],
    shot_log2_values: tuple[int, ...] | list[int],
    repeats_per_point: int,
    eps: float = SHOT_SWEEP_EPS,
    infid_tol: float = SHOT_SWEEP_INFID_TOL,
) -> dict[str, object]:
    counts_by_group: dict[tuple[str, int], int] = defaultdict(int)
    violations: list[dict[str, object]] = []
    for row in run_records or []:
        if not isinstance(row, dict):
            continue
        state = str(row.get("state", ""))
        log2_shots = int(row.get("log2_shots", -1))
        counts_by_group[(state, log2_shots)] += 1

        infidelity = _to_float_or_nan(row.get("infidelity"))
        if np.isfinite(infidelity) and (
            infidelity < (float(eps) - 1e-15)
            or infidelity > (1.0 + float(infid_tol))
        ):
            violations.append(dict(row))

    expected_pairs = [
        (str(state), int(log2_shots))
        for state in suite_states
        for log2_shots in shot_log2_values
    ]
    missing_groups = [
        {"state": state, "log2_shots": log2_shots}
        for state, log2_shots in expected_pairs
        if (state, log2_shots) not in counts_by_group
    ]
    bad_repeat_groups = [
        {
            "state": state,
            "log2_shots": log2_shots,
            "observed_repeats": int(count),
            "expected_repeats": int(repeats_per_point),
        }
        for (state, log2_shots), count in sorted(counts_by_group.items())
        if int(count) != int(repeats_per_point)
    ]
    return {
        "expected_group_count": int(len(expected_pairs)),
        "observed_group_count": int(len(counts_by_group)),
        "missing_groups": missing_groups,
        "bad_repeat_groups": bad_repeat_groups,
        "infidelity_range_violations": violations,
    }


@workflow.task
def validate_shot_sweep_run_records(
    run_records: list[dict[str, object]],
    suite_states: tuple[str, ...] | list[str],
    shot_log2_values: tuple[int, ...] | list[int],
    repeats_per_point: int,
    eps: float = SHOT_SWEEP_EPS,
    infid_tol: float = SHOT_SWEEP_INFID_TOL,
) -> dict[str, object]:
    """Validate shot-sweep coverage and infidelity range constraints."""
    return _validate_shot_sweep_run_records_impl(
        run_records=run_records,
        suite_states=suite_states,
        shot_log2_values=shot_log2_values,
        repeats_per_point=repeats_per_point,
        eps=eps,
        infid_tol=infid_tol,
    )


def _aggregate_shot_sweep_statistics_impl(
    run_records: list[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in run_records or []:
        if not isinstance(row, dict):
            continue
        grouped[(str(row.get("state", "")), int(row.get("log2_shots", -1)))].append(
            row
        )

    aggregated_rows: list[dict[str, object]] = []
    for (state, log2_shots), items in sorted(
        grouped.items(), key=lambda item: (item[0][0], item[0][1])
    ):
        if not items:
            continue
        inf_stats = _finite_stats(
            [
                row.get("infidelity")
                for row in items
                if np.isfinite(_to_float_or_nan(row.get("infidelity")))
            ]
        )
        log_stats = _finite_stats(
            [
                row.get("log10_infidelity")
                for row in items
                if np.isfinite(_to_float_or_nan(row.get("log10_infidelity")))
            ]
        )
        aggregated_rows.append(
            {
                "state": state,
                "log2_shots": int(log2_shots),
                "shots": int(items[0].get("shots", 0)),
                "n_total": int(len(items)),
                "n_valid_infidelity": int(inf_stats["count"]),
                "infid_mean": inf_stats["mean"],
                "infid_std": inf_stats["std"],
                "infid_sem": inf_stats["sem"],
                "infid_ci95": inf_stats["ci95"],
                "log10_infid_mean": log_stats["mean"],
                "log10_infid_std": log_stats["std"],
                "log10_infid_sem": log_stats["sem"],
                "log10_infid_ci95": log_stats["ci95"],
            }
        )
    return aggregated_rows


@workflow.task
def aggregate_shot_sweep_statistics(
    run_records: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Aggregate shot-sweep rows into per-state and per-shot summaries."""
    return _aggregate_shot_sweep_statistics_impl(run_records)


def _summarize_final_shot_sweep_impl(
    aggregated_stats: list[dict[str, object]],
    shot_log2_values: tuple[int, ...] | list[int],
) -> list[dict[str, object]]:
    if not shot_log2_values:
        return []
    last_log2 = max(int(value) for value in shot_log2_values)
    rows = []
    for row in aggregated_stats or []:
        if not isinstance(row, dict):
            continue
        if int(row.get("log2_shots", -1)) != last_log2:
            continue
        rows.append(
            {
                "state": str(row.get("state", "")),
                "n_total": int(row.get("n_total", 0)),
                "n_valid_infidelity": int(row.get("n_valid_infidelity", 0)),
                "infid_mean": row.get("infid_mean"),
                "infid_ci95": row.get("infid_ci95"),
                "log10_infid_mean": row.get("log10_infid_mean"),
                "log10_infid_ci95": row.get("log10_infid_ci95"),
            }
        )
    rows.sort(key=lambda item: item["state"])
    return rows


@workflow.task
def summarize_final_shot_sweep(
    aggregated_stats: list[dict[str, object]],
    shot_log2_values: tuple[int, ...] | list[int],
) -> list[dict[str, object]]:
    """Summarize the largest-shot operating point from shot-sweep stats."""
    return _summarize_final_shot_sweep_impl(
        aggregated_stats=aggregated_stats,
        shot_log2_values=shot_log2_values,
    )


@workflow.task
@with_plot_theme
def plot_convergence_suite_fidelity(
    statistical_convergence: dict[str, object],
) -> dict[str, mpl.figure.Figure]:
    """Plot product-state suite fidelity mean ± 95% CI."""
    per_state = (
        statistical_convergence.get("per_state", {})
        if isinstance(statistical_convergence, dict)
        else {}
    )
    states = sorted(per_state.keys())
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    if states:
        means = np.array(
            [
                per_state[state].get("fidelity_mean")
                if per_state[state].get("fidelity_mean") is not None
                else np.nan
                for state in states
            ],
            dtype=float,
        )
        errs = np.array(
            [
                per_state[state].get("fidelity_ci95")
                if per_state[state].get("fidelity_ci95") is not None
                else np.nan
                for state in states
            ],
            dtype=float,
        )
        x = np.arange(len(states))
        ax.errorbar(x, means, yerr=errs, fmt="o", capsize=4)
        ax.set_xticks(x, states, rotation=45)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Fidelity")
    ax.set_title("3Q product-state suite fidelity mean ± 95% CI")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    workflow.save_artifact("threeq_qst_convergence_fidelity", fig)
    return {"convergence_fidelity": fig}


@workflow.task
@with_plot_theme
def plot_shot_sweep_summary(
    aggregated_stats: list[dict[str, object]],
    suite_states: tuple[str, ...] | list[str],
) -> dict[str, mpl.figure.Figure]:
    """Plot infidelity and log10-infidelity versus log2(shots)."""
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in aggregated_stats or []:
        if not isinstance(row, dict):
            continue
        grouped[str(row.get("state", ""))].append(row)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    for state in suite_states:
        state_rows = sorted(
            grouped.get(str(state), []),
            key=lambda row: int(row.get("log2_shots", -1)),
        )
        if not state_rows:
            continue
        x = np.asarray(
            [row.get("log2_shots", np.nan) for row in state_rows],
            dtype=float,
        )

        y_infid = np.asarray(
            [row.get("infid_mean", np.nan) for row in state_rows],
            dtype=float,
        )
        e_infid = np.asarray(
            [row.get("infid_ci95", np.nan) for row in state_rows],
            dtype=float,
        )
        mask_infid = np.isfinite(y_infid)
        if np.any(mask_infid):
            axes[0].plot(
                x[mask_infid],
                y_infid[mask_infid],
                "o--",
                label=f"|{state}><{state}|",
            )
            low = y_infid[mask_infid] - np.nan_to_num(e_infid[mask_infid], nan=0.0)
            high = y_infid[mask_infid] + np.nan_to_num(e_infid[mask_infid], nan=0.0)
            axes[0].fill_between(x[mask_infid], low, high, alpha=0.2)

        y_log = np.asarray(
            [row.get("log10_infid_mean", np.nan) for row in state_rows],
            dtype=float,
        )
        e_log = np.asarray(
            [row.get("log10_infid_ci95", np.nan) for row in state_rows],
            dtype=float,
        )
        mask_log = np.isfinite(y_log)
        if np.any(mask_log):
            axes[1].plot(
                x[mask_log],
                y_log[mask_log],
                "o--",
                label=f"|{state}><{state}|",
            )
            low = y_log[mask_log] - np.nan_to_num(e_log[mask_log], nan=0.0)
            high = y_log[mask_log] + np.nan_to_num(e_log[mask_log], nan=0.0)
            axes[1].fill_between(x[mask_log], low, high, alpha=0.2)

    axes[0].set_title("Infidelity vs Log(Number of shots)")
    axes[0].set_xlabel("Log(Number of shots) = log2(shots)")
    axes[0].set_ylabel("Infidelity")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=9)

    axes[1].set_title("Log10(Infidelity) vs Log(Number of shots)")
    axes[1].set_xlabel("Log(Number of shots) = log2(shots)")
    axes[1].set_ylabel("Log10(Infidelity)")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=9)

    fig.suptitle("3Q QST Product-State Suite Shot Sweep", y=1.02, fontsize=14)
    fig.tight_layout()
    workflow.save_artifact("threeq_qst_shot_sweep_summary", fig)
    return {"shot_sweep_summary": fig}


__all__ = [
    "DEFAULT_PRODUCT_SUITE_STATES",
    "DEFAULT_SHOT_SWEEP_LOG2_VALUES",
    "SHOT_SWEEP_EPS",
    "SHOT_SWEEP_INFID_TOL",
    "ThreeQQstAnalysisOptions",
    "analysis_workflow",
    "analyze_tomography_run",
    "collect_convergence_run_record",
    "collect_shot_sweep_run_record",
    "extract_main_run_optimization_convergence",
    "summarize_statistical_convergence",
    "validate_shot_sweep_run_records",
    "aggregate_shot_sweep_statistics",
    "summarize_final_shot_sweep",
    "plot_convergence_suite_fidelity",
    "plot_shot_sweep_summary",
    "_summarize_statistical_convergence_impl",
    "_validate_shot_sweep_run_records_impl",
    "_aggregate_shot_sweep_statistics_impl",
    "_summarize_final_shot_sweep_impl",
]

"""This module defines the analysis for 2-qubit state tomography with readout mitigation and MLE."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from laboneq import workflow

from experiments.two_qubit_tomography_common import (
    OUTCOME_LABELS,
    READOUT_CALIBRATION_STATES,
    TOMOGRAPHY_SETTINGS,
    readout_calibration_handle,
    tomography_handle,
)
from laboneq_applications.core.validation import validate_result

if TYPE_CHECKING:
    import matplotlib as mpl
    from laboneq.workflow.tasks.run_experiment import RunExperimentResults


@workflow.workflow_options
class TwoQStateTomographyAnalysisOptions:
    """Options for 2Q tomography analysis."""

    do_plotting: bool = workflow.option_field(
        True,
        description="Whether to generate density-matrix and counts plots.",
    )
    max_mle_iterations: int = workflow.option_field(
        2000,
        description="Maximum iterations for MLE optimization.",
    )


@workflow.workflow(name="analysis_two_qubit_state_tomography")
def analysis_workflow(
    tomography_result: RunExperimentResults,
    ctrl,
    targ,
    readout_calibration_result: RunExperimentResults | None = None,
    target_state=None,
    bitflip_ctrl: bool = False,
    bitflip_targ: bool = False,
    options: TwoQStateTomographyAnalysisOptions | None = None,
) -> None:
    """Run IQ-probability readout-mitigated MLE analysis for 2Q tomography data."""
    options = (
        TwoQStateTomographyAnalysisOptions() if options is None else options
    )

    discriminator = fit_discriminator_from_readout_calibration(
        readout_calibration_result=readout_calibration_result,
        ctrl_uid=ctrl.uid,
        targ_uid=targ.uid,
    )
    assignment = extract_assignment_matrix(
        readout_calibration_result=readout_calibration_result,
        ctrl_uid=ctrl.uid,
        targ_uid=targ.uid,
        discriminator=discriminator,
    )
    tomography_counts = collect_tomography_counts(
        tomography_result=tomography_result,
        ctrl_uid=ctrl.uid,
        targ_uid=targ.uid,
        discriminator=discriminator,
    )
    mle_result = maximum_likelihood_reconstruct(
        tomography_counts=tomography_counts,
        assignment=assignment,
        max_iterations=options.max_mle_iterations,
    )
    state_metrics = calculate_state_metrics(
        rho_hat_real=mle_result["rho_hat_real"],
        rho_hat_imag=mle_result["rho_hat_imag"],
        target_state=target_state,
    )

    with workflow.if_(options.do_plotting):
        plot_density_matrix(
            rho_hat_real=mle_result["rho_hat_real"],
            rho_hat_imag=mle_result["rho_hat_imag"],
        )
        plot_counts(
            observed_counts=tomography_counts["counts"],
            predicted_counts=mle_result["predicted_counts"],
            setting_labels=tomography_counts["setting_labels"],
        )

    workflow.return_(
        {
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
            "bitflip_ctrl": bool(bitflip_ctrl),
            "bitflip_targ": bool(bitflip_targ),
        }
    )


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


def _collect_calibration_training_sets(
    readout_calibration_result: RunExperimentResults,
    ctrl_uid: str,
    targ_uid: str,
) -> dict[str, np.ndarray]:
    ctrl_by_state = {"g": [], "e": []}
    targ_by_state = {"g": [], "e": []}
    for prepared_label, (ctrl_state, targ_state) in READOUT_CALIBRATION_STATES:
        ctrl_shots = np.asarray(
            readout_calibration_result[
                readout_calibration_handle(ctrl_uid, prepared_label)
            ].data
        ).reshape(-1)
        targ_shots = np.asarray(
            readout_calibration_result[
                readout_calibration_handle(targ_uid, prepared_label)
            ].data
        ).reshape(-1)
        nshots = min(len(ctrl_shots), len(targ_shots))
        if nshots == 0:
            continue
        ctrl_by_state[ctrl_state].append(ctrl_shots[:nshots])
        targ_by_state[targ_state].append(targ_shots[:nshots])

    if not all(ctrl_by_state[s] for s in ("g", "e")):
        raise ValueError("Insufficient ctrl calibration samples for g/e states.")
    if not all(targ_by_state[s] for s in ("g", "e")):
        raise ValueError("Insufficient targ calibration samples for g/e states.")

    return {
        "ctrl_g": np.concatenate(ctrl_by_state["g"]),
        "ctrl_e": np.concatenate(ctrl_by_state["e"]),
        "targ_g": np.concatenate(targ_by_state["g"]),
        "targ_e": np.concatenate(targ_by_state["e"]),
    }


def _hard_counts_from_posteriors(posteriors: np.ndarray) -> np.ndarray:
    hard_outcomes = np.argmax(posteriors, axis=1)
    return np.bincount(hard_outcomes, minlength=4).astype(int)


def _classification_diagnostics(
    readout_calibration_result: RunExperimentResults,
    ctrl_uid: str,
    targ_uid: str,
    ctrl_model: dict,
    targ_model: dict,
) -> dict[str, object]:
    state_to_bit = {"g": 0, "e": 1}
    ctrl_true, ctrl_pred = [], []
    targ_true, targ_pred = [], []
    for prepared_label, (ctrl_state, targ_state) in READOUT_CALIBRATION_STATES:
        ctrl_shots = np.asarray(
            readout_calibration_result[
                readout_calibration_handle(ctrl_uid, prepared_label)
            ].data
        ).reshape(-1)
        targ_shots = np.asarray(
            readout_calibration_result[
                readout_calibration_handle(targ_uid, prepared_label)
            ].data
        ).reshape(-1)
        nshots = min(len(ctrl_shots), len(targ_shots))
        if nshots == 0:
            continue
        p_ctrl_e = _predict_p_e(ctrl_shots[:nshots], ctrl_model)
        p_targ_e = _predict_p_e(targ_shots[:nshots], targ_model)
        ctrl_true.append(np.full(nshots, state_to_bit[ctrl_state], dtype=int))
        targ_true.append(np.full(nshots, state_to_bit[targ_state], dtype=int))
        ctrl_pred.append((p_ctrl_e >= 0.5).astype(int))
        targ_pred.append((p_targ_e >= 0.5).astype(int))

    ctrl_true_arr = np.concatenate(ctrl_true)
    ctrl_pred_arr = np.concatenate(ctrl_pred)
    targ_true_arr = np.concatenate(targ_true)
    targ_pred_arr = np.concatenate(targ_pred)

    ctrl_cm = np.zeros((2, 2), dtype=float)
    targ_cm = np.zeros((2, 2), dtype=float)
    for t, p in zip(ctrl_true_arr, ctrl_pred_arr):
        ctrl_cm[t, p] += 1
    for t, p in zip(targ_true_arr, targ_pred_arr):
        targ_cm[t, p] += 1
    ctrl_cm /= np.maximum(np.sum(ctrl_cm, axis=1, keepdims=True), 1.0)
    targ_cm /= np.maximum(np.sum(targ_cm, axis=1, keepdims=True), 1.0)

    return {
        "ctrl_confusion_matrix": ctrl_cm.tolist(),
        "targ_confusion_matrix": targ_cm.tolist(),
        "ctrl_accuracy": float(np.mean(ctrl_true_arr == ctrl_pred_arr)),
        "targ_accuracy": float(np.mean(targ_true_arr == targ_pred_arr)),
    }


@workflow.task
def fit_discriminator_from_readout_calibration(
    readout_calibration_result: RunExperimentResults | None,
    ctrl_uid: str,
    targ_uid: str,
) -> dict[str, object]:
    """Fit qubit-wise IQ discriminators from 2Q readout calibration shots."""
    readout_calibration_result = _unwrap_result_like(readout_calibration_result)
    if readout_calibration_result is None:
        raise ValueError(
            "readout_calibration_result is required for INTEGRATION-based tomography analysis."
        )
    validate_result(readout_calibration_result)

    train = _collect_calibration_training_sets(
        readout_calibration_result=readout_calibration_result,
        ctrl_uid=ctrl_uid,
        targ_uid=targ_uid,
    )
    ctrl_model = _fit_binary_gaussian_discriminator(train["ctrl_g"], train["ctrl_e"])
    targ_model = _fit_binary_gaussian_discriminator(train["targ_g"], train["targ_e"])
    diagnostics = _classification_diagnostics(
        readout_calibration_result=readout_calibration_result,
        ctrl_uid=ctrl_uid,
        targ_uid=targ_uid,
        ctrl_model=ctrl_model,
        targ_model=targ_model,
    )

    model_export = {
        "ctrl": {
            "w": np.asarray(ctrl_model["w"], dtype=float).tolist(),
            "b": float(ctrl_model["b"]),
            "mu_g": np.asarray(ctrl_model["mu_g"], dtype=float).tolist(),
            "mu_e": np.asarray(ctrl_model["mu_e"], dtype=float).tolist(),
            "sigma": np.asarray(ctrl_model["sigma"], dtype=float).tolist(),
        },
        "targ": {
            "w": np.asarray(targ_model["w"], dtype=float).tolist(),
            "b": float(targ_model["b"]),
            "mu_g": np.asarray(targ_model["mu_g"], dtype=float).tolist(),
            "mu_e": np.asarray(targ_model["mu_e"], dtype=float).tolist(),
            "sigma": np.asarray(targ_model["sigma"], dtype=float).tolist(),
        },
    }
    return {
        "model": model_export,
        "internal": {"ctrl": ctrl_model, "targ": targ_model},
        "diagnostics": diagnostics,
    }


@workflow.task
def infer_bitflip_from_readout_calibration(
    readout_calibration_result: RunExperimentResults | None,
    ctrl_uid: str,
    targ_uid: str,
) -> dict[str, bool | float]:
    """Compatibility task: keep API but bitflip inference is disabled in IQ flow."""
    if readout_calibration_result is None:
        return {
            "bitflip_ctrl": False,
            "bitflip_targ": False,
            "accuracy_ctrl_direct": 0.0,
            "accuracy_ctrl_flip": 0.0,
            "accuracy_targ_direct": 0.0,
            "accuracy_targ_flip": 0.0,
        }
    readout_calibration_result = _unwrap_result_like(readout_calibration_result)
    validate_result(readout_calibration_result)
    train = _collect_calibration_training_sets(
        readout_calibration_result=readout_calibration_result,
        ctrl_uid=ctrl_uid,
        targ_uid=targ_uid,
    )
    ctrl_model = _fit_binary_gaussian_discriminator(train["ctrl_g"], train["ctrl_e"])
    targ_model = _fit_binary_gaussian_discriminator(train["targ_g"], train["targ_e"])
    diag = _classification_diagnostics(
        readout_calibration_result=readout_calibration_result,
        ctrl_uid=ctrl_uid,
        targ_uid=targ_uid,
        ctrl_model=ctrl_model,
        targ_model=targ_model,
    )
    acc_ctrl = float(diag.get("ctrl_accuracy", 0.0))
    acc_targ = float(diag.get("targ_accuracy", 0.0))
    return {
        "bitflip_ctrl": False,
        "bitflip_targ": False,
        "accuracy_ctrl_direct": acc_ctrl,
        "accuracy_ctrl_flip": 1.0 - acc_ctrl,
        "accuracy_targ_direct": acc_targ,
        "accuracy_targ_flip": 1.0 - acc_targ,
    }


@workflow.task
def resolve_bitflip_settings(
    bitflip_ctrl: bool,
    bitflip_targ: bool,
    auto_bitflip_from_calibration: bool,
    inferred_bitflip: dict[str, bool | float] | None = None,
) -> dict[str, bool]:
    """Compatibility task: always returns manual bitflip values unchanged."""
    del auto_bitflip_from_calibration, inferred_bitflip
    return {"bitflip_ctrl": bool(bitflip_ctrl), "bitflip_targ": bool(bitflip_targ)}


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


@workflow.task
def collect_tomography_counts(
    tomography_result: RunExperimentResults,
    ctrl_uid: str,
    targ_uid: str,
    discriminator: dict[str, object],
) -> dict[str, list]:
    """Collect 4-outcome soft counts per tomography setting from IQ posteriors."""
    tomography_result = _unwrap_result_like(tomography_result)
    validate_result(tomography_result)
    internal = discriminator["internal"]
    ctrl_model = internal["ctrl"]
    targ_model = internal["targ"]

    counts = []
    counts_hard = []
    shots_per_setting = []
    setting_labels = []

    for setting_label, _axes in TOMOGRAPHY_SETTINGS:
        ctrl_shots = np.asarray(
            tomography_result[tomography_handle(ctrl_uid, setting_label)].data
        ).reshape(-1)
        targ_shots = np.asarray(
            tomography_result[tomography_handle(targ_uid, setting_label)].data
        ).reshape(-1)
        nshots = min(len(ctrl_shots), len(targ_shots))
        ctrl_shots = ctrl_shots[:nshots]
        targ_shots = targ_shots[:nshots]

        p_ctrl_e = _predict_p_e(ctrl_shots, ctrl_model)
        p_targ_e = _predict_p_e(targ_shots, targ_model)
        p_ctrl_g = 1.0 - p_ctrl_e
        p_targ_g = 1.0 - p_targ_e

        posteriors = np.column_stack(
            [
                p_ctrl_g * p_targ_g,  # 00
                p_ctrl_g * p_targ_e,  # 01
                p_ctrl_e * p_targ_g,  # 10
                p_ctrl_e * p_targ_e,  # 11
            ]
        )
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
    ctrl_uid: str,
    targ_uid: str,
    discriminator: dict[str, object],
) -> dict[str, list]:
    """Extract 4x4 assignment matrix A_{ik} from IQ posterior probabilities."""
    readout_calibration_result = _unwrap_result_like(readout_calibration_result)
    if readout_calibration_result is None:
        raise ValueError(
            "readout_calibration_result is required for INTEGRATION-based assignment extraction."
        )

    validate_result(readout_calibration_result)
    internal = discriminator["internal"]
    ctrl_model = internal["ctrl"]
    targ_model = internal["targ"]

    counts_matrix_soft = np.zeros((4, 4), dtype=float)
    counts_matrix_hard = np.zeros((4, 4), dtype=int)
    for k, (prepared_label, _states) in enumerate(READOUT_CALIBRATION_STATES):
        ctrl_shots = np.asarray(
            readout_calibration_result[
                readout_calibration_handle(ctrl_uid, prepared_label)
            ].data
        ).reshape(-1)
        targ_shots = np.asarray(
            readout_calibration_result[
                readout_calibration_handle(targ_uid, prepared_label)
            ].data
        ).reshape(-1)
        nshots = min(len(ctrl_shots), len(targ_shots))
        ctrl_shots = ctrl_shots[:nshots]
        targ_shots = targ_shots[:nshots]

        p_ctrl_e = _predict_p_e(ctrl_shots, ctrl_model)
        p_targ_e = _predict_p_e(targ_shots, targ_model)
        p_ctrl_g = 1.0 - p_ctrl_e
        p_targ_g = 1.0 - p_targ_e
        posteriors = np.column_stack(
            [
                p_ctrl_g * p_targ_g,  # 00
                p_ctrl_g * p_targ_e,  # 01
                p_ctrl_e * p_targ_g,  # 10
                p_ctrl_e * p_targ_e,  # 11
            ]
        )
        counts_matrix_soft[:, k] = np.sum(posteriors, axis=0)
        counts_matrix_hard[:, k] = _hard_counts_from_posteriors(posteriors)

    with np.errstate(invalid="ignore", divide="ignore"):
        assignment_matrix = counts_matrix_soft / np.maximum(
            np.sum(counts_matrix_soft, axis=0, keepdims=True), 1.0
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


def _single_qubit_prerotation_unitary(axis: str) -> np.ndarray:
    if axis == "X":
        return _rotation_y(-np.pi / 2)
    if axis == "Y":
        return _rotation_x(np.pi / 2)
    if axis == "Z":
        return np.eye(2, dtype=complex)
    raise ValueError(f"Unsupported axis: {axis!r}.")


def _computational_projectors_2q() -> list[np.ndarray]:
    projectors = []
    for i in range(4):
        ket = np.zeros((4, 1), dtype=complex)
        ket[i, 0] = 1.0
        projectors.append(ket @ ket.conj().T)
    return projectors


def _build_noisy_povm(assignment_matrix: np.ndarray) -> np.ndarray:
    ideal_projectors = _computational_projectors_2q()
    noisy_povm = np.zeros((len(TOMOGRAPHY_SETTINGS), 4, 4, 4), dtype=complex)

    for s_idx, (_label, (ctrl_axis, targ_axis)) in enumerate(TOMOGRAPHY_SETTINGS):
        u_ctrl = _single_qubit_prerotation_unitary(ctrl_axis)
        u_targ = _single_qubit_prerotation_unitary(targ_axis)
        u_setting = np.kron(u_ctrl, u_targ)
        ideal_povm = [u_setting.conj().T @ p @ u_setting for p in ideal_projectors]

        for i in range(4):
            e_meas = np.zeros((4, 4), dtype=complex)
            for k in range(4):
                e_meas += assignment_matrix[i, k] * ideal_povm[k]
            noisy_povm[s_idx, i] = e_meas

    return noisy_povm


def _theta_to_density_matrix(theta: np.ndarray, dim: int = 4) -> np.ndarray:
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
        rho = _theta_to_density_matrix(theta)
        probs = _predict_probabilities(noisy_povm, rho)
        return float(-np.sum(counts * np.log(probs)))

    initial_theta = np.zeros(16, dtype=float)
    result = minimize(
        objective,
        initial_theta,
        method="L-BFGS-B",
        options={"maxiter": int(max_iterations)},
    )

    rho_hat = _theta_to_density_matrix(result.x)
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
        if key in {"plus_plus", "++"}:
            psi = np.array([1, 1, 1, 1], dtype=complex) / 2
            return np.outer(psi, psi.conj())
        if key in {"bell_phi_plus", "phi_plus", "phiplus"}:
            psi = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
            return np.outer(psi, psi.conj())
        raise ValueError(f"Unsupported target_state string: {target_state!r}.")

    arr = np.asarray(target_state, dtype=complex)
    if arr.shape == (4,):
        psi = arr / np.linalg.norm(arr)
        return np.outer(psi, psi.conj())
    if arr.shape == (4, 4):
        rho_target = (arr + arr.conj().T) / 2.0
        tr = np.trace(rho_target)
        return rho_target / tr if tr != 0 else rho_target

    raise ValueError(
        "target_state must be None, a known string, a length-4 statevector, or a 4x4 density matrix."
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
        rho_hat_imag, dtype=float
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
            op = np.kron(paulis[a], paulis[b])
            correlators[f"{a}{b}"] = float(np.real(np.trace(op @ rho)))

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
def plot_density_matrix(
    rho_hat_real: list[list[float]],
    rho_hat_imag: list[list[float]],
) -> dict[str, mpl.figure.Figure]:
    """Plot real/imaginary parts of reconstructed 2Q density matrix."""
    rho_real = np.asarray(rho_hat_real, dtype=float)
    rho_imag = np.asarray(rho_hat_imag, dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    labels = [f"|{s}>" for s in OUTCOME_LABELS]

    im0 = axes[0].imshow(rho_real, cmap="RdBu_r")
    axes[0].set_title("Re[rho]")
    axes[0].set_xticks(range(4), labels, rotation=45)
    axes[0].set_yticks(range(4), labels)
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(rho_imag, cmap="RdBu_r")
    axes[1].set_title("Im[rho]")
    axes[1].set_xticks(range(4), labels, rotation=45)
    axes[1].set_yticks(range(4), labels)
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    fig.tight_layout()
    workflow.save_artifact("twoq_tomography_density_matrix", fig)
    return {"density_matrix": fig}


@workflow.task
def plot_counts(
    observed_counts: list[list[int]],
    predicted_counts: list[list[float]],
    setting_labels: list[str],
) -> dict[str, mpl.figure.Figure]:
    """Plot observed and MLE-predicted counts for each setting/outcome."""
    observed = np.asarray(observed_counts, dtype=float)
    predicted = np.asarray(predicted_counts, dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    im0 = axes[0].imshow(observed, aspect="auto", cmap="viridis")
    axes[0].set_title("Observed Counts")
    axes[0].set_xlabel("Outcome")
    axes[0].set_ylabel("Setting")
    axes[0].set_xticks(range(4), OUTCOME_LABELS)
    axes[0].set_yticks(range(len(setting_labels)), setting_labels)
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(predicted, aspect="auto", cmap="viridis")
    axes[1].set_title("Predicted Counts (MLE)")
    axes[1].set_xlabel("Outcome")
    axes[1].set_xticks(range(4), OUTCOME_LABELS)
    axes[1].set_yticks(range(len(setting_labels)), setting_labels)
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    fig.tight_layout()
    workflow.save_artifact("twoq_tomography_counts", fig)
    return {"counts": fig}

"""Analysis workflow for IQ-cloud readout calibration."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from laboneq import workflow
from matplotlib.patches import Ellipse

from experiments.iq_cloud_common import (
    JOINT_LABELS_2Q,
    prepared_labels_for_num_qubits,
    validate_supported_num_qubits,
)
from laboneq_applications.core.validation import (
    validate_and_convert_qubits_sweeps,
    validate_result,
)

if TYPE_CHECKING:
    import matplotlib as mpl
    from laboneq.workflow.tasks.run_experiment import RunExperimentResults

    from laboneq_applications.typing import QuantumElements


_EPS = 1e-12


@workflow.workflow_options
class IQCloudAnalysisWorkflowOptions:
    """Options for IQ-cloud analysis workflow."""

    do_plotting: bool = workflow.option_field(
        True,
        description="Whether to create IQ/assignment plots.",
    )
    do_plotting_iq_clouds: bool = workflow.option_field(
        True,
        description="Whether to plot IQ cloud scatter + boundary + projection histograms.",
    )
    do_plotting_assignment_matrices: bool = workflow.option_field(
        True,
        description="Whether to plot assignment matrices.",
    )
    do_plotting_bootstrap_summary: bool = workflow.option_field(
        True,
        description="Whether to plot bootstrap uncertainty summary.",
    )
    ridge_target_condition: float = workflow.option_field(
        1e6,
        description="Target upper bound for covariance condition number in ridge regularization.",
    )
    bootstrap_samples: int = workflow.option_field(
        2000,
        description="Number of bootstrap repetitions.",
    )
    bootstrap_confidence_level: float = workflow.option_field(
        0.95,
        description="Bootstrap confidence interval level.",
    )
    bootstrap_seed: int | None = workflow.option_field(
        None,
        description="Random seed for bootstrap resampling.",
    )
    enforce_constant_kernel: bool = workflow.option_field(
        True,
        description=(
            "Whether to enforce default integration kernels when preparing threshold "
            "update payload."
        ),
    )


@workflow.workflow(name="analysis_iq_cloud")
def analysis_workflow(
    result: RunExperimentResults,
    qubits: QuantumElements,
    options: IQCloudAnalysisWorkflowOptions | None = None,
) -> None:
    """Analyze IQ-cloud data with shared-covariance Gaussian Bayes classifier."""
    options = IQCloudAnalysisWorkflowOptions() if options is None else options

    processed_data = collect_shots(result=result, qubits=qubits)
    decision_model = fit_decision_models(
        processed_data=processed_data,
        qubits=qubits,
        ridge_target_condition=options.ridge_target_condition,
    )
    thresholds = extract_thresholds(decision_model=decision_model)
    assignment_bundle = calculate_confusion_and_fidelity(
        processed_data=processed_data,
        decision_model=decision_model,
        qubits=qubits,
    )
    confusion_matrices = select_confusion_matrices(assignment_bundle=assignment_bundle)
    assignment_fidelity = select_assignment_fidelity(assignment_bundle=assignment_bundle)
    separation_metrics = calculate_separation_metrics(decision_model=decision_model)
    bootstrap = bootstrap_metrics(
        processed_data=processed_data,
        qubits=qubits,
        ridge_target_condition=options.ridge_target_condition,
        bootstrap_samples=options.bootstrap_samples,
        bootstrap_confidence_level=options.bootstrap_confidence_level,
        bootstrap_seed=options.bootstrap_seed,
    )
    qubit_parameters = extract_qubit_parameters_for_discrimination(
        qubits=qubits,
        decision_model=decision_model,
        enforce_constant_kernel=options.enforce_constant_kernel,
    )

    with workflow.if_(options.do_plotting):
        with workflow.if_(options.do_plotting_iq_clouds):
            plot_iq_clouds(
                processed_data=processed_data,
                decision_model=decision_model,
                qubits=qubits,
                bootstrap=bootstrap,
            )
        with workflow.if_(options.do_plotting_assignment_matrices):
            plot_assignment_matrices(
                confusion_matrices=confusion_matrices,
                assignment_fidelity=assignment_fidelity,
                qubits=qubits,
                separation_metrics=separation_metrics,
                bootstrap=bootstrap,
            )
        with workflow.if_(options.do_plotting_bootstrap_summary):
            plot_bootstrap_summary(
                bootstrap=bootstrap,
                qubits=qubits,
            )

    workflow.return_(
        {
            "decision_model": decision_model,
            "thresholds": thresholds,
            "confusion_matrices": confusion_matrices,
            "assignment_fidelity": assignment_fidelity,
            "separation_metrics": separation_metrics,
            "bootstrap": bootstrap,
            "qubit_parameters": qubit_parameters,
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
        if isinstance(current, dict) and "result" in current:
            current = current["result"]
            continue
        return current
    return current


def _label_to_bits(prepared_label: str) -> list[int]:
    return [0 if ch == "g" else 1 for ch in prepared_label]


def _real2(shots: np.ndarray) -> np.ndarray:
    return np.column_stack([np.real(shots), np.imag(shots)])


def _safe_cov(x: np.ndarray) -> np.ndarray:
    if x.shape[0] <= 1:
        return np.zeros((2, 2), dtype=float)
    cov = np.cov(x, rowvar=False, bias=False)
    return np.asarray(cov, dtype=float).reshape(2, 2)


def _ridge_lambda_trace_scaled(
    pooled_covariance: np.ndarray,
    target_condition: float,
) -> tuple[float, float]:
    cov = 0.5 * (pooled_covariance + pooled_covariance.T)
    eig = np.linalg.eigvalsh(cov)
    max_eig = float(np.max(eig))
    min_eig = float(np.min(eig))
    trace_scale = max(float(np.trace(cov) / 2.0), _EPS)

    lambda_pd = max(0.0, -min_eig + _EPS)
    if target_condition <= 1.0:
        lambda_cond = 0.0
    else:
        lambda_cond = max(
            0.0,
            (max_eig - target_condition * min_eig) / (target_condition - 1.0),
        )
    ridge_lambda = max(lambda_pd, lambda_cond)
    ridge_alpha = ridge_lambda / trace_scale
    return float(ridge_lambda), float(ridge_alpha)


def _fit_binary_shared_cov_model(
    shots_g: np.ndarray,
    shots_e: np.ndarray,
    target_condition: float,
) -> dict[str, float | list]:
    xg = _real2(shots_g)
    xe = _real2(shots_e)
    if xg.shape[0] < 1 or xe.shape[0] < 1:
        raise ValueError(
            "IQ-cloud model fitting requires non-empty g/e shots for each qubit."
        )

    mu_g = np.mean(xg, axis=0)
    mu_e = np.mean(xe, axis=0)
    cov_g = _safe_cov(xg)
    cov_e = _safe_cov(xe)
    ng = xg.shape[0]
    ne = xe.shape[0]
    dof = max(ng + ne - 2, 1)
    pooled_cov = ((max(ng - 1, 0) * cov_g) + (max(ne - 1, 0) * cov_e)) / dof

    ridge_lambda, ridge_alpha = _ridge_lambda_trace_scaled(
        pooled_covariance=pooled_cov,
        target_condition=target_condition,
    )
    sigma = pooled_cov + ridge_lambda * np.eye(2)
    inv_sigma = np.linalg.inv(sigma)

    delta = mu_e - mu_g
    w = inv_sigma @ delta
    b = -0.5 * (mu_e @ inv_sigma @ mu_e - mu_g @ inv_sigma @ mu_g)

    w_norm = float(np.linalg.norm(w))
    if w_norm < _EPS:
        axis_unit = np.array([1.0, 0.0], dtype=float)
        threshold = float(np.mean([mu_g[0], mu_e[0]]))
    else:
        axis_unit = w / w_norm
        threshold = float(-b / w_norm)

    return {
        "mu_g": [float(mu_g[0]), float(mu_g[1])],
        "mu_e": [float(mu_e[0]), float(mu_e[1])],
        "sigma": sigma.tolist(),
        "inv_sigma": inv_sigma.tolist(),
        "w": [float(w[0]), float(w[1])],
        "b": float(b),
        "t": float(threshold),
        "axis_unit": [float(axis_unit[0]), float(axis_unit[1])],
        "ridge_lambda": float(ridge_lambda),
        "ridge_alpha": float(ridge_alpha),
    }


def _predict_bits(shots: np.ndarray, model: dict[str, float | list]) -> np.ndarray:
    x = _real2(shots)
    w = np.asarray(model["w"], dtype=float)
    b = float(model["b"])
    return (x @ w + b >= 0.0).astype(int)


def _counts_to_normalized(counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, dtype=float)
    rowsum = counts.sum(axis=1, keepdims=True)
    normalized = np.divide(
        counts,
        rowsum,
        out=np.zeros_like(counts, dtype=float),
        where=rowsum > 0.0,
    )
    return normalized


def _build_shot_arrays(
    processed_data: dict,
    qubit_uids: list[str],
    prepared_labels: tuple[str, ...],
) -> dict[str, dict[str, np.ndarray]]:
    return {
        uid: {
            label: np.asarray(processed_data["shots_per_qubit"][uid][label], dtype=complex)
            for label in prepared_labels
        }
        for uid in qubit_uids
    }


def _fit_models_core(
    shot_arrays: dict[str, dict[str, np.ndarray]],
    qubit_uids: list[str],
    prepared_labels: tuple[str, ...],
    target_condition: float,
) -> dict[str, dict]:
    decision_model = {}
    for q_index, uid in enumerate(qubit_uids):
        g_labels = [label for label in prepared_labels if _label_to_bits(label)[q_index] == 0]
        e_labels = [label for label in prepared_labels if _label_to_bits(label)[q_index] == 1]
        shots_g = np.concatenate([shot_arrays[uid][label] for label in g_labels])
        shots_e = np.concatenate([shot_arrays[uid][label] for label in e_labels])
        decision_model[uid] = _fit_binary_shared_cov_model(
            shots_g=shots_g,
            shots_e=shots_e,
            target_condition=target_condition,
        )
    return decision_model


def _assignment_core(
    shot_arrays: dict[str, dict[str, np.ndarray]],
    decision_model: dict[str, dict],
    qubit_uids: list[str],
    prepared_labels: tuple[str, ...],
) -> dict[str, dict]:
    per_qubit_counts: dict[str, np.ndarray] = {
        uid: np.zeros((2, 2), dtype=int) for uid in qubit_uids
    }

    for label in prepared_labels:
        bits = _label_to_bits(label)
        for q_index, uid in enumerate(qubit_uids):
            true_bit = bits[q_index]
            pred_bits = _predict_bits(shot_arrays[uid][label], decision_model[uid])
            for pred_bit in pred_bits:
                per_qubit_counts[uid][true_bit, int(pred_bit)] += 1

    confusion_per_qubit = {}
    fidelity_per_qubit = {}
    for uid in qubit_uids:
        counts = per_qubit_counts[uid]
        normalized = _counts_to_normalized(counts)
        total = float(np.sum(counts))
        fidelity_per_qubit[uid] = float(np.trace(counts) / total) if total > 0 else 0.0
        confusion_per_qubit[uid] = {
            "counts": counts.tolist(),
            "normalized": normalized.tolist(),
            "labels": ["g", "e"],
        }

    out = {
        "confusion_matrices": {"per_qubit": confusion_per_qubit},
        "assignment_fidelity": {"per_qubit": fidelity_per_qubit},
    }

    if len(qubit_uids) == 2:
        uid0, uid1 = qubit_uids
        joint_counts = np.zeros((4, 4), dtype=int)
        for label in prepared_labels:
            true_bits = _label_to_bits(label)
            true_idx = 2 * true_bits[0] + true_bits[1]
            shots0 = shot_arrays[uid0][label]
            shots1 = shot_arrays[uid1][label]
            n = min(len(shots0), len(shots1))
            if n < 1:
                continue
            pred0 = _predict_bits(shots0[:n], decision_model[uid0])
            pred1 = _predict_bits(shots1[:n], decision_model[uid1])
            pred_idx = 2 * pred0 + pred1
            for p in pred_idx:
                joint_counts[true_idx, int(p)] += 1

        joint_norm = _counts_to_normalized(joint_counts)
        joint_total = float(np.sum(joint_counts))
        joint_fidelity = float(np.trace(joint_counts) / joint_total) if joint_total > 0 else 0.0
        average_fidelity = float(np.mean([fidelity_per_qubit[uid0], fidelity_per_qubit[uid1]]))

        out["confusion_matrices"]["joint"] = {
            "counts": joint_counts.tolist(),
            "normalized": joint_norm.tolist(),
            "labels": list(JOINT_LABELS_2Q),
        }
        out["assignment_fidelity"]["joint"] = joint_fidelity
        out["assignment_fidelity"]["average"] = average_fidelity

    return out


def _separation_core(
    decision_model: dict[str, dict],
) -> dict[str, dict[str, float]]:
    separation = {}
    for uid, model in decision_model.items():
        mu_g = np.asarray(model["mu_g"], dtype=float)
        mu_e = np.asarray(model["mu_e"], dtype=float)
        sigma = np.asarray(model["sigma"], dtype=float)
        w = np.asarray(model["w"], dtype=float)
        delta = mu_e - mu_g
        num = abs(float(w.T @ delta))
        den = np.sqrt(max(float(w.T @ sigma @ w), 0.0))
        delta_mu_over_sigma = float(num / den) if den > _EPS else 0.0
        separation[uid] = {
            "delta_mu_over_sigma": delta_mu_over_sigma,
        }
    return separation


def _snr_delta_mu_over_sigma(model: dict) -> float:
    mu_g = np.asarray(model["mu_g"], dtype=float)
    mu_e = np.asarray(model["mu_e"], dtype=float)
    sigma = np.asarray(model["sigma"], dtype=float)
    w = np.asarray(model["w"], dtype=float)
    delta = mu_e - mu_g
    num = abs(float(w.T @ delta))
    den = np.sqrt(max(float(w.T @ sigma @ w), 0.0))
    return float(num / den) if den > _EPS else 0.0


def _ci_summary(values: list[float], confidence_level: float) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "confidence_level": float(confidence_level),
        }
    alpha = (1.0 - confidence_level) / 2.0
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "ci_low": float(np.quantile(arr, alpha)),
        "ci_high": float(np.quantile(arr, 1.0 - alpha)),
        "confidence_level": float(confidence_level),
    }


def _format_ci(
    point: float,
    ci: dict | None,
    digits: int = 3,
) -> str:
    if not isinstance(ci, dict):
        return f"{point:.{digits}f}"
    mean = float(ci.get("mean", point))
    low = float(ci.get("ci_low", mean))
    high = float(ci.get("ci_high", mean))
    return f"{mean:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def _bootstrap_per_qubit_entry(
    bootstrap: dict | None,
    uid: str,
    key: str,
) -> dict | None:
    if not isinstance(bootstrap, dict):
        return None
    return bootstrap.get("per_qubit", {}).get(uid, {}).get(key)


def _ellipse_from_covariance(mu: np.ndarray, cov: np.ndarray, n_std: float = 2.0) -> Ellipse:
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    width, height = 2.0 * n_std * np.sqrt(np.maximum(eigvals, _EPS))
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    return Ellipse(
        xy=(float(mu[0]), float(mu[1])),
        width=float(width),
        height=float(height),
        angle=float(angle),
        fill=False,
    )


def _save_artifact_if_available(name: str, artifact) -> None:
    """Save workflow artifact when executed inside workflow task context.

    Plot tasks in this module are also reused directly in notebooks for synthetic
    validation. In that path, workflow recorders are not active and save_artifact
    raises RuntimeError. We skip artifact saving in that case.
    """
    try:
        workflow.save_artifact(name, artifact)
    except RuntimeError as exc:
        if "not supported outside of tasks" in str(exc):
            return
        raise


@workflow.task
def collect_shots(
    result: RunExperimentResults,
    qubits: QuantumElements,
) -> dict:
    """Collect integrated complex IQ shots for g/e-only 1Q or 2Q prepared states."""
    result = _unwrap_result_like(result)
    validate_result(result)
    qubits = validate_and_convert_qubits_sweeps(qubits)
    validate_supported_num_qubits(len(qubits))
    prepared_labels = prepared_labels_for_num_qubits(len(qubits))

    payload = {
        "prepared_labels": list(prepared_labels),
        "shots_per_qubit": {q.uid: {} for q in qubits},
    }
    from experiments.iq_cloud_common import iq_cloud_handle

    for label in prepared_labels:
        for q in qubits:
            data = np.asarray(result[iq_cloud_handle(q.uid, label)].data).reshape(-1)
            payload["shots_per_qubit"][q.uid][label] = data.tolist()
    return payload


@workflow.task
def fit_decision_models(
    processed_data: dict,
    qubits: QuantumElements,
    ridge_target_condition: float = 1e6,
) -> dict:
    """Fit per-qubit shared-covariance Gaussian Bayes decision models."""
    qubits = validate_and_convert_qubits_sweeps(qubits)
    validate_supported_num_qubits(len(qubits))
    prepared_labels = prepared_labels_for_num_qubits(len(qubits))
    qubit_uids = [q.uid for q in qubits]
    shot_arrays = _build_shot_arrays(processed_data, qubit_uids, prepared_labels)
    return _fit_models_core(
        shot_arrays=shot_arrays,
        qubit_uids=qubit_uids,
        prepared_labels=prepared_labels,
        target_condition=float(ridge_target_condition),
    )


@workflow.task
def extract_thresholds(decision_model: dict) -> dict:
    """Extract threshold representation (w, b, t) from decision models."""
    out = {}
    for uid, model in decision_model.items():
        out[uid] = {
            "w": [float(model["w"][0]), float(model["w"][1])],
            "b": float(model["b"]),
            "t": float(model["t"]),
            "axis_unit": [float(model["axis_unit"][0]), float(model["axis_unit"][1])],
        }
    return out


@workflow.task
def calculate_confusion_and_fidelity(
    processed_data: dict,
    decision_model: dict,
    qubits: QuantumElements,
) -> dict:
    """Compute confusion matrices (counts+normalized) and assignment fidelities."""
    qubits = validate_and_convert_qubits_sweeps(qubits)
    validate_supported_num_qubits(len(qubits))
    prepared_labels = prepared_labels_for_num_qubits(len(qubits))
    qubit_uids = [q.uid for q in qubits]
    shot_arrays = _build_shot_arrays(processed_data, qubit_uids, prepared_labels)
    return _assignment_core(
        shot_arrays=shot_arrays,
        decision_model=decision_model,
        qubit_uids=qubit_uids,
        prepared_labels=prepared_labels,
    )


@workflow.task
def select_confusion_matrices(assignment_bundle: dict) -> dict:
    """Select confusion-matrix payload from assignment bundle."""
    return assignment_bundle.get("confusion_matrices", {})


@workflow.task
def select_assignment_fidelity(assignment_bundle: dict) -> dict:
    """Select assignment-fidelity payload from assignment bundle."""
    return assignment_bundle.get("assignment_fidelity", {})


@workflow.task
def calculate_separation_metrics(decision_model: dict) -> dict:
    """Compute separation metrics per qubit."""
    return {"per_qubit": _separation_core(decision_model)}


@workflow.task
def bootstrap_metrics(
    processed_data: dict,
    qubits: QuantumElements,
    ridge_target_condition: float = 1e6,
    bootstrap_samples: int = 2000,
    bootstrap_confidence_level: float = 0.95,
    bootstrap_seed: int | None = None,
) -> dict:
    """Estimate bootstrap uncertainty for fidelity, threshold, and SNR metrics."""
    qubits = validate_and_convert_qubits_sweeps(qubits)
    validate_supported_num_qubits(len(qubits))
    prepared_labels = prepared_labels_for_num_qubits(len(qubits))
    qubit_uids = [q.uid for q in qubits]
    shot_arrays = _build_shot_arrays(processed_data, qubit_uids, prepared_labels)

    rng = np.random.default_rng(bootstrap_seed)
    b = int(max(0, bootstrap_samples))
    cl = float(bootstrap_confidence_level)
    cl = min(max(cl, 1e-6), 1.0 - 1e-6)

    per_qubit_vals = {
        uid: {
            "fidelity": [],
            "threshold": [],
            "delta_mu_over_sigma": [],
        }
        for uid in qubit_uids
    }
    joint_fidelity_vals: list[float] = []
    average_fidelity_vals: list[float] = []

    for _ in range(b):
        sampled = {uid: {} for uid in qubit_uids}
        for label in prepared_labels:
            if len(qubit_uids) == 1:
                uid = qubit_uids[0]
                shots = shot_arrays[uid][label]
                n = len(shots)
                idx = rng.integers(0, n, size=n) if n > 0 else np.array([], dtype=int)
                sampled[uid][label] = shots[idx]
            else:
                uid0, uid1 = qubit_uids
                shots0 = shot_arrays[uid0][label]
                shots1 = shot_arrays[uid1][label]
                n = min(len(shots0), len(shots1))
                idx = rng.integers(0, n, size=n) if n > 0 else np.array([], dtype=int)
                sampled[uid0][label] = shots0[:n][idx]
                sampled[uid1][label] = shots1[:n][idx]

        model_bs = _fit_models_core(
            shot_arrays=sampled,
            qubit_uids=qubit_uids,
            prepared_labels=prepared_labels,
            target_condition=float(ridge_target_condition),
        )
        assignment_bs = _assignment_core(
            shot_arrays=sampled,
            decision_model=model_bs,
            qubit_uids=qubit_uids,
            prepared_labels=prepared_labels,
        )
        separation_bs = _separation_core(model_bs)

        for uid in qubit_uids:
            per_qubit_vals[uid]["fidelity"].append(
                float(assignment_bs["assignment_fidelity"]["per_qubit"][uid])
            )
            per_qubit_vals[uid]["threshold"].append(float(model_bs[uid]["t"]))
            per_qubit_vals[uid]["delta_mu_over_sigma"].append(
                float(separation_bs[uid]["delta_mu_over_sigma"])
            )

        if len(qubit_uids) == 2:
            joint_fidelity_vals.append(float(assignment_bs["assignment_fidelity"]["joint"]))
            average_fidelity_vals.append(
                float(assignment_bs["assignment_fidelity"]["average"])
            )

    out = {"per_qubit": {}}
    for uid in qubit_uids:
        out["per_qubit"][uid] = {
            "fidelity": _ci_summary(per_qubit_vals[uid]["fidelity"], cl),
            "threshold": _ci_summary(per_qubit_vals[uid]["threshold"], cl),
            "delta_mu_over_sigma": _ci_summary(
                per_qubit_vals[uid]["delta_mu_over_sigma"], cl
            ),
        }

    if len(qubit_uids) == 2:
        out["joint"] = {"fidelity": _ci_summary(joint_fidelity_vals, cl)}
        out["average"] = {"fidelity": _ci_summary(average_fidelity_vals, cl)}

    out["settings"] = {
        "bootstrap_samples": b,
        "confidence_level": cl,
        "seed": bootstrap_seed,
    }
    return out


@workflow.task
def extract_qubit_parameters_for_discrimination(
    qubits: QuantumElements,
    decision_model: dict,
    enforce_constant_kernel: bool = True,
) -> dict:
    """Build old/new parameter payload for discrimination threshold updates."""
    qubits = validate_and_convert_qubits_sweeps(qubits)
    out = {
        "old_parameter_values": {q.uid: {} for q in qubits},
        "new_parameter_values": {q.uid: {} for q in qubits},
    }
    for q in qubits:
        out["old_parameter_values"][q.uid] = {
            "readout_integration_discrimination_thresholds": (
                q.parameters.readout_integration_discrimination_thresholds
            ),
            "readout_integration_kernels_type": (
                q.parameters.readout_integration_kernels_type
            ),
            "readout_integration_kernels": q.parameters.readout_integration_kernels,
        }
        threshold = float(decision_model[q.uid]["t"])
        out["new_parameter_values"][q.uid][
            "readout_integration_discrimination_thresholds"
        ] = [threshold]
        if enforce_constant_kernel:
            out["new_parameter_values"][q.uid]["readout_integration_kernels_type"] = (
                "default"
            )
            out["new_parameter_values"][q.uid]["readout_integration_kernels"] = None
    return out


@workflow.task
def plot_iq_clouds(
    processed_data: dict,
    decision_model: dict,
    qubits: QuantumElements,
    bootstrap: dict | None = None,
) -> dict[str, mpl.figure.Figure]:
    """Plot IQ clouds with Bayes boundary and projected histograms."""
    qubits = validate_and_convert_qubits_sweeps(qubits)
    validate_supported_num_qubits(len(qubits))
    prepared_labels = prepared_labels_for_num_qubits(len(qubits))
    qubit_uids = [q.uid for q in qubits]
    shot_arrays = _build_shot_arrays(processed_data, qubit_uids, prepared_labels)

    figures = {}
    for q_index, uid in enumerate(qubit_uids):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        ax_iq, ax_proj = axes
        model = decision_model[uid]
        w = np.asarray(model["w"], dtype=float)
        b = float(model["b"])
        axis = np.asarray(model["axis_unit"], dtype=float)
        mu_g = np.asarray(model["mu_g"], dtype=float)
        mu_e = np.asarray(model["mu_e"], dtype=float)
        sigma = np.asarray(model["sigma"], dtype=float)
        threshold = float(model["t"])
        snr = _snr_delta_mu_over_sigma(model)
        snr_ci = _bootstrap_per_qubit_entry(bootstrap, uid, "delta_mu_over_sigma")
        thr_ci = _bootstrap_per_qubit_entry(bootstrap, uid, "threshold")
        thr_center = float(thr_ci.get("mean", threshold)) if isinstance(thr_ci, dict) else threshold

        g_labels = [label for label in prepared_labels if _label_to_bits(label)[q_index] == 0]
        e_labels = [label for label in prepared_labels if _label_to_bits(label)[q_index] == 1]
        g_shots = np.concatenate([shot_arrays[uid][label] for label in g_labels])
        e_shots = np.concatenate([shot_arrays[uid][label] for label in e_labels])

        ax_iq.scatter(
            np.real(g_shots),
            np.imag(g_shots),
            s=8,
            alpha=0.25,
            label="g",
            color="tab:blue",
        )
        ax_iq.scatter(
            np.real(e_shots),
            np.imag(e_shots),
            s=8,
            alpha=0.25,
            label="e",
            color="tab:red",
        )
        ax_iq.plot(mu_g[0], mu_g[1], "o", color="tab:blue", ms=8)
        ax_iq.plot(mu_e[0], mu_e[1], "o", color="tab:red", ms=8)

        ell_g = _ellipse_from_covariance(mu_g, sigma)
        ell_e = _ellipse_from_covariance(mu_e, sigma)
        ell_g.set_edgecolor("tab:blue")
        ell_e.set_edgecolor("tab:red")
        ax_iq.add_patch(ell_g)
        ax_iq.add_patch(ell_e)

        x_min, x_max = ax_iq.get_xlim()
        y_min, y_max = ax_iq.get_ylim()
        if abs(w[1]) > _EPS:
            xs = np.linspace(x_min, x_max, 200)
            ys = -(w[0] * xs + b) / w[1]
            ax_iq.plot(xs, ys, "--", color="k", lw=1.5, label="Bayes boundary")
        elif abs(w[0]) > _EPS:
            x0 = -b / w[0]
            ax_iq.axvline(x0, linestyle="--", color="k", lw=1.5, label="Bayes boundary")

        ax_iq.set_xlim(x_min, x_max)
        ax_iq.set_ylim(y_min, y_max)
        ax_iq.set_title(f"{uid}: IQ cloud (SNR={_format_ci(snr, snr_ci, digits=3)})")
        ax_iq.set_xlabel("I")
        ax_iq.set_ylabel("Q")
        ax_iq.set_aspect("equal", adjustable="box")
        ax_iq.legend(frameon=False)

        proj_g = _real2(g_shots) @ axis
        proj_e = _real2(e_shots) @ axis
        ax_proj.hist(proj_g, bins=80, density=True, alpha=0.45, label="g", color="tab:blue")
        ax_proj.hist(proj_e, bins=80, density=True, alpha=0.45, label="e", color="tab:red")
        ax_proj.axvline(thr_center, linestyle="--", color="k", label=f"t={thr_center:.4g}")
        if isinstance(thr_ci, dict):
            t_low = float(thr_ci.get("ci_low", thr_center))
            t_high = float(thr_ci.get("ci_high", thr_center))
            ax_proj.axvspan(t_low, t_high, color="k", alpha=0.12, label="t 95% CI")
        ax_proj.set_title(
            f"Projection on LDA axis (SNR={_format_ci(snr, snr_ci, digits=3)})"
        )
        ax_proj.set_xlabel("Projected coordinate")
        ax_proj.legend(frameon=False)

        fig.tight_layout()
        _save_artifact_if_available(f"iq_cloud_{uid}", fig)
        figures[uid] = fig
    return figures


@workflow.task
def plot_assignment_matrices(
    confusion_matrices: dict,
    assignment_fidelity: dict,
    qubits: QuantumElements,
    separation_metrics: dict | None = None,
    bootstrap: dict | None = None,
) -> dict[str, mpl.figure.Figure]:
    """Plot assignment matrices for per-qubit and optional 2Q joint metrics."""
    qubits = validate_and_convert_qubits_sweeps(qubits)
    validate_supported_num_qubits(len(qubits))
    qubit_uids = [q.uid for q in qubits]

    panel_count = len(qubit_uids) + (1 if len(qubit_uids) == 2 else 0)
    fig, axes = plt.subplots(1, panel_count, figsize=(4.6 * panel_count, 3.8), squeeze=False)
    axes = axes.ravel()

    for i, uid in enumerate(qubit_uids):
        matrix = np.asarray(confusion_matrices["per_qubit"][uid]["normalized"], dtype=float)
        ax = axes[i]
        im = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="Blues")
        fid = float(assignment_fidelity["per_qubit"][uid])
        fid_ci = _bootstrap_per_qubit_entry(bootstrap, uid, "fidelity")
        snr = 0.0
        if separation_metrics is not None:
            snr = float(
                separation_metrics.get("per_qubit", {})
                .get(uid, {})
                .get("delta_mu_over_sigma", 0.0)
            )
        snr_ci = _bootstrap_per_qubit_entry(bootstrap, uid, "delta_mu_over_sigma")
        ax.set_title(
            f"{uid} fidelity={_format_ci(fid, fid_ci, digits=3)}, "
            f"SNR={_format_ci(snr, snr_ci, digits=3)}"
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Prepared")
        ax.set_xticks([0, 1], ["g", "e"])
        ax.set_yticks([0, 1], ["g", "e"])
        for r in range(2):
            for c in range(2):
                v = matrix[r, c]
                ax.text(
                    c,
                    r,
                    f"{v:.3f}",
                    ha="center",
                    va="center",
                    color="white" if v > 0.6 else "black",
                )
        fig.colorbar(im, ax=ax, fraction=0.046)

    if len(qubit_uids) == 2:
        ax = axes[-1]
        matrix = np.asarray(confusion_matrices["joint"]["normalized"], dtype=float)
        im = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="Reds")
        fid = float(assignment_fidelity["joint"])
        avg_fid = float(assignment_fidelity["average"])
        joint_ci = bootstrap.get("joint", {}).get("fidelity") if isinstance(bootstrap, dict) else None
        avg_ci = bootstrap.get("average", {}).get("fidelity") if isinstance(bootstrap, dict) else None
        ax.set_title(
            f"Joint={_format_ci(fid, joint_ci, digits=3)}, "
            f"Avg={_format_ci(avg_fid, avg_ci, digits=3)}"
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Prepared")
        labels = list(JOINT_LABELS_2Q)
        ax.set_xticks(range(4), labels)
        ax.set_yticks(range(4), labels)
        for r in range(4):
            for c in range(4):
                v = matrix[r, c]
                ax.text(
                    c,
                    r,
                    f"{v:.3f}",
                    ha="center",
                    va="center",
                    color="white" if v > 0.6 else "black",
                    fontsize=8,
                )
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.tight_layout()
    _save_artifact_if_available("iq_cloud_assignment_matrices", fig)
    return {"assignment_matrices": fig}


@workflow.task
def plot_bootstrap_summary(
    bootstrap: dict,
    qubits: QuantumElements,
) -> dict[str, mpl.figure.Figure]:
    """Plot bootstrap uncertainty summary for fidelity/threshold/SNR."""
    qubits = validate_and_convert_qubits_sweeps(qubits)
    qubit_uids = [q.uid for q in qubits]
    if not isinstance(bootstrap, dict):
        return {}

    metrics = [
        ("fidelity", "Fidelity"),
        ("threshold", "Threshold"),
        ("delta_mu_over_sigma", "SNR (delta_mu_over_sigma)"),
    ]
    x = np.arange(len(qubit_uids))

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), squeeze=False)
    axes = axes.ravel()
    for ax, (key, title) in zip(axes, metrics):
        means = []
        low_err = []
        high_err = []
        for uid in qubit_uids:
            entry = bootstrap.get("per_qubit", {}).get(uid, {}).get(key, {})
            mean = float(entry.get("mean", 0.0))
            lo = float(entry.get("ci_low", mean))
            hi = float(entry.get("ci_high", mean))
            means.append(mean)
            low_err.append(max(mean - lo, 0.0))
            high_err.append(max(hi - mean, 0.0))

        yerr = np.vstack([low_err, high_err])
        ax.errorbar(
            x,
            means,
            yerr=yerr,
            fmt="o",
            capsize=4,
            linestyle="None",
            color="tab:blue",
        )
        ax.set_xticks(x, qubit_uids)
        ax.set_title(title)
        ax.grid(alpha=0.2)
        if key == "fidelity":
            ax.set_ylim(0.0, 1.02)
            if len(qubit_uids) == 2:
                joint = bootstrap.get("joint", {}).get("fidelity")
                avg = bootstrap.get("average", {}).get("fidelity")
                if isinstance(joint, dict):
                    ax.axhline(float(joint.get("mean", 0.0)), color="tab:red", linestyle="--", label="joint mean")
                if isinstance(avg, dict):
                    ax.axhline(float(avg.get("mean", 0.0)), color="tab:green", linestyle=":", label="avg mean")
                if isinstance(joint, dict) or isinstance(avg, dict):
                    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    _save_artifact_if_available("iq_cloud_bootstrap_summary", fig)
    return {"bootstrap_summary": fig}

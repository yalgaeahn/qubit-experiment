"""Shared helpers for readout sweep analyses."""

from __future__ import annotations

import numpy as np
from laboneq.simple import dsl

_EPS = 1e-12


def unwrap_result_like(result_like):
    """Unwrap nested workflow payloads until a result-like object is reached."""
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


def as_1d_complex(data) -> np.ndarray:
    """Convert shot data into a flat complex-valued vector."""
    return np.asarray(data, dtype=complex).reshape(-1)


def extract_sweep_shots(raw_data, n_points: int) -> np.ndarray:
    """Extract sweep-point shot vectors as an array of shape (n_points, n_shots)."""
    if int(n_points) < 1:
        raise ValueError("n_points must be >= 1.")

    n_points = int(n_points)
    arr = np.asarray(raw_data, dtype=complex)

    if arr.ndim == 0:
        if n_points != 1:
            raise ValueError("Scalar shot data cannot be mapped to multiple sweep points.")
        return arr.reshape(1, 1)

    if arr.ndim == 1:
        if n_points == 1:
            return arr.reshape(1, -1)
        if arr.size % n_points == 0:
            return arr.reshape(n_points, -1)
        raise ValueError(
            f"Could not infer sweep axis: raw size {arr.size} is not divisible by {n_points}."
        )

    for axis, size in enumerate(arr.shape):
        if int(size) == n_points:
            moved = np.moveaxis(arr, axis, 0)
            return moved.reshape(n_points, -1)

    if arr.size % n_points == 0:
        return arr.reshape(n_points, -1)

    raise ValueError(
        "Could not extract sweep-point shots because no axis matches the sweep length "
        f"{n_points} and raw size {arr.size} is not divisible by {n_points}."
    )


def calibration_shots_by_state_and_sweep(
    result,
    qubit_uid: str,
    states: tuple[str, str] = ("g", "e"),
    n_points: int = 1,
) -> dict[str, np.ndarray]:
    """Return calibration shots per state and sweep point."""
    out = {}
    for state in states:
        raw = result[dsl.handles.calibration_trace_handle(qubit_uid, state)].data
        out[state] = extract_sweep_shots(raw, n_points=n_points)
    return out


def calibration_shots_by_state(
    result,
    qubit_uid: str,
    states: tuple[str, str] = ("g", "e"),
) -> dict[str, np.ndarray]:
    """Return flattened calibration shots per state."""
    out = {}
    for state in states:
        raw = result[dsl.handles.calibration_trace_handle(qubit_uid, state)].data
        out[state] = as_1d_complex(raw)
    return out


def _real2(shots: np.ndarray) -> np.ndarray:
    return np.column_stack([np.real(shots), np.imag(shots)])


def _safe_cov(x: np.ndarray) -> np.ndarray:
    if x.shape[0] <= 1:
        return np.zeros((2, 2), dtype=float)
    return np.asarray(np.cov(x, rowvar=False, bias=False), dtype=float).reshape(2, 2)


def _ridge_lambda_trace_scaled(
    pooled_covariance: np.ndarray,
    target_condition: float,
) -> float:
    cov = 0.5 * (pooled_covariance + pooled_covariance.T)
    eig = np.linalg.eigvalsh(cov)
    max_eig = float(np.max(eig))
    min_eig = float(np.min(eig))

    lambda_pd = max(0.0, -min_eig + _EPS)
    if target_condition <= 1.0:
        lambda_cond = 0.0
    else:
        lambda_cond = max(
            0.0,
            (max_eig - target_condition * min_eig) / (target_condition - 1.0),
        )
    return float(max(lambda_pd, lambda_cond))


def _fit_binary_shared_cov_model(
    shots_g: np.ndarray,
    shots_e: np.ndarray,
    target_condition: float,
) -> dict[str, np.ndarray | float]:
    xg = _real2(shots_g)
    xe = _real2(shots_e)
    if xg.shape[0] < 1 or xe.shape[0] < 1:
        raise ValueError("Non-empty g/e shots are required for model fitting.")

    mu_g = np.mean(xg, axis=0)
    mu_e = np.mean(xe, axis=0)
    cov_g = _safe_cov(xg)
    cov_e = _safe_cov(xe)
    ng = xg.shape[0]
    ne = xe.shape[0]
    dof = max(ng + ne - 2, 1)
    pooled_cov = ((max(ng - 1, 0) * cov_g) + (max(ne - 1, 0) * cov_e)) / dof

    ridge_lambda = _ridge_lambda_trace_scaled(
        pooled_covariance=pooled_cov,
        target_condition=target_condition,
    )
    sigma = pooled_cov + ridge_lambda * np.eye(2)
    inv_sigma = np.linalg.inv(sigma)

    delta = mu_e - mu_g
    w = inv_sigma @ delta
    b = -0.5 * (mu_e @ inv_sigma @ mu_e - mu_g @ inv_sigma @ mu_g)
    return {"w": w, "b": float(b), "sigma": sigma, "mu_g": mu_g, "mu_e": mu_e}


def _predict_bits(shots: np.ndarray, model: dict[str, np.ndarray | float]) -> np.ndarray:
    x = _real2(shots)
    w = np.asarray(model["w"], dtype=float)
    b = float(model["b"])
    return (x @ w + b >= 0.0).astype(int)


def evaluate_iq_binary(
    shots_g: np.ndarray,
    shots_e: np.ndarray,
    target_condition: float = 1e6,
) -> dict[str, float]:
    """Compute assignment fidelity and delta_mu_over_sigma for binary g/e IQ data."""
    shots_g = as_1d_complex(shots_g)
    shots_e = as_1d_complex(shots_e)
    if shots_g.size < 1 or shots_e.size < 1:
        return {"assignment_fidelity": 0.0, "delta_mu_over_sigma": 0.0}

    model = _fit_binary_shared_cov_model(
        shots_g=shots_g,
        shots_e=shots_e,
        target_condition=float(target_condition),
    )
    pred_g = _predict_bits(shots_g, model)
    pred_e = _predict_bits(shots_e, model)
    p00 = float(np.mean(pred_g == 0)) if pred_g.size > 0 else 0.0
    p11 = float(np.mean(pred_e == 1)) if pred_e.size > 0 else 0.0
    fidelity = 0.5 * (p00 + p11)

    w = np.asarray(model["w"], dtype=float)
    sigma = np.asarray(model["sigma"], dtype=float)
    delta = np.asarray(model["mu_e"], dtype=float) - np.asarray(model["mu_g"], dtype=float)
    num = abs(float(w.T @ delta))
    den = np.sqrt(max(float(w.T @ sigma @ w), 0.0))
    snr = float(num / den) if den > _EPS else 0.0
    return {"assignment_fidelity": float(fidelity), "delta_mu_over_sigma": float(snr)}


def select_best_index(
    assignment_fidelity: np.ndarray,
    delta_mu_over_sigma: np.ndarray,
    fidelity_tolerance: float = 5e-4,
    prefer_smallest: bool = True,
) -> dict[str, int | float | str]:
    """Select best sweep index using fidelity first, then SNR tie-break."""
    fid = np.asarray(assignment_fidelity, dtype=float).reshape(-1)
    snr = np.asarray(delta_mu_over_sigma, dtype=float).reshape(-1)
    n = fid.size
    if n < 1:
        raise ValueError("Cannot select best index from empty metric arrays.")

    finite = np.isfinite(fid)
    fid_clean = np.where(finite, fid, -np.inf)
    idx_argmax = int(np.argmax(fid_clean))
    max_fid = float(fid_clean[idx_argmax])
    tol = max(0.0, float(fidelity_tolerance))

    candidate_idx = np.where(fid_clean >= max_fid - tol)[0]
    if candidate_idx.size == 0:
        candidate_idx = np.array([idx_argmax], dtype=int)

    candidate_snr = np.asarray(snr[candidate_idx], dtype=float)
    candidate_snr = np.where(np.isfinite(candidate_snr), candidate_snr, -np.inf)
    max_candidate_snr = float(np.max(candidate_snr))
    top_snr_idx = candidate_idx[candidate_snr >= max_candidate_snr - 1e-12]

    if prefer_smallest:
        best_idx = int(np.min(top_snr_idx))
    else:
        center = 0.5 * (n - 1)
        best_idx = int(top_snr_idx[np.argmin(np.abs(top_snr_idx - center))])

    sorted_fid = np.sort(fid_clean[np.isfinite(fid_clean)])
    second_best = float(sorted_fid[-2]) if sorted_fid.size >= 2 else -np.inf
    low_margin = bool(max_fid - second_best <= tol)
    flat_optimum = bool(candidate_idx.size > 1)
    edge_hit = bool(best_idx in (0, n - 1))

    if edge_hit:
        quality_flag = "edge_hit"
    elif low_margin:
        quality_flag = "low_margin"
    elif flat_optimum:
        quality_flag = "flat_optimum"
    else:
        quality_flag = "ok"

    return {
        "index": int(best_idx),
        "max_assignment_fidelity": float(max_fid),
        "quality_flag": quality_flag,
        "num_candidates": int(candidate_idx.size),
    }

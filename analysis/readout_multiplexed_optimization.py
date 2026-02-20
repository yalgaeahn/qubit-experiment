"""Analysis helpers/workflow for 2Q multiplexed readout optimization."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from laboneq import workflow

if TYPE_CHECKING:
    import matplotlib as mpl


_EPS = 1e-12


def _deep_find_analysis_payload(obj, *, max_depth: int = 24):
    queue = [(obj, 0)]
    seen = set()
    attrs = ("output", "analysis_result", "result", "analysis_workflow_result")

    while queue:
        current, depth = queue.pop(0)
        if current is None or depth > max_depth:
            continue
        oid = id(current)
        if oid in seen:
            continue
        seen.add(oid)

        if isinstance(current, dict):
            af = current.get("assignment_fidelity")
            if isinstance(af, dict) and "per_qubit" in af:
                return current
            for k in attrs:
                if k in current:
                    queue.append((current[k], depth + 1))
            for v in current.values():
                queue.append((v, depth + 1))
            continue

        if isinstance(current, (list, tuple, set)):
            for v in current:
                queue.append((v, depth + 1))
            continue

        for a in attrs:
            if hasattr(current, a):
                try:
                    queue.append((getattr(current, a), depth + 1))
                except Exception:
                    pass

        d = getattr(current, "__dict__", None)
        if isinstance(d, dict):
            for v in d.values():
                queue.append((v, depth + 1))

    return None


def extract_candidate_metrics(
    candidate_result,
    *,
    ctrl_uid: str,
    targ_uid: str,
) -> dict:
    """Extract per-candidate fidelity metrics from iq_cloud workflow result."""
    payload = _deep_find_analysis_payload(candidate_result)
    if not isinstance(payload, dict):
        return {
            "valid": False,
            "error": "analysis_payload_not_found",
            "F_ctrl": float("nan"),
            "F_targ": float("nan"),
            "F_joint": float("nan"),
            "F_avg": float("nan"),
            "F_min": float("nan"),
            "overrange_count": None,
        }

    af = payload.get("assignment_fidelity", {})
    per = af.get("per_qubit", {}) if isinstance(af, dict) else {}
    try:
        f_ctrl = float(per.get(ctrl_uid, np.nan))
        f_targ = float(per.get(targ_uid, np.nan))
        f_joint = float(af.get("joint", np.nan))
        f_avg = float(af.get("average", np.nan))
    except Exception:
        f_ctrl = np.nan
        f_targ = np.nan
        f_joint = np.nan
        f_avg = np.nan

    if not np.isfinite(f_avg):
        vals = [x for x in (f_ctrl, f_targ) if np.isfinite(x)]
        f_avg = float(np.mean(vals)) if vals else np.nan
    f_min = min(f_ctrl, f_targ) if np.isfinite(f_ctrl) and np.isfinite(f_targ) else np.nan
    valid = bool(np.isfinite(f_ctrl) and np.isfinite(f_targ) and np.isfinite(f_joint))

    return {
        "valid": valid,
        "error": None if valid else "invalid_fidelity",
        "F_ctrl": float(f_ctrl),
        "F_targ": float(f_targ),
        "F_joint": float(f_joint),
        "F_avg": float(f_avg),
        "F_min": float(f_min),
        # Placeholder for future hardware counter ingestion.
        "overrange_count": None,
    }


def _stage_cost(
    state: dict[str, dict[str, float]],
    *,
    stage: str,
    target_uid: str | None,
    center: dict[str, dict[str, float]] | None,
) -> float:
    stage_l = stage.lower()
    if "frequency" in stage_l:
        if center is None:
            return 0.0
        total = 0.0
        for uid, q_state in state.items():
            c = center.get(uid, {})
            total += abs(
                float(q_state.get("readout_resonator_frequency", 0.0))
                - float(c.get("readout_resonator_frequency", 0.0))
            )
        return float(total)
    if "amplitude" in stage_l:
        uid = target_uid
        if uid is None:
            return float(sum(float(s.get("readout_amplitude", 0.0)) for s in state.values()))
        return float(state[uid].get("readout_amplitude", 0.0))
    if "length" in stage_l and "integration-window" not in stage_l:
        uid = target_uid
        if uid is None:
            return float(sum(float(s.get("readout_length", 0.0)) for s in state.values()))
        return float(state[uid].get("readout_length", 0.0))
    if "integration-window" in stage_l:
        uid = target_uid
        if uid is None:
            return float(
                sum(
                    float(s.get("readout_integration_delay", 0.0))
                    + float(s.get("readout_integration_length", 0.0))
                    for s in state.values()
                )
            )
        s = state[uid]
        return float(s.get("readout_integration_delay", 0.0)) + float(
            s.get("readout_integration_length", 0.0)
        )
    return 0.0


def select_best_candidate(
    candidates: list[dict],
    *,
    stage: str,
    center: dict[str, dict[str, float]] | None = None,
    target_uid: str | None = None,
    low_margin_tol: float = 5e-4,
) -> dict:
    """Select best candidate with maximin objective + deterministic tie-break."""
    if len(candidates) < 1:
        raise ValueError("select_best_candidate requires at least one candidate.")

    rows = []
    for idx, cand in enumerate(candidates):
        metrics = cand.get("metrics", {})
        state = cand.get("state", {})
        fmin = float(metrics.get("F_min", np.nan))
        fj = float(metrics.get("F_joint", np.nan))
        cost = _stage_cost(state, stage=stage, target_uid=target_uid, center=center)
        valid = bool(metrics.get("valid", False)) and np.isfinite(fmin) and np.isfinite(fj)
        rows.append(
            {
                "idx": int(idx),
                "valid": valid,
                "F_min": float(fmin if valid else -np.inf),
                "F_joint": float(fj if valid else -np.inf),
                "cost": float(cost if np.isfinite(cost) else np.inf),
            }
        )

    # Lexicographic: max F_min, max F_joint, min cost, min idx
    best_idx = min(
        range(len(rows)),
        key=lambda i: (
            -rows[i]["F_min"],
            -rows[i]["F_joint"],
            rows[i]["cost"],
            rows[i]["idx"],
        ),
    )
    best_row = rows[best_idx]
    best_candidate = deepcopy(candidates[int(best_row["idx"])])

    fmins = np.array([r["F_min"] for r in rows], dtype=float)
    finite = np.isfinite(fmins)
    quality = "ok"
    if not np.any(finite):
        quality = "invalid_data"
    else:
        f_sorted = np.sort(fmins[finite])
        if f_sorted.size >= 2 and float(f_sorted[-1] - f_sorted[-2]) <= float(low_margin_tol):
            quality = "low_margin"

    ranking = {
        "index": int(best_row["idx"]),
        "F_min": float(best_row["F_min"]),
        "F_joint": float(best_row["F_joint"]),
        "cost": float(best_row["cost"]),
        "quality": quality,
    }
    return {"best_candidate": best_candidate, "ranking": ranking, "rows": rows}


def _stage_metric_arrays(stage_item: dict) -> dict[str, np.ndarray]:
    ms = stage_item.get("metrics_summary", {})
    return {
        "F_min": np.asarray(ms.get("F_min", []), dtype=float).reshape(-1),
        "F_joint": np.asarray(ms.get("F_joint", []), dtype=float).reshape(-1),
        "cost": np.asarray(ms.get("cost", []), dtype=float).reshape(-1),
    }


@workflow.task
def build_summary(result: dict, *, ctrl_uid: str, targ_uid: str) -> dict:
    """Build compact summary payload from optimization result."""
    final_state = deepcopy(result.get("final_state", {}))
    history = list(result.get("history", []))
    diagnostics = deepcopy(result.get("diagnostics", {}))
    best_point = deepcopy(result.get("best_point", {}))
    quality_flag = str(result.get("quality_flag", "unknown"))

    return {
        "final_state": final_state,
        "history": history,
        "diagnostics": diagnostics,
        "best_point": best_point,
        "quality_flag": quality_flag,
        "old_parameter_values": deepcopy(result.get("old_parameter_values", {})),
        "new_parameter_values": deepcopy(result.get("new_parameter_values", {})),
        "ctrl_uid": ctrl_uid,
        "targ_uid": targ_uid,
    }


@workflow.task
def plot_stage_diagnostics(summary: dict) -> dict[str, mpl.figure.Figure]:
    """Create stage-wise diagnostic plots (2D frequency heatmaps + 1D curves)."""
    history = summary.get("history", [])
    figures: dict[str, mpl.figure.Figure] = {}

    for item in history:
        step = str(item.get("step", "stage"))
        arrays = _stage_metric_arrays(item)
        f_min = arrays["F_min"]
        f_joint = arrays["F_joint"]
        cost = arrays["cost"]
        if f_min.size < 1:
            continue

        grid = item.get("grid", {})
        x = np.asarray(grid.get("x", []), dtype=float).reshape(-1)
        y = np.asarray(grid.get("y", []), dtype=float).reshape(-1)
        is_freq = "frequency" in step
        is_2d = bool(is_freq and x.size > 1 and y.size > 1 and f_min.size == x.size * y.size)

        if is_2d:
            z = f_min.reshape(len(x), len(y))
            fig, ax = plt.subplots(1, 1, figsize=(5.8, 4.6))
            im = ax.imshow(
                z.T,
                origin="lower",
                aspect="auto",
                extent=[x.min() * 1e-6, x.max() * 1e-6, y.min() * 1e-6, y.max() * 1e-6],
                cmap="viridis",
            )
            ax.set_title(f"{step}: min(F_ctrl, F_targ)")
            ax.set_xlabel("ctrl readout f (MHz)")
            ax.set_ylabel("targ readout f (MHz)")
            fig.colorbar(im, ax=ax, fraction=0.046, label="min fidelity")
        else:
            axis = np.asarray(grid.get("axis", []), dtype=float).reshape(-1)
            if axis.size != f_min.size:
                axis = np.arange(f_min.size, dtype=float)
            fig, axes = plt.subplots(2, 1, figsize=(6.2, 5.4), sharex=True)
            axes[0].plot(axis, f_min, marker="o", label="min(F_ctrl,F_targ)")
            axes[0].plot(axis, f_joint, marker="s", label="F_joint", alpha=0.75)
            axes[0].legend(frameon=False)
            axes[0].grid(alpha=0.25)
            axes[0].set_ylabel("fidelity")
            axes[0].set_title(step)
            axes[1].plot(axis, cost, marker="^", color="tab:orange")
            axes[1].grid(alpha=0.25)
            axes[1].set_ylabel("stage cost")
            axes[1].set_xlabel(str(grid.get("axis_label", "candidate axis")))

        fig.tight_layout()
        artifact_name = f"readout_mxopt_{step.replace(' ', '_')}"
        workflow.save_artifact(artifact_name, fig)
        figures[artifact_name] = fig

    return figures


@workflow.task
def plot_final_summary(summary: dict) -> mpl.figure.Figure:
    """Plot baseline vs final fidelity summary."""
    diag = summary.get("diagnostics", {})
    base = diag.get("baseline_metrics", {})
    fin = diag.get("final_metrics", {})

    labels = ["F_ctrl", "F_targ", "F_joint", "F_min"]
    base_vals = [float(base.get(k, np.nan)) for k in labels]
    fin_vals = [float(fin.get(k, np.nan)) for k in labels]
    x = np.arange(len(labels), dtype=float)
    w = 0.36

    fig, ax = plt.subplots(1, 1, figsize=(6.2, 4.2))
    ax.bar(x - w / 2, base_vals, width=w, label="baseline")
    ax.bar(x + w / 2, fin_vals, width=w, label="final")
    ax.set_xticks(x, labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("fidelity")
    ax.set_title("2Q multiplexed readout optimization summary")
    ax.grid(alpha=0.2, axis="y")
    ax.legend(frameon=False)
    fig.tight_layout()
    workflow.save_artifact("readout_mxopt_final_summary", fig)
    return fig


@workflow.workflow_options
class ReadoutMultiplexedOptimizationAnalysisOptions:
    do_plotting: bool = workflow.option_field(
        True,
        description="Whether to create stage/final diagnostic plots.",
    )


@workflow.workflow(name="analysis_readout_multiplexed_optimization")
def analysis_workflow(
    result: dict,
    ctrl_uid: str,
    targ_uid: str,
    options: ReadoutMultiplexedOptimizationAnalysisOptions | None = None,
) -> None:
    """Analysis workflow for 2Q multiplexed readout optimization result payload."""
    options = (
        ReadoutMultiplexedOptimizationAnalysisOptions()
        if options is None
        else options
    )
    summary = build_summary(result=result, ctrl_uid=ctrl_uid, targ_uid=targ_uid)
    with workflow.if_(options.do_plotting):
        plot_stage_diagnostics(summary=summary)
        plot_final_summary(summary=summary)
    workflow.return_(summary)


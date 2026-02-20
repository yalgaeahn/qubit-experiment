"""2Q multiplexed readout optimization workflow."""

from __future__ import annotations

from copy import deepcopy
from itertools import product
from typing import TYPE_CHECKING

import numpy as np
from laboneq import workflow

from analysis import readout_multiplexed_optimization as mx_analysis
from experiments import iq_cloud
from laboneq_applications.tasks.parameter_updating import update_qpu

if TYPE_CHECKING:
    from laboneq.dsl.quantum import QuantumParameters
    from laboneq.dsl.quantum.qpu import QPU
    from laboneq.dsl.session import Session


DEFAULT_CONFIG = {
    "max_readout_length": 2.048e-6,
    "freq_coarse_span": 1.0e6,
    "freq_coarse_points": 7,
    "freq_refine_span": 250e3,
    "freq_refine_points": 5,
    "freq_fine_span": 80e3,
    "freq_fine_points": 5,
    "window_delay_span": 30e-9,
    "window_delay_points": 9,
    "window_length_factors_coarse": (0.7, 0.85, 1.0, 1.15),
    "window_length_factors_refine": (0.9, 1.0, 1.1),
    "amp_abs_range": (0.4, 1.0),
    "amp_coarse_points": 9,
    "amp_refine_frac": 0.12,
    "amp_refine_points": 7,
    "length_abs_range": (0.8e-6, 2.048e-6),
    "length_coarse_points": 6,
    "length_refine_span": 0.2e-6,
    "length_refine_points": 5,
    "count_coarse": 1024,
    "count_refine": 2048,
    "count_verify": 4096,
}


@workflow.workflow_options
class ReadoutMultiplexedOptimizationWorkflowOptions:
    """Workflow options for 2Q multiplexed readout optimization."""

    do_analysis: bool = workflow.option_field(
        True,
        description="Whether to run analysis workflow on optimization result.",
    )
    do_plotting: bool = workflow.option_field(
        True,
        description="Whether to emit diagnostic plots in analysis workflow.",
    )
    update: bool = workflow.option_field(
        False,
        description="Whether to apply final parameters to input qpu.",
    )


def _normalize_config(config: dict | None) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if isinstance(config, dict):
        cfg.update(config)
    return cfg


def _state_from_qubits(ctrl, targ) -> dict[str, dict[str, float]]:
    keys = (
        "readout_resonator_frequency",
        "readout_amplitude",
        "readout_length",
        "readout_integration_delay",
        "readout_integration_length",
    )
    state = {}
    for q in (ctrl, targ):
        state[q.uid] = {k: float(getattr(q.parameters, k)) for k in keys}
    return state


def _merge_temp_parameters(
    base: dict[str | tuple[str, str, str], dict | QuantumParameters] | None,
    state: dict[str, dict[str, float]],
) -> dict[str | tuple[str, str, str], dict | QuantumParameters]:
    merged: dict[str | tuple[str, str, str], dict | QuantumParameters] = {}
    if base is not None:
        for k, v in base.items():
            merged[k] = dict(v) if isinstance(v, dict) else deepcopy(v)

    for uid, q_state in state.items():
        per = merged.get(uid, {})
        if not isinstance(per, dict):
            per = {}
        per = dict(per)
        per.update({k: float(v) for k, v in q_state.items()})
        merged[uid] = per
    return merged


def _candidate_record(
    *,
    state: dict[str, dict[str, float]],
    metrics: dict,
    coord: dict | None,
    target_uid: str | None,
) -> dict:
    return {
        "state": deepcopy(state),
        "metrics": deepcopy(metrics),
        "coord": {} if coord is None else dict(coord),
        "target_uid": target_uid,
    }


def _to_float_list(values) -> list[float]:
    return [float(x) for x in np.asarray(values, dtype=float).reshape(-1)]


@workflow.task
def _run_optimization(
    session: Session,
    qpu: QPU,
    ctrl,
    targ,
    config: dict | None = None,
    temporary_parameters: dict[str | tuple[str, str, str], dict | QuantumParameters]
    | None = None,
) -> dict:
    cfg = _normalize_config(config)
    ctrl_uid = ctrl.uid
    targ_uid = targ.uid

    initial_state = _state_from_qubits(ctrl, targ)
    state = deepcopy(initial_state)
    history: list[dict] = []
    diagnostics: dict = {
        "stage_sequence": [],
        "overrange_records": [],
        "candidate_failures": [],
    }

    def _run_iq_candidate(
        candidate_state: dict[str, dict[str, float]],
        *,
        count: int,
    ) -> dict:
        opts = iq_cloud.experiment_workflow.options()
        opts.count(int(count))
        opts.do_analysis(True)
        opts.update(False)
        opts.do_plotting(False)
        if hasattr(opts, "do_plotting_iq_clouds"):
            opts.do_plotting_iq_clouds(False)
        if hasattr(opts, "do_plotting_assignment_matrices"):
            opts.do_plotting_assignment_matrices(False)
        if hasattr(opts, "do_plotting_bootstrap_summary"):
            opts.do_plotting_bootstrap_summary(False)

        run_result = iq_cloud.experiment_workflow(
            session=session,
            qpu=qpu,
            qubits=[ctrl, targ],
            temporary_parameters=_merge_temp_parameters(temporary_parameters, candidate_state),
            options=opts,
        ).run()
        metrics = mx_analysis.extract_candidate_metrics(
            run_result,
            ctrl_uid=ctrl_uid,
            targ_uid=targ_uid,
        )
        return metrics

    def _stage(
        *,
        step: str,
        candidates: list[dict],
        center: dict[str, dict[str, float]] | None,
        target_uid: str | None,
        count: int,
        grid: dict | None = None,
    ) -> dict[str, dict[str, float]]:
        scored: list[dict] = []
        for idx, cand_state in enumerate(candidates):
            try:
                metrics = _run_iq_candidate(cand_state, count=count)
            except Exception as ex:  # pragma: no cover - runtime hardware failure path
                metrics = {
                    "valid": False,
                    "error": str(ex),
                    "F_ctrl": np.nan,
                    "F_targ": np.nan,
                    "F_joint": np.nan,
                    "F_avg": np.nan,
                    "F_min": np.nan,
                    "overrange_count": None,
                }
                diagnostics["candidate_failures"].append(
                    {"step": step, "index": int(idx), "error": str(ex)}
                )
            diagnostics["overrange_records"].append(
                {
                    "step": step,
                    "index": int(idx),
                    "overrange_count": metrics.get("overrange_count"),
                }
            )
            scored.append(
                _candidate_record(
                    state=cand_state,
                    metrics=metrics,
                    coord={"index": idx},
                    target_uid=target_uid,
                )
            )

        selection = mx_analysis.select_best_candidate(
            scored,
            stage=step,
            center=center,
            target_uid=target_uid,
        )
        best = selection["best_candidate"]
        ranking = selection["ranking"]
        rows = selection["rows"]

        metrics_summary = {
            "F_ctrl": [float(x["metrics"].get("F_ctrl", np.nan)) for x in scored],
            "F_targ": [float(x["metrics"].get("F_targ", np.nan)) for x in scored],
            "F_joint": [float(x["metrics"].get("F_joint", np.nan)) for x in scored],
            "F_avg": [float(x["metrics"].get("F_avg", np.nan)) for x in scored],
            "F_min": [float(x["metrics"].get("F_min", np.nan)) for x in scored],
            "valid": [bool(x["metrics"].get("valid", False)) for x in scored],
            "cost": [float(r.get("cost", np.nan)) for r in rows],
        }

        item = {
            "step": step,
            "candidate_count": len(scored),
            "best_candidate": {
                "index": int(ranking["index"]),
                "metrics": deepcopy(best["metrics"]),
                "state": deepcopy(best["state"]),
                "cost": float(ranking["cost"]),
            },
            "metrics_summary": metrics_summary,
            "state_snapshot": deepcopy(best["state"]),
            "count": int(count),
            "quality": str(ranking["quality"]),
            "grid": {} if grid is None else deepcopy(grid),
        }
        history.append(item)
        diagnostics["stage_sequence"].append(step)
        return deepcopy(best["state"])

    # Stage 0: baseline
    state = _stage(
        step="baseline",
        candidates=[deepcopy(state)],
        center=deepcopy(state),
        target_uid=None,
        count=int(cfg["count_coarse"]),
    )
    baseline_metrics = deepcopy(history[-1]["best_candidate"]["metrics"])

    def _joint_frequency_stage(step: str, span: float, points: int, count: int) -> None:
        nonlocal state
        center = deepcopy(state)
        fx = np.linspace(
            float(center[ctrl_uid]["readout_resonator_frequency"]) - float(span),
            float(center[ctrl_uid]["readout_resonator_frequency"]) + float(span),
            int(points),
        )
        fy = np.linspace(
            float(center[targ_uid]["readout_resonator_frequency"]) - float(span),
            float(center[targ_uid]["readout_resonator_frequency"]) + float(span),
            int(points),
        )
        candidates = []
        for f_ctrl, f_targ in product(fx, fy):
            cand = deepcopy(center)
            cand[ctrl_uid]["readout_resonator_frequency"] = float(f_ctrl)
            cand[targ_uid]["readout_resonator_frequency"] = float(f_targ)
            candidates.append(cand)
        state = _stage(
            step=step,
            candidates=candidates,
            center=center,
            target_uid=None,
            count=count,
            grid={
                "x": _to_float_list(fx),
                "y": _to_float_list(fy),
                "axis_label": "joint_frequency_grid_hz",
            },
        )

    def _integration_window_stage(
        step_prefix: str,
        *,
        length_factors: tuple[float, ...],
        count: int,
    ) -> None:
        nonlocal state
        max_len = float(cfg["max_readout_length"])
        for uid in (ctrl_uid, targ_uid):
            center = deepcopy(state)
            c_delay = float(center[uid]["readout_integration_delay"])
            c_len = float(center[uid]["readout_integration_length"])
            delays = np.linspace(
                max(0.0, c_delay - float(cfg["window_delay_span"])),
                c_delay + float(cfg["window_delay_span"]),
                int(cfg["window_delay_points"]),
            )
            lengths = np.asarray(length_factors, dtype=float) * c_len
            lengths = np.unique(np.clip(lengths, 0.0, max_len))
            candidates = []
            for d, l in product(delays, lengths):
                cand = deepcopy(center)
                cand[uid]["readout_integration_delay"] = float(d)
                cand[uid]["readout_integration_length"] = float(l)
                candidates.append(cand)
            state = _stage(
                step=f"{step_prefix}-{uid}",
                candidates=candidates,
                center=center,
                target_uid=uid,
                count=count,
                grid={
                    "axis": list(range(len(candidates))),
                    "axis_label": f"{uid} integration-window candidate",
                },
            )

    def _amplitude_stage(step_prefix: str, *, refine: bool, count: int) -> None:
        nonlocal state
        for uid in (ctrl_uid, targ_uid):
            center = deepcopy(state)
            if not refine:
                lo, hi = [float(x) for x in cfg["amp_abs_range"]]
                amps = np.linspace(lo, hi, int(cfg["amp_coarse_points"]))
            else:
                base = float(center[uid]["readout_amplitude"])
                span = max(0.02, abs(base) * float(cfg["amp_refine_frac"]))
                amps = np.linspace(
                    max(0.0, base - span),
                    min(1.0, base + span),
                    int(cfg["amp_refine_points"]),
                )
            amps = np.unique(np.clip(amps, 0.0, 1.0))
            candidates = []
            for amp in amps:
                cand = deepcopy(center)
                cand[uid]["readout_amplitude"] = float(amp)
                candidates.append(cand)
            state = _stage(
                step=f"{step_prefix}-{uid}",
                candidates=candidates,
                center=center,
                target_uid=uid,
                count=count,
                grid={
                    "axis": _to_float_list(amps),
                    "axis_label": f"{uid} readout_amplitude",
                },
            )

    def _length_stage(step_prefix: str, *, refine: bool, count: int) -> None:
        nonlocal state
        max_len = float(cfg["max_readout_length"])
        for uid in (ctrl_uid, targ_uid):
            center = deepcopy(state)
            if not refine:
                lo, hi = [float(x) for x in cfg["length_abs_range"]]
                lengths = np.linspace(lo, hi, int(cfg["length_coarse_points"]))
            else:
                base = float(center[uid]["readout_length"])
                span = float(cfg["length_refine_span"])
                lengths = np.linspace(
                    max(0.0, base - span),
                    min(max_len, base + span),
                    int(cfg["length_refine_points"]),
                )
            lengths = np.unique(np.clip(lengths, 0.0, max_len))
            candidates = []
            for l in lengths:
                cand = deepcopy(center)
                cand[uid]["readout_length"] = float(l)
                candidates.append(cand)
            state = _stage(
                step=f"{step_prefix}-{uid}",
                candidates=candidates,
                center=center,
                target_uid=uid,
                count=count,
                grid={
                    "axis": _to_float_list(lengths),
                    "axis_label": f"{uid} readout_length (s)",
                },
            )

    # Stage 1~10
    _joint_frequency_stage(
        "joint-frequency-coarse",
        span=float(cfg["freq_coarse_span"]),
        points=int(cfg["freq_coarse_points"]),
        count=int(cfg["count_coarse"]),
    )
    _joint_frequency_stage(
        "joint-frequency-refine",
        span=float(cfg["freq_refine_span"]),
        points=int(cfg["freq_refine_points"]),
        count=int(cfg["count_refine"]),
    )
    _integration_window_stage(
        "integration-window-coarse",
        length_factors=tuple(cfg["window_length_factors_coarse"]),
        count=int(cfg["count_coarse"]),
    )
    _integration_window_stage(
        "integration-window-refine",
        length_factors=tuple(cfg["window_length_factors_refine"]),
        count=int(cfg["count_refine"]),
    )
    _amplitude_stage(
        "amplitude-coarse",
        refine=False,
        count=int(cfg["count_coarse"]),
    )
    _amplitude_stage(
        "amplitude-refine",
        refine=True,
        count=int(cfg["count_refine"]),
    )
    _length_stage(
        "length-coarse",
        refine=False,
        count=int(cfg["count_coarse"]),
    )
    _length_stage(
        "length-refine",
        refine=True,
        count=int(cfg["count_refine"]),
    )
    _joint_frequency_stage(
        "joint-frequency-fine",
        span=float(cfg["freq_fine_span"]),
        points=int(cfg["freq_fine_points"]),
        count=int(cfg["count_refine"]),
    )
    state = _stage(
        step="final-verify",
        candidates=[deepcopy(state)],
        center=deepcopy(state),
        target_uid=None,
        count=int(cfg["count_verify"]),
    )

    final_metrics = deepcopy(history[-1]["best_candidate"]["metrics"])
    best_point = {
        "step": "final-verify",
        "F_ctrl": float(final_metrics.get("F_ctrl", np.nan)),
        "F_targ": float(final_metrics.get("F_targ", np.nan)),
        "F_joint": float(final_metrics.get("F_joint", np.nan)),
        "F_avg": float(final_metrics.get("F_avg", np.nan)),
        "F_min": float(final_metrics.get("F_min", np.nan)),
        "state": deepcopy(state),
    }

    diagnostics.update(
        {
            "baseline_metrics": deepcopy(baseline_metrics),
            "final_metrics": deepcopy(final_metrics),
            "improvement": {
                "dF_ctrl": float(final_metrics.get("F_ctrl", np.nan))
                - float(baseline_metrics.get("F_ctrl", np.nan)),
                "dF_targ": float(final_metrics.get("F_targ", np.nan))
                - float(baseline_metrics.get("F_targ", np.nan)),
                "dF_joint": float(final_metrics.get("F_joint", np.nan))
                - float(baseline_metrics.get("F_joint", np.nan)),
                "dF_min": float(final_metrics.get("F_min", np.nan))
                - float(baseline_metrics.get("F_min", np.nan)),
            },
            "config_used": deepcopy(cfg),
        }
    )

    quality_flag = str(history[-1].get("quality", "unknown"))

    old_parameter_values = {
        ctrl_uid: deepcopy(initial_state[ctrl_uid]),
        targ_uid: deepcopy(initial_state[targ_uid]),
    }
    new_parameter_values = {
        ctrl_uid: deepcopy(state[ctrl_uid]),
        targ_uid: deepcopy(state[targ_uid]),
    }

    return {
        "final_state": deepcopy(state),
        "history": history,
        "diagnostics": diagnostics,
        "best_point": best_point,
        "quality_flag": quality_flag,
        "old_parameter_values": old_parameter_values,
        "new_parameter_values": new_parameter_values,
    }


@workflow.workflow(name="readout_multiplexed_optimization")
def experiment_workflow(
    session: Session,
    qpu: QPU,
    ctrl,
    targ,
    config: dict | None = None,
    temporary_parameters: dict[str | tuple[str, str, str], dict | QuantumParameters]
    | None = None,
    options: ReadoutMultiplexedOptimizationWorkflowOptions | None = None,
) -> None:
    """Run 2Q multiplexed readout optimization."""
    options = (
        ReadoutMultiplexedOptimizationWorkflowOptions()
        if options is None
        else options
    )

    result = _run_optimization(
        session=session,
        qpu=qpu,
        ctrl=ctrl,
        targ=targ,
        config=config,
        temporary_parameters=temporary_parameters,
    )

    final_output = result
    with workflow.if_(options.do_analysis):
        analysis_opts = mx_analysis.analysis_workflow.options()
        analysis_opts.do_plotting(bool(options.do_plotting))
        final_output = mx_analysis.analysis_workflow(
            result=result,
            ctrl_uid=ctrl.uid,
            targ_uid=targ.uid,
            options=analysis_opts,
        )

    with workflow.if_(options.update):
        update_qpu(qpu, result["new_parameter_values"])

    workflow.return_(final_output)

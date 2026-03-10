# Copyright 2026 AHNYALGAE
# SPDX-License-Identifier: Apache-2.0

"""Analysis for residual ZZ extraction via conditional Ramsey-echo."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np
import uncertainties as unc
from laboneq import workflow
from laboneq.simple import dsl

from .fitting_helpers import cosine_oscillatory_decay_fit
from laboneq_applications.analysis.calibration_traces_rotation import calculate_population_1d
from laboneq_applications.analysis.options import TuneUpAnalysisWorkflowOptions
from .plotting_helpers import timestamped_title
from laboneq_applications.core.validation import (
    validate_and_convert_qubits_sweeps,
    validate_result,
)
from .plot_theme import get_semantic_color, get_state_color, with_plot_theme

if TYPE_CHECKING:
    import lmfit
    import matplotlib as mpl
    from laboneq.workflow.tasks.run_experiment import RunExperimentResults
    from numpy.typing import ArrayLike

    from laboneq_applications.typing import QuantumElements, QubitSweepPoints


def _nan_ufloat() -> unc.core.Variable:
    return unc.ufloat(np.nan, np.nan)


def _safe_ufloat(value: float | None, stderr: float | None) -> unc.core.Variable:
    if value is None or not np.isfinite(value):
        return _nan_ufloat()
    if stderr is None or not np.isfinite(stderr):
        stderr = 0.0
    return unc.ufloat(float(value), abs(float(stderr)))


def _relative_error(value: float | None, stderr: float | None) -> float:
    if (
        value is None
        or stderr is None
        or not np.isfinite(value)
        or not np.isfinite(stderr)
        or value == 0
    ):
        return np.inf
    return abs(float(stderr) / float(value))


def _combine_sigmas(sigma_model_hz: float, sigma_boot_hz: float) -> float:
    candidates = [
        s for s in [sigma_model_hz, sigma_boot_hz] if s is not None and np.isfinite(s)
    ]
    if not candidates:
        return float("nan")
    return float(max(candidates))


def _validate_ctrl_states(ctrl_states: Sequence[str]) -> tuple[str, str]:
    states = tuple(str(s).lower() for s in ctrl_states)
    if states != ("g", "e"):
        raise ValueError("ctrl_states must be exactly ('g', 'e').")
    return ("g", "e")


def _normalize_mapping_mode(mapping_mode: str) -> str:
    mode = str(mapping_mode).lower().strip()
    if mode not in {"all_pairs", "pairwise"}:
        raise ValueError(
            "mapping_mode must be either 'all_pairs' or 'pairwise', "
            f"but got {mapping_mode!r}."
        )
    return mode


def _build_pair_plan_entries(
    ctrl_qubits: Sequence[object],
    targ_qubits: Sequence[object],
    delays: Sequence[Sequence[float] | np.ndarray],
    detunings: Sequence[float],
    mapping_mode: str,
) -> list[dict[str, object]]:
    mode = _normalize_mapping_mode(mapping_mode)
    plan: list[dict[str, object]] = []

    if mode == "pairwise":
        if len(ctrl_qubits) != len(targ_qubits):
            raise ValueError(
                "pairwise mapping requires len(ctrl) == len(targ), but got "
                f"len(ctrl)={len(ctrl_qubits)} and len(targ)={len(targ_qubits)}."
            )
        iterator = [(idx, idx) for idx in range(len(targ_qubits))]
    else:
        iterator = [
            (ctrl_idx, targ_idx)
            for ctrl_idx in range(len(ctrl_qubits))
            for targ_idx in range(len(targ_qubits))
        ]

    for ctrl_idx, targ_idx in iterator:
        q_c = ctrl_qubits[ctrl_idx]
        q_t = targ_qubits[targ_idx]
        if q_c.uid == q_t.uid:
            continue
        q_delay = np.asarray(delays[targ_idx], dtype=float).ravel().tolist()
        if len(q_delay) < 1:
            raise ValueError(f"delays for target {q_t.uid!r} must be non-empty.")
        plan.append(
            {
                "pair_index": int(len(plan)),
                "pair_key": f"{q_c.uid}->{q_t.uid}",
                "ctrl_uid": q_c.uid,
                "targ_uid": q_t.uid,
                "delay_values": q_delay,
                "detuning_hz": float(detunings[targ_idx]),
            }
        )

    if len(plan) == 0:
        raise ValueError(
            "No valid ctrl->targ pairs are available after automatic self-pair "
            "removal (ctrl_uid == targ_uid)."
        )

    return plan


def _echo_pulse_length(qubit: object, transition: str) -> float:
    if "f" in transition:
        return float(qubit.parameters.ef_drive_length)
    ge_pi = qubit.parameters.ge_drive_length_pi
    if ge_pi is not None:
        return float(ge_pi)
    return float(qubit.parameters.ge_drive_length)


def validate_and_convert_detunings(
    qubits: QuantumElements,
    detunings: float | Sequence[float] | None = None,
) -> Sequence[float]:
    """Validate detuning list against target qubits."""
    if not isinstance(qubits, Sequence):
        qubits = [qubits]

    if detunings is None:
        detunings = [0.0] * len(qubits)
    elif not isinstance(detunings, Sequence):
        detunings = [float(detunings)]
    else:
        detunings = [float(d) for d in detunings]

    if len(detunings) != len(qubits):
        raise ValueError(
            "Length of qubits and detunings must match, but got "
            f"len(qubits)={len(qubits)} and len(detunings)={len(detunings)}."
        )
    return detunings


@workflow.task(save=False)
def build_pair_plan(
    ctrl: QuantumElements,
    targ: QuantumElements,
    delays: QubitSweepPoints,
    detunings: float | Sequence[float] | None = None,
    mapping_mode: str = "all_pairs",
) -> list[dict[str, object]]:
    """Build ordered pair plan for pairwise/all-pairs execution."""
    ctrl = validate_and_convert_qubits_sweeps(ctrl)
    targ, delays = validate_and_convert_qubits_sweeps(targ, delays)
    detunings = validate_and_convert_detunings(targ, detunings)
    return _build_pair_plan_entries(
        ctrl_qubits=ctrl,
        targ_qubits=targ,
        delays=delays,
        detunings=detunings,
        mapping_mode=mapping_mode,
    )


def _split_state_axis(
    raw_data: ArrayLike,
    *,
    num_states: int = 2,
    expected_points: int | None = None,
) -> np.ndarray:
    """Return a state-major 2D array shaped as (state, points)."""
    arr = np.asarray(raw_data)
    if arr.ndim != 2:
        raise ValueError(
            f"Expected 2D state-sweep data, got shape={arr.shape} with ndim={arr.ndim}."
        )
    candidates: list[int] = []
    for axis in (0, 1):
        if arr.shape[axis] != num_states:
            continue
        if expected_points is None or arr.shape[1 - axis] == expected_points:
            candidates.append(axis)
    if not candidates:
        raise ValueError(
            "Could not identify state axis in raw data shape "
            f"{arr.shape}. Expected one axis of length {num_states}."
        )
    axis = candidates[0]
    return arr if axis == 0 else np.swapaxes(arr, 0, 1)


def _bootstrap_zz_uncertainty(
    x_s: np.ndarray,
    y_by_state: dict[str, np.ndarray],
    fit_by_state: dict[str, lmfit.model.ModelResult | None],
    *,
    old_freq_hz: float,
    introduced_detuning_hz: float,
    bootstrap_samples: int,
    bootstrap_seed: int | None,
) -> dict[str, object]:
    if bootstrap_samples <= 0:
        return {
            "sigma_boot_hz": float("nan"),
            "ci95_hz": (float("nan"), float("nan")),
            "num_valid_samples": 0,
        }

    labels = ("g", "e")
    for state in labels:
        if fit_by_state.get(state) is None:
            return {
                "sigma_boot_hz": float("nan"),
                "ci95_hz": (float("nan"), float("nan")),
                "num_valid_samples": 0,
            }

    rng = np.random.default_rng(bootstrap_seed)
    model_y: dict[str, np.ndarray] = {}
    residuals: dict[str, np.ndarray] = {}
    param_hints: dict[str, dict[str, dict[str, float | bool | str]]] = {}

    for state in labels:
        fit_res = fit_by_state[state]
        y = np.asarray(y_by_state[state], dtype=float)
        model_eval = np.asarray(
            fit_res.model.func(x_s, **fit_res.best_values),
            dtype=float,
        )
        res = y - model_eval
        if res.size < 2:  # noqa: PLR2004
            return {
                "sigma_boot_hz": float("nan"),
                "ci95_hz": (float("nan"), float("nan")),
                "num_valid_samples": 0,
            }
        model_y[state] = model_eval
        residuals[state] = res
        hints = {
            name: {"value": float(param.value)}
            for name, param in fit_res.params.items()
            if param.value is not None and np.isfinite(param.value)
        }
        if "decay_time" in hints:
            hints["decay_time"]["min"] = 0.0
        param_hints[state] = hints

    zz_samples: list[float] = []
    for _ in range(int(bootstrap_samples)):
        fit_bs: dict[str, lmfit.model.ModelResult] = {}
        for state in labels:
            idx = rng.integers(0, residuals[state].size, size=residuals[state].size)
            y_bs = model_y[state] + residuals[state][idx]
            try:
                fit_bs[state] = cosine_oscillatory_decay_fit(
                    x_s,
                    y_bs,
                    param_hints=param_hints[state],
                )
            except ValueError:
                fit_bs = {}
                break
        if len(fit_bs) != 2:  # noqa: PLR2004
            continue

        fg = fit_bs["g"].params["frequency"].value
        fe = fit_bs["e"].params["frequency"].value
        if not np.isfinite(fg) or not np.isfinite(fe):
            continue

        cond_g = old_freq_hz + introduced_detuning_hz - float(fg)
        cond_e = old_freq_hz + introduced_detuning_hz - float(fe)
        zz_samples.append(cond_e - cond_g)

    if len(zz_samples) < 2:  # noqa: PLR2004
        return {
            "sigma_boot_hz": float("nan"),
            "ci95_hz": (float("nan"), float("nan")),
            "num_valid_samples": len(zz_samples),
        }

    zz_arr = np.asarray(zz_samples, dtype=float)
    ci_low, ci_high = np.nanpercentile(zz_arr, [2.5, 97.5])
    sigma_boot = float(np.nanstd(zz_arr, ddof=1))
    return {
        "sigma_boot_hz": sigma_boot,
        "ci95_hz": (float(ci_low), float(ci_high)),
        "num_valid_samples": int(np.sum(np.isfinite(zz_arr))),
    }


def _fit_pair(
    x_s: np.ndarray,
    y_by_state: dict[str, np.ndarray],
    *,
    old_freq_hz: float,
    introduced_detuning_hz: float,
    transition: str,
    do_pca: bool,
    min_decay_time: float,
    max_rel_freq_err: float,
    bootstrap_samples: int,
    bootstrap_seed: int | None,
    fit_parameters_hints: dict[str, dict[str, float | bool | str]] | None = None,
) -> dict[str, object]:
    ctrl_states = ("g", "e")
    fit_by_state: dict[str, lmfit.model.ModelResult | None] = {}
    fit_success: dict[str, bool] = {}
    fit_diagnostics: dict[str, dict[str, float | bool]] = {}
    cond_frequency: dict[str, unc.core.Variable] = {}

    for state in ctrl_states:
        y = np.asarray(y_by_state[state], dtype=float)
        param_hints = {
            "amplitude": {"value": 0.5, "vary": do_pca},
            "oscillation_offset": {"value": 0.0, "vary": "f" in transition},
        }
        if fit_parameters_hints is not None:
            param_hints.update(fit_parameters_hints)

        try:
            fit_res = cosine_oscillatory_decay_fit(
                x_s,
                y,
                param_hints=param_hints,
            )
        except ValueError as err:
            logging.error("Residual ZZ fit failed for ctrl=%s: %s", state, err)
            fit_res = None

        fit_by_state[state] = fit_res
        if fit_res is None:
            fit_success[state] = False
            cond_frequency[state] = _nan_ufloat()
            fit_diagnostics[state] = {
                "fit_success": False,
                "frequency_hz": float("nan"),
                "frequency_stderr_hz": float("nan"),
                "decay_time_s": float("nan"),
                "decay_time_stderr_s": float("nan"),
                "relative_frequency_error": float("inf"),
            }
            continue

        freq_val = fit_res.params["frequency"].value
        freq_err = fit_res.params["frequency"].stderr
        decay_time = fit_res.params["decay_time"].value
        decay_time_err = fit_res.params["decay_time"].stderr
        rel_freq_err = _relative_error(freq_val, freq_err)

        is_success = (
            getattr(fit_res, "success", True)
            and np.isfinite(freq_val)
            and np.isfinite(decay_time)
            and float(decay_time) > float(min_decay_time)
            and rel_freq_err <= float(max_rel_freq_err)
        )
        fit_success[state] = bool(is_success)

        fit_diagnostics[state] = {
            "fit_success": bool(is_success),
            "frequency_hz": float(freq_val) if np.isfinite(freq_val) else float("nan"),
            "frequency_stderr_hz": (
                float(freq_err)
                if freq_err is not None and np.isfinite(freq_err)
                else float("nan")
            ),
            "decay_time_s": float(decay_time) if np.isfinite(decay_time) else float("nan"),
            "decay_time_stderr_s": (
                float(decay_time_err)
                if decay_time_err is not None and np.isfinite(decay_time_err)
                else float("nan")
            ),
            "relative_frequency_error": float(rel_freq_err),
        }

        if is_success:
            freq_fit = _safe_ufloat(freq_val, freq_err)
            cond_frequency[state] = old_freq_hz + introduced_detuning_hz - freq_fit
        else:
            cond_frequency[state] = _nan_ufloat()

    zz_value = cond_frequency["e"] - cond_frequency["g"]
    sigma_model = (
        float(np.sqrt(cond_frequency["e"].s**2 + cond_frequency["g"].s**2))
        if np.isfinite(cond_frequency["e"].s) and np.isfinite(cond_frequency["g"].s)
        else float("nan")
    )

    bootstrap = _bootstrap_zz_uncertainty(
        x_s=x_s,
        y_by_state=y_by_state,
        fit_by_state=fit_by_state,
        old_freq_hz=old_freq_hz,
        introduced_detuning_hz=introduced_detuning_hz,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    sigma_boot = float(bootstrap["sigma_boot_hz"])
    sigma_final = _combine_sigmas(sigma_model, sigma_boot)

    quality_flag = (
        "ok"
        if fit_success.get("g", False)
        and fit_success.get("e", False)
        and np.isfinite(zz_value.n)
        else "fail"
    )

    return {
        "state_fits": fit_by_state,
        "fit_success": fit_success,
        "fit_diagnostics": fit_diagnostics,
        "conditional_frequency": cond_frequency,
        "residual_zz": {
            "value": zz_value,
            "nominal_hz": float(zz_value.n),
            "std_dev_hz": float(zz_value.s),
            "sigma_model_hz": float(sigma_model),
            "sigma_boot_hz": float(sigma_boot),
            "sigma_final_hz": float(sigma_final),
            "ci95_hz": tuple(bootstrap["ci95_hz"]),
            "quality_flag": quality_flag,
        },
    }


@workflow.task(save=False)
def _materialize_pair_results(
    result: object,
    pair_plan: list[dict[str, object]],
) -> dict[str, object]:
    pair_keys = [str(entry["pair_key"]) for entry in pair_plan]

    if isinstance(result, Mapping):
        source = result
        if "results_by_pair" in source and isinstance(source["results_by_pair"], Mapping):
            source = source["results_by_pair"]

        missing = [k for k in pair_keys if k not in source]
        if missing:
            raise ValueError(
                "Missing pair results for keys: "
                f"{missing}. Available keys: {list(source.keys())}."
            )
        return {k: source[k] for k in pair_keys}

    if len(pair_keys) == 1:
        return {pair_keys[0]: result}

    raise ValueError(
        "Analysis requires dict-like pair results when multiple pairs are requested. "
        "Expected keys like '<ctrl_uid>-><targ_uid>'."
    )


@workflow.task(save=False)
def _resolve_mapping_mode_for_result(
    result: object,
    ctrl: QuantumElements,
    targ: QuantumElements,
    requested_mode: str = "all_pairs",
) -> str:
    mode = _normalize_mapping_mode(requested_mode)
    if mode == "pairwise":
        return mode

    if not isinstance(result, Mapping):
        return mode

    source = result
    if "results_by_pair" in source and isinstance(source["results_by_pair"], Mapping):
        source = source["results_by_pair"]

    ctrl_qubits = validate_and_convert_qubits_sweeps(ctrl)
    targ_qubits = validate_and_convert_qubits_sweeps(targ)

    all_pairs = [
        f"{q_c.uid}->{q_t.uid}"
        for q_c in ctrl_qubits
        for q_t in targ_qubits
        if q_c.uid != q_t.uid
    ]
    if all(key in source for key in all_pairs):
        return "all_pairs"

    if len(ctrl_qubits) == len(targ_qubits):
        pairwise = [
            f"{ctrl_qubits[i].uid}->{targ_qubits[i].uid}"
            for i in range(len(targ_qubits))
            if ctrl_qubits[i].uid != targ_qubits[i].uid
        ]
        if all(key in source for key in pairwise):
            return "pairwise"

    return mode


@workflow.task(save=False)
def _select_pair_result(
    pair_results: dict[str, object],
    pair_key: str,
) -> object:
    if pair_key not in pair_results:
        raise ValueError(
            f"Pair result for key {pair_key!r} was not found. "
            f"Available keys: {list(pair_results.keys())}."
        )
    return pair_results[pair_key]


@workflow.task(save=False)
def _select_qubit_by_uid(
    qubits: QuantumElements,
    uid: str,
):
    qubits = validate_and_convert_qubits_sweeps(qubits)
    for q in qubits:
        if q.uid == uid:
            return q
    raise ValueError(f"Qubit uid {uid!r} was not found in provided qubits.")


@workflow.task(save=False)
def _append_item(items: list, item: object) -> None:
    items.append(item)


@workflow.task(save=False)
def _materialize_list(items: list) -> list:
    return list(items)


@workflow.task
def _extract_pair_payload(
    pair_key: str,
    ctrl_uid: str,
    targ: QuantumElements,
    fit_results: dict[str, dict[str, object]],
    transition: Literal["ge", "ef"] = "ge",
) -> dict[str, object]:
    targ_qubits = validate_and_convert_qubits_sweeps(targ)
    if len(targ_qubits) != 1:
        raise ValueError(
            "_extract_pair_payload expects a single target qubit, but got "
            f"{len(targ_qubits)} targets."
        )
    q_t = targ_qubits[0]

    old_freq_hz = (
        q_t.parameters.resonance_frequency_ef
        if "f" in transition
        else q_t.parameters.resonance_frequency_ge
    )

    pair_fit = fit_results.get(q_t.uid, {})
    cond = pair_fit.get("conditional_frequency", {"g": _nan_ufloat(), "e": _nan_ufloat()})
    zz = pair_fit.get(
        "residual_zz",
        {
            "value": _nan_ufloat(),
            "nominal_hz": float("nan"),
            "std_dev_hz": float("nan"),
            "sigma_model_hz": float("nan"),
            "sigma_boot_hz": float("nan"),
            "sigma_final_hz": float("nan"),
            "ci95_hz": (float("nan"), float("nan")),
            "quality_flag": "fail",
        },
    )

    return {
        "pair_key": pair_key,
        "old_parameter_values": {
            f"resonance_frequency_{transition}": old_freq_hz,
        },
        "new_parameter_values": {},
        "conditional_frequency": {
            "g": cond["g"],
            "e": cond["e"],
        },
        "residual_zz": {
            "ctrl_uid": ctrl_uid,
            "targ_uid": q_t.uid,
            **zz,
        },
        "fit_diagnostics": pair_fit.get("fit_diagnostics", {}),
    }


@workflow.task(save=False)
def _merge_pair_payloads(
    pair_payloads: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    output: dict[str, dict[str, object]] = {
        "old_parameter_values": {},
        "new_parameter_values": {},
        "conditional_frequency": {},
        "residual_zz": {},
        "fit_diagnostics": {},
    }

    for payload in pair_payloads:
        pair_key = str(payload["pair_key"])
        output["old_parameter_values"][pair_key] = dict(payload["old_parameter_values"])
        output["new_parameter_values"][pair_key] = dict(payload["new_parameter_values"])
        output["conditional_frequency"][pair_key] = dict(payload["conditional_frequency"])
        output["residual_zz"][pair_key] = dict(payload["residual_zz"])
        output["fit_diagnostics"][pair_key] = dict(payload["fit_diagnostics"])

    return output


@workflow.workflow_options(base_class=TuneUpAnalysisWorkflowOptions)
class ResidualZZEchoAnalysisWorkflowOptions:
    """Top-level options for residual-ZZ echo analysis."""

    transition: Literal["ge", "ef"] = workflow.option_field(
        "ge",
        description="Transition used for echo Ramsey fit/parameter interpretation.",
    )
    do_rotation: bool = workflow.option_field(
        True,
        description="Use rotated population (True) or raw trace (False) for fitting.",
    )
    do_pca: bool = workflow.option_field(
        False,
        description="Force PCA projection even when calibration traces are present.",
    )
    use_cal_traces: bool = workflow.option_field(
        True,
        description="Whether calibration traces are expected in the result.",
    )
    cal_states: str | tuple = workflow.option_field(
        "ge",
        description="Calibration states used for population rotation/projection.",
    )
    bootstrap_samples: int = workflow.option_field(
        400,
        description="Number of residual-bootstrap refits per target pair.",
    )
    bootstrap_seed: int | None = workflow.option_field(
        None,
        description="Seed for reproducible bootstrap sampling.",
    )
    min_decay_time: float = workflow.option_field(
        0.1e-6,
        description="Minimum acceptable fitted decay_time (seconds).",
    )
    max_rel_freq_err: float = workflow.option_field(
        0.5,
        description="Maximum allowed relative frequency standard error.",
    )
    ctrl_states: tuple[Literal["g"], Literal["e"]] = workflow.option_field(
        ("g", "e"),
        description="Control states used in outer sweep, fixed to ('g','e').",
    )
    fit_parameters_hints: dict[str, dict[str, float | bool | str]] | None = (
        workflow.option_field(
            None,
            description="Optional lmfit parameter hints passed to each state fit.",
        )
    )
    save_figures: bool = workflow.option_field(
        True,
        description="Whether to save generated figures as workflow artifacts.",
    )
    close_figures: bool = workflow.option_field(
        True,
        description="Whether to close figures after creation.",
    )
    mapping_mode: Literal["all_pairs", "pairwise"] = workflow.option_field(
        "all_pairs",
        description="Pair mapping strategy for ctrl/targ combinations.",
    )


@workflow.workflow
def analysis_workflow(
    result: RunExperimentResults,
    ctrl: QuantumElements,
    targ: QuantumElements,
    delays: QubitSweepPoints,
    detunings: float | Sequence[float] | None = None,
    mapping_mode: Literal["all_pairs", "pairwise"] | None = None,
    options: ResidualZZEchoAnalysisWorkflowOptions | None = None,
) -> None:
    """Analyze conditional echo-Ramsey traces and estimate residual ZZ."""
    opts = ResidualZZEchoAnalysisWorkflowOptions() if options is None else options
    mapping_mode = opts.mapping_mode if mapping_mode is None else mapping_mode
    mapping_mode = _resolve_mapping_mode_for_result(
        result=result,
        ctrl=ctrl,
        targ=targ,
        requested_mode=mapping_mode,
    )
    pair_plan = build_pair_plan(
        ctrl=ctrl,
        targ=targ,
        delays=delays,
        detunings=detunings,
        mapping_mode=mapping_mode,
    )
    pair_results = _materialize_pair_results(result=result, pair_plan=pair_plan)

    pair_payloads: list = []
    with workflow.for_(pair_plan, lambda pair: pair["pair_key"]) as pair:
        pair_result = _select_pair_result(
            pair_results=pair_results,
            pair_key=pair["pair_key"],
        )
        pair_targ = _select_qubit_by_uid(
            qubits=targ,
            uid=pair["targ_uid"],
        )

        processed_data_dict = process_conditional_population(
            result=pair_result,
            targ=pair_targ,
            delays=pair["delay_values"],
            transition=opts.transition,
            do_pca=opts.do_pca,
            use_cal_traces=opts.use_cal_traces,
            cal_states=opts.cal_states,
            ctrl_states=opts.ctrl_states,
        )
        fit_results = fit_data(
            targ=pair_targ,
            processed_data_dict=processed_data_dict,
            detunings=pair["detuning_hz"],
            transition=opts.transition,
            do_rotation=opts.do_rotation,
            do_fitting=opts.do_fitting,
            do_pca=opts.do_pca,
            min_decay_time=opts.min_decay_time,
            max_rel_freq_err=opts.max_rel_freq_err,
            bootstrap_samples=opts.bootstrap_samples,
            bootstrap_seed=opts.bootstrap_seed,
            fit_parameters_hints=opts.fit_parameters_hints,
            ctrl_states=opts.ctrl_states,
        )
        pair_payload = _extract_pair_payload(
            pair_key=pair["pair_key"],
            ctrl_uid=pair["ctrl_uid"],
            targ=pair_targ,
            fit_results=fit_results,
            transition=opts.transition,
        )
        _append_item(pair_payloads, pair_payload)

        plot_population(
            pair_key=pair["pair_key"],
            ctrl_uid=pair["ctrl_uid"],
            targ=pair_targ,
            processed_data_dict=processed_data_dict,
            fit_results=fit_results,
            pair_payload=pair_payload,
            do_plotting=opts.do_plotting,
            do_qubit_population_plotting=opts.do_qubit_population_plotting,
            do_rotation=opts.do_rotation,
            save_figures=opts.save_figures,
            close_figures=opts.close_figures,
        )

    qubit_parameters = _merge_pair_payloads(_materialize_list(pair_payloads))

    with workflow.if_(opts.do_plotting):
        with workflow.if_(opts.do_qubit_population_plotting):
            plot_residual_zz_summary(
                qubit_parameters=qubit_parameters,
                save_figures=opts.save_figures,
                close_figures=opts.close_figures,
            )

    workflow.return_(qubit_parameters)


@workflow.task
def process_conditional_population(
    result: RunExperimentResults,
    targ: QuantumElements,
    delays: QubitSweepPoints,
    transition: Literal["ge", "ef"] = "ge",
    do_pca: bool = False,
    use_cal_traces: bool = True,
    cal_states: str | tuple = "ge",
    ctrl_states: Sequence[str] = ("g", "e"),
) -> dict[str, dict[str, object]]:
    """Extract per-target per-control-state traces from single-run state sweep."""
    ctrl_states = _validate_ctrl_states(ctrl_states)
    validate_result(result)
    targ, delays = validate_and_convert_qubits_sweeps(targ, delays)

    processed: dict[str, dict[str, object]] = {}
    for q_t, q_delays in zip(targ, delays):
        q_delays = np.asarray(q_delays, dtype=float).ravel()
        raw_data = result[dsl.handles.result_handle(q_t.uid)].data
        state_major = _split_state_axis(
            raw_data,
            num_states=len(ctrl_states),
            expected_points=len(q_delays),
        )

        if use_cal_traces:
            calibration_traces = [
                result[dsl.handles.calibration_trace_handle(q_t.uid, cs)].data
                for cs in cal_states
            ]
            use_pca = do_pca
        else:
            calibration_traces = []
            use_pca = True

        population_by_state: dict[str, np.ndarray] = {}
        raw_by_state: dict[str, np.ndarray] = {}
        pop_cal_by_state: dict[str, np.ndarray] = {}
        num_cal_traces = 0
        for state_idx, state_label in enumerate(ctrl_states):
            trace = np.asarray(state_major[state_idx, :]).ravel()
            data_dict = calculate_population_1d(
                raw_data=trace,
                sweep_points=q_delays,
                calibration_traces=calibration_traces,
                do_pca=use_pca,
            )
            population_by_state[state_label] = np.asarray(data_dict["population"])
            raw_by_state[state_label] = np.asarray(data_dict["data_raw"])
            pop_cal_by_state[state_label] = np.asarray(data_dict["population_cal_traces"])
            num_cal_traces = int(data_dict["num_cal_traces"])

        sweep_points_effective = q_delays + _echo_pulse_length(q_t, transition)
        processed[q_t.uid] = {
            "ctrl_states": ctrl_states,
            "sweep_points": q_delays,
            "sweep_points_effective": sweep_points_effective,
            "data_raw": raw_by_state,
            "population": population_by_state,
            "population_cal_traces": pop_cal_by_state,
            "num_cal_traces": num_cal_traces,
        }

    return processed


@workflow.task
def fit_data(
    targ: QuantumElements,
    processed_data_dict: dict[str, dict[str, object]],
    detunings: float | Sequence[float] | None = None,
    transition: Literal["ge", "ef"] = "ge",
    do_rotation: bool = True,
    do_fitting: bool = True,
    do_pca: bool = False,
    min_decay_time: float = 0.1e-6,
    max_rel_freq_err: float = 0.5,
    bootstrap_samples: int = 400,
    bootstrap_seed: int | None = None,
    fit_parameters_hints: dict[str, dict[str, float | bool | str]] | None = None,
    ctrl_states: Sequence[str] = ("g", "e"),
) -> dict[str, dict[str, object]]:
    """Fit per-state decaying-cosine traces and compute residual ZZ metrics."""
    targ = validate_and_convert_qubits_sweeps(targ)
    detunings = validate_and_convert_detunings(targ, detunings)
    fit_results: dict[str, dict[str, object]] = {}

    if not do_fitting:
        return fit_results

    ctrl_states = _validate_ctrl_states(ctrl_states)
    source_key = "population" if do_rotation else "data_raw"
    for i, q_t in enumerate(targ):
        if q_t.uid not in processed_data_dict:
            continue
        q_data = processed_data_dict[q_t.uid]
        old_freq_hz = (
            q_t.parameters.resonance_frequency_ef
            if "f" in transition
            else q_t.parameters.resonance_frequency_ge
        )

        x_s = np.asarray(q_data["sweep_points_effective"], dtype=float)
        y_source = q_data[source_key]
        y_by_state = {
            state: np.asarray(y_source[state], dtype=float) for state in ctrl_states
        }
        pair_result = _fit_pair(
            x_s=x_s,
            y_by_state=y_by_state,
            old_freq_hz=float(old_freq_hz),
            introduced_detuning_hz=float(detunings[i]),
            transition=transition,
            do_pca=do_pca,
            min_decay_time=float(min_decay_time),
            max_rel_freq_err=float(max_rel_freq_err),
            bootstrap_samples=int(bootstrap_samples),
            bootstrap_seed=bootstrap_seed,
            fit_parameters_hints=fit_parameters_hints,
        )
        fit_results[q_t.uid] = pair_result

    return fit_results


def _format_ufloat_hz(value: unc.core.Variable) -> str:
    if not np.isfinite(value.n):
        return "nan"
    if np.isfinite(value.s):
        return f"{value.n / 1e6:.6f} +- {value.s / 1e6:.6f} MHz"
    return f"{value.n / 1e6:.6f} MHz"


@workflow.task
@with_plot_theme
def plot_population(
    pair_key: str,
    ctrl_uid: str,
    targ: QuantumElements,
    processed_data_dict: dict[str, dict[str, object]],
    fit_results: dict[str, dict[str, object]],
    pair_payload: dict[str, object],
    do_plotting: bool = True,
    do_qubit_population_plotting: bool = True,
    do_rotation: bool = True,
    save_figures: bool = True,
    close_figures: bool = True,
) -> dict[str, mpl.figure.Figure]:
    """Plot conditional traces for one ctrl->targ pair with fitted curves."""
    if not (do_plotting and do_qubit_population_plotting):
        return {}

    targ_qubits = validate_and_convert_qubits_sweeps(targ)
    if len(targ_qubits) != 1:
        raise ValueError(
            "plot_population expects a single target qubit per call, but got "
            f"{len(targ_qubits)} targets."
        )
    q_t = targ_qubits[0]

    source_key = "population" if do_rotation else "data_raw"
    q_data = processed_data_dict[q_t.uid]
    x_s = np.asarray(q_data["sweep_points_effective"], dtype=float)
    y_by_state = q_data[source_key]
    pair_fit = fit_results.get(q_t.uid, {})

    fig, ax = plt.subplots()
    ax.set_title(timestamped_title(f"Residual ZZ Echo {q_t.uid} (ctrl={ctrl_uid})"))
    ax.set_xlabel("x90-Pulse Separation, tau_eff (us)")
    ax.set_ylabel(
        "Population"
        if do_rotation
        else "Raw amplitude (arb.)"
    )

    for state in ("g", "e"):
        color = get_state_color(state)
        y = np.asarray(y_by_state[state], dtype=float)
        ax.plot(
            x_s * 1e6,
            y,
            "o",
            ms=4,
            color=color,
            label=f"ctrl={state} data",
        )

        fit_obj = pair_fit.get("state_fits", {}).get(state)
        fit_ok = bool(pair_fit.get("fit_success", {}).get(state, False))
        if fit_obj is not None and fit_ok:
            x_fine = np.linspace(x_s[0], x_s[-1], 801)
            y_fit = fit_obj.model.func(x_fine, **fit_obj.best_values)
            ax.plot(
                x_fine * 1e6,
                y_fit,
                "-",
                lw=1.5,
                color=color,
                alpha=0.9,
                label=f"ctrl={state} fit",
            )

    cond = pair_payload["conditional_frequency"]
    zz = pair_payload["residual_zz"]
    textstr = (
        f"f|g: {_format_ufloat_hz(cond['g'])}\n"
        f"f|e: {_format_ufloat_hz(cond['e'])}\n"
        f"ZZ: {_format_ufloat_hz(zz['value'])}\n"
        f"sigma_final: {zz['sigma_final_hz'] / 1e6:.6f} MHz\n"
        f"quality: {zz['quality_flag']}"
    )
    ax.text(
        0.01,
        0.02,
        textstr,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.2",
            "facecolor": get_semantic_color("text_box"),
            "edgecolor": get_semantic_color("text_box_edge"),
            "alpha": 0.85,
        },
    )
    ax.grid(alpha=0.25)
    ax.legend(loc="best")

    if save_figures:
        workflow.save_artifact(f"ResidualZZEcho_{ctrl_uid}_to_{q_t.uid}", fig)
    if close_figures:
        plt.close(fig)

    return {pair_key: fig}


@workflow.task
@with_plot_theme
def plot_residual_zz_summary(
    qubit_parameters: dict[str, dict[str, object]],
    save_figures: bool = True,
    close_figures: bool = True,
) -> dict[str, mpl.figure.Figure]:
    """Plot pair-wise residual ZZ values with error bars."""
    labels: list[str] = []
    zz_mhz: list[float] = []
    err_mhz: list[float] = []
    ok_mask: list[bool] = []

    for pair_key, zz_entry in qubit_parameters["residual_zz"].items():
        labels.append(str(pair_key))
        nominal_hz = float(zz_entry.get("nominal_hz", np.nan))
        sigma_hz = float(zz_entry.get("sigma_final_hz", np.nan))
        quality_flag = str(zz_entry.get("quality_flag", "fail")).lower()
        zz_mhz.append(nominal_hz / 1e6)
        err_mhz.append(sigma_hz / 1e6)
        ok_mask.append(
            quality_flag == "ok"
            and np.isfinite(nominal_hz)
            and np.isfinite(sigma_hz)
        )

    fig, ax = plt.subplots()
    x = np.arange(len(labels), dtype=int)
    ok = np.asarray(ok_mask, dtype=bool)
    if np.any(ok):
        ax.errorbar(
            x[ok],
            np.asarray(zz_mhz)[ok],
            yerr=np.asarray(err_mhz)[ok],
            fmt="o",
            capsize=3,
            label="ZZ (fit ok)",
        )
    if np.any(~ok):
        ax.plot(
            x[~ok],
            np.zeros(np.count_nonzero(~ok)),
            linestyle="None",
            marker="x",
            color=get_semantic_color("fail"),
            label="fit failed",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Residual ZZ (MHz)")
    ax.set_title(timestamped_title("Residual ZZ Summary"))
    ax.axhline(0.0, color=get_semantic_color("boundary"), lw=0.8, alpha=0.4)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")

    if save_figures:
        workflow.save_artifact("ResidualZZEcho_summary", fig)
    if close_figures:
        plt.close(fig)

    return {"summary": fig}

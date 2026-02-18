"""Analysis workflow for readout-line MID sweep."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np
from laboneq import workflow
from laboneq.simple import dsl

from analysis.fitting_helpers import exponential_decay_fit
from analysis.readout_sweep_common import (
    as_1d_complex,
    calibration_shots_by_state,
    extract_sweep_shots,
    unwrap_result_like,
)
from laboneq_applications.core.validation import validate_result

if TYPE_CHECKING:
    import matplotlib as mpl
    from laboneq.dsl.quantum.quantum_element import QuantumElement
    from laboneq.workflow.tasks.run_experiment import RunExperimentResults
    from numpy.typing import ArrayLike


@workflow.workflow_options
class ReadoutMidSweepAnalysisOptions:
    """Options for readout MID-sweep analysis."""

    do_plotting: bool = workflow.option_field(
        True, description="Whether to create MID analysis plots."
    )
    do_plotting_mid_curves: bool = workflow.option_field(
        True,
        description="Whether to plot 1D MID curves.",
    )
    do_plotting_heatmap: bool = workflow.option_field(
        True,
        description="Whether to plot 2D MID heatmaps for two-axis sweeps.",
    )
    do_plotting_diagnostics: bool = workflow.option_field(
        True,
        description="Whether to plot per-candidate diagnostic traces.",
    )
    do_mapping_validation: bool = workflow.option_field(
        False,
        description="Enable classifier mapping validation for calibration_traces -> p_e(delay).",
    )
    do_plotting_mapping_validation: bool = workflow.option_field(
        True,
        description="Whether to plot mapping-validation summary when enabled.",
    )
    mapping_validation_split_mode: Literal["even_odd"] = workflow.option_field(
        "even_odd",
        description="Train/test split strategy for calibration-shot holdout validation.",
    )
    mapping_validation_min_fidelity: float = workflow.option_field(
        0.95,
        description="Minimum holdout assignment fidelity for mapping pass.",
    )
    mapping_validation_min_dynamic_range: float = workflow.option_field(
        0.05,
        description="Minimum dynamic range of p_e(delay) for mapping pass.",
    )
    max_mid_tolerance: float = workflow.option_field(
        1e-4,
        description="Tolerance for near-optimal selection on MID rate.",
    )
    max_mid_rate: float | None = workflow.option_field(
        None,
        description="Hard maximum MID constraint. Candidates above this are discarded.",
    )
    prefer_lower_frequency_within_tolerance: bool = workflow.option_field(
        False,
        description="Prefer lower readout frequency among tied candidates.",
    )
    prefer_lower_amplitude_within_tolerance: bool = workflow.option_field(
        False,
        description="Prefer lower readout amplitude among tied candidates.",
    )
    relative_mid_increase_limit: float = workflow.option_field(
        0.2,
        description="Relative MID increase limit used for pass/fail gate.",
    )
    use_cal_traces: bool = workflow.option_field(
        True,
        description="Reserved option for compatibility (must remain True).",
    )
    cal_states: str | tuple[str, ...] = workflow.option_field(
        "ge",
        description="Reserved option for compatibility (must remain 'ge').",
    )
    do_rotation: bool = workflow.option_field(
        True,
        description="Reserved option for compatibility.",
    )
    do_pca: bool = workflow.option_field(
        False,
        description="Reserved option for compatibility.",
    )


def _coerce_points(points: ArrayLike | None, name: str) -> np.ndarray:
    arr = np.asarray(points if points is not None else [], dtype=float).reshape(-1)
    if arr.size < 1:
        raise ValueError(f"{name} must contain at least one value.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite.")
    return arr


def _coerce_mid_sweep_points(
    frequency_points: ArrayLike | None,
    amplitude_points: ArrayLike | None,
    sweep_axis: Literal["frequency", "amplitude", "both"],
) -> tuple[np.ndarray, np.ndarray]:
    if sweep_axis == "frequency":
        return _coerce_points(frequency_points, "frequency"), np.array([np.nan])
    if sweep_axis == "amplitude":
        return np.array([np.nan]), _coerce_points(amplitude_points, "amplitude")
    if sweep_axis == "both":
        return (
            _coerce_points(frequency_points, "frequency"),
            _coerce_points(amplitude_points, "amplitude"),
        )
    raise ValueError("sweep_axis must be 'frequency', 'amplitude', or 'both'.")


@workflow.task(save=False)
def _resolve_axis_points(
    explicit_points: ArrayLike | None,
    mid_sweep_points: dict[str, ArrayLike] | None,
    axis: Literal["frequency", "amplitude"],
):
    if explicit_points is not None:
        return explicit_points
    if isinstance(mid_sweep_points, dict):
        return mid_sweep_points.get(axis)
    return None


def _select_best_index(
    primary: np.ndarray,
    secondary: np.ndarray,
    tolerance: float,
    prefer_lower: bool = False,
) -> tuple[int, int, str]:
    """Select best index using MID primary and decay-rate secondary criteria."""
    primary = np.asarray(primary, dtype=float)
    secondary = np.asarray(secondary, dtype=float)
    if primary.shape != secondary.shape:
        raise ValueError("primary and secondary metrics must have the same shape.")
    if primary.ndim != 2:
        raise ValueError("primary must be a 2D array.")
    if primary.size < 1:
        return 0, 0, "invalid_data"

    valid_primary = np.isfinite(primary)
    if not valid_primary.any():
        return 0, 0, "invalid_data"

    score_primary = np.where(valid_primary, primary, float("inf"))
    best_primary = float(np.min(score_primary))
    tol = abs(float(tolerance))
    candidate_mask = score_primary <= best_primary + tol
    candidates = np.argwhere(candidate_mask)
    if candidates.size == 0:
        candidates = np.argwhere(valid_primary)
    if candidates.size == 0:
        return 0, 0, "invalid_data"

    score_secondary = np.where(np.isfinite(secondary), secondary, float("inf"))
    sec_values = score_secondary[tuple(candidates.T)]
    if np.any(np.isfinite(sec_values)):
        best_secondary = float(np.min(sec_values))
        tied = np.isclose(sec_values, best_secondary, atol=tol, rtol=0.0)
        candidates = candidates[tied]

    if candidates.size == 0:
        return 0, 0, "invalid_data"

    if prefer_lower:
        order = candidates[:, 0] * primary.shape[1] + candidates[:, 1]
        idx = int(np.argmin(order))
    else:
        center = (np.asarray(primary.shape, dtype=float) - 1.0) / 2.0
        dist = np.linalg.norm(candidates.astype(float) - center, axis=1)
        idx = int(np.argmin(dist))

    li, ai = int(candidates[idx][0]), int(candidates[idx][1])
    edge_hit = (li in (0, primary.shape[0] - 1)) or (ai in (0, primary.shape[1] - 1))
    if edge_hit:
        quality = "edge_hit"
    elif candidates.shape[0] > 1:
        quality = "flat_optimum"
    else:
        quality = "ok"
    return li, ai, quality


def _collapse_population_to_1d(population: np.ndarray, n_points: int) -> np.ndarray:
    """Reduce population data to 1D sweep trace.

    For single-shot acquisitions population can have shape (n_points, n_shots)
    or an equivalent reshaping. We average over the non-sweep dimensions.
    """
    pop = np.asarray(population, dtype=float)
    if pop.ndim == 1:
        if pop.size != n_points:
            if pop.size % n_points != 0:
                raise ValueError(
                    f"1D population length {pop.size} is not compatible with sweep length {n_points}."
                )
            return pop.reshape(n_points, -1).mean(axis=1)
        return pop

    if pop.shape[0] == n_points:
        return pop.reshape(n_points, -1).mean(axis=1)
    if pop.shape[-1] == n_points:
        moved = np.moveaxis(pop, -1, 0)
        return moved.reshape(n_points, -1).mean(axis=1)
    if pop.size % n_points == 0:
        return pop.reshape(n_points, -1).mean(axis=1)
    raise ValueError(
        f"Population shape {pop.shape} is not compatible with sweep length {n_points}."
    )


def _extract_population_trace(
    result,
    qubit_uid: str,
    n_points: int,
) -> np.ndarray:
    """Extract e-population per delay from single-shot IQ and g/e calibration clouds."""
    raw_main = result[dsl.handles.result_handle(qubit_uid)].data
    main_shots = extract_sweep_shots(raw_main, n_points=n_points)  # (n_points, n_shots)

    cal = calibration_shots_by_state(result, qubit_uid, states=("g", "e"))
    shots_g = as_1d_complex(cal["g"])
    shots_e = as_1d_complex(cal["e"])
    if shots_g.size < 1 or shots_e.size < 1:
        raise ValueError("Missing g/e calibration shots for population extraction.")

    mu_g = np.mean(shots_g)
    mu_e = np.mean(shots_e)
    v = mu_e - mu_g
    v_norm = abs(v)
    if v_norm <= 1e-15:
        raise ValueError("Calibration clusters are degenerate (|mu_e-mu_g| ~ 0).")

    def _project(z: np.ndarray) -> np.ndarray:
        return np.real((z - mu_g) * np.conj(v)) / v_norm

    score_g = _project(shots_g)
    score_e = _project(shots_e)
    threshold = 0.5 * (np.mean(score_g) + np.mean(score_e))

    score_main = _project(main_shots)
    return np.mean(score_main > threshold, axis=1).astype(float)


def _fit_projection_classifier(
    shots_g: np.ndarray,
    shots_e: np.ndarray,
) -> dict[str, float | complex]:
    shots_g = as_1d_complex(shots_g)
    shots_e = as_1d_complex(shots_e)
    if shots_g.size < 2 or shots_e.size < 2:
        raise ValueError("Need at least 2 shots per state for classifier fitting.")

    mu_g = np.mean(shots_g)
    mu_e = np.mean(shots_e)
    v = mu_e - mu_g
    v_norm = abs(v)
    if v_norm <= 1e-15:
        raise ValueError("Calibration clusters are degenerate (|mu_e-mu_g| ~ 0).")

    score_g = np.real((shots_g - mu_g) * np.conj(v)) / v_norm
    score_e = np.real((shots_e - mu_g) * np.conj(v)) / v_norm
    threshold = 0.5 * (float(np.mean(score_g)) + float(np.mean(score_e)))
    return {
        "mu_g": complex(mu_g),
        "mu_e": complex(mu_e),
        "v": complex(v),
        "v_norm": float(v_norm),
        "threshold": float(threshold),
    }


def _predict_projection_bits(
    shots: np.ndarray,
    classifier: dict[str, float | complex],
) -> np.ndarray:
    arr = as_1d_complex(shots)
    mu_g = complex(classifier["mu_g"])
    v = complex(classifier["v"])
    v_norm = float(classifier["v_norm"])
    threshold = float(classifier["threshold"])
    score = np.real((arr - mu_g) * np.conj(v)) / v_norm
    return (score > threshold).astype(int)


def _extract_population_trace_with_classifier(
    result,
    qubit_uid: str,
    n_points: int,
    classifier: dict[str, float | complex],
) -> np.ndarray:
    raw_main = result[dsl.handles.result_handle(qubit_uid)].data
    main_shots = extract_sweep_shots(raw_main, n_points=n_points)
    bits = _predict_projection_bits(main_shots, classifier)
    return np.mean(bits, axis=1).astype(float)


def _even_odd_train_test(shots: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = as_1d_complex(shots)
    if arr.size < 4:
        raise ValueError("Need at least 4 shots for even/odd train-test split.")
    train = arr[::2]
    test = arr[1::2]
    if train.size < 2 or test.size < 2:
        raise ValueError("Insufficient train/test shots after even/odd split.")
    return train, test


@workflow.task(save=False)
def _mapping_validation_disabled() -> dict:
    return {"enabled": False}


@workflow.task(save=False)
def _attach_mapping_validation(metrics: dict, mapping_validation: dict) -> dict:
    merged = dict(metrics)
    merged["mapping_validation"] = mapping_validation
    return merged


@workflow.task(save=False)
def validate_mapping_pipeline(
    results: list[RunExperimentResults],
    qubits: list[QuantumElement],
    delays: ArrayLike,
    frequency_points: ArrayLike | None = None,
    amplitude_points: ArrayLike | None = None,
    sweep_axis: Literal["frequency", "amplitude", "both"] = "both",
    split_mode: Literal["even_odd"] = "even_odd",
    min_fidelity: float = 0.95,
    min_dynamic_range: float = 0.05,
) -> dict:
    if split_mode != "even_odd":
        raise ValueError("Unsupported split_mode. Expected 'even_odd'.")

    frequency_points, amplitude_points = _coerce_mid_sweep_points(
        frequency_points=frequency_points,
        amplitude_points=amplitude_points,
        sweep_axis=sweep_axis,
    )
    delay_points = np.asarray(delays, dtype=float).reshape(-1)
    if delay_points.size < 1:
        raise ValueError("delays must contain at least one value.")

    if sweep_axis == "frequency":
        n_freq, n_amp = frequency_points.size, 1
    elif sweep_axis == "amplitude":
        n_freq, n_amp = 1, amplitude_points.size
    else:
        n_freq, n_amp = frequency_points.size, amplitude_points.size
    if n_freq * n_amp != len(results):
        raise ValueError(
            "Result count does not match sweep-grid size for mapping validation: "
            f"{len(results)} != {n_freq} x {n_amp}."
        )

    per_candidate: list[dict] = []
    pass_count = 0
    fail_count = 0
    fail_reasons: dict[str, int] = {}

    for idx, (result_like, qubit) in enumerate(zip(results, qubits)):
        li, ai = divmod(idx, n_amp)
        result = unwrap_result_like(result_like)
        validate_result(result)

        report: dict[str, object] = {
            "candidate_key": f"{li}_{ai}",
            "index_frequency": int(li),
            "index_amplitude": int(ai),
            "readout_resonator_frequency": (
                float(frequency_points[li]) if sweep_axis != "amplitude" else float("nan")
            ),
            "readout_amplitude": (
                float(amplitude_points[ai]) if sweep_axis != "frequency" else float("nan")
            ),
            "mapping_pass": False,
        }

        try:
            raw_main = result[dsl.handles.result_handle(qubit.uid)].data
            main_shots = extract_sweep_shots(raw_main, n_points=delay_points.size)
            cal = calibration_shots_by_state(result, qubit.uid, states=("g", "e"))
            shots_g = as_1d_complex(cal["g"])
            shots_e = as_1d_complex(cal["e"])
            train_g, test_g = _even_odd_train_test(shots_g)
            train_e, test_e = _even_odd_train_test(shots_e)

            holdout_classifier = _fit_projection_classifier(train_g, train_e)
            pred_g = _predict_projection_bits(test_g, holdout_classifier)
            pred_e = _predict_projection_bits(test_e, holdout_classifier)
            p00 = float(np.mean(pred_g == 0))
            p11 = float(np.mean(pred_e == 1))
            p01 = 1.0 - p00
            p10 = 1.0 - p11
            fidelity = 0.5 * (p00 + p11)

            full_classifier = _fit_projection_classifier(shots_g, shots_e)
            pop = _extract_population_trace_with_classifier(
                result=result,
                qubit_uid=qubit.uid,
                n_points=delay_points.size,
                classifier=full_classifier,
            )
            n_shots_per_delay = int(main_shots.shape[1])
            stderr = np.sqrt(np.clip(pop * (1.0 - pop), 0.0, None) / max(n_shots_per_delay, 1))
            dynamic_range = float(np.max(pop) - np.min(pop))

            pass_fidelity = bool(fidelity >= float(min_fidelity))
            pass_dynamic = bool(dynamic_range >= float(min_dynamic_range))
            mapping_pass = pass_fidelity and pass_dynamic

            if mapping_pass:
                pass_count += 1
            else:
                fail_count += 1
                reason = "classifier_bad" if not pass_fidelity else "mapping_flat"
                fail_reasons[reason] = fail_reasons.get(reason, 0) + 1

            report.update(
                {
                    "assignment_fidelity_test": float(fidelity),
                    "p_g_as_e": float(p01),
                    "p_e_as_g": float(p10),
                    "calibration_separation_abs": float(abs(complex(full_classifier["v"]))),
                    "pe_delay_mean": [float(x) for x in pop],
                    "pe_delay_stderr": [float(x) for x in stderr],
                    "pe_dynamic_range": float(dynamic_range),
                    "n_shots_per_delay": int(n_shots_per_delay),
                    "mapping_pass": bool(mapping_pass),
                    "status_flag": "ok"
                    if mapping_pass
                    else ("classifier_bad" if not pass_fidelity else "mapping_flat"),
                    "split_mode": split_mode,
                }
            )
        except Exception as exc:
            fail_count += 1
            fail_reasons["error"] = fail_reasons.get("error", 0) + 1
            report.update(
                {
                    "status_flag": "error",
                    "error": str(exc),
                }
            )

        per_candidate.append(report)

    total = len(per_candidate)
    return {
        "enabled": True,
        "summary": {
            "total_candidates": int(total),
            "pass_candidates": int(pass_count),
            "fail_candidates": int(fail_count),
            "fail_reasons": fail_reasons,
            "pass_fraction": float(pass_count / total) if total > 0 else 0.0,
            "split_mode": split_mode,
            "min_fidelity": float(min_fidelity),
            "min_dynamic_range": float(min_dynamic_range),
        },
        "per_candidate": per_candidate,
    }


@workflow.workflow(name="analysis_readout_mid_sweep")
def analysis_workflow(
    results: list[RunExperimentResults],
    qubits: list[QuantumElement],
    delays: ArrayLike,
    frequency_points: ArrayLike | None = None,
    amplitude_points: ArrayLike | None = None,
    mid_sweep_points: dict[str, ArrayLike] | None = None,
    sweep_axis: Literal["frequency", "amplitude", "both"] = "both",
    reference_qubit: QuantumElement | None = None,
    options: ReadoutMidSweepAnalysisOptions | None = None,
) -> None:
    """Evaluate MID for each readout candidate and select best point."""
    options = ReadoutMidSweepAnalysisOptions() if options is None else options
    resolved_frequency_points = _resolve_axis_points(
        explicit_points=frequency_points,
        mid_sweep_points=mid_sweep_points,
        axis="frequency",
    )
    resolved_amplitude_points = _resolve_axis_points(
        explicit_points=amplitude_points,
        mid_sweep_points=mid_sweep_points,
        axis="amplitude",
    )
    metrics = calculate_metrics(
        results=results,
        qubits=qubits,
        delays=delays,
        frequency_points=resolved_frequency_points,
        amplitude_points=resolved_amplitude_points,
        sweep_axis=sweep_axis,
        reference_qubit=reference_qubit,
        max_mid_tolerance=options.max_mid_tolerance,
        max_mid_rate=options.max_mid_rate,
        prefer_lower_frequency_within_tolerance=options.prefer_lower_frequency_within_tolerance,
        prefer_lower_amplitude_within_tolerance=options.prefer_lower_amplitude_within_tolerance,
        relative_mid_increase_limit=options.relative_mid_increase_limit,
        use_cal_traces=options.use_cal_traces,
        cal_states=options.cal_states,
        do_rotation=options.do_rotation,
        do_pca=options.do_pca,
    )
    mapping_validation = _mapping_validation_disabled()
    with workflow.if_(options.do_mapping_validation):
        mapping_validation = validate_mapping_pipeline(
            results=results,
            qubits=qubits,
            delays=delays,
            frequency_points=resolved_frequency_points,
            amplitude_points=resolved_amplitude_points,
            sweep_axis=sweep_axis,
            split_mode=options.mapping_validation_split_mode,
            min_fidelity=options.mapping_validation_min_fidelity,
            min_dynamic_range=options.mapping_validation_min_dynamic_range,
        )
    metrics = _attach_mapping_validation(
        metrics=metrics,
        mapping_validation=mapping_validation,
    )

    with workflow.if_(options.do_plotting):
        with workflow.if_(options.do_plotting_mid_curves):
            with workflow.if_(sweep_axis != "both"):
                plot_mid_curves_1d(
                    sweep_axis=sweep_axis,
                    frequency_points=metrics["sweep_points"]["readout_resonator_frequency"],
                    amplitude_points=metrics["sweep_points"]["readout_amplitude"],
                    metrics=metrics,
                )
        with workflow.if_(options.do_plotting_heatmap):
            with workflow.if_(sweep_axis == "both"):
                plot_mid_heatmap(metrics=metrics)
        with workflow.if_(options.do_plotting_diagnostics):
            plot_mid_diagnostics(metrics=metrics, delays=delays)
        with workflow.if_(options.do_mapping_validation):
            with workflow.if_(options.do_plotting_mapping_validation):
                plot_mapping_validation_summary(
                    metrics=metrics,
                    delays=delays,
                )

    workflow.return_(metrics)


@workflow.task(save=False)
def calculate_metrics(
    results: list[RunExperimentResults],
    qubits: list[QuantumElement],
    delays: ArrayLike,
    frequency_points: ArrayLike | None = None,
    amplitude_points: ArrayLike | None = None,
    sweep_axis: Literal["frequency", "amplitude", "both"] = "both",
    reference_qubit: QuantumElement | None = None,
    max_mid_tolerance: float = 1e-4,
    max_mid_rate: float | None = None,
    prefer_lower_frequency_within_tolerance: bool = False,
    prefer_lower_amplitude_within_tolerance: bool = False,
    relative_mid_increase_limit: float = 0.2,
    use_cal_traces: bool = True,
    cal_states: str | tuple[str, ...] = "ge",
    do_rotation: bool = True,
    do_pca: bool = False,
) -> dict:
    frequency_points, amplitude_points = _coerce_mid_sweep_points(
        frequency_points=frequency_points,
        amplitude_points=amplitude_points,
        sweep_axis=sweep_axis,
    )
    delay_points = np.asarray(delays, dtype=float).reshape(-1)
    if delay_points.size < 1:
        raise ValueError("delays must contain at least one value.")
    if not np.all(np.isfinite(delay_points)):
        raise ValueError("delays must be finite.")

    if len(results) != len(qubits):
        raise ValueError(
            "Result and qubit counts must match: "
            f"{len(results)} != {len(qubits)}."
        )

    if sweep_axis == "frequency":
        n_freq, n_amp = frequency_points.size, 1
    elif sweep_axis == "amplitude":
        n_freq, n_amp = 1, amplitude_points.size
    else:
        n_freq, n_amp = frequency_points.size, amplitude_points.size
    if n_freq * n_amp != len(results):
        raise ValueError(
            "Result count does not match sweep-grid size: "
            f"{len(results)} != {n_freq} x {n_amp}."
        )

    if reference_qubit is None:
        reference_qubit = qubits[0]

    decay_rate = np.full((n_freq, n_amp), np.nan, dtype=float)
    decay_rate_stderr = np.full((n_freq, n_amp), np.nan, dtype=float)
    fit_success = np.zeros((n_freq, n_amp), dtype=bool)
    fit_payload: dict[str, dict] = {}

    for idx, (result_like, qubit) in enumerate(zip(results, qubits)):
        li, ai = divmod(idx, n_amp)
        result = unwrap_result_like(result_like)
        validate_result(result)
        key = f"{li}_{ai}"
        raw_diag: dict[str, object] = {}

        try:
            raw = np.asarray(result[dsl.handles.result_handle(qubit.uid)].data, dtype=complex)
            raw_diag = {
                "raw_abs": _collapse_population_to_1d(np.abs(raw), delay_points.size).tolist(),
                "raw_i": _collapse_population_to_1d(np.real(raw), delay_points.size).tolist(),
                "raw_q": _collapse_population_to_1d(np.imag(raw), delay_points.size).tolist(),
            }
        except Exception as raw_exc:
            raw_diag = {"raw_extract_error": str(raw_exc)}

        swpts = np.asarray(delay_points, dtype=float).reshape(-1)
        pop = np.array([], dtype=float)
        pop_error = None
        try:
            pop = _extract_population_trace(
                result=result,
                qubit_uid=qubit.uid,
                n_points=swpts.size,
            )
        except Exception as exc:
            pop_error = str(exc)

        if pop_error is None:
            try:
                fit_res = exponential_decay_fit(swpts, pop)
                gamma = float(fit_res.params["decay_rate"].value)
                gamma_err = fit_res.params["decay_rate"].stderr
                if gamma_err is None or not np.isfinite(gamma_err):
                    gamma_err = float("nan")
                if not np.isfinite(gamma) or gamma <= 0:
                    raise ValueError("Invalid fitted decay rate.")
                decay_rate[li, ai] = gamma
                decay_rate_stderr[li, ai] = float(gamma_err)
                fit_success[li, ai] = True
                fit_payload[key] = {
                    "sweep_points": swpts.tolist(),
                    "population": pop.tolist(),
                    "fit_success": True,
                    "decay_rate": gamma,
                    "decay_rate_stderr": float(gamma_err),
                    **raw_diag,
                }
            except Exception as exc:
                fit_payload[key] = {
                    "sweep_points": swpts.tolist(),
                    "population": pop.tolist(),
                    "fit_success": False,
                    "error": str(exc),
                    "decay_rate": float("nan"),
                    "decay_rate_stderr": float("nan"),
                    **raw_diag,
                }
        else:
            fit_payload[key] = {
                "sweep_points": swpts.tolist(),
                "population": [],
                "fit_success": False,
                "error": pop_error,
                "decay_rate": float("nan"),
                "decay_rate_stderr": float("nan"),
                **raw_diag,
            }

    finite_decay = np.isfinite(decay_rate) & (decay_rate > 0)
    gamma_baseline = float(np.min(decay_rate[finite_decay])) if finite_decay.any() else float("nan")
    mid_rate = np.full((n_freq, n_amp), float("inf"), dtype=float)
    if np.isfinite(gamma_baseline):
        mid_rate[finite_decay] = np.maximum(decay_rate[finite_decay] - gamma_baseline, 0.0)
    if max_mid_rate is not None:
        mid_rate = np.where(mid_rate <= float(max_mid_rate), mid_rate, float("inf"))

    if np.all(~np.isfinite(mid_rate)):
        quality = "invalid_data"
        best_li, best_ai = 0, 0
    else:
        best_li, best_ai, quality = _select_best_index(
            primary=mid_rate,
            secondary=decay_rate,
            tolerance=max_mid_tolerance,
            prefer_lower=(
                prefer_lower_frequency_within_tolerance
                if sweep_axis == "frequency"
                else prefer_lower_amplitude_within_tolerance
            )
            if sweep_axis != "both"
            else False,
        )

    freq_points = (
        frequency_points
        if sweep_axis != "amplitude"
        else np.array([float(reference_qubit.parameters.readout_resonator_frequency)])
    )
    amp_points = (
        amplitude_points
        if sweep_axis != "frequency"
        else np.array([float(reference_qubit.parameters.readout_amplitude)])
    )

    if sweep_axis == "frequency":
        new_frequency = float(freq_points[best_li])
        new_amplitude = float(reference_qubit.parameters.readout_amplitude)
    elif sweep_axis == "amplitude":
        new_frequency = float(reference_qubit.parameters.readout_resonator_frequency)
        new_amplitude = float(amp_points[best_ai])
    else:
        new_frequency = float(freq_points[best_li])
        new_amplitude = float(amp_points[best_ai])

    best_decay = float(decay_rate[best_li, best_ai])
    best_mid = float(mid_rate[best_li, best_ai])
    if np.isfinite(gamma_baseline) and gamma_baseline > 0 and np.isfinite(best_decay):
        relative_increase = float(best_decay / gamma_baseline - 1.0)
    else:
        relative_increase = float("nan")
    mid_gate_pass = bool(
        np.isfinite(relative_increase)
        and relative_increase <= float(relative_mid_increase_limit)
    )

    old_values = {
        reference_qubit.uid: {
            "readout_resonator_frequency": float(
                reference_qubit.parameters.readout_resonator_frequency
            ),
            "readout_amplitude": float(reference_qubit.parameters.readout_amplitude),
        }
    }
    new_values = {
        reference_qubit.uid: {
            "readout_resonator_frequency": new_frequency,
            "readout_amplitude": new_amplitude,
        }
    }
    if sweep_axis == "frequency":
        new_values[reference_qubit.uid] = {
            "readout_resonator_frequency": new_frequency,
        }
    elif sweep_axis == "amplitude":
        new_values[reference_qubit.uid] = {
            "readout_amplitude": new_amplitude,
        }

    return {
        "sweep_parameter": "readout_mid_sweep",
        "sweep_axis": sweep_axis,
        "sweep_points": {
            "readout_resonator_frequency": freq_points,
            "readout_amplitude": amp_points,
        },
        "metrics_vs_sweep": {
            "mid_rate_at_best": mid_rate,
            "decay_rate": decay_rate,
            "decay_rate_stderr": decay_rate_stderr,
            "fit_success": fit_success,
            "gamma_baseline": gamma_baseline,
            "relative_mid_increase_limit": float(relative_mid_increase_limit),
            "selection_primary": "mid_rate_at_best",
            "selection_secondary": "decay_rate",
        },
        "best_point": {
            "index_frequency": int(best_li),
            "index_amplitude": int(best_ai),
            "readout_resonator_frequency": new_frequency,
            "readout_amplitude": new_amplitude,
            "mid_rate_at_best": best_mid,
            "decay_rate": best_decay,
            "decay_rate_stderr": float(decay_rate_stderr[best_li, best_ai]),
            "gamma_baseline": gamma_baseline,
            "relative_mid_increase": relative_increase,
            "mid_gate_pass": mid_gate_pass,
        },
        "quality_flag": str(quality),
        "selection_reason": "Primary=mid_rate_at_best, secondary=decay_rate.",
        "old_parameter_values": old_values,
        "new_parameter_values": new_values,
        "candidate_fit_results": fit_payload,
    }


@workflow.task(save=False)
def plot_mid_curves_1d(
    sweep_axis: Literal["frequency", "amplitude"],
    frequency_points: np.ndarray,
    amplitude_points: np.ndarray,
    metrics: dict,
) -> mpl.figure.Figure:
    """Plot 1D MID and decay-rate curves with error bars."""
    best = metrics["best_point"]
    ms = metrics["metrics_vs_sweep"]
    mid = np.asarray(ms["mid_rate_at_best"], dtype=float)
    gamma = np.asarray(ms["decay_rate"], dtype=float)
    gamma_err = np.asarray(ms["decay_rate_stderr"], dtype=float)

    if sweep_axis == "frequency":
        x = np.asarray(frequency_points, dtype=float) * 1e-9
        mid_y = mid[:, 0]
        gamma_y = gamma[:, 0]
        gamma_e = gamma_err[:, 0]
        xlabel = "Readout resonator frequency (GHz)"
        best_x = float(best["readout_resonator_frequency"]) * 1e-9
    else:
        x = np.asarray(amplitude_points, dtype=float)
        mid_y = mid[0, :]
        gamma_y = gamma[0, :]
        gamma_e = gamma_err[0, :]
        xlabel = "Readout amplitude (a.u.)"
        best_x = float(best["readout_amplitude"])

    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    axes[0].plot(x, mid_y, marker="o")
    axes[0].axvline(best_x, linestyle="--", color="gray")
    axes[0].set_ylabel("mid_rate_at_best")
    axes[0].grid(alpha=0.25)

    axes[1].errorbar(x, gamma_y, yerr=gamma_e, fmt="o-", capsize=3)
    axes[1].axvline(best_x, linestyle="--", color="gray")
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel("decay_rate")
    axes[1].grid(alpha=0.25)
    axes[1].text(
        0.02,
        0.03,
        (
            f"MID gate pass={best['mid_gate_pass']}, "
            f"relative increase={best['relative_mid_increase']:.4g}, "
            f"baseline={best['gamma_baseline']:.4g}"
        ),
        transform=axes[1].transAxes,
        fontsize=8,
    )

    fig.suptitle(f"Readout MID sweep ({metrics['sweep_axis']})", fontsize=12)
    workflow.save_artifact("readout_mid_sweep_curves", fig)
    return fig


@workflow.task(save=False)
def plot_mid_heatmap(metrics: dict) -> mpl.figure.Figure:
    """Plot 2D MID heatmap for frequency × amplitude sweep."""
    sweep = metrics["sweep_points"]
    freqs_hz = np.asarray(sweep["readout_resonator_frequency"], dtype=float)
    amps = np.asarray(sweep["readout_amplitude"], dtype=float)
    mid = np.asarray(metrics["metrics_vs_sweep"]["mid_rate_at_best"], dtype=float)
    if mid.ndim != 2:
        raise ValueError("Expected 2D MID metric for heatmap.")

    fig, ax = plt.subplots(figsize=(7, 5))
    extent = [
        float(np.min(amps)),
        float(np.max(amps)),
        float(np.min(freqs_hz / 1e9)),
        float(np.max(freqs_hz / 1e9)),
    ]
    mid_plot = np.where(np.isfinite(mid), mid, np.nan)
    im = ax.imshow(
        mid_plot,
        aspect="auto",
        origin="lower",
        extent=extent,
    )
    ax.set_xlabel("Readout amplitude")
    ax.set_ylabel("Readout resonator frequency (GHz)")
    ax.set_title("Readout MID: mid_rate_at_best")
    ax.scatter(
        float(metrics["best_point"]["readout_amplitude"]),
        float(metrics["best_point"]["readout_resonator_frequency"]) / 1e9,
        c="w",
        s=45,
        edgecolors="k",
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("mid_rate_at_best")
    fig.suptitle(
        (
            f"Readout MID sweep (quality={metrics['quality_flag']}, "
            f"gate={metrics['best_point']['mid_gate_pass']})"
        ),
        fontsize=12,
    )
    workflow.save_artifact("readout_mid_sweep_heatmap", fig)
    return fig


@workflow.task(save=False)
def plot_mapping_validation_summary(
    metrics: dict,
    delays: ArrayLike,
) -> mpl.figure.Figure:
    """Plot mapping-validation summary and best-candidate p_e(delay)."""
    report = metrics.get("mapping_validation", {})
    if not isinstance(report, dict) or not bool(report.get("enabled", False)):
        raise ValueError("mapping_validation report is not enabled.")

    candidates = report.get("per_candidate", [])
    if not isinstance(candidates, list) or len(candidates) < 1:
        raise ValueError("No mapping-validation candidates available.")

    labels: list[str] = []
    fidelity: list[float] = []
    dyn_range: list[float] = []
    status: list[str] = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        labels.append(str(item.get("candidate_key", f"idx{len(labels)}")))
        fidelity.append(float(item.get("assignment_fidelity_test", float("nan"))))
        dyn_range.append(float(item.get("pe_dynamic_range", float("nan"))))
        status.append(str(item.get("status_flag", "unknown")))

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=False)
    x = np.arange(len(labels), dtype=float)

    colors = ["tab:green" if s == "ok" else "tab:red" for s in status]
    axes[0].bar(x, fidelity, color=colors, alpha=0.85)
    axes[0].set_ylabel("Holdout fidelity")
    axes[0].set_xticks(x, labels, rotation=45, ha="right")
    axes[0].grid(alpha=0.25)

    axes[1].bar(x, dyn_range, color=colors, alpha=0.85)
    axes[1].set_ylabel("p_e dynamic range")
    axes[1].set_xticks(x, labels, rotation=45, ha="right")
    axes[1].grid(alpha=0.25)

    best = metrics.get("best_point", {})
    best_key = f"{int(best.get('index_frequency', 0))}_{int(best.get('index_amplitude', 0))}"
    best_item = None
    for item in candidates:
        if isinstance(item, dict) and str(item.get("candidate_key", "")) == best_key:
            best_item = item
            break

    x_delay = np.asarray(delays, dtype=float).reshape(-1) * 1e6
    if isinstance(best_item, dict):
        pe = np.asarray(best_item.get("pe_delay_mean", []), dtype=float).reshape(-1)
        err = np.asarray(best_item.get("pe_delay_stderr", []), dtype=float).reshape(-1)
        if pe.size > 0 and x_delay.size == pe.size:
            axes[2].errorbar(x_delay, pe, yerr=err, fmt="o-", capsize=2)
            axes[2].set_title(
                f"Best candidate {best_key} ({best_item.get('status_flag', 'unknown')})"
            )
        else:
            axes[2].text(0.03, 0.5, "Best-candidate p_e(delay) unavailable", transform=axes[2].transAxes)
    else:
        axes[2].text(0.03, 0.5, "Best candidate not found in mapping report", transform=axes[2].transAxes)
    axes[2].set_xlabel("Delay (us)")
    axes[2].set_ylabel("p_e")
    axes[2].grid(alpha=0.25)

    summary = report.get("summary", {})
    fig.suptitle(
        (
            "Readout MID mapping validation "
            f"(pass={summary.get('pass_candidates', 0)}/{summary.get('total_candidates', 0)})"
        ),
        fontsize=12,
    )
    fig.tight_layout()
    workflow.save_artifact("readout_mid_sweep_mapping_validation", fig)
    return fig


@workflow.task(save=False)
def plot_mid_diagnostics(
    metrics: dict,
    delays: ArrayLike,
) -> mpl.figure.Figure:
    """Plot per-candidate population-vs-delay traces for failure diagnosis."""
    sweep = metrics["sweep_points"]
    n_freq = len(np.asarray(sweep["readout_resonator_frequency"], dtype=float).reshape(-1))
    n_amp = len(np.asarray(sweep["readout_amplitude"], dtype=float).reshape(-1))
    fit_payload = metrics.get("candidate_fit_results", {})

    fig, axes = plt.subplots(
        n_freq,
        n_amp,
        figsize=(4.6 * n_amp, 2.8 * n_freq),
        squeeze=False,
        sharex=False,
    )
    default_delays = np.asarray(delays, dtype=float).reshape(-1) * 1e6

    for li in range(n_freq):
        for ai in range(n_amp):
            ax = axes[li][ai]
            item = fit_payload.get(f"{li}_{ai}", {})
            swpts = np.asarray(item.get("sweep_points", []), dtype=float).reshape(-1)
            x = swpts * 1e6 if swpts.size > 0 else default_delays

            pop = np.asarray(item.get("population", []), dtype=float).reshape(-1)

            if pop.size > 0 and x.size == pop.size:
                ax.plot(x, pop, "o-", label="population")
            if pop.size > 0 and x.size != pop.size:
                xp = np.linspace(default_delays.min(), default_delays.max(), pop.size)
                ax.plot(xp, pop, "o-", label="population")
            if pop.size == 0:
                ax.text(
                    0.03,
                    0.5,
                    "No population trace (extraction failed)",
                    transform=ax.transAxes,
                    fontsize=8,
                )

            fit_ok = bool(item.get("fit_success", False))
            err = str(item.get("error", ""))
            if len(err) > 64:
                err = err[:61] + "..."
            title = f"[{li},{ai}] {'ok' if fit_ok else 'fail'}"
            if err:
                title = f"{title} | {err}"
            ax.set_title(title, fontsize=9)
            ax.grid(alpha=0.25)
            ax.set_xlabel("Delay (us)")
            if pop.size > 0:
                ax.legend(loc="best", fontsize=8)

    fig.suptitle("Readout MID candidate diagnostics", fontsize=12)
    fig.tight_layout()
    workflow.save_artifact("readout_mid_sweep_diagnostics", fig)
    return fig

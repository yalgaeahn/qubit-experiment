"""Loaded readout-mode linewidth from CW complex reflection, not notch S21.

NPL MAT58 (2021), eq.26/NLQFIT7: complex offset + complex pole, with cable delay.
A bounded complex least-squares refinement provides covariance and residuals;
fit-window stability and low-power convergence gate the representative result.
A two-pole comparison addresses overlapping filter modes when identifiable.
No QPU updates; kappa is an energy decay rate in rad/s, kappa_over_2pi_hz in Hz.
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
from laboneq import workflow
from laboneq.simple import dsl
from laboneq_applications.analysis.options import BasePlottingOptions
from laboneq_applications.core import validation
from scipy.ndimage import uniform_filter1d
from scipy.optimize import least_squares
from scipy.signal import find_peaks

from .plot_theme import with_plot_theme

MODEL_REFERENCE = "https://eprintspublications.npl.co.uk/9304/ (MAT58 eq.26)"


@workflow.workflow_options
class LinewidthAnalysisOptions:
    do_plotting: bool = workflow.option_field(True)
    geometry: str = workflow.option_field("reflection")
    fit_tolerance: float = workflow.option_field(0.05)
    allow_two_pole: bool = workflow.option_field(True)


@workflow.task_options(base_class=BasePlottingOptions)
class LinewidthPlotOptions:
    pass


def _response(x, p):
    """x in MHz, p=(fr MHz, log(width MHz), b complex, c complex, delay ns, ...)."""
    fr, lw, br, bi, cr, ci, tau = p[:7]
    signal = br + 1j * bi + (cr + 1j * ci) / (1 + 2j * (x - fr) / np.exp(lw))
    if len(p) == 11:
        fr2, lw2, c2r, c2i = p[7:]
        signal += (c2r + 1j * c2i) / (1 + 2j * (x - fr2) / np.exp(lw2))
    return signal * np.exp(-2j * np.pi * x * tau * 1e-3)


def _initial_pole(f, z, reference):
    x = (f - reference) / 1e6
    smooth = uniform_filter1d(z.real, 5) + 1j * uniform_filter1d(z.imag, 5)
    slope = np.abs(np.gradient(smooth, x))
    # Stay near the requested mode even when the survey contains other resonances.
    near = np.abs(x) < max(np.ptp(x) / 4, 4 * np.median(np.diff(x)))
    near[:3] = False
    near[-3:] = False
    indices = np.flatnonzero(near)
    if indices.size == 0:
        raise ValueError("Insufficient samples around the requested resonance.")
    peak = int(indices[np.argmax(slope[indices])])
    half = slope[peak] / 2
    lo = peak
    hi = peak
    while lo > 0 and slope[lo - 1] >= half:
        lo -= 1
    while hi < len(f) - 1 and slope[hi + 1] >= half:
        hi += 1
    width = max(x[hi] - x[lo], 8 * np.median(np.diff(x)))
    width = min(width, np.ptp(x) / 3)
    edge = max(3, len(z) // 10)
    b = np.mean(np.r_[z[:edge], z[-edge:]])
    c = z[peak] - b
    return np.array([x[peak], np.log(width), b.real, b.imag, c.real, c.imag, 0.0])


def _npl_seed(f, z, reference, seed):
    """Bound the number of NPL iterations; scipy refinement checks convergence.

    Installed skrf.fitted_s omits the NLQFIT7 line-delay term; never use it to
    reconstruct an NLQFIT7 fit. m7a is phase slope in rad/Hz, not a time constant.
    """
    import skrf
    from skrf.qfactor import Qfactor

    ntwk = skrf.Network(frequency=skrf.Frequency.from_f(f, unit="hz"), s=z[:, None, None])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        q = Qfactor(
            ntwk,
            "reflection",
            Q_L0=reference / (np.exp(seed[1]) * 1e6),
            f_L0=reference + seed[0] * 1e6,
        )
        fit = q.fit(method="NLQFIT7", loop_plan="fff")
    if fit.Q_L <= 0 or not f.min() < fit.f_L < f.max():
        raise ValueError("NPL seed returned an invalid resonance.")
    return np.array(
        [
            (fit.f_L - reference) / 1e6,
            np.log(fit.f_L / fit.Q_L / 1e6),
            fit.m1,
            fit.m2,
            fit.m3,
            fit.m4,
            -fit.m7a * 1e9 / (2 * np.pi),
        ]
    )


def _solve(f, z, reference, seed, max_nfev=1200):
    x = (f - reference) / 1e6
    step = np.median(np.diff(x))
    lower = [x.min(), np.log(step), -10, -10, -20, -20, -500]
    upper = [x.max(), np.log(2 * np.ptp(x)), 10, 10, 20, 20, 500]
    if len(seed) == 11:
        lower += [x.min(), np.log(step), -20, -20]
        upper += [x.max(), np.log(4 * np.ptp(x)), 20, 20]
    seed = np.clip(seed, np.asarray(lower) + 1e-10, np.asarray(upper) - 1e-10)

    def residual(p):
        e = _response(x, p) - z
        return np.r_[e.real, e.imag]

    fit = least_squares(residual, seed, bounds=(lower, upper), x_scale="jac", max_nfev=max_nfev)
    if not fit.success:
        raise ValueError("Bounded complex fit did not converge.")
    p = fit.x
    e = _response(x, p) - z
    dof = max(1, 2 * len(f) - len(p))
    _, singular, vh = np.linalg.svd(fit.jac, full_matrices=False)
    rank_ok = singular[-1] > singular[0] * 1e-10
    covariance = (vh.T / np.maximum(singular, 1e-30) ** 2) @ vh * np.sum(np.abs(e) ** 2) / dof
    fr = reference + p[0] * 1e6
    width = np.exp(p[1]) * 1e6
    width_se = width * np.sqrt(max(0, covariance[1, 1]))
    # Q=fr/width; retain the covariance of frequency and log(width).
    q_gradient = np.zeros(len(p))
    q_gradient[0] = 1e6 / width
    q_gradient[1] = -fr / width
    q_se = np.sqrt(max(0, q_gradient @ covariance @ q_gradient))
    structure = 0.0
    if np.mean(np.abs(e) ** 2) > 1e-25:
        structure = float(abs(np.mean(e[1:] * e[:-1].conj())) / np.mean(np.abs(e) ** 2))
    rss = max(float(np.sum(np.abs(e) ** 2)), 1e-30)
    return {
        "parameters": p,
        "covariance": covariance,
        "rank_ok": bool(rank_ok),
        "fr_hz": float(fr),
        "fr_stderr_hz": float(np.sqrt(max(0, covariance[0, 0])) * 1e6),
        "linewidth_hz": float(width),
        "linewidth_stderr_hz": float(width_se),
        "q_loaded": float(fr / width),
        "q_loaded_stderr": float(q_se),
        "kappa_rad_s": float(2 * np.pi * width),
        "kappa_rad_s_stderr": float(2 * np.pi * width_se),
        "kappa_over_2pi_hz": float(width),
        "kappa_over_2pi_stderr_hz": float(width_se),
        "energy_decay_time_s": float(1 / (2 * np.pi * width)),
        "energy_decay_time_stderr_s": float(width_se / (2 * np.pi * width**2)),
        "bic": float(2 * len(f) * np.log(rss / (2 * len(f))) + len(p) * np.log(2 * len(f))),
        "residual_rms_normalized": float(np.sqrt(np.mean(np.abs(e) ** 2))),
        "residual_lag1": structure,
    }


def _window_fits(f, z, reference, fit):
    p = fit["parameters"]
    center = fit["fr_hz"]
    width = fit["linewidth_hz"]
    rows = []
    for multiple in (1.5, 2.0, 3.0):
        mask = np.abs(f - center) <= multiple * width
        if (
            mask.sum() < 21
            or f.min() > center - multiple * width
            or f.max() < center + multiple * width
        ):
            rows.append({"half_width_linewidths": multiple, "status": "insufficient_coverage"})
            continue
        try:
            local_z = z[mask]
            seed = p[:7].copy()
            if len(p) == 11:
                x = (f[mask] - reference) / 1e6
                second = (p[9] + 1j * p[10]) / (1 + 2j * (x - p[7]) / np.exp(p[8]))
                # Filter contribution is constrained by the full-band two-pole fit.
                local_z = local_z - second * np.exp(-2j * np.pi * x * p[6] * 1e-3)
            local = _solve(f[mask], local_z, reference, seed)
            rows.append(
                {
                    "half_width_linewidths": multiple,
                    "status": "ok",
                    "q_loaded": local["q_loaded"],
                    "fr_hz": local["fr_hz"],
                }
            )
        except (ValueError, np.linalg.LinAlgError) as exc:
            rows.append({"half_width_linewidths": multiple, "status": "failed", "reason": str(exc)})
    return rows


def _quality(fit, windows, tolerance):
    reasons = []
    if not fit["rank_ok"]:
        reasons.append("rank_deficient_fit")
    if fit["q_loaded_stderr"] / fit["q_loaded"] > tolerance:
        reasons.append("large_statistical_error")
    if fit["residual_lag1"] > 0.3:
        reasons.append("structured_residual")
    if any(w["status"] != "ok" for w in windows):
        reasons.append("insufficient_fit_window_coverage")
    elif max(abs(w["q_loaded"] / fit["q_loaded"] - 1) for w in windows) > tolerance:
        reasons.append("window_dependent_q")
    return reasons


def fit_trace(
    frequencies,
    raw,
    reference_frequency_hz,
    stored_delay_s=None,
    tolerance=0.05,
    allow_two_pole=True,
):
    """Fit one complex reflection trace. Failure is data, not a fabricated kappa."""
    f = np.asarray(frequencies, float)
    raw = np.asarray(raw, complex)
    order = np.argsort(f)
    f, raw = f[order], raw[order]
    if f.size < 21 or raw.shape != f.shape or not np.all(np.isfinite(raw)):
        raise ValueError("A trace must have at least 21 finite complex samples.")
    edge = max(3, len(f) // 10)
    if stored_delay_s is None:
        # Estimate each off-resonant edge independently: a phase winding across
        # the resonator is not cable delay and must not join the two edge offsets.
        slopes = [
            np.polyfit(f[sl] - reference_frequency_hz, np.unwrap(np.angle(raw[sl])), 1)[0]
            for sl in (slice(0, edge), slice(-edge, None))
        ]
        tau0 = -float(np.mean(slopes)) / (2 * np.pi)
    else:
        tau0 = float(stored_delay_s)
    if not np.isfinite(tau0):
        raise ValueError("stored_delay_s must be finite or None.")
    scale = float(np.max(np.abs(raw)))
    if scale <= 0:
        raise ValueError("Trace is zero; a resonator pole cannot be identified.")
    z = raw / scale * np.exp(2j * np.pi * (f - reference_frequency_hz) * tau0)
    seed = _initial_pole(f, z, reference_frequency_hz)
    # A derivative peak can be a noise spike at low power. Fit every candidate on
    # the same full data, including reference-centered linewidth seeds; never
    # compare residuals computed over different frequency windows.
    mask = np.ones(len(f), dtype=bool)
    seeds = [seed]
    x = (f - reference_frequency_hz) / 1e6
    for width in np.ptp(x) * np.array([1 / 32, 1 / 16, 1 / 8]):
        basis = 1 / (1 + 2j * x / width)
        b0, c0 = np.linalg.lstsq(np.column_stack([np.ones(len(x)), basis]), z, rcond=None)[0]
        seeds.append(np.array([0.0, np.log(width), b0.real, b0.imag, c0.real, c0.imag, 0.0]))
    seed_method = "NPL_NLQFIT7_plus_bounded_complex_refinement"
    try:
        seeds.append(_npl_seed(f[mask], z[mask], reference_frequency_hz, seed))
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        seed_method = "NPL_equation26_bounded_complex_fit_NLQFIT7_seed_failed"
    fits = []
    for start in seeds:
        try:
            fits.append(_solve(f[mask], z[mask], reference_frequency_hz, start))
        except (ValueError, np.linalg.LinAlgError):
            continue
    if not fits:
        raise ValueError("No converged complex reflection fit.")
    fit = min(fits, key=lambda value: value["residual_rms_normalized"])
    # Evaluate windows on available data; do not conceal a broad filter by cropping.
    windows = _window_fits(f, z, reference_frequency_hz, fit)
    reasons = _quality(fit, windows, tolerance)
    model = "NPL_single_pole_reflection_with_delay"
    comparisons = []
    if reasons and allow_two_pole:
        # Wide data are necessary to identify a filter pole, not just its slope.
        full_single = _solve(f, z, reference_frequency_hz, fit["parameters"])
        x = (f - reference_frequency_hz) / 1e6
        remainder = z - _response(x, full_single["parameters"])
        peaks, _ = find_peaks(np.abs(np.gradient(remainder, x)), distance=max(3, len(f) // 10))
        candidates = sorted(peaks, key=lambda i: abs(remainder[i]), reverse=True)
        candidates = [i for i in candidates if abs(f[i] - fit["fr_hz"]) > 2 * fit["linewidth_hz"]][
            :2
        ]
        for i in candidates:
            for width in (np.ptp(x) / 5, np.ptp(x) / 2):
                start = np.r_[
                    full_single["parameters"],
                    x[i],
                    np.log(width),
                    remainder[i].real,
                    remainder[i].imag,
                ]
                try:
                    double = _solve(f, z, reference_frequency_hz, start)
                    p = double["parameters"]
                    errors = np.sqrt(np.maximum(0, np.diag(double["covariance"])))
                    identifiable = (
                        double["rank_ok"] and errors[8] < 0.5 and errors[7] < np.ptp(x) / 4
                    )
                    w = _window_fits(f, z, reference_frequency_hz, double)
                    why = _quality(double, w, tolerance)
                    comparisons.append(
                        {
                            "model": "two_pole",
                            "bic": double["bic"],
                            "identifiable": bool(identifiable),
                            "reasons": why,
                        }
                    )
                    if identifiable and not why and full_single["bic"] - double["bic"] > 10:
                        if model.startswith("NPL") or double["bic"] < fit["bic"]:
                            fit, windows, reasons = double, w, []
                            model = "two_complex_poles_with_delay"
                except (ValueError, np.linalg.LinAlgError):
                    continue
    p = fit["parameters"]
    modeled = (
        _response((f - reference_frequency_hz) / 1e6, p)
        * scale
        * np.exp(-2j * np.pi * (f - reference_frequency_hz) * tau0)
    )
    fit.update(
        {
            "status": "ok" if not reasons else "unreliable",
            "reasons": reasons,
            "model": model,
            "model_reference": MODEL_REFERENCE,
            "seed_method": seed_method,
            "delay_s": float(tau0 + p[6] * 1e-9),
            "delay_initial_s": float(tau0),
            "fit_window_hz": [float(f[mask].min()), float(f[mask].max())]
            if len(p) == 7
            else [float(f.min()), float(f.max())],
            "window_checks": windows,
            "model_comparisons": comparisons,
            "fitted_complex": modeled[np.argsort(order)],
            "residual_complex": (raw - modeled)[np.argsort(order)],
        }
    )
    return fit


def choose_representative(points, tolerance=0.05):
    """Require the lowest two measured powers to pass and agree; never skip failures."""
    ordered = sorted(points, key=lambda p: p["amplitude"])
    if len(ordered) < 2:
        return None, {"status": "not_verified", "reason": "at_least_two_powers_required"}
    low, other = ordered[:2]
    if low["status"] != "ok" or other["status"] != "ok":
        return None, {"status": "not_verified", "reason": "lowest_power_fit_unreliable"}
    q_limit = max(
        tolerance * low["q_loaded"], 2 * np.hypot(low["q_loaded_stderr"], other["q_loaded_stderr"])
    )
    f_limit = max(
        tolerance * low["linewidth_hz"], 2 * np.hypot(low["fr_stderr_hz"], other["fr_stderr_hz"])
    )
    stable = (
        abs(low["q_loaded"] - other["q_loaded"]) <= q_limit
        and abs(low["fr_hz"] - other["fr_hz"]) <= f_limit
    )
    convergence = {
        "status": "converged" if stable else "not_converged",
        "q_difference": float(abs(low["q_loaded"] - other["q_loaded"])),
        "frequency_difference_hz": float(abs(low["fr_hz"] - other["fr_hz"])),
        "q_limit": float(q_limit),
        "frequency_limit_hz": float(f_limit),
    }
    if not stable:
        return None, convergence
    keys = (
        "amplitude",
        "output_power_dbm",
        "fr_hz",
        "fr_stderr_hz",
        "q_loaded",
        "q_loaded_stderr",
        "linewidth_hz",
        "linewidth_stderr_hz",
        "kappa_rad_s",
        "kappa_rad_s_stderr",
        "kappa_over_2pi_hz",
        "kappa_over_2pi_stderr_hz",
        "energy_decay_time_s",
        "energy_decay_time_stderr_s",
        "model",
        "delay_s",
    )
    representative = {k: low[k] for k in keys}
    representative["power_variation_fraction"] = abs(low["q_loaded"] / other["q_loaded"] - 1)
    return representative, convergence


@workflow.task
def calculate_metrics(
    result,
    qubit,
    frequencies,
    amplitudes,
    reference_frequency_hz,
    geometry="reflection",
    fit_tolerance=0.05,
    allow_two_pole=True,
):
    if geometry != "reflection":
        raise ValueError("Only reflection geometry is supported.")
    validation.validate_result(result)
    qubit = validation.validate_and_convert_single_qubit_sweeps(qubit)
    f, a = np.asarray(frequencies, float), np.asarray(amplitudes, float)
    if (
        f.ndim != 1
        or f.size < 21
        or not np.all(np.isfinite(f))
        or not (np.all(np.diff(f) > 0) or np.all(np.diff(f) < 0))
    ):
        raise ValueError("frequencies must contain at least 21 finite monotonic points.")
    if a.ndim != 1 or len(a) == 0 or not np.all(np.isfinite(a)) or np.any((a <= 0) | (a > 1)):
        raise ValueError("amplitudes must be a nonempty 1D array in (0,1].")
    if not f.min() < reference_frequency_hz < f.max():
        raise ValueError("Reference frequency must be inside the sweep.")
    if not 0 < fit_tolerance < 1:
        raise ValueError("fit_tolerance must be in (0,1).")
    raw = np.asarray(result[dsl.handles.result_handle(qubit.uid)].data)
    if raw.shape != (len(a), len(f)):
        raise ValueError("Raw data shape must be (number of amplitudes, number of frequencies).")
    points = []
    for amplitude, trace in zip(a, raw, strict=True):
        point = {
            "amplitude": float(amplitude),
            "output_power_dbm": float(
                qubit.parameters.readout_range_out + 20 * np.log10(amplitude)
            ),
        }
        try:
            point.update(
                fit_trace(
                    f,
                    trace,
                    reference_frequency_hz,
                    qubit.parameters.readout_phase_delay,
                    fit_tolerance,
                    allow_two_pole,
                )
            )
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            point.update(status="failed", reasons=[str(exc)])
        points.append(point)
    representative, convergence = choose_representative(points, fit_tolerance)
    return {
        "qubit_uid": qubit.uid,
        "geometry": geometry,
        "reference_frequency_hz": float(reference_frequency_hz),
        "frequencies_hz": f,
        "amplitudes": a,
        "raw_complex": raw.copy(),
        "power_points": tuple(points),
        "representative": representative,
        "power_convergence": convergence,
        "interpretation": "loaded hybridized readout-mode linewidth; natural qubit state distribution",
        "uncertainty_note": "Jacobian standard errors are conditional on the response model; power/window/repeat variation is reported separately.",
        "units": {
            "fr_hz": "Hz",
            "kappa_over_2pi_hz": "Hz",
            "kappa_rad_s": "rad/s",
            "energy_decay_time_s": "s",
        },
    }


@workflow.workflow(name="analysis_resonator_linewidth_cw")
def analysis_workflow(
    result,
    qubit,
    frequencies,
    amplitudes,
    reference_frequency_hz,
    options: LinewidthAnalysisOptions | None = None,
):
    metrics = calculate_metrics(
        result,
        qubit,
        frequencies,
        amplitudes,
        reference_frequency_hz,
        options.geometry,
        options.fit_tolerance,
        options.allow_two_pole,
    )
    with workflow.if_(options.do_plotting):
        plot_fits(metrics)
    workflow.return_(metrics)


@workflow.task
@with_plot_theme
def plot_fits(metrics, options: LinewidthPlotOptions | None = None):
    opts = LinewidthPlotOptions() if options is None else options
    f = metrics["frequencies_hz"]
    reference = metrics["reference_frequency_hz"]
    x = (f - reference) / 1e6
    figures = {}
    for i, point in enumerate(metrics["power_points"]):
        raw = metrics["raw_complex"][i]
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
        power = point["output_power_dbm"]
        title = f"{metrics['qubit_uid']} CW reflection: {power:.2f} dBm (instrument port), {point['status']}"
        if "q_loaded" in point:
            title += f"\nf = {point['fr_hz'] / 1e9:.9f} GHz; QL = {point['q_loaded']:.1f}; κ/2π = {point['linewidth_hz'] / 1e6:.4f} MHz"
        fig.suptitle(title)
        axes[0, 0].plot(x, np.abs(raw), ".", ms=2, label="data")
        tau = point.get("delay_s", 0)
        rotation = np.exp(2j * np.pi * (f - reference) * tau)
        axes[0, 1].plot(x, np.angle(raw * rotation), ".", ms=2)
        axes[1, 0].plot((raw * rotation).real, (raw * rotation).imag, ".", ms=2)
        if "fitted_complex" in point:
            fitted = point["fitted_complex"]
            axes[0, 0].plot(x, np.abs(fitted), "-", label="fit")
            axes[0, 1].plot(x, np.angle(fitted * rotation), "-")
            axes[1, 0].plot((fitted * rotation).real, (fitted * rotation).imag, "-")
            error = point["residual_complex"]
            axes[1, 1].plot(x, error.real, label="I")
            axes[1, 1].plot(x, error.imag, label="Q")
            axes[1, 1].legend()
        else:
            axes[1, 1].text(
                0.5,
                0.5,
                "\n".join(point["reasons"]),
                transform=axes[1, 1].transAxes,
                ha="center",
                wrap=True,
            )
        axes[0, 0].set_ylabel("Magnitude (a.u.)")
        axes[0, 1].set_ylabel("Delay-corrected phase (rad)")
        axes[1, 0].set(xlabel="I (a.u.)", ylabel="Q (a.u.)", aspect="equal")
        axes[1, 1].set_ylabel("Complex residual (a.u.)")
        for ax in (axes[0, 0], axes[0, 1], axes[1, 1]):
            ax.set_xlabel(f"Frequency − {reference / 1e9:.7f} GHz (MHz)")
        axes[0, 0].legend()
        key = f"Linewidth_CW_{metrics['qubit_uid']}_{i}"
        if opts.save_figures:
            workflow.save_artifact(key, fig)
        if opts.close_figures:
            plt.close(fig)
        figures[key] = fig
    return figures

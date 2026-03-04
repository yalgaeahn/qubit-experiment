from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from analysis.fitting_helpers import cosine_oscillatory_decay
from analysis.residual_zz_echo import (
    _build_pair_plan_entries,
    _fit_pair,
    _split_state_axis,
)


@dataclass
class _QubitStub:
    uid: str


def _make_trace(
    x_s: np.ndarray,
    freq_hz: float,
    *,
    decay_time_s: float = 18e-6,
    phase: float = 0.3,
    amplitude: float = 0.45,
    oscillation_offset: float = 0.0,
    exponential_offset: float = 0.5,
) -> np.ndarray:
    return np.asarray(
        cosine_oscillatory_decay(
            x_s,
            frequency=freq_hz,
            phase=phase,
            decay_time=decay_time_s,
            amplitude=amplitude,
            oscillation_offset=oscillation_offset,
            exponential_offset=exponential_offset,
            decay_exponent=1.0,
        ),
        dtype=float,
    )


def test_build_pair_plan_all_pairs_order() -> None:
    q0, q3 = _QubitStub("q0"), _QubitStub("q3")
    q1, q2 = _QubitStub("q1"), _QubitStub("q2")
    delays = [np.array([0.0, 1.0]), np.array([0.0, 2.0])]
    detunings = [1.0, 2.0]

    plan = _build_pair_plan_entries(
        ctrl_qubits=[q0, q3],
        targ_qubits=[q1, q2],
        delays=delays,
        detunings=detunings,
        mapping_mode="all_pairs",
    )

    assert [entry["pair_key"] for entry in plan] == [
        "q0->q1",
        "q0->q2",
        "q3->q1",
        "q3->q2",
    ]
    assert [entry["detuning_hz"] for entry in plan] == [1.0, 2.0, 1.0, 2.0]


def test_build_pair_plan_pairwise_order() -> None:
    q0, q3 = _QubitStub("q0"), _QubitStub("q3")
    q1, q2 = _QubitStub("q1"), _QubitStub("q2")
    delays = [np.array([0.0, 1.0]), np.array([0.0, 2.0])]
    detunings = [1.0, 2.0]

    plan = _build_pair_plan_entries(
        ctrl_qubits=[q0, q3],
        targ_qubits=[q1, q2],
        delays=delays,
        detunings=detunings,
        mapping_mode="pairwise",
    )

    assert [entry["pair_key"] for entry in plan] == ["q0->q1", "q3->q2"]


def test_build_pair_plan_pairwise_rejects_mismatch() -> None:
    q0, q3 = _QubitStub("q0"), _QubitStub("q3")
    q1 = _QubitStub("q1")

    with pytest.raises(ValueError, match="pairwise mapping requires"):
        _build_pair_plan_entries(
            ctrl_qubits=[q0, q3],
            targ_qubits=[q1],
            delays=[np.array([0.0, 1.0])],
            detunings=[1.0],
            mapping_mode="pairwise",
        )


def test_build_pair_plan_all_pairs_excludes_self_pairs_by_default() -> None:
    q0, q1, q2 = _QubitStub("q0"), _QubitStub("q1"), _QubitStub("q2")
    delays = [np.array([0.0, 1.0])] * 3
    detunings = [1.0, 2.0, 3.0]

    plan = _build_pair_plan_entries(
        ctrl_qubits=[q0, q1, q2],
        targ_qubits=[q0, q1, q2],
        delays=delays,
        detunings=detunings,
        mapping_mode="all_pairs",
    )

    assert [entry["pair_key"] for entry in plan] == [
        "q0->q1",
        "q0->q2",
        "q1->q0",
        "q1->q2",
        "q2->q0",
        "q2->q1",
    ]


def test_build_pair_plan_raises_when_only_self_pairs_remain() -> None:
    q0 = _QubitStub("q0")

    with pytest.raises(ValueError, match="automatic self-pair removal"):
        _build_pair_plan_entries(
            ctrl_qubits=[q0],
            targ_qubits=[q0],
            delays=[np.array([0.0, 1.0])],
            detunings=[1.0],
            mapping_mode="all_pairs",
        )


def test_split_state_axis_parses_single_run_state_sweep() -> None:
    raw = np.arange(2 * 11, dtype=float).reshape(2, 11) + 1j * 0.0
    split = _split_state_axis(raw, num_states=2, expected_points=11)
    assert split.shape == (2, 11)
    assert np.allclose(split, raw)

    split_t = _split_state_axis(raw.T, num_states=2, expected_points=11)
    assert split_t.shape == (2, 11)
    assert np.allclose(split_t, raw)


def test_split_state_axis_raises_on_unsupported_shape() -> None:
    raw = np.ones((3, 11), dtype=complex)
    with pytest.raises(ValueError, match="state axis"):
        _split_state_axis(raw, num_states=2, expected_points=11)


def test_fit_pair_recovers_known_residual_zz_from_synthetic_data() -> None:
    x_s = np.linspace(0.0, 24e-6, 301)
    fit_freq_g = 0.90e6
    fit_freq_e = 1.20e6
    y_g = _make_trace(x_s, fit_freq_g, phase=0.2)
    y_e = _make_trace(x_s, fit_freq_e, phase=1.1)

    out = _fit_pair(
        x_s=x_s,
        y_by_state={"g": y_g, "e": y_e},
        old_freq_hz=5.0e9,
        introduced_detuning_hz=0.0,
        transition="ge",
        do_pca=False,
        min_decay_time=0.2e-6,
        max_rel_freq_err=0.8,
        bootstrap_samples=0,
        bootstrap_seed=7,
        fit_parameters_hints=None,
    )

    assert out["residual_zz"]["quality_flag"] == "ok"
    expected_zz_hz = fit_freq_g - fit_freq_e
    assert abs(out["residual_zz"]["nominal_hz"] - expected_zz_hz) < 1.5e4


def test_fit_pair_applies_detuning_correction_formula() -> None:
    x_s = np.linspace(0.0, 20e-6, 251)
    fit_freq_g = 0.80e6
    fit_freq_e = 1.10e6
    old_freq_hz = 4.95e9
    detuning_hz = 1.50e6

    out = _fit_pair(
        x_s=x_s,
        y_by_state={
            "g": _make_trace(x_s, fit_freq_g, phase=0.0),
            "e": _make_trace(x_s, fit_freq_e, phase=0.7),
        },
        old_freq_hz=old_freq_hz,
        introduced_detuning_hz=detuning_hz,
        transition="ge",
        do_pca=False,
        min_decay_time=0.2e-6,
        max_rel_freq_err=0.8,
        bootstrap_samples=0,
        bootstrap_seed=3,
        fit_parameters_hints=None,
    )

    cond_g = out["conditional_frequency"]["g"]
    cond_e = out["conditional_frequency"]["e"]
    assert abs(cond_g.n - (old_freq_hz + detuning_hz - fit_freq_g)) < 2.0e4
    assert abs(cond_e.n - (old_freq_hz + detuning_hz - fit_freq_e)) < 2.0e4


def test_fit_pair_bootstrap_sigma_combines_with_model_sigma() -> None:
    rng = np.random.default_rng(123)
    x_s = np.linspace(0.0, 25e-6, 321)
    fit_freq_g = 1.30e6
    fit_freq_e = 1.55e6
    y_g = _make_trace(x_s, fit_freq_g, phase=0.4) + rng.normal(0.0, 0.01, size=x_s.size)
    y_e = _make_trace(x_s, fit_freq_e, phase=1.0) + rng.normal(0.0, 0.01, size=x_s.size)

    out = _fit_pair(
        x_s=x_s,
        y_by_state={"g": y_g, "e": y_e},
        old_freq_hz=5.0e9,
        introduced_detuning_hz=0.0,
        transition="ge",
        do_pca=False,
        min_decay_time=0.2e-6,
        max_rel_freq_err=0.9,
        bootstrap_samples=80,
        bootstrap_seed=456,
        fit_parameters_hints=None,
    )

    sigma_model = out["residual_zz"]["sigma_model_hz"]
    sigma_boot = out["residual_zz"]["sigma_boot_hz"]
    sigma_final = out["residual_zz"]["sigma_final_hz"]
    if np.isfinite(sigma_model):
        assert sigma_final >= sigma_model
    if np.isfinite(sigma_boot):
        assert sigma_final >= sigma_boot


def test_fit_pair_flags_failure_on_unfit_state_trace() -> None:
    x_s = np.linspace(0.0, 20e-6, 251)
    y_flat = np.full(x_s.size, 0.5, dtype=float)
    out = _fit_pair(
        x_s=x_s,
        y_by_state={"g": y_flat, "e": y_flat},
        old_freq_hz=5.0e9,
        introduced_detuning_hz=0.0,
        transition="ge",
        do_pca=False,
        min_decay_time=0.2e-6,
        max_rel_freq_err=0.5,
        bootstrap_samples=40,
        bootstrap_seed=999,
        fit_parameters_hints=None,
    )

    assert out["residual_zz"]["quality_flag"] == "fail"
    assert np.isnan(out["residual_zz"]["nominal_hz"])

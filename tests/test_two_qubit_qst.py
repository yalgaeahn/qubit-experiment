from __future__ import annotations

from types import SimpleNamespace

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

pytest.importorskip("laboneq")
pytest.importorskip("laboneq_applications")
pytest.importorskip("scipy")

from qubit_experiment.analysis import two_qubit_qst as qst_analysis
from qubit_experiment.experiments import two_qubit_qst as qst


def test_workflow_options_expose_qst_and_nested_analysis_fields() -> None:
    options = qst.experiment_workflow.options()

    assert hasattr(options, "initial_state")
    assert hasattr(options, "custom_prep")
    assert hasattr(options, "do_convergence_validation")
    assert hasattr(options, "convergence_repeats_per_state")
    assert hasattr(options, "do_shot_sweep_convergence")
    assert hasattr(options, "shot_sweep_log2_values")
    assert hasattr(options, "count")
    assert hasattr(options, "active_reset")
    assert hasattr(options, "do_plotting")
    assert hasattr(options, "max_mle_iterations")


def test_workflow_options_do_not_expose_removed_rip_fields() -> None:
    options = qst.experiment_workflow.options()

    assert not hasattr(options, "validation_mode")
    assert not hasattr(options, "use_rip")
    assert not hasattr(options, "enforce_target_match")


def test_resolve_target_configuration_canonicalizes_matching_target() -> None:
    resolved = qst._resolve_target_configuration_impl(
        custom_prep=False,
        initial_state="ge",
        target_state=None,
    )

    assert resolved["initial_state"] == "01"
    assert resolved["target_state_effective"] == "01"
    assert resolved["custom_prep"] is False


def test_resolve_target_configuration_rejects_mismatched_target() -> None:
    with pytest.raises(ValueError, match="target_state must match initial_state"):
        qst._resolve_target_configuration_impl(
            custom_prep=False,
            initial_state="++",
            target_state="00",
        )


def test_resolve_target_configuration_rejects_custom_prep_for_now() -> None:
    with pytest.raises(
        NotImplementedError,
        match="independently of initial_state",
    ):
        qst._resolve_target_configuration_impl(
            custom_prep=True,
            initial_state="++",
            target_state=None,
        )


def test_create_experiment_rejects_custom_prep_until_implemented() -> None:
    with pytest.raises(
        NotImplementedError,
        match="independently of initial_state",
    ):
        qst._create_experiment_impl(
            qpu=SimpleNamespace(),
            qubits=[SimpleNamespace(uid="q0"), SimpleNamespace(uid="q1")],
            bus=SimpleNamespace(uid="b0"),
            custom_prep=True,
        )


def test_create_experiment_rejects_invalid_acquisition_type() -> None:
    options = SimpleNamespace(
        count=4096,
        acquisition_type="raw",
        averaging_mode=qst.AveragingMode.SINGLE_SHOT,
        repetition_mode=None,
        repetition_time=None,
        reset_oscillator_phase=False,
        active_reset=False,
        active_reset_repetitions=1,
        active_reset_states="ge",
    )

    with pytest.raises(ValueError, match="AcquisitionType.INTEGRATION"):
        qst._create_experiment_impl(
            qpu=SimpleNamespace(),
            qubits=[SimpleNamespace(uid="q0"), SimpleNamespace(uid="q1")],
            bus=SimpleNamespace(uid="b0"),
            options=options,
        )


def test_create_experiment_rejects_invalid_averaging_mode() -> None:
    options = SimpleNamespace(
        count=4096,
        acquisition_type=qst.AcquisitionType.INTEGRATION,
        averaging_mode="cyclic",
        repetition_mode=None,
        repetition_time=None,
        reset_oscillator_phase=False,
        active_reset=False,
        active_reset_repetitions=1,
        active_reset_states="ge",
    )

    with pytest.raises(ValueError, match="AveragingMode.SINGLE_SHOT"):
        qst._create_experiment_impl(
            qpu=SimpleNamespace(),
            qubits=[SimpleNamespace(uid="q0"), SimpleNamespace(uid="q1")],
            bus=SimpleNamespace(uid="b0"),
            options=options,
        )


def test_create_experiment_rejects_non_two_qubit_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        qst.validation,
        "validate_and_convert_qubits_sweeps",
        lambda qubits: list(qubits),
    )

    with pytest.raises(ValueError, match="expects exactly 2 qubits"):
        qst._create_experiment_impl(
            qpu=SimpleNamespace(),
            qubits=[SimpleNamespace(uid="q0")],
            bus=SimpleNamespace(uid="b0"),
        )


def test_summarize_statistical_convergence_matches_expected_keys() -> None:
    report = qst_analysis._summarize_statistical_convergence_impl(
        [
            {
                "state_label": "00",
                "repeat_index": 0,
                "fidelity_to_target": 0.95,
                "optimizer_success": True,
                "negative_log_likelihood": 10.0,
                "rho_min_eigenvalue": 0.0,
            },
            {
                "state_label": "00",
                "repeat_index": 1,
                "fidelity_to_target": 0.85,
                "optimizer_success": False,
                "negative_log_likelihood": 12.0,
                "rho_min_eigenvalue": -0.01,
            },
        ]
    )

    assert set(report.keys()) == {"per_state", "aggregate"}
    assert report["per_state"]["00"]["num_runs"] == 2
    assert report["per_state"]["00"]["optimizer_success_rate"] == pytest.approx(0.5)
    assert report["aggregate"]["num_total_runs"] == 2
    assert report["aggregate"]["pooled_fidelity_mean"] == pytest.approx(0.9)


def test_aggregate_shot_sweep_statistics_computes_notebook_schema() -> None:
    rows = [
        {
            "state": "00",
            "log2_shots": 3,
            "shots": 8,
            "repeat": 1,
            "fidelity": 0.9,
            "infidelity": 0.1,
            "log10_infidelity": np.log10(0.1),
            "nll": 10.0,
            "min_eig": 0.0,
        },
        {
            "state": "00",
            "log2_shots": 3,
            "shots": 8,
            "repeat": 2,
            "fidelity": 0.8,
            "infidelity": 0.2,
            "log10_infidelity": np.log10(0.2),
            "nll": 11.0,
            "min_eig": -0.01,
        },
    ]

    aggregated = qst_analysis._aggregate_shot_sweep_statistics_impl(rows)

    assert len(aggregated) == 1
    row = aggregated[0]
    assert row["state"] == "00"
    assert row["log2_shots"] == 3
    assert row["shots"] == 8
    assert row["n_total"] == 2
    assert row["n_valid_infidelity"] == 2
    assert row["infid_mean"] == pytest.approx(0.15)
    assert row["log10_infid_mean"] == pytest.approx(
        np.mean([np.log10(0.1), np.log10(0.2)])
    )


def test_validate_shot_sweep_run_records_reports_missing_bad_and_range_issues() -> None:
    rows = [
        {
            "state": "00",
            "log2_shots": 3,
            "shots": 8,
            "repeat": 1,
            "fidelity": 0.95,
            "infidelity": 0.05,
            "log10_infidelity": np.log10(0.05),
            "nll": 10.0,
            "min_eig": 0.0,
        },
        {
            "state": "00",
            "log2_shots": 3,
            "shots": 8,
            "repeat": 2,
            "fidelity": 0.95,
            "infidelity": 1.1,
            "log10_infidelity": np.log10(1.1),
            "nll": 11.0,
            "min_eig": 0.0,
        },
    ]

    checks = qst_analysis._validate_shot_sweep_run_records_impl(
        run_records=rows,
        suite_states=("00", "01"),
        shot_log2_values=(3, 4),
        repeats_per_point=2,
    )

    assert checks["expected_group_count"] == 4
    assert checks["observed_group_count"] == 1
    assert {"state": "01", "log2_shots": 3} in checks["missing_groups"]
    assert {"state": "00", "log2_shots": 4} in checks["missing_groups"]
    assert checks["bad_repeat_groups"] == []
    assert len(checks["infidelity_range_violations"]) == 1


def test_validate_shot_sweep_run_records_reports_bad_repeat_groups() -> None:
    rows = [
        {
            "state": "00",
            "log2_shots": 3,
            "shots": 8,
            "repeat": 1,
            "fidelity": 0.95,
            "infidelity": 0.05,
            "log10_infidelity": np.log10(0.05),
            "nll": 10.0,
            "min_eig": 0.0,
        }
    ]

    checks = qst_analysis._validate_shot_sweep_run_records_impl(
        run_records=rows,
        suite_states=("00",),
        shot_log2_values=(3,),
        repeats_per_point=2,
    )

    assert checks["missing_groups"] == []
    assert checks["bad_repeat_groups"] == [
        {
            "state": "00",
            "log2_shots": 3,
            "observed_repeats": 1,
            "expected_repeats": 2,
        }
    ]


def test_summarize_final_shot_sweep_selects_largest_log2() -> None:
    final_summary = qst_analysis._summarize_final_shot_sweep_impl(
        aggregated_stats=[
            {
                "state": "00",
                "log2_shots": 3,
                "shots": 8,
                "n_total": 2,
                "n_valid_infidelity": 2,
                "infid_mean": 0.1,
                "infid_ci95": 0.01,
                "log10_infid_mean": -1.0,
                "log10_infid_ci95": 0.1,
            },
            {
                "state": "00",
                "log2_shots": 4,
                "shots": 16,
                "n_total": 2,
                "n_valid_infidelity": 2,
                "infid_mean": 0.05,
                "infid_ci95": 0.005,
                "log10_infid_mean": -1.3,
                "log10_infid_ci95": 0.08,
            },
        ],
        shot_log2_values=(3, 4),
    )

    assert final_summary == [
        {
            "state": "00",
            "n_total": 2,
            "n_valid_infidelity": 2,
            "infid_mean": 0.05,
            "infid_ci95": 0.005,
            "log10_infid_mean": -1.3,
            "log10_infid_ci95": 0.08,
        }
    ]

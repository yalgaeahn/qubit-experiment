from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

pytest.importorskip("laboneq")
pytest.importorskip("laboneq_applications")
pytest.importorskip("uncertainties")

from analysis import coherence_tracking as tracking_analysis
from experiments import coherence_tracking as tracking


@dataclass
class _QubitStub:
    uid: str


def test_workflow_options_default_tracking_fields() -> None:
    options = tracking.experiment_workflow.options()

    assert hasattr(options, "run_t1")
    assert hasattr(options, "run_t2_star")
    assert hasattr(options, "run_t2")
    assert hasattr(options, "refocus_qop")
    assert hasattr(options, "history_path")
    assert hasattr(options, "transition")
    assert hasattr(options, "do_analysis")
    assert hasattr(options, "update")


def test_resolve_metric_flags_requires_enabled_branch() -> None:
    with pytest.raises(ValueError, match="At least one"):
        tracking._resolve_metric_flags_value(
            run_t1=False,
            run_t2_star=False,
            run_t2=False,
            t1_delays=None,
            t2_star_delays=None,
            t2_delays=None,
        )


def test_resolve_metric_flags_requires_delays_for_enabled_metric() -> None:
    with pytest.raises(ValueError, match="t2_delays"):
        tracking._resolve_metric_flags_value(
            run_t1=False,
            run_t2_star=False,
            run_t2=True,
            t1_delays=None,
            t2_star_delays=None,
            t2_delays=None,
        )


def test_merge_metric_outputs_preserves_all_parameter_keys() -> None:
    merged = tracking_analysis._merge_metric_outputs(
        {
            "old_parameter_values": {"q0": {"ge_T1": 10.0e-6}},
            "new_parameter_values": {"q0": {"ge_T1": 11.0e-6}},
        },
        {
            "old_parameter_values": {
                "q0": {
                    "ge_T2_star": 8.0e-6,
                    "resonance_frequency_ge": 5.0e9,
                }
            },
            "new_parameter_values": {
                "q0": {
                    "ge_T2_star": 9.0e-6,
                    "resonance_frequency_ge": 5.01e9,
                }
            },
        },
        {
            "old_parameter_values": {"q0": {"ge_T2": 15.0e-6}},
            "new_parameter_values": {"q0": {"ge_T2": 16.0e-6}},
        },
    )

    assert merged["new_parameter_values"]["q0"] == {
        "ge_T1": pytest.approx(11.0e-6),
        "ge_T2_star": pytest.approx(9.0e-6),
        "resonance_frequency_ge": pytest.approx(5.01e9),
        "ge_T2": pytest.approx(16.0e-6),
    }


def test_build_history_rows_includes_ramsey_frequency() -> None:
    metric_outputs = {
        "t1": {
            "old_parameter_values": {"q0": {"ge_T1": 10.0e-6}},
            "new_parameter_values": {"q0": {"ge_T1": 11.0e-6}},
        },
        "t2": {
            "old_parameter_values": {"q0": {"ge_T2": 15.0e-6}},
            "new_parameter_values": {"q0": {"ge_T2": 16.0e-6}},
        },
        "t2_star": {
            "old_parameter_values": {
                "q0": {
                    "ge_T2_star": 8.0e-6,
                    "resonance_frequency_ge": 5.0e9,
                }
            },
            "new_parameter_values": {
                "q0": {
                    "ge_T2_star": 9.0e-6,
                    "resonance_frequency_ge": 5.01e9,
                }
            },
        },
    }

    rows = tracking_analysis._build_history_rows(
        metric_outputs,
        timestamp_utc="2026-03-06T00:00:00Z",
    )
    t2_star_rows = [row for row in rows if row["metric"] == "t2_star"]

    assert len(rows) == 3
    assert t2_star_rows[0]["resonance_frequency_ge_hz"] == pytest.approx(5.01e9)


def test_load_history_rows_skips_malformed_lines(tmp_path: Path) -> None:
    path = tmp_path / "tracking.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"timestamp_utc": "2026-03-06T00:00:00Z", "metric": "t1"}),
                "{bad json",
                json.dumps(["not", "a", "dict"]),
            ]
        ),
        encoding="utf-8",
    )

    rows = tracking_analysis._load_history_rows(path)

    assert rows == [{"timestamp_utc": "2026-03-06T00:00:00Z", "metric": "t1"}]


def test_materialize_tracking_history_appends_rows(tmp_path: Path) -> None:
    path = tmp_path / "coherence_tracking.jsonl"
    metric_outputs = {
        "t1": {
            "old_parameter_values": {"q0": {"ge_T1": 10.0e-6}},
            "new_parameter_values": {"q0": {"ge_T1": 11.0e-6}},
        },
        "t2": tracking_analysis._empty_analysis_payload_value(),
        "t2_star": tracking_analysis._empty_analysis_payload_value(),
    }

    out = tracking_analysis.materialize_tracking_history(metric_outputs, path)

    assert out["history_path"] == str(path.resolve())
    assert len(out["history_entries"]) == 1
    assert path.exists()


def test_plot_tracking_history_saves_one_artifact_per_qubit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    saved: list[str] = []
    monkeypatch.setattr(
        tracking_analysis,
        "validate_and_convert_qubits_sweeps",
        lambda qubits: list(qubits),
    )
    monkeypatch.setattr(
        tracking_analysis.workflow,
        "save_artifact",
        lambda name, fig: saved.append(name),
    )

    figures = tracking_analysis.plot_tracking_history(
        qubits=[_QubitStub("q0")],
        history_rows=[
            {
                "timestamp_utc": "2026-03-06T00:00:00Z",
                "qubit_uid": "q0",
                "metric": "t1",
                "value_s": 10.0e-6,
                "std_dev_s": None,
                "previous_value_s": 9.0e-6,
                "history_version": 1,
            },
            {
                "timestamp_utc": "2026-03-06T00:01:00Z",
                "qubit_uid": "q0",
                "metric": "t2",
                "value_s": 20.0e-6,
                "std_dev_s": None,
                "previous_value_s": 19.0e-6,
                "history_version": 1,
            },
            {
                "timestamp_utc": "2026-03-06T00:02:00Z",
                "qubit_uid": "q0",
                "metric": "t2_star",
                "value_s": 8.0e-6,
                "std_dev_s": None,
                "previous_value_s": 7.5e-6,
                "history_version": 1,
            },
        ],
    )

    assert "q0" in figures
    assert saved == ["Coherence_tracking_q0"]


def test_build_workflow_output_keeps_analysis_none() -> None:
    out = tracking._build_workflow_output(None, None, None, None)

    assert out["analysis"] is None
    assert out["results"] == {"t1": None, "t2_star": None, "t2": None}

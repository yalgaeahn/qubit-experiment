from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

pytest.importorskip("laboneq")
pytest.importorskip("laboneq_applications")

from laboneq.simple import dsl

import analysis.plotting_helpers as ph


@dataclass
class _Qubit:
    uid: str


class _Node:
    def __init__(self, data):
        self.data = data


class _FakeResult(dict):
    pass


def _setup_validation_passthrough(monkeypatch, qubits, sweep_points):
    monkeypatch.setattr(ph, "validate_result", lambda result: None)
    monkeypatch.setattr(
        ph,
        "validate_and_convert_qubits_sweeps",
        lambda qb, sw: (qubits, sweep_points),
    )


def test_timestamped_title_format() -> None:
    fixed = datetime(2026, 3, 5, 12, 34, 56)
    assert ph.timestamped_title("My Plot", dt=fixed) == "My Plot (2026-03-05 12:34:56)"


def test_plot_raw_complex_data_1d_single_trace(monkeypatch) -> None:
    qubits = [_Qubit(uid="q0")]
    sweep_points = [np.linspace(0.0, 1.0, 11)]
    _setup_validation_passthrough(monkeypatch, qubits, sweep_points)

    saved = []
    monkeypatch.setattr(ph.workflow, "save_artifact", lambda name, fig: saved.append(name))

    result = _FakeResult()
    result[dsl.handles.result_handle("q0")] = _Node(
        np.linspace(0.0, 1.0, 11) + 1j * np.linspace(0.2, 0.4, 11)
    )

    figures = ph.plot_raw_complex_data_1d.func(
        qubits=qubits,
        result=result,
        sweep_points=sweep_points,
        xlabel="Delay (us)",
        xscaling=1e6,
        options=SimpleNamespace(
            save_figures=True,
            close_figures=False,
            use_cal_traces=False,
            cal_states="ge",
        ),
    )

    assert "q0" in figures
    fig = figures["q0"]
    assert len(fig.axes) == 2
    assert fig.axes[1].get_xlabel() == "Delay (us)"
    assert saved == ["Raw_data_q0"]
    plt.close(fig)


def test_plot_raw_complex_data_1d_suffix_traces(monkeypatch) -> None:
    qubits = [_Qubit(uid="q0")]
    sweep_points = [np.linspace(0.0, 1.0, 9)]
    _setup_validation_passthrough(monkeypatch, qubits, sweep_points)

    saved = []
    monkeypatch.setattr(ph.workflow, "save_artifact", lambda name, fig: saved.append(name))

    result = _FakeResult()
    result[dsl.handles.result_handle("q0")] = ["y180", "my180"]
    result[dsl.handles.result_handle("q0", suffix="y180")] = _Node(
        np.linspace(0.0, 1.0, 9) + 1j * np.linspace(0.1, 0.2, 9)
    )
    result[dsl.handles.result_handle("q0", suffix="my180")] = _Node(
        np.linspace(1.0, 0.0, 9) + 1j * np.linspace(0.3, 0.4, 9)
    )

    figures = ph.plot_raw_complex_data_1d.func(
        qubits=qubits,
        result=result,
        sweep_points=sweep_points,
        options=SimpleNamespace(
            save_figures=True,
            close_figures=False,
            use_cal_traces=False,
            cal_states="ge",
        ),
    )

    assert "q0_multi" in figures
    fig = figures["q0_multi"]
    assert len(fig.axes[0].lines) == 2
    assert len(fig.axes[1].lines) == 2
    assert saved == ["Raw_data_q0_y180_my180"]
    plt.close(fig)


def test_plot_raw_complex_data_1d_calibration_lines(monkeypatch) -> None:
    qubits = [_Qubit(uid="q0")]
    sweep_points = [np.linspace(0.0, 1.0, 7)]
    _setup_validation_passthrough(monkeypatch, qubits, sweep_points)

    monkeypatch.setattr(ph.workflow, "save_artifact", lambda *_: None)

    result = _FakeResult()
    result[dsl.handles.result_handle("q0")] = _Node(
        np.linspace(0.0, 1.0, 7) + 1j * np.linspace(0.1, 0.2, 7)
    )
    result[dsl.handles.calibration_trace_handle("q0", "g")] = _Node(
        np.array([0.2 + 0.3j, 0.2 + 0.3j])
    )
    result[dsl.handles.calibration_trace_handle("q0", "e")] = _Node(
        np.array([0.7 + 0.8j, 0.7 + 0.8j])
    )

    figures = ph.plot_raw_complex_data_1d.func(
        qubits=qubits,
        result=result,
        sweep_points=sweep_points,
        options=SimpleNamespace(
            save_figures=False,
            close_figures=False,
            use_cal_traces=True,
            cal_states="ge",
        ),
    )

    fig = figures["q0"]
    line_styles_i = [line.get_linestyle() for line in fig.axes[0].lines]
    line_styles_q = [line.get_linestyle() for line in fig.axes[1].lines]
    assert "--" in line_styles_i
    assert "--" in line_styles_q
    plt.close(fig)

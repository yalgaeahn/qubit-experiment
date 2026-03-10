from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("laboneq")
pytest.importorskip("laboneq_applications")
pytest.importorskip("uncertainties")

from qubit_experiment.experiments import ramsey


def test_workflow_options_default_echo_false() -> None:
    options = ramsey.experiment_workflow.options()
    assert hasattr(options, "echo")
    assert hasattr(options, "refocus_qop")
    assert hasattr(options, "count")
    assert hasattr(options, "use_cal_traces")

def test_phase_values_ramsey_vs_echo_modes() -> None:
    delays = np.array([0.0, 1.0e-6, 2.0e-6, 3.0e-6], dtype=float)
    detuning = 1.0e6

    ramsey_phase = ramsey._compute_phase_values_for_mode(
        delays,
        detuning,
        echo=False,
    )
    echo_phase = ramsey._compute_phase_values_for_mode(
        delays,
        detuning,
        echo=True,
    )

    expected = ((delays - delays[0]) * detuning * 2 * np.pi) % (2 * np.pi)
    assert np.allclose(ramsey_phase, expected)
    assert np.allclose(echo_phase, 0.0)


def test_sequence_kwargs_include_echo_pulse_only_in_echo_mode() -> None:
    ramsey_kwargs = ramsey._ramsey_qop_kwargs(
        transition="ge",
        echo=False,
        refocus_qop="y180",
    )
    echo_kwargs = ramsey._ramsey_qop_kwargs(
        transition="ge",
        echo=True,
        refocus_qop="y180",
    )

    assert ramsey_kwargs == {"transition": "ge"}
    assert echo_kwargs == {"transition": "ge", "echo_pulse": "y180"}


def test_analysis_routing_switches_with_echo_mode() -> None:
    assert (
        ramsey._analysis_workflow_for_mode(echo=False)
        is ramsey.ramsey_analysis_workflow
    )
    assert (
        ramsey._analysis_workflow_for_mode(echo=True)
        is ramsey.echo_analysis_workflow
    )


def test_echo_detuning_warning_is_emitted_once_for_nonzero_detuning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, str, tuple[object, ...]]] = []

    def _fake_log(level: int, msg: str, *args: object) -> None:
        calls.append((level, msg, args))

    monkeypatch.setattr(ramsey.workflow, "log", _fake_log)

    ramsey._maybe_log_ignored_detunings(echo=True, detunings=[0.3e6, 0.0])
    ramsey._maybe_log_ignored_detunings(echo=True, detunings=[0.0, 0.0])
    ramsey._maybe_log_ignored_detunings(echo=False, detunings=[0.3e6, 0.0])

    assert len(calls) == 1
    _, message, args = calls[0]
    assert "ignores detunings" in message
    assert args == ([0.3e6, 0.0],)

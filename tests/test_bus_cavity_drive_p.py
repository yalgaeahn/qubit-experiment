from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("laboneq")

from laboneq.dsl.calibration import Calibration, SignalCalibration

from qubit_experiment.qpu_types.bus_cavity import BusCavity, BusCavityParameters
from qubit_experiment.qpu_types.bus_cavity import operations as bus_ops_module
from qubit_experiment.qpu_types.bus_cavity.operations import BusCavityOperations


class _FakeLogicalSignal:
    def __init__(self, path: str):
        self.path = path


class _FakeLogicalSignalGroup:
    def __init__(self, **signals: str):
        self.logical_signals = {
            name: _FakeLogicalSignal(path) for name, path in signals.items()
        }


def _bus_with_drive_p() -> BusCavity:
    return BusCavity(
        uid="b0",
        signals={
            "drive": "b0/drive",
            "drive_p": "b0/drive_p",
        },
        parameters=BusCavityParameters(
            rip_amplitude=0.09,
            rip_length=4.8e-7,
            rip_detuning=-11e6,
            drive_lo_frequency=5.2e9,
            drive_range=10,
            resonance_frequency_bus=5.5055e9,
            rip_p_amplitude=0.05,
            rip_p_length=8.0e-7,
            rip_p_pulse={"function": "NestedCosine", "tau_rel": 0.7},
            rip_p_detuning=-20e6,
            drive_p_lo_frequency=5.8e9,
            drive_p_range=5,
            resonance_frequency_bus_p=6.1e9,
        ),
    )


def test_bus_cavity_from_device_setup_preserves_primary_drive_only() -> None:
    device_setup = SimpleNamespace(
        logical_signal_groups={
            "b0": _FakeLogicalSignalGroup(
                drive="/logical_signal_groups/b0/drive",
            )
        }
    )

    [bus] = BusCavity.from_device_setup(device_setup, qubit_uids=["b0"])

    assert bus.signals == {"drive": "b0/drive"}


def test_bus_cavity_from_device_setup_accepts_optional_drive_p() -> None:
    device_setup = SimpleNamespace(
        logical_signal_groups={
            "b0": _FakeLogicalSignalGroup(
                drive="/logical_signal_groups/b0/drive",
                drive_p="/logical_signal_groups/b0/drive_p",
            )
        }
    )

    [bus] = BusCavity.from_device_setup(device_setup, qubit_uids=["b0"])

    assert bus.signals == {
        "drive": "b0/drive",
        "drive_p": "b0/drive_p",
    }


def test_bus_cavity_calibration_builds_independent_drive_and_drive_p() -> None:
    bus = _bus_with_drive_p()

    calibration = bus.calibration()
    primary = calibration["b0/drive"]
    secondary = calibration["b0/drive_p"]

    assert primary.oscillator.frequency == pytest.approx(294.5e6)
    assert primary.local_oscillator.frequency == pytest.approx(5.2e9)
    assert primary.range == pytest.approx(10)
    assert primary.automute is True

    assert secondary.oscillator.frequency == pytest.approx(280e6)
    assert secondary.local_oscillator.frequency == pytest.approx(5.8e9)
    assert secondary.range == pytest.approx(5)
    assert secondary.automute is True


def test_rip_defaults_to_primary_drive(monkeypatch: pytest.MonkeyPatch) -> None:
    bus = _bus_with_drive_p()
    operations = BusCavityOperations()
    plays: list[tuple[str, dict]] = []

    monkeypatch.setattr(
        bus_ops_module.dsl,
        "play",
        lambda signal, **kwargs: plays.append((signal, kwargs)),
    )

    with bus_ops_module.dsl.section(uid="root"):
        operations.rip(bus)

    assert plays[0][0] == "b0/drive"
    assert plays[0][1]["amplitude"] == pytest.approx(0.09)
    assert plays[0][1]["length"] == pytest.approx(4.8e-7)


def test_rip_can_target_drive_p(monkeypatch: pytest.MonkeyPatch) -> None:
    bus = _bus_with_drive_p()
    operations = BusCavityOperations()
    plays: list[tuple[str, dict]] = []

    monkeypatch.setattr(
        bus_ops_module.dsl,
        "play",
        lambda signal, **kwargs: plays.append((signal, kwargs)),
    )

    with bus_ops_module.dsl.section(uid="root"):
        operations.rip(bus, line="drive_p")

    assert plays[0][0] == "b0/drive_p"
    assert plays[0][1]["amplitude"] == pytest.approx(0.05)
    assert plays[0][1]["length"] == pytest.approx(8.0e-7)
    assert plays[0][1]["pulse"].function == "NestedCosine"
    assert plays[0][1]["pulse"].pulse_parameters == {"tau_rel": 0.7}


def test_bus_delay_can_target_drive_p(monkeypatch: pytest.MonkeyPatch) -> None:
    bus = _bus_with_drive_p()
    operations = BusCavityOperations()
    delays: list[tuple[str, float]] = []

    monkeypatch.setattr(
        bus_ops_module.dsl,
        "delay",
        lambda signal, *, time: delays.append((signal, time)),
    )

    with bus_ops_module.dsl.section(uid="root"):
        operations.bus_delay(bus, 120e-9, line="drive_p")

    assert delays == [("b0/drive_p", 120e-9)]


def test_set_bus_frequency_allows_one_call_per_line() -> None:
    bus = _bus_with_drive_p()
    operations = BusCavityOperations()
    calibration = Calibration(
        {
            "b0/drive": SignalCalibration(),
            "b0/drive_p": SignalCalibration(),
        }
    )

    with bus_ops_module.dsl.section(uid="root"):
        operations.set_bus_frequency(bus, 5.49e9, calibration=calibration, line="drive")
        operations.set_bus_frequency(
            bus,
            6.03e9,
            calibration=calibration,
            line="drive_p",
        )

    assert calibration["b0/drive"].oscillator.frequency == pytest.approx(290e6)
    assert calibration["b0/drive_p"].oscillator.frequency == pytest.approx(230e6)

    with pytest.raises(RuntimeError, match="multiple times"):
        with bus_ops_module.dsl.section(uid="root_repeat"):
            operations.set_bus_frequency(
                bus,
                5.48e9,
                calibration=calibration,
                line="drive",
            )


def test_invalid_drive_line_raises_value_error() -> None:
    bus = _bus_with_drive_p()

    with pytest.raises(ValueError, match="not one of"):
        bus.rip_parameters(line="not_a_line")

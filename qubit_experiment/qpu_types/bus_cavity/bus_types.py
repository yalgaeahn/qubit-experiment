# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Tunable coupler parameters and elements."""

from __future__ import annotations

from typing import ClassVar, Literal

import attrs
from laboneq.core.utilities.dsl_dataclass_decorator import classformatter
from laboneq.dsl.calibration import Calibration, Oscillator, SignalCalibration
from laboneq.dsl.enums import ModulationType
from laboneq.dsl.quantum import (
    QuantumElement,
    QuantumParameters,
)


@classformatter
@attrs.define(kw_only=True)
class BusCavityParameters(QuantumParameters):
    """Tunable coupler parameters.

    Attributes:
        gate_parameters:
            Dictionary of parameters specific to each implemented gate. The dictionary
            keys are the names of the gates, and the dictionary values are the
            corresponding parameter dictionaries. Typical parameters include: "pulse",
            "amplitude", "length", and "frequency". By default, only the iSWAP gate
            parameters are initialized.
        dc_slot:
            Slot number on the DC source used for applying a DC voltage to the coupler.
        flux_offset_voltage:
            Offset voltage for flux control line - defaults to 0 volts.
        dc_voltage_parking:
            Coupler DC parking voltage.
    """

    # RIP parameters
    rip_amplitude: float = 1.0
    rip_length: float = 1e-6
    rip_phase: float = 1.5707963267948966
    rip_pulse: dict = attrs.field(
        factory=lambda: {"function": "NestedCosine"}
    )

    rip_detuning: float | None = None

    
    rip_p_amplitude: float = 1.0
    rip_p_length: float = 1e-6
    rip_p_phase: float = 1.5707963267948966
    rip_p_pulse: dict = attrs.field(
        factory=lambda: {"function": "NestedCosine"}
    )
    rip_p_detuning: float | None = None

    drive_lo_frequency: float | None = None
    drive_range: float = 10
    drive_p_lo_frequency: float | None = None
    drive_p_range: float = 10

    # spectroscopy parameters
    spectroscopy_length: float | None = 100e-6
    spectroscopy_amplitude: float | None = 1
    spectroscopy_pulse: dict = attrs.field(
        factory=lambda: {
            "function": "const",
            "can_compress": True,
        },
    )
    # spectroscopy_pulse: dict = attrs.field(
    #     factory=lambda: {
    #         "function": "GaussianSquare",
    #         "sigma":0.2,
    #         "risefall_sigma_ratio":0.5,
    #         "can_compress": True
    #     },
    # )

    # characterization parameter

    resonance_frequency_bus: float | None = None
    resonance_frequency_bus_p: float | None = None
    kappa: float | None = None

    @staticmethod
    def _drive_frequency(
        lo_frequency: float | None,
        resonance_frequency: float | None,
        detuning: float | None,
    ) -> float | None:
        if lo_frequency is None or resonance_frequency is None or detuning is None:
            return None
        return resonance_frequency + detuning - lo_frequency

    @property
    def drive_frequency_bus(self) -> float | None:
        return self._drive_frequency(
            self.drive_lo_frequency,
            self.resonance_frequency_bus,
            self.rip_detuning,
        )

    @property
    def drive_frequency_bus_p(self) -> float | None:
        return self._drive_frequency(
            self.drive_p_lo_frequency,
            self.resonance_frequency_bus_p,
            self.rip_p_detuning,
        )



@classformatter
@attrs.define
class BusCavity(QuantumElement):
    """Tunable coupler."""

    PARAMETERS_TYPE = BusCavityParameters
    REQUIRED_SIGNALS = ("drive",)
    OPTIONAL_SIGNALS = ("drive_p",)
    DRIVE_LINES: ClassVar[tuple[Literal["drive", "drive_p"], ...]] = (
        "drive",
        "drive_p",
    )
    _LINE_PARAMETER_FIELDS: ClassVar[
        dict[Literal["drive", "drive_p"], dict[str, str]]
    ] = {
        "drive": {
            "amplitude": "rip_amplitude",
            "length": "rip_length",
            "pulse": "rip_pulse",
            "phase": "rip_phase",
            "detuning": "rip_detuning",
            "lo_frequency": "drive_lo_frequency",
            "range": "drive_range",
            "resonance_frequency": "resonance_frequency_bus",
            "frequency": "drive_frequency_bus",
        },
        "drive_p": {
            "amplitude": "rip_p_amplitude",
            "length": "rip_p_length",
            "pulse": "rip_p_pulse",
            "phase": "rip_p_phase",
            "detuning": "rip_p_detuning",
            "lo_frequency": "drive_p_lo_frequency",
            "range": "drive_p_range",
            "resonance_frequency": "resonance_frequency_bus_p",
            "frequency": "drive_frequency_bus_p",
        },
    }

    #SIGNAL_ALIASES: ClassVar = {"drive_bus": "drive"}

    def _validate_drive_line(
        self,
        line: Literal["drive", "drive_p"],
        *,
        require_signal: bool = False,
    ) -> None:
        if line not in self.DRIVE_LINES:
            raise ValueError(
                f"Bus drive line {line!r} is not one of {self.DRIVE_LINES!r}."
            )
        if require_signal and line not in self.signals:
            raise ValueError(
                f"Bus {self.uid!r} does not define the {line!r} logical signal."
            )

    def _line_parameter(
        self,
        line: Literal["drive", "drive_p"],
        parameter: str,
    ):
        self._validate_drive_line(line)
        return getattr(
            self.parameters,
            self._LINE_PARAMETER_FIELDS[line][parameter],
        )

    def rip_parameters(
        self,
        line: Literal["drive", "drive_p"] = "drive",
    ) -> tuple[str, dict]:
        self._validate_drive_line(line, require_signal=True)
        param_keys = ["amplitude", "length", "pulse"]
        params = {k: self._line_parameter(line, k) for k in param_keys}
        return line, params

    def spectroscopy_parameters(self) -> tuple[str, dict]:
        """Return the qubit-spectroscopy line and the spectroscopy-pulse parameters.

        Returns:
           line:
               The qubit-spectroscopy drive line of the qubit.
           params:
               The spectroscopy-pulse parameters.
        """
        param_keys = ["amplitude", "length", "pulse"]
        params = {k: getattr(self.parameters, f"spectroscopy_{k}") for k in param_keys}
        return "drive", params

    def calibration(self) -> Calibration:
        """Returns the calibration for the tunable coupler.

        Returns:
            Prefilled calibration object from coupler parameters.
        """
        calibration_items = {}
        for line in self.DRIVE_LINES:
            if line not in self.signals:
                continue

            sig_cal = SignalCalibration()
            drive_frequency = self._line_parameter(line, "frequency")
            if drive_frequency is not None:
                sig_cal.oscillator = Oscillator(
                    uid=f"{self.uid}_{line}_osc",
                    frequency=drive_frequency,
                    modulation_type=ModulationType.AUTO,
                )
                lo_frequency = self._line_parameter(line, "lo_frequency")
                if lo_frequency is not None:
                    sig_cal.local_oscillator = Oscillator(
                        uid=f"{self.uid}_{line}_local_osc",
                        frequency=lo_frequency,
                    )
                sig_cal.range = self._line_parameter(line, "range")
                sig_cal.automute = True
            calibration_items[self.signals[line]] = sig_cal

        return Calibration(calibration_items)

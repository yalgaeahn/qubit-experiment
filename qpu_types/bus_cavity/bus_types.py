# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Tunable coupler parameters and elements."""

from __future__ import annotations

from typing import ClassVar

import attrs
from laboneq.core.utilities.dsl_dataclass_decorator import classformatter
from laboneq.dsl.calibration import (
    Calibration,
    SignalCalibration,
    Oscillator,
)
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
    rip_pulse: dict = attrs.field(
        factory=lambda: {"function": "NestedCosine"}
    )

    rip_detuning: float | None = None


    drive_lo_frequency : float | None = None
    drive_range: float = 10

    #spectroscopy parameters
    spectroscopy_length: float | None = 5e-6
    spectroscopy_amplitude: float | None = 1
    spectroscopy_pulse: dict = attrs.field(
        factory=lambda: {
            "function": "GaussianSquare",
            "sigma":0.2,
            "risefall_sigma_ratio":1.0, 
            "can_compress": True
        },
    )
    
    
    
    # characterization parameter
    
    resonance_frequency_bus: float | None = None
    kappa : float | None = None


    @property
    def drive_frequency_bus(self) -> float | None:
        if self.drive_lo_frequency is None or self.resonance_frequency_bus is None or self.rip_detuning is None:
            return None
        return self.resonance_frequency_bus + self.rip_detuning - self.drive_lo_frequency



@classformatter
@attrs.define
class BusCavity(QuantumElement):
    """Tunable coupler."""

    PARAMETERS_TYPE = BusCavityParameters
    REQUIRED_SIGNALS = ("drive",)

    #SIGNAL_ALIASES: ClassVar = {"drive_bus": "drive"}

    def rip_parameters(self) -> tuple[str, dict]:
        param_keys = ["amplitude", "length", "pulse"]
        params = {k: getattr(self.parameters, f"rip_{k}") for k in param_keys}
        return "drive", params
    
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
        drive_lo = None

        if self.parameters.drive_lo_frequency is not None:
            drive_lo = Oscillator(
                uid=f"{self.uid}_drive_local_osc",
                frequency=self.parameters.drive_lo_frequency,
            )

        calibration_items = {}
        if "drive" in self.signals:
            sig_cal = SignalCalibration()
            if self.parameters.drive_frequency_bus is not None:
                sig_cal.oscillator = Oscillator(
                    uid=f"{self.uid}_drive_osc",
                    frequency=self.parameters.drive_frequency_bus,
                    modulation_type=ModulationType.AUTO,
                )
                sig_cal.local_oscillator = drive_lo
                sig_cal.range = self.parameters.drive_range
                sig_cal.automute = True
                calibration_items[self.signals["drive"]] = sig_cal
            calibration_items[self.signals["drive"]] = sig_cal

        return Calibration(calibration_items)

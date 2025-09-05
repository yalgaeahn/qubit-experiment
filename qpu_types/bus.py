from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import attrs
from laboneq.core.utilities.dsl_dataclass_decorator import classformatter
from laboneq.dsl.calibration import Calibration, Oscillator, SignalCalibration
from laboneq.dsl.enums import ModulationType
from laboneq.dsl.quantum import (
    QuantumElement,
    QuantumParameters,
)
from laboneq.simple import dsl

if TYPE_CHECKING:
    from laboneq.dsl.experiment.pulse import Pulse

# TODO: Add support for specifying integration kernels as a list of sample
#       values.

@classformatter
@attrs.define()
class BusParameters(QuantumParameters):
    

    resonance_frequency : float
 
    # free-form dictionary of user-defined parameters
    user_defined: dict = attrs.field(factory=dict)
    
    ########################################################################################
    @property
    def drive_frequency_cr(self) -> float | None: 
        """should be (targ) resonance_frequency_ge - (ctrl) drive lo frequency"""
        if self.resonance_targ_frequency is None or self.drive_lo_frequency is None:
            return None
        return self.resonance_targ_frequency - self.drive_lo_frequency
    
    @property
    def drive_frequency_cr_cancel(self) -> float | None:
        if self.drive_lo_frequency is None or self.resonance_frequency_ge is None:
            return None
        return self.resonance_frequency_ge - self.drive_lo_frequency
    #########################################################################################
    @property
    def drive_frequency_ge(self) -> float | None:
        """Qubit drive frequency for the g-e transition."""
        if self.drive_lo_frequency is None or self.resonance_frequency_ge is None:
            return None
        return self.resonance_frequency_ge - self.drive_lo_frequency

    @property
    def drive_frequency_ef(self) -> float | None:
        """Qubit drive frequency for the e-f transition."""
        if self.drive_lo_frequency is None or self.resonance_frequency_ef is None:
            return None
        return self.resonance_frequency_ef - self.drive_lo_frequency

    @property
    def readout_frequency(self) -> float | None:
        """Readout baseband frequency."""
        if (
            self.readout_lo_frequency is None
            or self.readout_resonator_frequency is None
        ):
            return None
        return self.readout_resonator_frequency - self.readout_lo_frequency


@classformatter
@attrs.define()
class TransmonQubit(QuantumElement):
    """A class for a superconducting, fixed frequency Transmon Qubit."""

    PARAMETERS_TYPE = TransmonQubitParameters
    REQUIRED_SIGNALS = (
        "acquire",
        "drive",
        "measure",
    )
    OPTIONAL_SIGNALS = (
        "drive_ef",
        "drive_cr"
    )

    TRANSITIONS = ("ge", "ef")
# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Tunable transmon operations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, ClassVar, TYPE_CHECKING

import numpy as np
from laboneq.dsl.calibration import Calibration, Oscillator
from laboneq.dsl.enums import ModulationType
from laboneq.dsl.parameter import SweepParameter
from laboneq.simple import SectionAlignment, dsl

from laboneq_applications.typing import QuantumElements

# from .qubit_types import FixedTransmonQubit
from .bus_types import BusCavity

import custom_pulse_library
from laboneq.dsl.experiment import builtins

# TODO: Implement multistate 0-1-2 measurement operation

# TODO: Add rotate_xy gate that performs a rotation about an axis in the xy-plane.


class BusCavityOperations(dsl.QuantumOperations):
    """Operations for FixedTransmonQubits."""

    QUBIT_TYPES = BusCavity

    # common angles used by rx, ry and rz.
    _PI = np.pi
    _PI_BY_2 = np.pi / 2

    @dsl.quantum_operation
    def bus_spectroscopy_drive(
        self,
        b: BusCavity,
        amplitude: float | SweepParameter | None = None,
        phase: float = 0.0,
        length: float | SweepParameter | None = None,
        pulse: dict | None = None,
    ) -> None:
        """Long pulse used for qubit spectroscopy that emulates a coherent field.

        Arguments:
            q:
                The qubit to apply the spectroscopy drive.
            amplitude:
                The amplitude of the pulse. By default, the
                qubit parameter "spectroscopy_amplitude".
            phase:
                The phase of the pulse in radians. By default,
                this is 0.0.
            length:
                The duration of the rotation pulse. By default, this
                is determined by the qubit parameters.
            pulse:
                A dictionary of overrides for the qubit-spectroscopy pulse parameters.

                The dictionary may contain sweep parameters for the pulse
                parameters other than `function`.

                If the `function` parameter is different to the one
                specified for the qubit, then this override dictionary
                completely replaces the existing pulse parameters.

                Otherwise, the values override or extend the existing ones.
        """
        spec_line, params = b.spectroscopy_parameters()
        if amplitude is None:
            amplitude = params["amplitude"]
        if length is None:
            length = params["length"]

        spectroscopy_pulse = dsl.create_pulse(
            params["pulse"], pulse, name="bus_spectroscopy_pulse"
        )

        dsl.play(
            b.signals[spec_line],
            amplitude=amplitude,
            phase=phase,
            length=length,
            pulse=spectroscopy_pulse,
        )

    @dsl.quantum_operation
    def rip(self, 
            b: BusCavity,
            amplitude: float | None = None,
            phase: float = _PI_BY_2,
            increment_oscillator_phase: float | SweepParameter | None = None,
            length: float | None = None,
            pulse: dict | None = None,
        ) -> None:

        rip_line, params = b.rip_parameters()
        rip_amplitude = params["amplitude"] if amplitude is None else amplitude
        rip_length = params["length"] if length is None else length

        rip_pulse = dsl.create_pulse(
            params["pulse"], pulse, name="rip_pulse"
        )

        dsl.play(
            b.signals[rip_line],
            amplitude=rip_amplitude,
            phase=phase,
            length=rip_length,
            pulse=rip_pulse,
            increment_oscillator_phase=increment_oscillator_phase
        )

    @dsl.quantum_operation
    def bus_barrier(self, b: BusCavity) -> None:
        """Add a barrier on all the qubit signals.

        Arguments:
            q:
                The qubit to block on.

        Note:
            A barrier returns an empty section that
            reserves all the qubit signals. The
            signals are reserved via `@dsl.quantum_operation` so
            the implementation of this operations is just
            `pass`.
        """

    @dsl.quantum_operation
    def bus_delay(self, b: BusCavity, time: float) -> None:
        """Add a delay on the qubit drive signal.

        Arguments:
            q:
                The qubit to delay on.
            time:
                The duration of the delay in seconds.
        """
        # Delaying on a single line is sufficient since the operation
        # section automatically reserves all lines.
        signal_line, _ = b.rip_parameters()
        dsl.delay(b.signals[signal_line], time=time)

    @dsl.quantum_operation
    def set_bus_frequency(
        self,
        b: BusCavity,
        frequency: float | SweepParameter,
        *,
        readout: bool = False,
        rf: bool = True,
        calibration: Calibration | None = None,
    ) -> None:
        """Sets the frequency of the given qubit drive line or readout line.

        Arguments:
            q:
                The qubit to set the transition or readout frequency of.
            frequency:
                The frequency to set in Hz.
                By default the frequency specified is the RF frequency.
                The oscillator frequency may be set directly instead
                by passing `rf=False`.
            transition:
                The transition to rotate. By default this is "ge"
                (i.e. the 0-1 transition).
            readout:
                If true, the frequency of the readout line is set
                instead. Setting the readout frequency to a sweep parameter
                is only supported in spectroscopy mode. The LabOne Q compiler
                will raise an error in other modes.
            rf:
                If True, set the RF frequency of the transition.
                If False, set the oscillator frequency directly instead.
                The default is to set the RF frequency.
            calibration:
                The experiment calibration to update (see the note below).
                By default, the calibration from the currently active
                experiment context is used. If no experiment context is
                active, for example when using
                `@qubit_experiment(context=False)`, the calibration
                object may be passed explicitly.

        Raises:
            RuntimeError:
                If there is an attempt to call `set_frequency` more than
                once on the same signal. See notes below for details.

        Notes:
            Currently `set_frequency` is implemented by setting the
            appropriate oscillator frequencies in the experiment calibration.
            This has two important consequences:

            * Each experiment may only set one frequency per signal line,
              although this may be a parameter sweep.

            * The set frequency or sweep applies for the whole experiment
              regardless of where in the experiment the frequency is set.

            This will be improved in a future release.
        """
   
        signal_line, _ = b.rip_parameters()
        lo_frequency = b.parameters.drive_lo_frequency

        if rf:
            # This subtraction works for both numbers and SweepParameters
            frequency -= lo_frequency

        if calibration is None:
            calibration = dsl.experiment_calibration()
        signal_calibration = calibration[b.signals[signal_line]]
        oscillator = signal_calibration.oscillator

        if oscillator is None:
            oscillator = signal_calibration.oscillator = Oscillator(frequency=frequency)
        if getattr(oscillator, "_set_frequency", False):
            # We mark the oscillator with a _set_frequency attribute to ensure that
            # set_frequency isn't performed on the same oscillator twice. Ideally
            # LabOne Q would provide a set_frequency DSL method that removes the
            # need for setting the frequency on the experiment calibration.
            raise RuntimeError(
                f"Frequency of qubit {b.uid} {signal_line} line was set multiple times"
                f" using the set_frequency operation.",
            )

        oscillator._set_frequency = True
        oscillator.frequency = frequency
        if readout:
            # LabOne Q does not support software modulation of measurement
            # signal sweeps because it results in multiple readout waveforms
            # on the same readout signal. Ideally the LabOne Q compiler would
            # sort this out for us when the modulation type is AUTO, but currently
            # it does not.
            oscillator.modulation_type = ModulationType.HARDWARE



##################################################################################
###create_pulse + _PulseCache 기존 dsl.QuantumOperations 에 있는 함수 오버라이딩 ###
##################################################################################
class _PulseCache:
    """A cache for pulses to ensure that each unique pulse is only created once."""

    GLOBAL_CACHE: ClassVar[dict[tuple, Pulse]] = {}

    def __init__(self, cache: dict | None = None):
        if cache is None:
            cache = {}
        self.cache = cache

    @classmethod
    def experiment_or_global_cache(cls) -> _PulseCache:
        """Return a pulse cache.

        If there is an active experiment context, return its cache. Otherwise
        return the global pulse cache.
        """
        context = builtins.current_experiment_context()
        if context is None:
            return cls(cls.GLOBAL_CACHE)
        if not hasattr(context, "_pulse_cache"):
            context._pulse_cache = cls()
        return context._pulse_cache

    @classmethod
    def reset_global_cache(cls) -> None:
        cls.GLOBAL_CACHE.clear()

    def _parameter_value_key(self, key: str, value: object) -> object:
        if isinstance(value, Parameter):
            return (value.uid, tuple(value.values))
        if isinstance(value, list):
            if all(isinstance(x, Number) for x in value):
                return tuple(value)
            raise ValueError(
                f"Pulse parameter {key!r} is a list of values that are not all numbers."
                " It cannot be cached by create_pulse(...)."
            )
        if isinstance(value, np.ndarray):
            if np.issubdtype(value.dtype, np.number) and len(value.shape) == 1:
                return tuple(value)
            raise ValueError(
                f"Pulse parameter {key!r} is a numpy array whose values are not all"
                " numbers or whose dimension is not one. It cannot be cached by"
                " create_pulse(...)."
            )
        return value

    def _key(self, name: str, function: str, parameters: dict) -> tuple:
        parameters = {k: self._parameter_value_key(k, v) for k, v in parameters.items()}
        return (name, function, tuple(sorted(parameters.items())))

    def get(self, name: str, function: str, parameters: dict) -> Pulse | None:
        """Return the cache pulse or `None`."""
        key = self._key(name, function, parameters)
        return self.cache.get(key, None)

    def store(self, pulse: Pulse, name: str, function: str, parameters: dict) -> None:
        """Store the given pulse in the cache."""
        key = self._key(name, function, parameters)
        self.cache[key] = pulse

def create_pulse(
    parameters: dict,
    overrides: dict | None = None,
    name: str | None = None,
) -> Pulse:
    """Create a pulse from the given parameters and parameter overrides.
    
    Note) JSAHN 2025-08-04 
        Originaly function is from dsl.quantum, I had to redefine function and override 
        original in order to use pulse from custom pulse library.
         
     
    The parameters are dictionary that contains:

      - a key `"function"` that specifies which function from the LabOne Q
        `pulse_library` to use to construct the pulse. The function may
        either be the name of a registered pulse functional or
        `"sampled_pulse"` which uses `pulse_library.sampled_pulse`.
      - any other parameters required by the given pulse function.

    Arguments:
        parameters:
            A dictionary of pulse parameters. If `None`, then the overrides
            must completely define the pulse.
        overrides:
            A dictionary of overrides for the pulse parameters.
            If the overrides changes the pulse function, then the
            overrides completely replace the existing pulse parameters.
            Otherwise they extend or override them.
            The dictionary of overrides may contain sweep parameters.
        name:
            The name of the pulse. This is used as a prefix to generate the
            pulse `uid`.

    Returns:
        pulse:
            The pulse described by the parameters.
    """
    if overrides is None:
        overrides = {}
    if "function" in overrides and overrides["function"] != parameters["function"]:
        parameters = overrides.copy()
    else:
        parameters = {**parameters, **overrides}

    function = parameters.pop("function")

    if function == "sampled_pulse":
        # special case the sampled_pulse function that is not registered as a
        # pulse functional:
        pulse_function = custom_pulse_library.sampled_pulse
    else:
        try:
            pulse_function = custom_pulse_library.pulse_factory(function)
        except KeyError as err:
            pulse_function = custom_pulse_library.pulse_factory(function)
            #raise ValueError(f"Unsupported pulse function {function!r}.") from err

    if name is None:
        name = "unnamed"

    pulse_cache = _PulseCache.experiment_or_global_cache()
    pulse = pulse_cache.get(name, function, parameters)
    if pulse is None:
        pulse = pulse_function(uid=builtins.uid(name), **parameters)
        pulse_cache.store(pulse, name, function, parameters)

    return pulse


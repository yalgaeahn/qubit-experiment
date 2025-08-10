from __future__ import annotations

from typing import Literal, TypeVar

import attrs
from laboneq.simple import AcquisitionType, AveragingMode, RepetitionMode
from laboneq.workflow import (
    option_field,
    task_options,
    workflow_options,
)

T = TypeVar("T")


def _parse_acquisition_type(v: str | AcquisitionType) -> AcquisitionType:
    return AcquisitionType(v)


def _parse_averaging_mode(v: str | AveragingMode) -> AveragingMode:
    return AveragingMode(v)


def _parse_repetition_mode(v: str | RepetitionMode) -> RepetitionMode:
    return RepetitionMode(v)


@task_options
class BaseExperimentOptions:
    """Base options for the experiment.

    Attributes:
        count:
            The number of repetitions.
            Default: A common choice in practice, 1024.
        averaging_mode:
            Averaging mode to use for the experiment.
            Default: `AveragingMode.CYCLIC`.
        acquisition_type:
            Acquisition type to use for the experiment.
            Default: `AcquisitionType.INTEGRATION`.
        repetition_mode:
            The repetition mode to use for the experiment.
            Default: `RepetitionMode.FASTEST`.
        repetition_time:
            The repetition time.
            Default: None.
        reset_oscillator_phase:
            Whether to reset the oscillator phase.
            Default: False.
        active_reset (bool):
            Whether to use active reset.
            Default: False.
        active_reset_repetitions (int):
            The number of times to repeat the active resets.
            Default: 1
        active_reset_states (str | tuple | None):
            The qubit states to actively reset.
            Default: "ge"
    """

    count: int = option_field(default=1024, description="The number of repetitions.")
    acquisition_type: str | AcquisitionType = option_field(
        AcquisitionType.INTEGRATION,
        description="Acquisition type to use for the experiment.",
        converter=_parse_acquisition_type,
    )
    averaging_mode: str | AveragingMode = option_field(
        AveragingMode.CYCLIC,
        description="Averaging mode to use for the experiment.",
        converter=_parse_averaging_mode,
    )
    repetition_mode: str | RepetitionMode = option_field(
        RepetitionMode.FASTEST,
        description="The repetition mode to use for the experiment.",
        converter=_parse_repetition_mode,
    )
    repetition_time: float | None = option_field(
        None, description="The repetition time."
    )
    reset_oscillator_phase: bool = option_field(
        False, description="Whether to reset the oscillator phase."
    )
    active_reset: bool = option_field(False, description="Whether to use active reset.")
    active_reset_repetitions: int = option_field(
        1, description="The number of times to repeat the active resets."
    )
    active_reset_states: str | tuple | None = option_field(
        "ge", description="The qubit states to actively reset."
    )


@task_options(base_class=BaseExperimentOptions)
class DirectCRHamiltonianTomographyOptions:
    """Base options for direct cr hamiltonian tomography experiment"""
    # test: float = option_field(default=0.0)
    # amplitudes :
    # lengths :

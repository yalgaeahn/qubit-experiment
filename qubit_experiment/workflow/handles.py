"""Acquisition handle helpers used by qubit_experiment workflows."""

from __future__ import annotations

CALIBRATION_TRACE_2Q_PREFIX = "cal_trace_2q"


def calibration_trace_2q_handle(
    qubit_name: str,
    prepared_label: str | None = None,
    prefix: str = CALIBRATION_TRACE_2Q_PREFIX,
) -> str:
    """Return the acquisition handle for a 2Q multiplexed calibration trace."""

    return (
        f"{qubit_name}/{prefix}/{prepared_label}"
        if prepared_label is not None
        else f"{qubit_name}/{prefix}"
    )

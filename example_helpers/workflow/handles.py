# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Module that defines acquisition handle formatters for LabOneQ experiments."""

from __future__ import annotations

RESULT_PREFIX = "result"
CALIBRATION_TRACE_PREFIX = "cal_trace"
CALIBRATION_TRACE_2Q_PREFIX = "cal_trace_2q"
ACTIVE_RESET_PREFIX = "active_reset"


def result_handle(
    qubit_name: str, prefix: str = RESULT_PREFIX, suffix: str | None = None
) -> str:
    """Return the acquisition handle for the main sweep result."""
    return (
        f"{qubit_name}/{prefix}"
        if suffix is None
        else f"{qubit_name}/{prefix}/{suffix}"
    )


def calibration_trace_handle(
    qubit_name: str,
    state: str | None = None,
    prefix: str = CALIBRATION_TRACE_PREFIX,
) -> str:
    """Return the acquisition handle for a calibration trace."""
    return (
        f"{qubit_name}/{prefix}/{state}"
        if state is not None
        else f"{qubit_name}/{prefix}"
    )


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


def active_reset_handle(
    qubit_name: str,
    prefix: str = ACTIVE_RESET_PREFIX,
    suffix: str | None = None,
) -> str:
    """Return the acquisition handle for an active reset."""
    res_handle_split = result_handle(qubit_name).split("/")
    res_handle = "/".join([h for h in res_handle_split if h != qubit_name])
    return (
        f"{qubit_name}/{prefix}/{res_handle}"
        if suffix is None
        else f"{qubit_name}/{prefix}/{res_handle}/{suffix}"
    )


def active_reset_calibration_trace_handle(
    qubit_name: str,
    state: str,
    prefix: str = ACTIVE_RESET_PREFIX,
    suffix: str | None = None,
) -> str:
    """Return the acquisition handle for active-reset calibration traces."""
    ct_handle_split = calibration_trace_handle(qubit_name, state).split("/")
    ct_handle = "/".join([h for h in ct_handle_split if h != qubit_name])
    return (
        f"{qubit_name}/{prefix}/{ct_handle}"
        if suffix is None
        else f"{qubit_name}/{prefix}/{ct_handle}/{suffix}"
    )

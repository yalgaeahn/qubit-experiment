"""This module defines shared constants and handle helpers for 2-qubit state tomography."""

from __future__ import annotations

TOMOGRAPHY_SETTINGS: tuple[tuple[str, tuple[str, str]], ...] = (
    ("XX", ("X", "X")),
    ("XY", ("X", "Y")),
    ("XZ", ("X", "Z")),
    ("YX", ("Y", "X")),
    ("YY", ("Y", "Y")),
    ("YZ", ("Y", "Z")),
    ("ZX", ("Z", "X")),
    ("ZY", ("Z", "Y")),
    ("ZZ", ("Z", "Z")),
)

READOUT_CALIBRATION_STATES: tuple[tuple[str, tuple[str, str]], ...] = (
    ("00", ("g", "g")),
    ("01", ("g", "e")),
    ("10", ("e", "g")),
    ("11", ("e", "e")),
)

OUTCOME_LABELS: tuple[str, ...] = ("00", "01", "10", "11")


def tomography_handle(qubit_uid: str, setting_label: str) -> str:
    """Result handle for tomography measurement."""
    return f"{qubit_uid}/tomo/{setting_label}"


def readout_calibration_handle(qubit_uid: str, prepared_label: str) -> str:
    """Result handle for readout calibration measurement."""
    return f"{qubit_uid}/readout_cal/{prepared_label}"

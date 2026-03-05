"""This module defines shared constants and handle helpers for 2-qubit state tomography."""

from __future__ import annotations

CANONICAL_2Q_STATE_ALIASES: dict[str, str] = {
    "00": "00",
    "01": "01",
    "10": "10",
    "11": "11",
    "++": "++",
    "+-": "+-",
    "-+": "-+",
    "--": "--",
    "gg": "00",
    "ge": "01",
    "eg": "10",
    "ee": "11",
    "plus_plus": "++",
    "plus_minus": "+-",
    "minus_plus": "-+",
    "minus_minus": "--",
}

SINGLE_QUBIT_TOKEN_SECTION_LABELS: dict[str, str] = {
    "g": "g",
    "e": "e",
    "+": "plus",
    "-": "minus",
}

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


def canonical_two_qubit_state_label(label: str) -> str:
    """Normalize supported 2Q product-state labels to canonical form."""
    if not isinstance(label, str):
        raise ValueError(f"label must be a string, got {type(label)!r}.")
    key = label.strip().lower().replace(" ", "")
    canonical = CANONICAL_2Q_STATE_ALIASES.get(key)
    if canonical is None:
        raise ValueError(
            "Unsupported 2Q state label. Use one of "
            "'00', '01', '10', '11', '++', '+-', '-+', '--', "
            "'gg', 'ge', 'eg', 'ee', "
            "'plus_plus', 'plus_minus', 'minus_plus', 'minus_minus'."
        )
    return canonical


def state_token_for_section_name(token: str) -> str:
    """Map single-qubit state tokens to section-name friendly labels."""
    try:
        return SINGLE_QUBIT_TOKEN_SECTION_LABELS[token]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported single-qubit state token for section naming: {token!r}."
        ) from exc


def tomography_handle(qubit_uid: str, setting_label: str) -> str:
    """Result handle for tomography measurement."""
    return f"{qubit_uid}/tomo/{setting_label}"


def readout_calibration_handle(qubit_uid: str, prepared_label: str) -> str:
    """Result handle for readout calibration measurement."""
    return f"{qubit_uid}/readout_cal/{prepared_label}"

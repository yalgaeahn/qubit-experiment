"""This module defines shared constants and handle helpers for 3-qubit state tomography."""

from __future__ import annotations

from itertools import product

_CANONICAL_BINARY_STATES: tuple[str, ...] = tuple(
    "".join(bits) for bits in product("01", repeat=3)
)
_CANONICAL_PLUS_MINUS_STATES: tuple[str, ...] = tuple(
    "".join(bits) for bits in product("+-", repeat=3)
)
_CANONICAL_GE_STATES: tuple[str, ...] = tuple(
    "".join(bits) for bits in product("ge", repeat=3)
)

CANONICAL_3Q_STATE_ALIASES: dict[str, str] = {}
for label in _CANONICAL_BINARY_STATES:
    CANONICAL_3Q_STATE_ALIASES[label] = label
for label in _CANONICAL_PLUS_MINUS_STATES:
    CANONICAL_3Q_STATE_ALIASES[label] = label
for label in _CANONICAL_GE_STATES:
    CANONICAL_3Q_STATE_ALIASES[label] = label.replace("g", "0").replace("e", "1")

_TOKEN_WORD = {
    "+": "plus",
    "-": "minus",
}
for label in _CANONICAL_PLUS_MINUS_STATES:
    key = "_".join(_TOKEN_WORD[ch] for ch in label)
    CANONICAL_3Q_STATE_ALIASES[key] = label

SINGLE_QUBIT_TOKEN_SECTION_LABELS: dict[str, str] = {
    "g": "g",
    "e": "e",
    "+": "plus",
    "-": "minus",
}

_TOMOGRAPHY_AXES = ("X", "Y", "Z")
TOMOGRAPHY_SETTINGS: tuple[tuple[str, tuple[str, str, str]], ...] = tuple(
    (
        f"{a0}{a1}{a2}",
        (a0, a1, a2),
    )
    for a0 in _TOMOGRAPHY_AXES
    for a1 in _TOMOGRAPHY_AXES
    for a2 in _TOMOGRAPHY_AXES
)

READOUT_CALIBRATION_STATES: tuple[tuple[str, tuple[str, str, str]], ...] = tuple(
    (
        label,
        tuple("g" if bit == "0" else "e" for bit in label),
    )
    for label in _CANONICAL_BINARY_STATES
)

OUTCOME_LABELS: tuple[str, ...] = _CANONICAL_BINARY_STATES


def canonical_three_qubit_state_label(label: str) -> str:
    """Normalize supported 3Q product-state labels to canonical form."""
    if not isinstance(label, str):
        raise ValueError(f"label must be a string, got {type(label)!r}.")
    key = label.strip().lower().replace(" ", "")
    canonical = CANONICAL_3Q_STATE_ALIASES.get(key)
    if canonical is None:
        raise ValueError(
            "Unsupported 3Q state label. Use one of binary labels "
            "('000'..'111'), +/- labels ('+++', ...), g/e labels ('ggg', ...), "
            "or word aliases like 'plus_plus_plus'."
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

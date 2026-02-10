"""This module defines shared constants and handle helpers for 2Q multiplexed IQ blobs."""

from __future__ import annotations

PREPARED_STATES_2Q: tuple[tuple[str, tuple[str, str]], ...] = (
    ("00", ("g", "g")),
    ("01", ("g", "e")),
    ("10", ("e", "g")),
    ("11", ("e", "e")),
)

OUTCOME_LABELS_2Q: tuple[str, ...] = ("00", "01", "10", "11")


def multiplexed_iq_blob_handle(qubit_uid: str, prepared_label: str) -> str:
    """Result handle for 2Q multiplexed IQ-blob acquisition."""
    return f"{qubit_uid}/multiplexed_iq_blob/{prepared_label}"

"""Shared helpers/constants for IQ-cloud calibration workflows."""

from __future__ import annotations

PREPARED_LABELS_1Q: tuple[str, ...] = ("g", "e")
PREPARED_LABELS_2Q: tuple[str, ...] = ("gg", "ge", "eg", "ee")
JOINT_LABELS_2Q: tuple[str, ...] = ("gg", "ge", "eg", "ee")


def validate_supported_num_qubits(num_qubits: int) -> None:
    """Validate IQ-cloud supported qubit counts."""
    if num_qubits not in (1, 2):
        raise ValueError(
            "iq_cloud supports only 1 or 2 qubits. "
            f"Received {num_qubits} qubits."
        )


def prepared_labels_for_num_qubits(num_qubits: int) -> tuple[str, ...]:
    """Return prepared-state labels for 1Q/2Q g/e-only experiments."""
    validate_supported_num_qubits(num_qubits)
    if num_qubits == 1:
        return PREPARED_LABELS_1Q
    return PREPARED_LABELS_2Q


def iq_cloud_handle(qubit_uid: str, prepared_label: str) -> str:
    """Result handle for one qubit and one prepared label."""
    return f"{qubit_uid}/iq_cloud/{prepared_label}"

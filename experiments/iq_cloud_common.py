"""Shared helpers/constants for IQ-cloud calibration workflows."""

from __future__ import annotations

from example_helpers.workflow.handles import calibration_trace_2q_handle as _cal_trace_2q
from laboneq.simple import dsl

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
    """Legacy IQ-cloud result handle for one qubit and one prepared label.

    Deprecated: kept for backward compatibility with historical datasets.
    Canonical handles are:
    - 1Q: dsl.handles.calibration_trace_handle(qubit_uid, state)
    - 2Q: calibration_trace_2q_handle(qubit_uid, prepared_label)
    """
    return f"{qubit_uid}/iq_cloud/{prepared_label}"


def iq_cloud_1q_cal_trace_handle(qubit_uid: str, state: str) -> str:
    """Canonical 1Q IQ-cloud handle."""
    return dsl.handles.calibration_trace_handle(qubit_uid, state)


def iq_cloud_2q_cal_trace_handle(qubit_uid: str, prepared_label: str) -> str:
    """Canonical 2Q IQ-cloud handle."""
    return _cal_trace_2q(qubit_uid, prepared_label)

from __future__ import annotations

import pytest

from experiments.iq_cloud_common import (
    JOINT_LABELS_2Q,
    joint_labels_for_num_qubits,
    prepared_labels_for_num_qubits,
    validate_supported_num_qubits,
)


def test_prepared_labels_follow_binary_order_up_to_3q() -> None:
    assert prepared_labels_for_num_qubits(1) == ("g", "e")
    assert prepared_labels_for_num_qubits(2) == ("gg", "ge", "eg", "ee")
    assert prepared_labels_for_num_qubits(3) == (
        "ggg",
        "gge",
        "geg",
        "gee",
        "egg",
        "ege",
        "eeg",
        "eee",
    )


def test_joint_labels_2q_alias_kept_for_compatibility() -> None:
    assert joint_labels_for_num_qubits(2) == JOINT_LABELS_2Q


def test_validate_supported_num_qubits_rejects_non_positive() -> None:
    with pytest.raises(ValueError):
        validate_supported_num_qubits(0)


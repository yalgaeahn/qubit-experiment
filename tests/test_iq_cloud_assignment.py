from __future__ import annotations

import numpy as np

from qubit_experiment.analysis.iq_cloud import _assignment_core
from qubit_experiment.experiments.iq_cloud_common import prepared_labels_for_num_qubits


def _bit_to_shots(bit: int, count: int) -> np.ndarray:
    real_center = 1.0 if bit else -1.0
    return np.full(count, real_center, dtype=float) + 1j * np.zeros(count, dtype=float)


def _build_perfect_shot_arrays(
    qubit_uids: list[str],
    prepared_labels: tuple[str, ...],
    shots_per_label: int,
    short_uid: str | None = None,
    short_count: int = 0,
) -> dict[str, dict[str, np.ndarray]]:
    shot_arrays: dict[str, dict[str, np.ndarray]] = {uid: {} for uid in qubit_uids}
    for label in prepared_labels:
        bits = [0 if ch == "g" else 1 for ch in label]
        for q_index, uid in enumerate(qubit_uids):
            count = short_count if uid == short_uid else shots_per_label
            shot_arrays[uid][label] = _bit_to_shots(bits[q_index], count)
    return shot_arrays


def _simple_decision_model(qubit_uids: list[str]) -> dict[str, dict]:
    return {uid: {"w": [1.0, 0.0], "b": 0.0} for uid in qubit_uids}


def test_assignment_core_2q_joint_mapping_regression() -> None:
    qubit_uids = ["q0", "q1"]
    prepared_labels = prepared_labels_for_num_qubits(len(qubit_uids))
    shot_arrays = _build_perfect_shot_arrays(
        qubit_uids=qubit_uids,
        prepared_labels=prepared_labels,
        shots_per_label=5,
    )

    bundle = _assignment_core(
        shot_arrays=shot_arrays,
        decision_model=_simple_decision_model(qubit_uids),
        qubit_uids=qubit_uids,
        prepared_labels=prepared_labels,
    )

    joint_counts = np.asarray(bundle["confusion_matrices"]["joint"]["counts"], dtype=int)
    assert joint_counts.shape == (4, 4)
    assert np.array_equal(joint_counts, 5 * np.eye(4, dtype=int))
    assert bundle["confusion_matrices"]["joint"]["labels"] == list(prepared_labels)
    assert bundle["assignment_fidelity"]["joint"] == 1.0
    assert bundle["assignment_fidelity"]["average"] == 1.0


def test_assignment_core_3q_joint_shape_labels_and_index_mapping() -> None:
    qubit_uids = ["q0", "q1", "q2"]
    prepared_labels = prepared_labels_for_num_qubits(len(qubit_uids))
    shot_arrays = _build_perfect_shot_arrays(
        qubit_uids=qubit_uids,
        prepared_labels=prepared_labels,
        shots_per_label=4,
    )

    bundle = _assignment_core(
        shot_arrays=shot_arrays,
        decision_model=_simple_decision_model(qubit_uids),
        qubit_uids=qubit_uids,
        prepared_labels=prepared_labels,
    )

    joint_counts = np.asarray(bundle["confusion_matrices"]["joint"]["counts"], dtype=int)
    assert joint_counts.shape == (8, 8)
    assert np.array_equal(joint_counts, 4 * np.eye(8, dtype=int))
    assert bundle["confusion_matrices"]["joint"]["labels"] == list(prepared_labels)
    assert bundle["assignment_fidelity"]["joint"] == 1.0
    assert bundle["assignment_fidelity"]["average"] == 1.0


def test_assignment_core_joint_uses_min_length_across_qubits() -> None:
    qubit_uids = ["q0", "q1", "q2"]
    prepared_labels = prepared_labels_for_num_qubits(len(qubit_uids))
    shot_arrays = _build_perfect_shot_arrays(
        qubit_uids=qubit_uids,
        prepared_labels=prepared_labels,
        shots_per_label=5,
        short_uid="q2",
        short_count=3,
    )

    bundle = _assignment_core(
        shot_arrays=shot_arrays,
        decision_model=_simple_decision_model(qubit_uids),
        qubit_uids=qubit_uids,
        prepared_labels=prepared_labels,
    )

    joint_counts = np.asarray(bundle["confusion_matrices"]["joint"]["counts"], dtype=int)
    assert joint_counts.shape == (8, 8)
    assert int(np.sum(joint_counts)) == len(prepared_labels) * 3
    assert np.array_equal(np.diag(joint_counts), np.full(8, 3, dtype=int))
    assert bundle["assignment_fidelity"]["joint"] == 1.0

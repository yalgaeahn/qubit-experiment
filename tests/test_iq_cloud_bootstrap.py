from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

import qubit_experiment.analysis.iq_cloud as iq_cloud
from qubit_experiment.experiments.iq_cloud_common import prepared_labels_for_num_qubits


@dataclass
class DummyQubit:
    uid: str


def _build_processed_data(
    num_qubits: int = 3,
    shots_per_label: int = 40,
    seed: int = 1234,
) -> tuple[list[DummyQubit], dict]:
    rng = np.random.default_rng(seed)
    qubits = [DummyQubit(uid=f"q{i}") for i in range(num_qubits)]
    labels = prepared_labels_for_num_qubits(num_qubits)
    payload: dict[str, Any] = {
        "prepared_labels": list(labels),
        "shots_per_qubit": {q.uid: {} for q in qubits},
    }

    for label in labels:
        bits = [0 if ch == "g" else 1 for ch in label]
        for q_index, q in enumerate(qubits):
            center_i = -1.4 if bits[q_index] == 0 else 1.4
            center_q = 0.25 * (q_index + 1)
            noise = 0.18 * (
                rng.standard_normal(shots_per_label)
                + 1j * rng.standard_normal(shots_per_label)
            )
            shots = (center_i + 1j * center_q) + noise
            payload["shots_per_qubit"][q.uid][label] = shots.tolist()
    return qubits, payload


def _assert_valid_ci(entry: dict) -> None:
    mean = float(entry["mean"])
    ci_low = float(entry["ci_low"])
    ci_high = float(entry["ci_high"])
    assert ci_low <= mean <= ci_high


def test_bootstrap_metrics_3q_contains_joint_average_and_valid_ci(
    monkeypatch,
) -> None:
    qubits, processed_data = _build_processed_data()
    monkeypatch.setattr(iq_cloud, "validate_and_convert_qubits_sweeps", lambda q: q)

    bootstrap = iq_cloud.bootstrap_metrics.func(
        processed_data=processed_data,
        qubits=qubits,
        ridge_target_condition=1e6,
        bootstrap_samples=80,
        bootstrap_confidence_level=0.95,
        bootstrap_seed=77,
    )

    assert "per_qubit" in bootstrap
    assert "joint" in bootstrap
    assert "average" in bootstrap
    assert "settings" in bootstrap

    for q in qubits:
        per_qubit = bootstrap["per_qubit"][q.uid]
        for key in ("fidelity", "threshold", "delta_mu_over_sigma"):
            _assert_valid_ci(per_qubit[key])

    _assert_valid_ci(bootstrap["joint"]["fidelity"])
    _assert_valid_ci(bootstrap["average"]["fidelity"])

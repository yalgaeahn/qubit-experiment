from __future__ import annotations

from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import analysis.iq_cloud as iq_cloud
from experiments.iq_cloud_common import prepared_labels_for_num_qubits


@dataclass
class DummyQubit:
    uid: str


def _bootstrap_payload_3q(qubit_uids: list[str]) -> dict:
    per_qubit = {}
    for uid in qubit_uids:
        per_qubit[uid] = {
            "fidelity": {
                "mean": 0.98,
                "ci_low": 0.97,
                "ci_high": 0.99,
                "confidence_level": 0.95,
            },
            "threshold": {
                "mean": 0.0,
                "ci_low": -0.1,
                "ci_high": 0.1,
                "confidence_level": 0.95,
            },
            "delta_mu_over_sigma": {
                "mean": 5.0,
                "ci_low": 4.8,
                "ci_high": 5.2,
                "confidence_level": 0.95,
            },
        }
    return {
        "per_qubit": per_qubit,
        "joint": {
            "fidelity": {
                "mean": 0.96,
                "ci_low": 0.95,
                "ci_high": 0.97,
                "confidence_level": 0.95,
            }
        },
        "average": {
            "fidelity": {
                "mean": 0.98,
                "ci_low": 0.97,
                "ci_high": 0.99,
                "confidence_level": 0.95,
            }
        },
    }


def test_plot_assignment_matrices_smoke_for_3q(monkeypatch) -> None:
    qubits = [DummyQubit(uid=f"q{i}") for i in range(3)]
    qubit_uids = [q.uid for q in qubits]
    labels = list(prepared_labels_for_num_qubits(3))
    monkeypatch.setattr(iq_cloud, "validate_and_convert_qubits_sweeps", lambda q: q)

    confusion_matrices = {
        "per_qubit": {
            uid: {
                "normalized": np.eye(2).tolist(),
            }
            for uid in qubit_uids
        },
        "joint": {
            "normalized": np.eye(8).tolist(),
            "labels": labels,
        },
    }
    assignment_fidelity = {
        "per_qubit": {uid: 1.0 for uid in qubit_uids},
        "joint": 1.0,
        "average": 1.0,
    }
    separation_metrics = {
        "per_qubit": {uid: {"delta_mu_over_sigma": 5.0} for uid in qubit_uids}
    }

    figures = iq_cloud.plot_assignment_matrices.func(
        confusion_matrices=confusion_matrices,
        assignment_fidelity=assignment_fidelity,
        qubits=qubits,
        separation_metrics=separation_metrics,
        bootstrap=_bootstrap_payload_3q(qubit_uids),
    )

    assert "assignment_matrices" in figures
    fig = figures["assignment_matrices"]
    assert fig is not None
    assert len(fig.axes) >= 4
    plt.close(fig)


def test_plot_bootstrap_summary_smoke_for_3q(monkeypatch) -> None:
    qubits = [DummyQubit(uid=f"q{i}") for i in range(3)]
    monkeypatch.setattr(iq_cloud, "validate_and_convert_qubits_sweeps", lambda q: q)

    figures = iq_cloud.plot_bootstrap_summary.func(
        bootstrap=_bootstrap_payload_3q([q.uid for q in qubits]),
        qubits=qubits,
    )

    assert "bootstrap_summary" in figures
    fig = figures["bootstrap_summary"]
    assert fig is not None
    assert len(fig.axes) == 3
    plt.close(fig)


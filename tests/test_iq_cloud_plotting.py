from __future__ import annotations

from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_hex

pytest.importorskip("laboneq")

import qubit_experiment.analysis.iq_cloud as iq_cloud
from qubit_experiment.analysis.plot_theme import (
    get_plot_theme_rc_params,
    get_semantic_color,
    get_state_color,
)
from qubit_experiment.experiments.iq_cloud_common import prepared_labels_for_num_qubits


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
    expected_face = to_hex(get_plot_theme_rc_params()["axes.facecolor"])
    assert to_hex(fig.axes[0].get_facecolor()) == expected_face
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
    expected_face = to_hex(get_plot_theme_rc_params()["axes.facecolor"])
    assert to_hex(fig.axes[0].get_facecolor()) == expected_face
    expected_g = to_hex(get_state_color("g"))
    marker_line = fig.axes[0].lines[0]
    assert to_hex(marker_line.get_color()) == expected_g
    plt.close(fig)


def test_plot_iq_clouds_theme_colors_smoke(monkeypatch) -> None:
    qubits = [DummyQubit(uid=f"q{i}") for i in range(2)]
    monkeypatch.setattr(iq_cloud, "validate_and_convert_qubits_sweeps", lambda q: q)

    rng = np.random.default_rng(123)
    labels = prepared_labels_for_num_qubits(2)
    processed_data = {
        "shots_per_qubit": {
            q.uid: {
                label: (
                    rng.normal(0.0, 0.15, size=128)
                    + 1j * rng.normal(0.0, 0.15, size=128)
                ).tolist()
                for label in labels
            }
            for q in qubits
        }
    }
    decision_model = {
        q.uid: {
            "w": np.array([1.0, -0.2]),
            "b": 0.1,
            "axis_unit": np.array([1.0, 0.0]),
            "mu_g": np.array([-0.2, 0.0]),
            "mu_e": np.array([0.3, 0.1]),
            "sigma": np.array([[0.05, 0.0], [0.0, 0.05]]),
            "t": 0.02,
        }
        for q in qubits
    }

    figures = iq_cloud.plot_iq_clouds.func(
        processed_data=processed_data,
        decision_model=decision_model,
        qubits=qubits,
        bootstrap=None,
    )
    fig = figures[qubits[0].uid]
    ax_iq = fig.axes[0]

    expected_axes_face = to_hex(get_plot_theme_rc_params()["axes.facecolor"])
    assert to_hex(ax_iq.get_facecolor()) == expected_axes_face

    collection_colors = {
        to_hex(collection.get_facecolor()[0])
        for collection in ax_iq.collections
        if hasattr(collection, "get_facecolor")
        and collection.get_facecolor().size > 0
    }
    assert to_hex(get_state_color("g")) in collection_colors
    assert to_hex(get_state_color("e")) in collection_colors

    line_colors = {to_hex(line.get_color()) for line in ax_iq.lines}
    assert to_hex(get_semantic_color("boundary")) in line_colors
    plt.close(fig)

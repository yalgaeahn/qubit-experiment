"""Durable local records for coherence monitoring; no hardware or QPU updates.

Raw LabOne Q results are saved before analysis. Core writes are fatal on failure;
plots are best effort. Each directory belongs to one run and one writer.
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from uuid import uuid4

import attrs
import numpy as np
from laboneq import serializers


class MonitoringStorageError(RuntimeError):
    """A required monitoring record could not be persisted."""


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def json_value(value):
    """Convert scientific payloads to strict JSON, preserving complex arrays."""
    if hasattr(value, "nominal_value"):
        return {"value": json_value(value.nominal_value), "std_dev": json_value(value.std_dev)}
    if attrs.has(type(value)):
        return json_value(attrs.asdict(value))
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, complex):
        return {"real": json_value(value.real), "imag": json_value(value.imag)}
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Enum):
        return json_value(value.value)
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Cannot persist {type(value).__name__} in monitoring metadata")


def atomic_write(path, payload, *, laboneq=False):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if laboneq:
            serializers.save(payload, temporary)
        else:
            temporary.write_text(
                json.dumps(json_value(payload), allow_nan=False, indent=2), encoding="utf-8"
            )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception as exc:
        raise MonitoringStorageError(f"Could not save {path}: {exc}") from exc
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def append_records(path, rows):
    if not rows:
        return
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(json_value(row), allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except Exception as exc:
        raise MonitoringStorageError(f"Could not append {path}: {exc}") from exc


def event(state, kind, **details):
    row = {"run_id": state["run_id"], "timestamp_utc": utc_now(), "event": kind, **details}
    if state.get("monitoring_directory"):
        append_records(Path(state["monitoring_directory"]) / "events.jsonl", [row])
    return row


def save_status(state, status, **details):
    if state.get("monitoring_directory"):
        atomic_write(
            Path(state["monitoring_directory"]) / "status.json",
            {
                "run_id": state["run_id"],
                "status": status,
                "updated_utc": utc_now(),
                "metric_counts": state["metric_counts"],
                "final_frequencies": state.get("frequencies", {}),
                **details,
            },
        )


def record_terminal(state, status, **details):
    """Attempt both terminal writes without masking an interrupt/original error."""
    errors = []
    for operation in (
        lambda: event(state, status, **details),
        lambda: save_status(state, status, **details),
    ):
        try:
            operation()
        except Exception as exc:
            errors.append(str(exc))
    return errors


def render_trends(state, tracked_parameters):
    """Scatter points deliberately do not interpolate over failures or gaps."""
    if not state.get("monitoring_directory"):
        return
    import matplotlib.dates as mdates
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    rows = state["history_entries"]
    for uid in state["initial_frequencies"]:
        fig = Figure(figsize=(9, 2.7 * len(tracked_parameters)))
        FigureCanvasAgg(fig)
        axes = fig.subplots(len(tracked_parameters), 1, squeeze=False)
        try:
            for axis, key in zip(axes[:, 0], tracked_parameters):
                points = [r for r in rows if r["qubit_uid"] == uid and r["parameter_key"] == key]
                scale = 1e-9 if key == "resonance_frequency_ge" else 1e6
                for warning, marker in ((False, "o"), (True, "x")):
                    selected = [r for r in points if bool(r.get("quality_reasons")) == warning]
                    if selected:
                        axis.scatter(
                            [
                                datetime.fromisoformat(r["timestamp_utc"].replace("Z", "+00:00"))
                                for r in selected
                            ],
                            [r["value"] * scale for r in selected],
                            marker=marker,
                            label="quality warning" if warning else "accepted",
                        )
                axis.set_ylabel(f"{key} ({'GHz' if scale == 1e-9 else 'us'})")
                axis.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=timezone.utc))
                if points:
                    axis.legend()
                axis.grid(alpha=0.2)
            axes[-1, 0].set_xlabel("UTC")
            fig.suptitle(uid)
            fig.tight_layout()
            path = Path(state["monitoring_directory"]) / f"trend_{uid}.png"
            temporary = path.with_suffix(".tmp.png")
            try:
                fig.savefig(temporary)
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
        finally:
            fig.clear()

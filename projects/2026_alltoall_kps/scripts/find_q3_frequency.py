#!/usr/bin/env python3
"""Bounded Q3 spectroscopy using the existing workflow, independent of notebooks."""

from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/kps-mpl")

import matplotlib.pyplot as plt
import numpy as np
from laboneq import workflow
from laboneq.simple import Session
from laboneq_applications.tasks.parameter_updating import temporary_qpu
from measure_resonator_linewidth_cw import PROJECT, load_platform

from qubit_experiment.experiments import qubit_spectroscopy as spec
from qubit_experiment.experiments.options import QubitSpectroscopyExperimentOptions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--start", type=float, default=4e9)
    parser.add_argument("--stop", type=float, default=5e9)
    parser.add_argument("--points", type=int, default=2001)
    parser.add_argument("--power", type=float, default=-43.46)
    parser.add_argument("--off", action="store_true")
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--readout-amplitude", type=float)
    parser.add_argument("--readout-frequency", type=float)
    parser.add_argument("--label", default="survey")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not (4e9 <= min(args.start, args.stop) < max(args.start, args.stop) <= 5e9):
        parser.error("Frequency limits must lie in 4–5 GHz")
    if not (-70 <= args.power <= -43.46 and 2 <= args.points <= 4001 and 1 <= args.count <= 8192):
        parser.error("Invalid bounded power, point count, or averages")
    snapshot = sorted((PROJECT / "qpu_parameters").glob("*.json"))[-1]
    qpu, setup = load_platform(snapshot, "192.168.0.83")
    q = qpu.groups.qubits[2]
    assert q.uid == "q3" and q.signals["drive"] == "q3/drive"
    assert setup.instruments[0].address == "DEV12256"
    p = deepcopy(q.parameters)
    p.drive_lo_frequency = round((args.start + args.stop) / 2 / 200e6) * 200e6
    p.spectroscopy_length = 10e-6
    p.spectroscopy_amplitude = 0 if args.off else 10 ** ((args.power - p.drive_range) / 20)
    if args.readout_amplitude is not None:
        if not 0 < args.readout_amplitude <= p.readout_amplitude:
            parser.error("Readout amplitude may only be reduced from the snapshot")
        p.readout_amplitude = args.readout_amplitude
    if args.readout_frequency is not None:
        if abs(args.readout_frequency - p.readout_resonator_frequency) > 10e6:
            parser.error("Readout frequency must stay within 10 MHz of the snapshot")
        p.readout_resonator_frequency = args.readout_frequency
    assert 0 <= p.spectroscopy_amplitude <= 1
    freq = np.linspace(args.start, args.stop, args.points)
    out = args.output or PROJECT / "data/q3_spectroscopy_search" / datetime.now().strftime(
        "%Y%m%d-%H%M%S"
    )
    out.mkdir(parents=True, exist_ok=True)
    stem = out / args.label
    if stem.with_suffix(".npz").exists():
        raise FileExistsError(stem)
    metadata = {
        "snapshot": str(snapshot),
        "target": q.uid,
        "execute": args.execute,
        "power_dbm": None if args.off else args.power,
        "count": args.count,
        "start_hz": args.start,
        "stop_hz": args.stop,
        "points": args.points,
        "drive_range_dbm": p.drive_range,
        "drive_lo_hz": p.drive_lo_frequency,
        "spectroscopy_amplitude": p.spectroscopy_amplitude,
        "spectroscopy_length_s": p.spectroscopy_length,
        "readout": q.readout_parameters(),
        "parameters_repr": repr(p),
    }
    stem.with_suffix(".json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")
    session = Session(setup, configure_logging=False)
    temp = temporary_qpu.func(qpu, {q.uid: p})
    exp = spec.create_experiment.func(
        temp, temp[q.uid], freq, options=QubitSpectroscopyExperimentOptions(count=2)
    )
    session.compile(exp)
    print(f"COMPILE_OK {out} {json.dumps(metadata, default=str)}", flush=True)
    if not args.execute:
        return
    options = spec.experiment_workflow.options()
    options.update(False)
    options.do_analysis(False)
    options.count(args.count)
    options.logstore(workflow.logbook.FolderStore(out / "workflow_logs"))
    connected = False
    try:
        session.connect(do_emulation=False, ignore_version_mismatch=False)
        connected = True
        # A second controller must not interrupt an active acquisition.
        device = session.devices["DEV12256"]
        for node in ("sgchannels/2/awg/enable", "qachannels/0/generator/enable"):
            if device.root.raw_path_to_node("/dev12256/" + node)():
                raise RuntimeError(f"Active acquisition detected: {node}")
        print("CONNECTED_IDLE; ACQUIRING", flush=True)
        run = spec.experiment_workflow(
            session, qpu, q.uid, freq, temporary_parameters={q.uid: p}, options=options
        ).run()
        z = np.asarray(run.output[f"{q.uid}/result"].data, dtype=complex)
        np.savez(stem.with_suffix(".npz"), frequency_hz=freq, signal=z)
        assert z.shape == freq.shape and np.isfinite(z).all()
        np.savetxt(
            stem.with_suffix(".csv"),
            np.column_stack([freq, z.real, z.imag]),
            delimiter=",",
            header="frequency_hz,I,Q",
            comments="",
        )
        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(11, 8))
        for ax, y, name in zip(axes, [z.real, z.imag, abs(z)], ["I", "Q", "Magnitude"]):
            ax.plot(freq / 1e9, y, ".-", ms=2, lw=0.5)
            ax.set_ylabel(name)
            ax.grid(alpha=0.3)
        axes[0].set_title(f"Q3 {args.label}: {metadata['power_dbm']} dBm, N={args.count}")
        axes[-1].set_xlabel("Drive frequency (GHz)")
        fig.tight_layout()
        fig.savefig(stem.with_suffix(".png"), dpi=160)
        plt.close(fig)
        print(f"SAVED {stem}", flush=True)
    finally:
        if connected:
            session.disconnect()


if __name__ == "__main__":
    main()

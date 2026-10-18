#!/usr/bin/env python3
"""Independent, sequential CW linewidth characterization; no notebook kernel.

Default compiles a real-device configuration offline. --execute acquires data.
All power values refer to the instrument output port; no QPU snapshot is updated.
"""

# ruff: noqa: E402 -- project bootstrap must precede package imports
from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
PROJECT = Path(__file__).resolve().parents[1]
REPOSITORY = PROJECT.parents[1]
sys.path.insert(0, str(REPOSITORY))

import matplotlib.pyplot as plt
import numpy as np
from laboneq import serializers, workflow
from laboneq.dsl.device import DeviceSetup
from laboneq.dsl.result import AcquiredResult, Results
from laboneq.dsl.session import Session

from qubit_experiment.analysis import resonator_linewidth_cw as analysis
from qubit_experiment.experiments import resonator_linewidth_cw as experiment


def json_safe(value):
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, dict):
        return {
            k: json_safe(v)
            for k, v in value.items()
            if k not in {"fitted_complex", "residual_complex", "raw_complex", "covariance"}
        }
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    path.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n")


def load_platform(snapshot, server_host):
    # Older snapshots used these import names. Register only serialization aliases.
    for legacy, canonical in {
        "qpu_types": "qubit_experiment.qpu_types",
        "qpu_types.fixed_transmon": "qubit_experiment.qpu_types.fixed_transmon",
        "qpu_types.fixed_transmon.qubit_types": "qubit_experiment.qpu_types.fixed_transmon.qubit_types",
        "qpu_types.fixed_transmon.operations": "qubit_experiment.qpu_types.fixed_transmon.operations",
    }.items():
        sys.modules.setdefault(legacy, importlib.import_module(canonical))
    qpu = serializers.load(snapshot)
    setup = DeviceSetup.from_yaml(
        str(PROJECT / "configs/descriptors/kps.yaml"), server_host=server_host
    )
    setup.instruments[0].device_options = "SHFQC/PLUS/QC6CH"
    return qpu, setup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubit", default="q2")
    parser.add_argument("--reference-frequency-hz", type=float, default=None)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--server-host", default="192.168.0.83")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--input-range-dbm", type=float, default=None)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args()
    if args.reference_frequency_hz is None:
        if args.qubit != "q2":
            parser.error("--reference-frequency-hz is required for qubits other than q2")
        args.reference_frequency_hz = 7.3452e9
    snapshot = args.snapshot or sorted((PROJECT / "qpu_parameters").glob("*.json"))[-1]
    out = args.output or PROJECT / "data/resonator_linewidth_cw" / datetime.now().strftime(
        "%Y%m%d-%H%M%S"
    )
    out.mkdir(parents=True, exist_ok=False)
    qpu, setup = load_platform(snapshot, args.server_host)
    q = qpu[args.qubit]
    reference = args.reference_frequency_hz
    base_amplitude = 0.001294196
    amplitudes = base_amplitude * np.array([0.5, 1, 2])
    overrides = {q.uid: {"readout_range_out": -10}}
    if args.input_range_dbm is not None:
        overrides[q.uid]["readout_range_in"] = args.input_range_dbm
    q_analysis = q.replace(**overrides[q.uid])
    session = Session(setup, configure_logging=False)
    metadata = {
        "qubit_uid": q.uid,
        "snapshot": str(snapshot.resolve()),
        "reference_frequency_hz": reference,
        "readout_operating_frequency_hz": q.parameters.readout_resonator_frequency,
        "output_range_dbm": -10,
        "input_range_dbm": q_analysis.parameters.readout_range_in,
        "amplitudes": amplitudes,
        "instrument_port_power_dbm": -10 + 20 * np.log10(amplitudes),
        "execute": args.execute,
        "server_host": args.server_host,
        "model_reference": analysis.MODEL_REFERENCE,
        "power_note": "instrument output, not attenuation-calibrated chip input",
        "interpretation": "loaded readout hybrid mode; qubit not actively prepared",
    }
    write_json(out / "request.json", metadata)
    # Real setup and real QPU: compiler only; no device connection/output.
    opts = experiment.LinewidthExperimentOptions(count=2)
    from laboneq_applications.tasks.parameter_updating import temporary_qpu

    offline_qpu = temporary_qpu.func(qpu, overrides)
    exp = experiment.create_experiment.func(
        offline_qpu,
        q_analysis,
        np.linspace(reference - 8e6, reference + 8e6, 801),
        amplitudes,
        reference,
        options=opts,
    )
    compiled = session.compile(exp)
    write_json(
        out / "offline_compile.json",
        {"status": "passed", "experiment_uid": compiled.experiment.uid},
    )
    print(f"OUTPUT {out.resolve()}", flush=True)
    print("Offline compile passed.", flush=True)
    if not args.execute:
        return
    logstore = workflow.logbook.FolderStore(str(out / "workflow_logs"))
    logstore.activate()
    connected = False
    cleanup_errors = []
    stages = []

    def acquire(label, f, a, count=2048, integration=2e-6):
        print(
            f"ACQUIRE {label}: {len(a)} powers × {len(f)} frequencies, count={count}, integration={integration:g}s",
            flush=True,
        )
        opts = experiment.experiment_workflow.options()
        opts.do_analysis(False)
        opts.count(count)
        opts.integration_length(integration)
        overrange = []

        class CaptureOverrange(logging.Handler):
            def emit(self, record):
                if "overrange count" in record.getMessage().lower():
                    overrange.append(record.getMessage())

        handler = CaptureOverrange()
        logger = logging.getLogger("laboneq")
        logger.addHandler(handler)
        try:
            result = (
                experiment.experiment_workflow(
                    session,
                    qpu,
                    q.uid,
                    f,
                    a,
                    reference,
                    temporary_parameters=overrides,
                    options=opts,
                )
                .run()
                .output
            )
        finally:
            logger.removeHandler(handler)
        raw = result["raw_result"]
        # Persist acquisition before invoking any numerical analysis.
        serializers.save(raw, out / f"{label}.raw.json")
        data = np.asarray(raw[f"{q.uid}/result"].data)
        if data.shape == (len(f),) and len(a) == 1:
            data = data[None, :]
        if data.shape != (len(a), len(f)) or not np.all(np.isfinite(data)):
            raise ValueError(f"Invalid acquired data: {data.shape}")
        np.savez_compressed(out / f"{label}.npz", frequencies_hz=f, amplitudes=a, raw_complex=data)
        if overrange:
            write_json(
                out / f"{label}.invalid.json",
                {"reason": "ADC/DAC overrange", "warnings": overrange},
            )
            raise ValueError("Overrange detected; raw data retained but excluded from fitting.")
        stage = {
            "label": label,
            "count": count,
            "integration_length_s": integration,
            "frequency_start_hz": f[0],
            "frequency_stop_hz": f[-1],
            "frequency_points": len(f),
            "amplitudes": a,
        }
        stages.append(stage)
        write_json(out / "stages.json", stages)
        return raw, data

    def analyze(label, f, a, data):
        raw = Results(
            acquired_results={
                f"{q.uid}/result": AcquiredResult(
                    data=data, axis=[a, f], axis_name=["amplitudes", "frequencies"]
                )
            }
        )
        opts = analysis.analysis_workflow.options()
        opts.save_figures(False)
        run = analysis.analysis_workflow(raw, q_analysis, f, a, reference, options=opts).run()
        metrics = run.output
        write_json(out / f"{label}.fit.json", metrics)
        for key, figure in run.tasks["plot_fits"].output.items():
            figure.savefig(out / f"{label}.{key}.png", dpi=150)
            plt.close(figure)
        print(
            f"FIT {label}: {[(p['amplitude'], p['status'], p.get('linewidth_hz'), p.get('reasons')) for p in metrics['power_points']]}",
            flush=True,
        )
        return metrics

    try:
        session.connect(do_emulation=False)
        connected = True
        if args.resume_from is not None:
            previous = json.loads((args.resume_from / "request.json").read_text())
            if previous["qubit_uid"] != q.uid or previous["reference_frequency_hz"] != reference:
                raise ValueError("Resume pilot must match the qubit and reference frequency.")
            if (
                previous["snapshot"] != str(snapshot.resolve())
                or previous.get("input_range_dbm") != q_analysis.parameters.readout_range_in
            ):
                raise ValueError("Resume requires the same QPU snapshot and receiver range.")
            if (args.resume_from / "02_pilot_retry_8192.invalid.json").exists():
                raise ValueError("Cannot resume an acquisition flagged as overrange.")
            saved = np.load(args.resume_from / "02_pilot_retry_8192.npz")
            np.testing.assert_allclose(saved["amplitudes"], [base_amplitude], rtol=1e-12)
            pilot_f, pilot_data = saved["frequencies_hz"], saved["raw_complex"]
            write_json(
                out / "resume.json",
                {
                    "source": str(args.resume_from.resolve()),
                    "note": "local pilot had no input overrange; broad survey excluded",
                },
            )
            pilot = analyze("02_resumed_pilot", pilot_f, np.array([base_amplitude]), pilot_data)
            point = pilot["power_points"][0]
        else:
            acquire(
                "01_survey",
                np.linspace(reference - 100e6, reference + 100e6, 2001),
                np.array([base_amplitude]),
                count=256,
            )
            pilot_f = np.linspace(reference - 8e6, reference + 8e6, 801)
            _, pilot_data = acquire("02_pilot", pilot_f, np.array([base_amplitude]), count=512)
            pilot = analyze("02_pilot", pilot_f, np.array([base_amplitude]), pilot_data)
            point = pilot["power_points"][0]
            for count in (2048, 8192):
                if point["status"] == "ok":
                    break
                _, pilot_data = acquire(
                    f"02_pilot_retry_{count}", pilot_f, np.array([base_amplitude]), count=count
                )
                pilot = analyze(
                    f"02_pilot_retry_{count}", pilot_f, np.array([base_amplitude]), pilot_data
                )
                point = pilot["power_points"][0]
        # A pilot is a scan-placement estimate, not the final reported kappa.
        # A statistically imprecise but window-stable pole can center denser data.
        seed_ok = (
            "q_loaded" in point
            and set(point["reasons"]) <= {"large_statistical_error"}
            and point["q_loaded_stderr"] / point["q_loaded"] < 0.25
        )
        if not seed_ok:
            raise ValueError(
                "Pilot does not identify a stable resonance; do not center on a noise fit."
            )
        center = point["fr_hz"]
        halfspan = min(8e6, max(1e6, 4 * point["linewidth_hz"]))
        f = np.linspace(center - halfspan, center + halfspan, 1601)
        traces = []
        for index, amplitude in enumerate(amplitudes):
            count = int(np.clip(2048 * (base_amplitude / amplitude) ** 2, 512, 8192))
            _, trace = acquire(
                f"03_low_power_{index}", f, np.array([amplitude]), count=count, integration=16e-6
            )
            traces.append(trace[0])
        data = np.stack(traces)
        metrics = analyze("03_low_power", f, amplitudes, data)
        # At most two lower powers. Keep all raw traces, including failed fits.
        for lower in range(2):
            if metrics["representative"] is not None:
                break
            if any(p["status"] != "ok" for p in metrics["power_points"][:2]):
                # Integrate more at the same power before sacrificing SNR further.
                for count in (4096, 8192):
                    _, data = acquire(f"04_retry_{lower}_{count}", f, amplitudes, count=count)
                    metrics = analyze(f"04_retry_{lower}_{count}", f, amplitudes, data)
                    if all(p["status"] == "ok" for p in metrics["power_points"][:2]):
                        break
                if any(p["status"] != "ok" for p in metrics["power_points"][:2]):
                    break
            if metrics["representative"] is not None:
                break
            extra_amp = np.array([amplitudes.min() * 0.5])
            _, extra_data = acquire(f"05_lower_{lower}", f, extra_amp)
            amplitudes = np.r_[extra_amp, amplitudes]
            data = np.vstack([extra_data, data])
            metrics = analyze(f"05_combined_{lower}", f, amplitudes, data)
        validation_rows = []
        representative = metrics["representative"]
        if representative is not None:
            candidate = np.array([representative["amplitude"]])
            for label, scan_f, integration in [
                ("06_long_integration", f, 4e-6),
                ("07_reverse", f[::-1], 2e-6),
            ]:
                _, check_data = acquire(label, scan_f, candidate, integration=integration)
                check = analyze(label, scan_f, candidate, check_data)["power_points"][0]
                passed = (
                    check["status"] == "ok"
                    and abs(check["q_loaded"] / representative["q_loaded"] - 1) <= 0.05
                )
                validation_rows.append({"label": label, "passed": passed, "fit": check})
            if not all(row["passed"] for row in validation_rows):
                metrics["representative"] = None
                metrics["validation_status"] = "integration_or_direction_dependence"
            else:
                metrics["validation_status"] = "passed"
        else:
            metrics["validation_status"] = "no_reliable_low_power_plateau"
        metrics["repeat_validation"] = validation_rows
        metrics["request"] = metadata
        write_json(out / "summary.json", metrics)
        print(
            "FINAL "
            + json.dumps(
                json_safe(
                    {
                        "representative": metrics["representative"],
                        "validation_status": metrics["validation_status"],
                        "power_convergence": metrics["power_convergence"],
                    }
                )
            ),
            flush=True,
        )
    finally:
        if connected:
            # Stop CW explicitly, including after an acquisition/analysis exception.
            for path, value in [
                ("qachannels/0/generator/enable", 0),
                ("qachannels/0/spectroscopy/envelope/enable", 1),
                ("qachannels/0/output/on", 0),
            ]:
                try:
                    session.devices["DEV12256"].root.raw_path_to_node("/dev12256/" + path)(value)
                except Exception as exc:
                    cleanup_errors.append(f"{path}: {exc}")
            try:
                session.disconnect()
            except Exception as exc:
                cleanup_errors.append(f"disconnect: {exc}")
        write_json(out / "cleanup.json", {"connected": connected, "errors": cleanup_errors})
        if cleanup_errors:
            print("CLEANUP ERRORS " + repr(cleanup_errors), flush=True)


if __name__ == "__main__":
    main()

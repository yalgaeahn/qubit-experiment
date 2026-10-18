"""Hardware-free CW reflection acquisition, analysis, and failure contracts."""

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest
from laboneq import workflow
from laboneq.dsl.result import AcquiredResult, Results
from laboneq.dsl.session import Session
from laboneq.simple import AcquisitionType, AveragingMode, dsl
from laboneq.workflow.reference import Reference

from qubit_experiment.analysis import resonator_linewidth_cw as analysis
from qubit_experiment.experiments import resonator_linewidth_cw as experiment
from qubit_experiment.qpu_types.fixed_transmon.demo_qpus import demo_platform

REFERENCE = 7.3452e9


def synthetic(f, width=0.5e6, delay=80e-9, residue=-1.2 * np.exp(0.3j), noise=0.0005):
    rng = np.random.default_rng(42)
    x = f - REFERENCE
    z = (1 + residue / (1 + 2j * x / width)) * np.exp(-2j * np.pi * x * delay)
    return z + noise * (rng.normal(size=f.size) + 1j * rng.normal(size=f.size))


def resolved(value):
    assert not isinstance(value, Reference)
    if isinstance(value, dict):
        for child in value.values():
            resolved(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            resolved(child)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("residue", [-1.2 * np.exp(0.3j), -2.0, -0.05])
def test_complex_reflection_recovery(reverse, residue):
    f = np.linspace(REFERENCE - 4e6, REFERENCE + 4e6, 301)
    z = synthetic(f, residue=residue, noise=1e-4)
    if reverse:
        f, z = f[::-1], z[::-1]
    original = z.copy()
    fit = analysis.fit_trace(f, z, REFERENCE, 80e-9)
    assert fit["status"] == "ok", fit["reasons"]
    assert fit["linewidth_hz"] == pytest.approx(0.5e6, rel=0.015)
    assert fit["fr_hz"] == pytest.approx(REFERENCE, abs=3000)
    assert fit["delay_s"] == pytest.approx(80e-9, abs=1e-9)
    assert fit["kappa_rad_s"] == pytest.approx(2 * np.pi * fit["linewidth_hz"])
    assert fit["energy_decay_time_s"] == pytest.approx(1 / fit["kappa_rad_s"])
    np.testing.assert_allclose(z - fit["fitted_complex"], fit["residual_complex"])
    np.testing.assert_array_equal(z, original)
    resolved(fit)


def test_no_delay_seed_and_unidentifiable_data():
    f = np.linspace(REFERENCE - 4e6, REFERENCE + 4e6, 301)
    fit = analysis.fit_trace(f, synthetic(f), REFERENCE)
    assert fit["linewidth_hz"] == pytest.approx(0.5e6, rel=0.02)
    with pytest.raises(ValueError, match="zero"):
        analysis.fit_trace(f, np.zeros_like(f), REFERENCE)


@pytest.mark.parametrize("analyze", [False, True])
def test_workflow_temporary_parameters_compile_and_offline_reanalysis(monkeypatch, analyze):
    platform = demo_platform(1)
    q = platform.qpu.quantum_elements[0]
    q.parameters.readout_amplitude = 0.4
    q.parameters.readout_phase_delay = 80e-9
    original_frequency = q.parameters.readout_resonator_frequency
    f = np.linspace(REFERENCE - 4e6, REFERENCE + 4e6, 301)
    amps = np.array([0.001, 0.002, 0.004])
    raw = np.stack([synthetic(f) * a for a in amps])
    result = Results(
        acquired_results={
            dsl.handles.result_handle(q.uid): AcquiredResult(
                data=raw, axis=[amps, f], axis_name=["amplitudes", "frequencies"]
            )
        }
    )

    @workflow.task(save=False)
    def acquire(session, compiled_experiment):
        return result

    monkeypatch.setattr(experiment, "run_experiment", acquire)
    opts = experiment.experiment_workflow.options()
    opts.logstore([])
    opts.count(2)
    opts.do_analysis(analyze)
    opts.do_plotting(False)
    run = experiment.experiment_workflow(
        Session(platform.setup, configure_logging=False),
        platform.qpu,
        q.uid,
        f,
        amps,
        REFERENCE,
        temporary_parameters={q.uid: {"readout_amplitude": 0.9, "readout_range_out": -10}},
        options=opts,
    ).run()
    resolved(run.output)
    assert (
        run.output["analysis"] is None
        if not analyze
        else run.output["analysis"]["representative"] is not None
    )
    assert q.parameters.readout_amplitude == 0.4
    assert q.parameters.readout_resonator_frequency == original_frequency
    exp = run.tasks["create_experiment"].output
    np.testing.assert_array_equal(
        exp.get_calibration()[q.signals["measure"]].amplitude.values, amps
    )
    assert exp.get_calibration()[q.signals["acquire"]].port_delay == 0
    assert exp.get_calibration()[q.signals["measure"]].automute is False
    if analyze:
        metrics = run.output["analysis"]
        assert metrics["representative"]["output_power_dbm"] == pytest.approx(-70)
        np.testing.assert_array_equal(metrics["raw_complex"], raw)
        options = analysis.analysis_workflow.options()
        options.logstore([])
        options.do_plotting(False)
        # Standalone analysis accepts the raw saved result and the temporary qubit.
        metrics2 = (
            analysis.analysis_workflow(
                result, q.replace(readout_range_out=-10), f, amps, REFERENCE, options=options
            )
            .run()
            .output
        )
        assert metrics2["representative"]["q_loaded"] == pytest.approx(
            metrics["representative"]["q_loaded"]
        )


@pytest.mark.parametrize(
    "change",
    [
        dict(acquisition_type=AcquisitionType.INTEGRATION),
        dict(averaging_mode=AveragingMode.SINGLE_SHOT),
        dict(integration_length=0.0),
        dict(settling_time=-1.0),
    ],
)
def test_reject_invalid_acquisition(change):
    platform = demo_platform(1)
    q = platform.qpu.quantum_elements[0]
    with pytest.raises(ValueError):
        experiment.create_experiment.func(
            platform.qpu,
            q,
            np.linspace(REFERENCE - 1e6, REFERENCE + 1e6, 31),
            [0.01],
            REFERENCE,
            options=experiment.LinewidthExperimentOptions(**change),
        )


def test_update_and_sweep_errors():
    with pytest.raises(ValueError, match="persistent"):
        experiment.validate_request.func("q2", [1] * 21, [0.01], 1, True)
    for amps in ([0], [-0.1], [1.1], [0.1, 0.1]):
        with pytest.raises(ValueError):
            experiment.validate_sweeps(
                np.linspace(REFERENCE - 1e6, REFERENCE + 1e6, 31), amps, REFERENCE
            )


def test_failed_low_power_not_silently_skipped():
    points = [{"amplitude": 0.001, "status": "failed"}, {"amplitude": 0.01, "status": "ok"}]
    representative, convergence = analysis.choose_representative(points)
    assert representative is None
    assert convergence["reason"] == "lowest_power_fit_unreliable"


def test_overlapping_filter_requires_two_poles():
    f = np.linspace(REFERENCE - 10e6, REFERENCE + 10e6, 601)
    x = (f - REFERENCE) / 1e6
    rng = np.random.default_rng(8)
    z = (1 - 1.1 / (1 + 2j * x / 0.5) + 0.6j / (1 + 2j * (x - 2) / 4)) * np.exp(
        -2j * np.pi * x * 0.08
    )
    z += 1e-4 * (rng.normal(size=len(f)) + 1j * rng.normal(size=len(f)))
    single = analysis.fit_trace(f, z, REFERENCE, 80e-9, allow_two_pole=False)
    assert single["status"] == "unreliable"
    fitted = analysis.fit_trace(f, z, REFERENCE, 80e-9)
    assert fitted["status"] == "ok"
    assert fitted["model"] == "two_complex_poles_with_delay"
    assert fitted["linewidth_hz"] == pytest.approx(0.5e6, rel=0.01)


def test_insufficient_window_not_reported_as_reliable():
    f = np.linspace(REFERENCE - 0.6e6, REFERENCE + 0.6e6, 101)
    fit = analysis.fit_trace(f, synthetic(f), REFERENCE, 80e-9, allow_two_pole=False)
    assert fit["status"] == "unreliable"
    assert "insufficient_fit_window_coverage" in fit["reasons"]


def test_long_spectroscopy_integration_does_not_require_readout_lrt():
    platform = demo_platform(1)
    q = platform.qpu.quantum_elements[0]
    f = np.linspace(REFERENCE - 2e6, REFERENCE + 2e6, 101)
    exp = experiment.create_experiment.func(
        platform.qpu,
        q,
        f,
        [0.001],
        REFERENCE,
        options=experiment.LinewidthExperimentOptions(integration_length=16e-6, count=2),
    )
    compiled = Session(platform.setup, configure_logging=False).compile(exp)
    assert not compiled.scheduled_experiment.artifacts.requires_long_readout
    acquire_lengths = []

    def walk(section):
        for child in getattr(section, "children", []):
            if type(child).__name__ == "Acquire":
                acquire_lengths.append(child.length)
                assert child.kernel is None
            walk(child)

    for section in exp.sections:
        walk(section)
    assert acquire_lengths == [16e-6]


def test_analysis_folderstore_persists_power_fits(tmp_path, caplog):
    q = demo_platform(1).qpu.quantum_elements[0]
    q.parameters.readout_phase_delay = 80e-9
    f = np.linspace(REFERENCE - 4e6, REFERENCE + 4e6, 101)
    a = np.array([0.001, 0.002])
    raw = Results(
        acquired_results={
            dsl.handles.result_handle(q.uid): AcquiredResult(
                data=np.stack([synthetic(f) * amp for amp in a]),
                axis=[a, f],
                axis_name=["amplitudes", "frequencies"],
            )
        }
    )
    opts = analysis.analysis_workflow.options()
    opts.logstore([workflow.logbook.FolderStore(str(tmp_path / "logs"))])
    opts.do_plotting(False)
    output = analysis.analysis_workflow(raw, q, f, a, REFERENCE, options=opts).run().output
    assert len(output["power_points"]) == 2
    assert not any("not supported by the serializer" in r.message for r in caplog.records)
    assert list((tmp_path / "logs").rglob("*power-points*"))


def test_backend_overrange_blocks_fit_and_retains_raw(monkeypatch):
    import logging

    marker = Results(acquired_results={})

    @workflow.task(save=False)
    def acquire(session, compiled_experiment):
        logging.getLogger("laboneq.controller.devices.device_zi").warning(
            "Channel 0 Input overrange count: 17"
        )
        return marker

    monkeypatch.setattr(experiment, "run_experiment", acquire)
    logger = logging.getLogger("laboneq")
    before = list(logger.handlers)
    acquired = experiment.acquire_cw.func(None, None)
    assert list(logger.handlers) == before
    assert acquired["raw_result"] is marker
    with pytest.raises(ValueError, match="overrange"):
        experiment.validated_acquisition.func(acquired)

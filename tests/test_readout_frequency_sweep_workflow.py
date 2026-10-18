"""Execute readout-frequency workflows with UID inputs without hardware."""

import numpy as np
import pytest
from laboneq import workflow
from laboneq.dsl.result import AcquiredResult, Results
from laboneq.dsl.session import Session
from laboneq.simple import dsl
from laboneq.workflow.reference import Reference

from qubit_experiment.experiments import readout_frequency_sweep
from qubit_experiment.qpu_types.fixed_transmon.demo_qpus import demo_platform


def _assert_resolved(value):
    assert not isinstance(value, Reference)
    if isinstance(value, dict):
        for item in value.values():
            _assert_resolved(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_resolved(item)


@pytest.mark.parametrize("parameter_object", [False, True])
@pytest.mark.parametrize("analyze,update", [(False, False), (True, False), (True, True)])
def test_uid_sweep_runs_and_preserves_temporary_parameters(
    monkeypatch, parameter_object, analyze, update
):
    platform = demo_platform(1)
    qpu = platform.qpu
    q = qpu.quantum_elements[0]
    original_frequency = q.parameters.readout_resonator_frequency
    frequencies = original_frequency + np.array([-1e6, 0.25e6, 1e6])
    overrides = {"readout_range_out": 0, "readout_amplitude": 0.012}
    temporary = q.parameters.replace(**overrides) if parameter_object else overrides
    session = Session(platform.setup, configure_logging=False)
    compile_original = session.compile
    compiled_frequencies = []

    def compile_offline(*, experiment):
        calibration = experiment.get_calibration()[q.signals["measure"]]
        assert calibration.range == overrides["readout_range_out"]
        compiled_frequencies.append(
            calibration.local_oscillator.frequency + calibration.oscillator.frequency
        )
        return compile_original(experiment=experiment)

    monkeypatch.setattr(session, "compile", compile_offline)

    @workflow.task(save=False)
    def acquire_synthetic_shots(session, compiled_experiment):
        index = len(compiled_frequencies) - 1
        rng = np.random.default_rng(100 + index)
        separation = 3.0 if index == 1 else 0.2
        shots = {}
        for state, mean in (("g", 0.0), ("e", separation)):
            samples = mean + rng.normal(0, 0.2, 128) + 1j * rng.normal(0, 0.2, 128)
            shots[dsl.handles.calibration_trace_handle(q.uid, state)] = AcquiredResult(
                data=samples, axis_name=["shot"], axis=[np.arange(128)]
            )
        return Results(acquired_results=shots)

    monkeypatch.setattr(
        readout_frequency_sweep, "_run_experiment_no_log", acquire_synthetic_shots
    )
    options = readout_frequency_sweep.experiment_workflow.options()
    options.logstore([])
    options.count(8)
    options.do_analysis(analyze)
    options.update(update)
    options.do_plotting(False)
    options.bootstrap_samples(0)
    result = readout_frequency_sweep.experiment_workflow(
        session=session,
        qpu=qpu,
        qubit=q.uid,
        frequencies=frequencies,
        temporary_parameters={q.uid: temporary},
        options=options,
    ).run()

    np.testing.assert_allclose(compiled_frequencies, frequencies)
    _assert_resolved(result.output)
    assert result.output == {"status": "completed"}
    if analyze:
        metrics = result.tasks["analysis_readout_frequency_sweep"].output
        _assert_resolved(metrics)
        np.testing.assert_allclose(metrics["sweep_points"], frequencies)
        assert metrics["new_parameter_values"][q.uid][
            "readout_resonator_frequency"
        ] == frequencies[1]
    expected = frequencies[1] if update else original_frequency
    assert q.parameters.readout_resonator_frequency == expected
    assert q.parameters.readout_range_out != overrides["readout_range_out"]

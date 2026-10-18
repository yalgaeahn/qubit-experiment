"""Cable delay correction and three-panel analysis without instrument access."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from laboneq.dsl.result import AcquiredResult, Results
from laboneq.simple import dsl

from qubit_experiment.analysis import resonator_spectroscopy as one_d
from qubit_experiment.analysis import resonator_spectroscopy_amplitude as two_d
from qubit_experiment.analysis.dispersive_shift import correct_readout_phase_delay
from qubit_experiment.qpu_types.fixed_transmon.demo_qpus import demo_platform


@pytest.mark.parametrize("tau", [80e-9, 0.0, None])
@pytest.mark.parametrize("amplitude_sweep", [False, True])
def test_analysis_workflow_three_panels(tau, amplitude_sweep):
    q = demo_platform(1).qpu.quantum_elements[0]
    q.parameters.readout_phase_delay = tau
    frequencies = np.linspace(7e9, 7.01e9, 31)
    amplitudes = np.array([0.1, 0.4, 0.7])
    expected_phase = 0.2
    trace = np.exp(
        1j * (expected_phase - 2 * np.pi * (frequencies - frequencies.mean()) * (tau or 0))
    )
    raw = amplitudes[:, None] * trace if amplitude_sweep else trace
    original = raw.copy()
    result = Results(
        acquired_results={
            dsl.handles.result_handle(q.uid): AcquiredResult(
                data=raw,
                axis_name=["amplitude", "frequency"] if amplitude_sweep else ["frequency"],
                axis=[amplitudes, frequencies] if amplitude_sweep else [frequencies],
            )
        }
    )
    module = two_d if amplitude_sweep else one_d
    options = module.analysis_workflow.options()
    options.logstore([])
    options.do_raw_data_plotting(False)
    options.save_figures(False)
    options.close_figures(False)
    if amplitude_sweep:
        run = module.analysis_workflow(
            result, q, frequencies, amplitudes, "Frequency (GHz)", "Amplitude", options=options
        ).run()
        fig = run.tasks["plot_signal_magnitude_and_phase_2d"].output[q.uid]
        axes = [ax for ax in fig.axes if ax.get_title()]
        assert len(axes) == 3
        if tau is not None:
            phase = np.asarray(axes[2].collections[0].get_array()).reshape(raw.shape)
            np.testing.assert_allclose(phase, expected_phase, atol=1e-12)
        else:
            assert "unavailable" in axes[2].texts[0].get_text()
    else:
        options.do_plotting_real_imaginary(False)
        run = module.analysis_workflow(result, q, frequencies, options=options).run()
        fig = run.tasks["plot_magnitude_phase"].output
        assert len(fig.axes) == 3
        data = run.tasks["calculate_signal_magnitude_and_phase"].output
        assert data["phase_delay"]["applied"] == (tau is not None)
        if tau is not None:
            np.testing.assert_allclose(fig.axes[2].lines[0].get_ydata(), expected_phase, atol=1e-12)
        else:
            assert data["phase_corrected"] is None
            assert "unavailable" in fig.axes[2].texts[0].get_text()
    np.testing.assert_array_equal(raw, original)
    plt.close(fig)


def test_phase_correction_checks_frequency_axis():
    q = demo_platform(1).qpu.quantum_elements[0]
    with pytest.raises(ValueError, match="last data axis"):
        correct_readout_phase_delay(q, [1, 2], np.ones((2, 3)))


@pytest.mark.parametrize("amplitude_sweep", [False, True])
def test_experiment_uses_temporary_delay(monkeypatch, amplitude_sweep):
    from laboneq import workflow
    from laboneq.dsl.session import Session

    from qubit_experiment.experiments import (
        resonator_spectroscopy,
        resonator_spectroscopy_amplitude,
    )

    platform = demo_platform(1)
    q = platform.qpu.quantum_elements[0]
    q.parameters.readout_phase_delay = None
    tau = 80e-9
    freqs = q.parameters.readout_resonator_frequency + np.linspace(-1e6, 1e6, 11)
    amps = np.array([0.1, 0.3])
    trace = np.exp(-2j * np.pi * (freqs - freqs.mean()) * tau)
    raw = amps[:, None] * trace if amplitude_sweep else trace
    acquired = AcquiredResult(
        data=raw,
        axis_name=["amplitudes", "frequencies"] if amplitude_sweep else ["frequencies"],
        axis=[amps, freqs] if amplitude_sweep else [freqs],
    )

    @workflow.task(save=False)
    def run_synthetic(session, compiled_experiment):
        return Results(acquired_results={dsl.handles.result_handle(q.uid): acquired})

    module = resonator_spectroscopy_amplitude if amplitude_sweep else resonator_spectroscopy
    monkeypatch.setattr(module, "run_experiment", run_synthetic)
    opts = module.experiment_workflow.options()
    opts.logstore([])
    opts.count(4)
    opts.update(False)
    opts.use_cw(False)
    opts.do_raw_data_plotting(False)
    opts.save_figures(False)
    opts.close_figures(False)
    kwargs = {}
    if amplitude_sweep:
        kwargs["amplitudes"] = amps
    else:
        opts.do_plotting_real_imaginary(False)
    run = module.experiment_workflow(
        session=Session(platform.setup, configure_logging=False),
        qpu=platform.qpu,
        qubit=q.uid,
        frequencies=freqs,
        temporary_parameters={q.uid: {"readout_phase_delay": tau}},
        options=opts,
        **kwargs,
    ).run()
    analysis = run.tasks["analysis_workflow"]
    if amplitude_sweep:
        fig = analysis.tasks["plot_signal_magnitude_and_phase_2d"].output[q.uid]
        corrected = fig.axes[2].collections[0].get_array()
    else:
        fig = analysis.tasks["plot_magnitude_phase"].output
        corrected = fig.axes[2].lines[0].get_ydata()
    np.testing.assert_allclose(corrected, 0, atol=1e-12)
    assert q.parameters.readout_phase_delay is None
    plt.close(fig)

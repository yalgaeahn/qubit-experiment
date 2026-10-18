"""CW reflection linewidth measurement for one qubit at a time.

Sweep absolute CW amplitude (near time) and frequency (real time), settle, acquire.
Only SPECTROSCOPY/CYCLIC is accepted; no qubit drive or permanent QPU updates.
Dedicated analysis reports the loaded readout-mode Q and kappa in rad/s (kappa/2pi
in Hz), including delay, fit-window and low-power convergence diagnostics.
"""

from __future__ import annotations

import logging

import numpy as np
from laboneq import workflow
from laboneq.simple import AcquisitionType, AveragingMode, Experiment, SweepParameter, dsl
from laboneq.workflow.tasks import compile_experiment, run_experiment
from laboneq_applications.core import validation
from laboneq_applications.experiments.options import BaseExperimentOptions
from laboneq_applications.tasks.parameter_updating import (
    temporary_qpu,
    temporary_quantum_elements_from_qpu,
)

from qubit_experiment.analysis.resonator_linewidth_cw import analysis_workflow


@workflow.workflow_options
class LinewidthWorkflowOptions:
    do_analysis: bool = workflow.option_field(True)
    update: bool = workflow.option_field(False)


@workflow.task_options(base_class=BaseExperimentOptions)
class LinewidthExperimentOptions:
    acquisition_type: AcquisitionType = workflow.option_field(AcquisitionType.SPECTROSCOPY)
    averaging_mode: AveragingMode = workflow.option_field(AveragingMode.CYCLIC)
    count: int = workflow.option_field(2048)
    integration_length: float = workflow.option_field(2e-6)
    settling_time: float = workflow.option_field(5e-6)


def validate_sweeps(frequencies, amplitudes, reference_frequency_hz):
    f = np.asarray(frequencies, dtype=float)
    a = np.asarray(amplitudes, dtype=float)
    if f.ndim != 1 or f.size < 21 or not np.all(np.isfinite(f)) or np.any(f <= 0):
        raise ValueError("frequencies must contain at least 21 positive finite points.")
    if not (np.all(np.diff(f) > 0) or np.all(np.diff(f) < 0)):
        raise ValueError("frequencies must be strictly monotonic.")
    if a.ndim != 1 or a.size == 0 or not np.all(np.isfinite(a)) or np.any((a <= 0) | (a > 1)):
        raise ValueError("amplitudes must be a nonempty 1D array in (0, 1].")
    if len(np.unique(a)) != len(a):
        raise ValueError("amplitudes must be unique.")
    if not np.isfinite(reference_frequency_hz) or not f.min() < reference_frequency_hz < f.max():
        raise ValueError("reference_frequency_hz must be strictly inside the frequency sweep.")
    return f, a


@workflow.task(save=False)
def validate_request(qubit, frequencies, amplitudes, reference_frequency_hz, update):
    if not isinstance(qubit, str):
        raise ValueError("Pass one qubit UID string; measure multiple qubits sequentially.")
    if update:
        raise ValueError("resonator_linewidth_cw does not support persistent updates.")
    validate_sweeps(frequencies, amplitudes, reference_frequency_hz)
    return qubit


@workflow.task
def acquire_cw(session, compiled_experiment):
    """Retain raw acquisition and flag backend overrange warnings before fitting."""
    messages = []

    class CaptureOverrange(logging.Handler):
        def emit(self, record):
            if "overrange count" in record.getMessage().lower():
                messages.append(record.getMessage())

    logger = logging.getLogger("laboneq")
    handler = CaptureOverrange()
    logger.addHandler(handler)
    try:
        raw = run_experiment.func(session, compiled_experiment)
    finally:
        logger.removeHandler(handler)
    return {"raw_result": raw, "overrange_warnings": tuple(messages)}


@workflow.task(save=False)
def validated_acquisition(acquisition):
    if acquisition["overrange_warnings"]:
        raise ValueError(
            "ADC/DAC overrange detected; raw data retained in acquire_cw log, "
            "but excluded from linewidth fitting: " + "; ".join(acquisition["overrange_warnings"])
        )
    return acquisition["raw_result"]


@workflow.workflow(name="resonator_linewidth_cw")
def experiment_workflow(
    session,
    qpu,
    qubit,
    frequencies,
    amplitudes,
    reference_frequency_hz,
    temporary_parameters=None,
    options: LinewidthWorkflowOptions | None = None,
):
    """Acquire CW data and return resolved raw data plus optional analysis.

    Amplitudes are oscillator gains, independent of the qubit's readout_amplitude.
    Port power is readout_range_out + 20*log10(amplitude), not chip input power.
    """
    uid = validate_request(qubit, frequencies, amplitudes, reference_frequency_hz, options.update)
    temp_qpu = temporary_qpu(qpu, temporary_parameters)
    q = temporary_quantum_elements_from_qpu(temp_qpu, uid)
    exp = create_experiment(temp_qpu, q, frequencies, amplitudes, reference_frequency_hz)
    compiled = compile_experiment(session, exp)
    acquisition = acquire_cw(session, compiled)
    raw = validated_acquisition(acquisition)
    with workflow.if_(options.do_analysis):
        analysis = analysis_workflow(raw, q, frequencies, amplitudes, reference_frequency_hz)
        workflow.return_(
            bundle_result(raw, q, frequencies, amplitudes, reference_frequency_hz, analysis.output)
        )
    with workflow.else_():
        workflow.return_(bundle_result(raw, q, frequencies, amplitudes, reference_frequency_hz))


@workflow.task(save=False)
def bundle_result(raw, qubit, frequencies, amplitudes, reference_frequency_hz, analysis=None):
    return {
        "raw_result": raw,
        "analysis": analysis,
        "qubit_uid": qubit.uid,
        "frequencies_hz": np.asarray(frequencies),
        "amplitudes": np.asarray(amplitudes),
        "reference_frequency_hz": float(reference_frequency_hz),
        "readout_range_out_dbm": qubit.parameters.readout_range_out,
    }


@workflow.task
@dsl.qubit_experiment
def create_experiment(
    qpu,
    qubit,
    frequencies,
    amplitudes,
    reference_frequency_hz,
    options: LinewidthExperimentOptions | None = None,
) -> Experiment:
    opts = LinewidthExperimentOptions() if options is None else options
    qubit = validation.validate_and_convert_single_qubit_sweeps(qubit)
    f, a = validate_sweeps(frequencies, amplitudes, reference_frequency_hz)
    if AcquisitionType(opts.acquisition_type) != AcquisitionType.SPECTROSCOPY:
        raise ValueError("CW linewidth requires acquisition_type=SPECTROSCOPY.")
    if AveragingMode(opts.averaging_mode) != AveragingMode.CYCLIC:
        raise ValueError("CW linewidth requires averaging_mode=CYCLIC.")
    if opts.active_reset:
        raise ValueError("CW linewidth performs no qubit preparation or active reset.")
    if opts.count < 1 or not isinstance(opts.count, (int, np.integer)):
        raise ValueError("count must be a positive integer.")
    if not np.isfinite(opts.integration_length) or opts.integration_length <= 0:
        raise ValueError("integration_length must be positive and finite.")
    if not np.isfinite(opts.settling_time) or opts.settling_time < 0:
        raise ValueError("settling_time must be nonnegative and finite.")
    # The copy is local to this builder; original/temporary QPU remains untouched.
    q = qubit.replace(
        readout_integration_length=opts.integration_length,
        readout_integration_kernels=None,
        readout_integration_kernels_type="default",
    )
    qop = qpu.quantum_operations
    cal = dsl.experiment_calibration()
    # CW acquisition is not aligned to a pulse; do not inherit calibrated readout delay.
    cal[q.signals["acquire"]].port_delay = 0.0
    cal[q.signals["acquire"]].oscillator = cal[q.signals["measure"]].oscillator
    # CW has no play pulse to open a pulse-based automatic output mute gate.
    cal[q.signals["measure"]].automute = False
    with dsl.sweep(parameter=SweepParameter(f"amplitudes_{q.uid}", a)) as amplitude:
        with dsl.acquire_loop_rt(
            count=opts.count,
            acquisition_type=opts.acquisition_type,
            averaging_mode=opts.averaging_mode,
            repetition_mode=opts.repetition_mode,
            repetition_time=opts.repetition_time,
            reset_oscillator_phase=opts.reset_oscillator_phase,
        ):
            with dsl.sweep(parameter=SweepParameter(f"frequencies_{q.uid}", f)) as frequency:
                qop.set_readout_amplitude(q, amplitude=amplitude)
                qop.set_frequency(q, frequency=frequency, readout=True)
                qop.delay(q, opts.settling_time)
                # Spectroscopy uses its own hardware integrator, not readout
                # weights. A long default kernel incorrectly requests LRT even
                # for CW spectroscopy on a device without that readout option.
                with dsl.section(name=f"cw_acquire_{q.uid}"):
                    dsl.acquire(
                        signal=q.signals["acquire"],
                        handle=dsl.handles.result_handle(q.uid),
                        length=opts.integration_length,
                    )

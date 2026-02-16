"""Experiment workflow for readout-line MID sweep.

The experiment applies two readout pulses in each delay point:
1) readout kick on the measure line without acquisition,
2) final readout with acquisition.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from laboneq import workflow
from laboneq.dsl.enums import AcquisitionType, AveragingMode
from laboneq.simple import Experiment, SectionAlignment, SweepParameter, dsl
from laboneq.workflow.tasks import compile_experiment, run_experiment

from analysis.readout_mid_sweep import analysis_workflow
from laboneq_applications.core import validation
from laboneq_applications.experiments.options import BaseExperimentOptions
from laboneq_applications.tasks.parameter_updating import (
    temporary_qpu,
    temporary_quantum_elements_from_qpu,
    update_qpu,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from laboneq.dsl.quantum import QuantumParameters
    from laboneq.dsl.quantum.qpu import QPU
    from laboneq.dsl.quantum.quantum_element import QuantumElement
    from laboneq.dsl.session import Session
    from numpy.typing import ArrayLike


@workflow.task_options(base_class=BaseExperimentOptions)
class ReadoutMidSweepExperimentOptions:
    """Options for readout-line MID acquisition."""

    count: int = workflow.option_field(
        1024,
        description="Number of single shots per delay point.",
    )
    acquisition_type: AcquisitionType = workflow.option_field(
        AcquisitionType.INTEGRATION,
        description="Acquire integrated complex IQ samples.",
    )
    averaging_mode: AveragingMode = workflow.option_field(
        AveragingMode.SINGLE_SHOT,
        description="Single-shot acquisition mode.",
    )
    transition: Literal["ge"] = workflow.option_field(
        "ge",
        description="Transition used for Ramsey pulses in readout-line MID.",
    )
    use_cal_traces: bool = workflow.option_field(
        True,
        description="Whether to include calibration traces in the experiment.",
    )
    cal_states: str | tuple[str, ...] = workflow.option_field(
        "ge",
        description="Calibration states for trace-based rotation/projection.",
    )


@workflow.workflow_options
class ReadoutMidSweepWorkflowOptions:
    """Workflow options for readout-MID sweep."""

    do_analysis: bool = workflow.option_field(True)
    update: bool = workflow.option_field(False)


def _coerce_points(points: ArrayLike | None, name: str) -> np.ndarray:
    arr = np.asarray(points if points is not None else [], dtype=float).reshape(-1)
    if arr.size < 1:
        raise ValueError(f"{name} must contain at least one value.")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain finite values.")
    return arr


@workflow.task(save=False)
def _coerce_mid_frequencies(
    sweep_axis: Literal["frequency", "amplitude", "both"],
    frequencies: ArrayLike | None,
) -> list[float]:
    if sweep_axis == "frequency":
        return [float(x) for x in _coerce_points(frequencies, "readout_resonator_frequencies")]
    if sweep_axis == "both":
        return [float(x) for x in _coerce_points(frequencies, "readout_resonator_frequencies")]
    if sweep_axis == "amplitude":
        return [float("nan")]
    raise ValueError("sweep_axis must be 'frequency', 'amplitude', or 'both'.")


@workflow.task(save=False)
def _coerce_mid_amplitudes(
    sweep_axis: Literal["frequency", "amplitude", "both"],
    amplitudes: ArrayLike | None,
) -> list[float]:
    if sweep_axis == "amplitude":
        return [float(x) for x in _coerce_points(amplitudes, "readout_amplitudes")]
    if sweep_axis == "both":
        return [float(x) for x in _coerce_points(amplitudes, "readout_amplitudes")]
    if sweep_axis == "frequency":
        return [float("nan")]
    raise ValueError("sweep_axis must be 'frequency', 'amplitude', or 'both'.")


@workflow.task(save=False)
def _coerce_delays(delays: ArrayLike) -> list[float]:
    points = np.asarray(delays, dtype=float).reshape(-1)
    if points.size < 1:
        raise ValueError("delays must contain at least one value.")
    if not np.isfinite(points).all():
        raise ValueError("delays must contain finite values.")
    if np.any(points < 0):
        raise ValueError("delays must be non-negative.")
    return [float(x) for x in points]


@workflow.task(save=False)
def _merge_temp_parameters(
    base: dict[str | tuple[str, str, str], dict | QuantumParameters] | None,
    qubit_uid: str,
    readout_frequency: float | None = None,
    readout_amplitude: float | None = None,
) -> dict[str | tuple[str, str, str], dict | QuantumParameters]:
    merged: dict[str | tuple[str, str, str], dict | QuantumParameters] = {}
    if base is not None:
        for key, value in base.items():
            merged[key] = dict(value) if isinstance(value, dict) else value

    per_qubit = merged.get(qubit_uid, {})
    if not isinstance(per_qubit, dict):
        per_qubit = {}
    per_qubit = dict(per_qubit)
    if readout_frequency is not None:
        per_qubit["readout_resonator_frequency"] = float(readout_frequency)
    if readout_amplitude is not None:
        per_qubit["readout_amplitude"] = float(readout_amplitude)
    merged[qubit_uid] = per_qubit
    return merged


@workflow.task(save=False)
def _coalesce_readout_point(value: float) -> float | None:
    return None if not np.isfinite(value) else float(value)


@workflow.task(save=False)
def _append_item(items: list, item) -> None:
    items.append(item)


@workflow.task(save=False)
def _materialize_list(items: list) -> list:
    return list(items)


@workflow.workflow(name="readout_mid_sweep")
def experiment_workflow(
    session: Session,
    qpu: "QPU",
    qubit: QuantumElement,
    delays: ArrayLike,
    readout_resonator_frequencies: ArrayLike | None = None,
    readout_amplitudes: ArrayLike | None = None,
    sweep_axis: Literal["frequency", "amplitude", "both"] = "both",
    temporary_parameters: dict[str | tuple[str, str, str], dict | QuantumParameters]
    | None = None,
    options: ReadoutMidSweepWorkflowOptions | None = None,
) -> None:
    """Run candidate sweep for readout-line MID validation."""
    options = ReadoutMidSweepWorkflowOptions() if options is None else options
    base_temp_qpu = temporary_qpu(qpu, temporary_parameters)
    base_qubit = temporary_quantum_elements_from_qpu(base_temp_qpu, qubit)
    freq_points = _coerce_mid_frequencies(
        sweep_axis=sweep_axis,
        frequencies=readout_resonator_frequencies,
    )
    amp_points = _coerce_mid_amplitudes(
        sweep_axis=sweep_axis,
        amplitudes=readout_amplitudes,
    )
    delay_points = _coerce_delays(delays)

    results: list = []
    temp_qubits: list = []

    with workflow.for_(freq_points, lambda x: x) as readout_frequency:
        with workflow.for_(amp_points, lambda x: x) as readout_amplitude:
            run_frequency = _coalesce_readout_point(readout_frequency)
            run_amplitude = _coalesce_readout_point(readout_amplitude)
            per_run_params = _merge_temp_parameters(
                base=temporary_parameters,
                qubit_uid=qubit.uid,
                readout_frequency=run_frequency,
                readout_amplitude=run_amplitude,
            )
            per_run_temp_qpu = temporary_qpu(qpu, per_run_params)
            per_run_qubit = temporary_quantum_elements_from_qpu(
                per_run_temp_qpu,
                qubit,
            )
            exp = create_experiment(
                qpu=per_run_temp_qpu,
                qubit=per_run_qubit,
                delays=delay_points,
            )
            compiled = compile_experiment(session, exp)
            run = run_experiment(session, compiled)
            _append_item(results, run)
            _append_item(temp_qubits, per_run_qubit)

    collected_results = _materialize_list(results)
    collected_qubits = _materialize_list(temp_qubits)

    analysis_result = None
    with workflow.if_(options.do_analysis):
        analysis_result = analysis_workflow(
            results=collected_results,
            qubits=collected_qubits,
            delays=delay_points,
            frequency_points=freq_points,
            amplitude_points=amp_points,
            sweep_axis=sweep_axis,
            reference_qubit=base_qubit,
        )
        with workflow.if_(options.update):
            update_qpu(qpu, analysis_result.output["new_parameter_values"])

    workflow.return_({"analysis_result": analysis_result})


@workflow.task
@dsl.qubit_experiment
def create_experiment(
    qpu: "QPU",
    qubit: QuantumElement,
    delays: ArrayLike,
    options: ReadoutMidSweepExperimentOptions | None = None,
) -> Experiment:
    """Create a readout-line MID experiment with 2x readout pulses per point."""
    opts = ReadoutMidSweepExperimentOptions() if options is None else options
    if opts.transition != "ge":
        raise ValueError("readout_mid_sweep currently supports transition='ge' only.")
    if AcquisitionType(opts.acquisition_type) != AcquisitionType.INTEGRATION:
        raise ValueError("readout_mid_sweep requires acquisition_type=INTEGRATION.")
    if AveragingMode(opts.averaging_mode) != AveragingMode.SINGLE_SHOT:
        raise ValueError("readout_mid_sweep requires averaging_mode=SINGLE_SHOT.")
    if (
        opts.use_cal_traces
        and AveragingMode(opts.averaging_mode) == AveragingMode.SEQUENTIAL
    ):
        raise ValueError(
            "'AveragingMode.SEQUENTIAL' cannot be used with calibration traces."
        )

    qubit, delays = validation.validate_and_convert_single_qubit_sweeps(qubit, delays)
    delays = np.asarray(delays, dtype=float).reshape(-1)
    if delays.size < 1:
        raise ValueError("delays must contain at least one value.")
    if np.any(delays < 0):
        raise ValueError("delays must be non-negative.")

    half_delays = 0.5 * delays
    half_delay_sweep = SweepParameter(
        uid=f"mid_half_delay_{qubit.uid}",
        values=half_delays,
    )
    qop = qpu.quantum_operations
    measure_section_length = qop.measure_section_length(qubit)
    with dsl.acquire_loop_rt(
        count=opts.count,
        averaging_mode=opts.averaging_mode,
        acquisition_type=opts.acquisition_type,
        repetition_mode=opts.repetition_mode,
        repetition_time=opts.repetition_time,
        reset_oscillator_phase=opts.reset_oscillator_phase,
    ):
        with dsl.sweep(
            name=f"readout_mid_delay_sweep_{qubit.uid}",
            parameter=half_delay_sweep,
        ):
            if opts.active_reset:
                qop.active_reset(
                    qubit,
                    active_reset_states=opts.active_reset_states,
                    number_resets=opts.active_reset_repetitions,
                    measure_section_length=measure_section_length,
                )

            qop.prepare_state.omit_section(qubit, state="g")
            sec_x90_1 = qop.x90(qubit, transition=opts.transition)
            with dsl.section(
                name=f"mid_delay_1_{qubit.uid}",
                alignment=SectionAlignment.LEFT,
                play_after=sec_x90_1.uid,
            ) as sec_delay_1:
                qop.delay.omit_section(qubit, time=half_delay_sweep)
            with dsl.section(
                name=f"mid_readout_kick_{qubit.uid}",
                alignment=SectionAlignment.LEFT,
                play_after=sec_delay_1.uid,
            ) as sec_kick:
                qop.measure.omit_section(
                    qubit,
                    handle=dsl.handles.result_handle(qubit.uid, suffix="kick"),
                )
                sec_kick.length = measure_section_length
            with dsl.section(
                name=f"mid_delay_2_{qubit.uid}",
                alignment=SectionAlignment.LEFT,
                play_after=sec_kick.uid,
            ) as sec_delay_2:
                qop.delay.omit_section(qubit, time=half_delay_sweep)
            with dsl.section(
                name=f"mid_x90_2_{qubit.uid}",
                alignment=SectionAlignment.RIGHT,
                play_after=sec_delay_2.uid,
            ) as sec_x90_2:
                qop.x90.omit_section(qubit, transition=opts.transition)
            with dsl.section(
                name=f"mid_measure_{qubit.uid}",
                alignment=SectionAlignment.LEFT,
                play_after=sec_x90_2.uid,
            ) as sec_measure:
                qop.measure.omit_section(
                    qubit,
                    handle=dsl.handles.result_handle(qubit.uid),
                )
                sec_measure.length = measure_section_length
            with dsl.section(
                name=f"mid_reset_{qubit.uid}",
                alignment=SectionAlignment.LEFT,
                play_after=sec_measure.uid,
            ):
                qop.passive_reset.omit_section(qubit)

        if opts.use_cal_traces:
            qop.calibration_traces.omit_section(
                qubits=qubit,
                states=opts.cal_states,
                active_reset=opts.active_reset,
                active_reset_states=opts.active_reset_states,
                active_reset_repetitions=opts.active_reset_repetitions,
                measure_section_length=measure_section_length,
            )

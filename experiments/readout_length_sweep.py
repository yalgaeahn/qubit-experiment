"""Experiment workflow for readout pulse-length optimization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from laboneq import workflow
from laboneq.dsl.enums import AcquisitionType, AveragingMode
from laboneq.simple import Experiment, dsl
from laboneq.workflow.tasks import compile_experiment, run_experiment

from analysis.readout_length_sweep import analysis_workflow
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
class ReadoutLengthSweepExperimentOptions:
    """Options for readout length sweep acquisition."""

    count: int = workflow.option_field(
        4096,
        description="Number of single shots per prepared state.",
    )
    acquisition_type: AcquisitionType = workflow.option_field(
        AcquisitionType.INTEGRATION,
        description="Acquire integrated complex IQ samples.",
    )
    averaging_mode: AveragingMode = workflow.option_field(
        AveragingMode.SINGLE_SHOT,
        description="Single-shot acquisition mode.",
    )
    states: str | tuple[str, ...] = workflow.option_field(
        "ge",
        description="Calibration states. Only g/e is supported in this workflow.",
    )


@workflow.workflow_options
class ReadoutLengthSweepWorkflowOptions:
    """Workflow options for readout length optimization."""

    do_analysis: bool = workflow.option_field(
        True, description="Whether to run analysis."
    )
    update: bool = workflow.option_field(
        False, description="Whether to apply optimized parameters to the input qpu."
    )
    max_readout_length: float | None = workflow.option_field(
        None,
        description=(
            "Optional hard upper bound for readout_length sweep points in seconds. "
            "Use this to enforce hardware waveform limits before run submission."
        ),
    )


def _states_to_tuple(states: str | Sequence[str]) -> tuple[str, ...]:
    state_tuple = tuple(states) if isinstance(states, str) else tuple(states)
    if len(state_tuple) != 2 or set(state_tuple) != {"g", "e"}:
        raise ValueError("Only g/e states are supported. Use states='ge'.")
    return state_tuple


def _merged_temp_parameters(
    base: dict[str | tuple[str, str, str], dict | QuantumParameters] | None,
    qubit_uid: str,
    readout_length: float,
) -> dict[str | tuple[str, str, str], dict | QuantumParameters]:
    merged: dict[str | tuple[str, str, str], dict | QuantumParameters] = {}
    if base is not None:
        for k, v in base.items():
            merged[k] = dict(v) if isinstance(v, dict) else v
    per_qubit = merged.get(qubit_uid, {})
    if not isinstance(per_qubit, dict):
        per_qubit = {}
    per_qubit = dict(per_qubit)
    per_qubit["readout_length"] = float(readout_length)
    merged[qubit_uid] = per_qubit
    return merged


@workflow.task(save=False)
def _resolve_readout_lengths(
    readout_lengths: ArrayLike,
    max_readout_length: float | None = None,
) -> list[float]:
    points = np.asarray(readout_lengths, dtype=float).reshape(-1)
    if points.size < 1:
        raise ValueError("readout_lengths must contain at least one value.")
    if not np.isfinite(points).all():
        raise ValueError("readout_lengths must contain only finite values.")
    if max_readout_length is not None:
        max_len = float(max_readout_length)
        if max_len <= 0:
            raise ValueError("max_readout_length must be positive when specified.")
        if np.any(points > max_len):
            too_long = float(np.max(points))
            raise ValueError(
                "readout_lengths contains values exceeding max_readout_length: "
                f"max={too_long:.3e}s > limit={max_len:.3e}s."
            )
    return [float(x) for x in points]


@workflow.task(save=False)
def _validate_measure_section_limit(
    qubit: QuantumElement,
    max_readout_length: float | None = None,
) -> None:
    if max_readout_length is None:
        return
    max_len = float(max_readout_length)
    ro_int_delay = float(qubit.parameters.readout_integration_delay or 0.0)
    ro_int_length = float(qubit.parameters.readout_integration_length)
    min_section_len = ro_int_delay + ro_int_length
    if min_section_len > max_len:
        raise ValueError(
            "Integration window exceeds max_readout_length: "
            f"readout_integration_delay + readout_integration_length = "
            f"{min_section_len:.3e}s > limit={max_len:.3e}s."
        )


@workflow.task(save=False)
def _build_temp_params_for_readout_length(
    base: dict[str | tuple[str, str, str], dict | QuantumParameters] | None,
    qubit_uid: str,
    readout_length: float,
) -> dict[str | tuple[str, str, str], dict | QuantumParameters]:
    return _merged_temp_parameters(
        base=base,
        qubit_uid=qubit_uid,
        readout_length=float(readout_length),
    )


@workflow.task(save=False)
def _append_item(items: list, item) -> None:
    items.append(item)


@workflow.task(save=False)
def _materialize_list(items: list) -> list:
    return list(items)


@workflow.workflow(name="readout_length_sweep")
def experiment_workflow(
    session: Session,
    qpu: QPU,
    qubit: QuantumElement,
    readout_lengths: ArrayLike,
    temporary_parameters: dict[str | tuple[str, str, str], dict | QuantumParameters]
    | None = None,
    options: ReadoutLengthSweepWorkflowOptions | None = None,
) -> None:
    """Run repeated IQ-cloud acquisitions across readout-length candidates."""
    options = ReadoutLengthSweepWorkflowOptions() if options is None else options
    length_points = _resolve_readout_lengths(
        readout_lengths,
        max_readout_length=options.max_readout_length,
    )

    base_temp_qpu = temporary_qpu(qpu, temporary_parameters)
    base_qubit = temporary_quantum_elements_from_qpu(base_temp_qpu, qubit)
    _validate_measure_section_limit(
        qubit=base_qubit,
        max_readout_length=options.max_readout_length,
    )

    results = []
    temp_qubits = []
    with workflow.for_(length_points, lambda x: x) as readout_length:
        per_run_temp_parameters = _build_temp_params_for_readout_length(
            base=temporary_parameters,
            qubit_uid=qubit.uid,
            readout_length=readout_length,
        )
        per_run_qpu = temporary_qpu(qpu, per_run_temp_parameters)
        per_run_qubit = temporary_quantum_elements_from_qpu(per_run_qpu, qubit)
        exp = create_experiment(
            qpu=per_run_qpu,
            qubit=per_run_qubit,
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
            readout_lengths=length_points,
            reference_qubit=base_qubit,
        )
        with workflow.if_(options.update):
            update_qpu(qpu, analysis_result.output["new_parameter_values"])

    workflow.return_({"analysis_result": analysis_result})


@workflow.task
@dsl.qubit_experiment
def create_experiment(
    qpu: QPU,
    qubit: QuantumElement,
    options: ReadoutLengthSweepExperimentOptions | None = None,
) -> Experiment:
    """Create a calibration-trace experiment for a fixed readout length."""
    opts = ReadoutLengthSweepExperimentOptions() if options is None else options
    states = _states_to_tuple(opts.states)
    qubit = validation.validate_and_convert_qubits_sweeps(qubit)

    if AcquisitionType(opts.acquisition_type) != AcquisitionType.INTEGRATION:
        raise ValueError("readout_length_sweep requires acquisition_type=INTEGRATION.")
    if AveragingMode(opts.averaging_mode) != AveragingMode.SINGLE_SHOT:
        raise ValueError("readout_length_sweep requires averaging_mode=SINGLE_SHOT.")

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
        qop.calibration_traces.omit_section(
            qubits=qubit,
            states=states,
            active_reset=opts.active_reset,
            active_reset_states=opts.active_reset_states,
            active_reset_repetitions=opts.active_reset_repetitions,
            measure_section_length=measure_section_length,
        )

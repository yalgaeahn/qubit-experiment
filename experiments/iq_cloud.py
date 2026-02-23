"""Experiment workflow for IQ-cloud readout calibration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from laboneq import workflow
from laboneq.dsl.enums import AcquisitionType, AveragingMode
from laboneq.simple import Experiment, SectionAlignment, dsl
from laboneq.workflow.tasks import compile_experiment, run_experiment

from example_helpers.workflow.handles import calibration_trace_2q_handle
from analysis.iq_cloud import (
    analysis_workflow,
    collect_shots,
    extract_qubit_parameters_for_discrimination,
    fit_decision_models,
)
from experiments.iq_cloud_common import (
    prepared_labels_for_num_qubits,
    validate_supported_num_qubits,
)
from laboneq_applications.core import validation
from laboneq_applications.experiments.options import BaseExperimentOptions
from laboneq_applications.tasks.parameter_updating import (
    temporary_qpu,
    temporary_quantum_elements_from_qpu,
    update_qpu,
)

if TYPE_CHECKING:
    from laboneq.dsl.quantum import QuantumParameters
    from laboneq.dsl.quantum.qpu import QPU
    from laboneq.dsl.session import Session

    from laboneq_applications.typing import QuantumElements


@workflow.task_options(base_class=BaseExperimentOptions)
class IQCloudExperimentOptions:
    """Options for IQ-cloud experiment acquisition."""

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
        description="Use single-shot acquisition.",
    )


@workflow.workflow_options
class IQCloudWorkflowOptions:
    """Workflow options for IQ-cloud experiment."""

    do_analysis: bool = workflow.option_field(
        True,
        description="Whether to run IQ-cloud analysis.",
    )
    update: bool = workflow.option_field(
        False,
        description="Whether to update discrimination thresholds on the input qpu.",
    )
    enforce_constant_kernel: bool = workflow.option_field(
        True,
        description=(
            "Whether to enforce default integration kernels when applying threshold "
            "updates."
        ),
    )


@workflow.workflow(name="iq_cloud")
def experiment_workflow(
    session: Session,
    qpu: QPU,
    qubits: QuantumElements,
    temporary_parameters: dict[str | tuple[str, str, str], dict | QuantumParameters]
    | None = None,
    options: IQCloudWorkflowOptions | None = None,
) -> None:
    """Run IQ-cloud experiment (1Q or 2Q multiplexed) with optional analysis/update."""
    options = IQCloudWorkflowOptions() if options is None else options
    temp_qpu = temporary_qpu(qpu, temporary_parameters)
    qubits = temporary_quantum_elements_from_qpu(temp_qpu, qubits)

    exp = create_experiment(temp_qpu, qubits)
    compiled_exp = compile_experiment(session, exp)
    result = run_experiment(session, compiled_exp)

    analysis_result = None
    with workflow.if_(options.do_analysis):
        analysis_result = analysis_workflow(
            result=result,
            qubits=qubits,
        )

    with workflow.if_(options.update):
        processed_data = collect_shots(result=result, qubits=qubits)
        decision_model = fit_decision_models(processed_data=processed_data, qubits=qubits)
        qubit_parameters = extract_qubit_parameters_for_discrimination(
            qubits=qubits,
            decision_model=decision_model,
            enforce_constant_kernel=options.enforce_constant_kernel,
        )
        update_qpu(qpu, _new_parameter_values(qubit_parameters))

    workflow.return_({"result": result, "analysis_result": analysis_result})


@workflow.task
@dsl.qubit_experiment
def create_experiment(
    qpu: QPU,
    qubits: QuantumElements,
    options: IQCloudExperimentOptions | None = None,
) -> Experiment:
    """Create an IQ-cloud experiment for 1Q single or 2Q multiplexed readout."""
    opts = IQCloudExperimentOptions() if options is None else options
    qubits = validation.validate_and_convert_qubits_sweeps(qubits)
    num_qubits = len(qubits)
    validate_supported_num_qubits(num_qubits)
    prepared_labels = prepared_labels_for_num_qubits(num_qubits)

    qop = qpu.quantum_operations
    max_measure_section_length = qop.measure_section_length(qubits)

    with dsl.acquire_loop_rt(
        count=opts.count,
        averaging_mode=opts.averaging_mode,
        acquisition_type=opts.acquisition_type,
        repetition_mode=opts.repetition_mode,
        repetition_time=opts.repetition_time,
        reset_oscillator_phase=opts.reset_oscillator_phase,
    ):
        if num_qubits == 1:
            qop.calibration_traces.omit_section(
                qubits=qubits,
                states=prepared_labels,
                active_reset=opts.active_reset,
                active_reset_states=opts.active_reset_states,
                active_reset_repetitions=opts.active_reset_repetitions,
                measure_section_length=max_measure_section_length,
            )
        else:
            for prepared_label in prepared_labels:
                with dsl.section(
                    name=f"iq_cloud_{prepared_label}",
                    alignment=SectionAlignment.LEFT,
                ):
                    prep_play_after = None
                    if opts.active_reset:
                        active_reset_sec = qop.active_reset(
                            qubits,
                            active_reset_states=opts.active_reset_states,
                            number_resets=opts.active_reset_repetitions,
                            measure_section_length=max_measure_section_length,
                        )
                        prep_play_after = active_reset_sec.uid

                    prep_kwargs = {
                        "name": f"prep_{prepared_label}",
                        "alignment": SectionAlignment.LEFT,
                    }
                    if prep_play_after is not None:
                        prep_kwargs["play_after"] = prep_play_after

                    with dsl.section(**prep_kwargs) as prep_sec:
                        prev_uid = None
                        for idx, (q, state) in enumerate(zip(qubits, prepared_label)):
                            sec_kwargs = {
                                "name": f"prep_{q.uid}_{state}_{prepared_label}_{idx}",
                                "alignment": SectionAlignment.LEFT,
                            }
                            if prev_uid is not None:
                                sec_kwargs["play_after"] = prev_uid
                            with dsl.section(**sec_kwargs) as prep_single:
                                qop.prepare_state(q, state=state)
                            prev_uid = prep_single.uid

                    with dsl.section(
                        name=f"measure_{prepared_label}",
                        alignment=SectionAlignment.LEFT,
                        play_after=prep_sec.uid,
                    ):
                        for q in qubits:
                            sec = qop.measure(
                                q,
                                handle=calibration_trace_2q_handle(q.uid, prepared_label),
                            )
                            sec.length = max_measure_section_length
                        for q in qubits:
                            qop.passive_reset(q)


@workflow.task
def _new_parameter_values(qubit_parameters: dict) -> dict:
    """Select concrete update payload for update_qpu."""
    return qubit_parameters.get("new_parameter_values", {})

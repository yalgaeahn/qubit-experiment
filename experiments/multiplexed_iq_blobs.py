"""This module defines the experiments for 2Q multiplexed IQ blob threshold calibration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from laboneq import workflow
from laboneq.dsl.enums import AcquisitionType, AveragingMode
from laboneq.simple import Experiment, SectionAlignment, dsl
from laboneq.workflow.tasks import compile_experiment, run_experiment

from analysis.multiplexed_iq_blobs import analysis_workflow
from experiments.multiplexed_iq_blobs_common import (
    PREPARED_STATES_2Q,
    multiplexed_iq_blob_handle,
)
from laboneq_applications.core import validation
from laboneq_applications.experiments.options import BaseExperimentOptions
from laboneq_applications.tasks.parameter_updating import (
    temporary_qpu,
    temporary_quantum_elements_from_qpu,
)

if TYPE_CHECKING:
    from laboneq.dsl.quantum import QuantumParameters
    from laboneq.dsl.quantum.qpu import QPU
    from laboneq.dsl.session import Session

    from laboneq_applications.typing import QuantumElements


@workflow.task_options(base_class=BaseExperimentOptions)
class MultiplexedIQBlobExperimentOptions:
    """Options for 2Q multiplexed IQ blob experiment."""

    count: int = workflow.option_field(
        4096,
        description="Number of shots per prepared 2Q basis state.",
    )
    acquisition_type: AcquisitionType = workflow.option_field(
        AcquisitionType.INTEGRATION,
        description="Acquire integrated complex IQ values for threshold extraction.",
    )
    averaging_mode: AveragingMode = workflow.option_field(
        AveragingMode.SINGLE_SHOT,
        description="Use single-shot IQ points for threshold extraction.",
    )


@workflow.workflow_options
class MultiplexedIQBlobWorkflowOptions:
    """Workflow options for 2Q multiplexed IQ blob experiment."""

    do_analysis: bool = workflow.option_field(
        True,
        description="Whether to run multiplexed IQ blob analysis.",
    )


@workflow.workflow(name="multiplexed_iq_blobs")
def experiment_workflow(
    session: Session,
    qpu: QPU,
    ctrl: QuantumElements,
    targ: QuantumElements,
    temporary_parameters: dict[str | tuple[str, str, str], dict | QuantumParameters]
    | None = None,
    options: MultiplexedIQBlobWorkflowOptions | None = None,
) -> None:
    """Run 2Q multiplexed IQ blob experiment and optional threshold analysis."""
    options = MultiplexedIQBlobWorkflowOptions() if options is None else options
    temp_qpu = temporary_qpu(qpu, temporary_parameters)
    ctrl = temporary_quantum_elements_from_qpu(temp_qpu, ctrl)
    targ = temporary_quantum_elements_from_qpu(temp_qpu, targ)

    exp = create_experiment(temp_qpu, ctrl, targ)
    compiled_exp = compile_experiment(session, exp)
    result = run_experiment(session, compiled_exp)

    analysis_result = None
    with workflow.if_(options.do_analysis):
        analysis_result = analysis_workflow(
            result=result,
            ctrl=ctrl,
            targ=targ,
        )

    workflow.return_(
        {
            "result": result,
            "analysis_result": analysis_result,
        }
    )


@workflow.task
@dsl.qubit_experiment
def create_experiment(
    qpu: QPU,
    ctrl: QuantumElements,
    targ: QuantumElements,
    options: MultiplexedIQBlobExperimentOptions | None = None,
) -> Experiment:
    """Create 2Q multiplexed IQ blob experiment for threshold extraction."""
    opts = MultiplexedIQBlobExperimentOptions() if options is None else options
    ctrl = validation.validate_and_convert_single_qubit_sweeps(ctrl)
    targ = validation.validate_and_convert_single_qubit_sweeps(targ)

    qop = qpu.quantum_operations
    max_measure_section_length = qop.measure_section_length([ctrl, targ])

    with dsl.acquire_loop_rt(
        count=opts.count,
        averaging_mode=opts.averaging_mode,
        acquisition_type=opts.acquisition_type,
        repetition_mode=opts.repetition_mode,
        repetition_time=opts.repetition_time,
        reset_oscillator_phase=opts.reset_oscillator_phase,
    ):
        for prepared_label, (ctrl_state, targ_state) in PREPARED_STATES_2Q:
            with dsl.section(
                name=f"multiplexed_iq_blob_{prepared_label}",
                alignment=SectionAlignment.LEFT,
            ):
                prep_play_after = None
                if opts.active_reset:
                    active_reset_sec = qop.active_reset(
                        [ctrl, targ],
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
                    with dsl.section(
                        name=f"prep_ctrl_{prepared_label}",
                        alignment=SectionAlignment.LEFT,
                    ) as prep_ctrl:
                        qop.prepare_state(ctrl, state=ctrl_state)
                    with dsl.section(
                        name=f"prep_targ_{prepared_label}",
                        alignment=SectionAlignment.LEFT,
                        play_after=prep_ctrl.uid,
                    ):
                        qop.prepare_state(targ, state=targ_state)

                with dsl.section(
                    name=f"measure_{prepared_label}",
                    alignment=SectionAlignment.LEFT,
                    play_after=prep_sec.uid,
                ):
                    sec_ctrl = qop.measure(
                        ctrl,
                        handle=multiplexed_iq_blob_handle(ctrl.uid, prepared_label),
                    )
                    sec_targ = qop.measure(
                        targ,
                        handle=multiplexed_iq_blob_handle(targ.uid, prepared_label),
                    )
                    sec_ctrl.length = max_measure_section_length
                    sec_targ.length = max_measure_section_length
                    qop.passive_reset(ctrl)
                    qop.passive_reset(targ)

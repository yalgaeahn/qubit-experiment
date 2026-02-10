"""This module defines the experiments for 2-qubit computational-basis readout calibration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from laboneq import workflow
from laboneq.dsl.enums import AcquisitionType, AveragingMode
from laboneq.simple import Experiment, SectionAlignment, dsl
from laboneq.workflow.tasks import compile_experiment, run_experiment

from experiments.two_qubit_tomography_common import (
    READOUT_CALIBRATION_STATES,
    readout_calibration_handle,
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
class TwoQReadoutCalibrationExperimentOptions:
    """Options for 2Q readout calibration experiment."""

    count: int = workflow.option_field(
        4096,
        description="Number of shots per prepared basis state.",
    )
    acquisition_type: AcquisitionType = workflow.option_field(
        AcquisitionType.DISCRIMINATION,
        description="Acquire discrimination outcomes for assignment matrix calibration.",
    )
    averaging_mode: AveragingMode = workflow.option_field(
        AveragingMode.SINGLE_SHOT,
        description="Use single-shot outcomes for assignment matrix calibration.",
    )


@workflow.workflow(name="two_qubit_readout_calibration")
def experiment_workflow(
    session: Session,
    qpu: QPU,
    ctrl: QuantumElements,
    targ: QuantumElements,
    temporary_parameters: dict[str | tuple[str, str, str], dict | QuantumParameters]
    | None = None,
) -> None:
    """Run 2Q readout calibration (|00>,|01>,|10>,|11>)."""
    temp_qpu = temporary_qpu(qpu, temporary_parameters)
    ctrl = temporary_quantum_elements_from_qpu(temp_qpu, ctrl)
    targ = temporary_quantum_elements_from_qpu(temp_qpu, targ)

    exp = create_experiment(temp_qpu, ctrl, targ)
    compiled_exp = compile_experiment(session, exp)
    result = run_experiment(session, compiled_exp)
    workflow.return_(result)


@workflow.task
@dsl.qubit_experiment
def create_experiment(
    qpu: QPU,
    ctrl: QuantumElements,
    targ: QuantumElements,
    options: TwoQReadoutCalibrationExperimentOptions | None = None,
) -> Experiment:
    """Create 2Q readout calibration experiment."""
    opts = TwoQReadoutCalibrationExperimentOptions() if options is None else options
    ctrl = validation.validate_and_convert_single_qubit_sweeps(ctrl)
    targ = validation.validate_and_convert_single_qubit_sweeps(targ)

    qop = qpu.quantum_operations
    #max_measure_section_length = qpu.measure_section_length([ctrl, targ])

    with dsl.acquire_loop_rt(
        count=opts.count,
        averaging_mode=opts.averaging_mode,
        acquisition_type=opts.acquisition_type,
        repetition_mode=opts.repetition_mode,
        repetition_time=opts.repetition_time,
        reset_oscillator_phase=opts.reset_oscillator_phase,
    ):
        for prepared_label, (ctrl_state, targ_state) in READOUT_CALIBRATION_STATES:
            with dsl.section(
                name=f"readout_cal_{prepared_label}",
                alignment=SectionAlignment.LEFT,
            ):
                prep_play_after = None
                if opts.active_reset:
                    active_reset_sec = qop.active_reset(
                        [ctrl, targ],
                        active_reset_states=opts.active_reset_states,
                        number_resets=opts.active_reset_repetitions,
                        #measure_section_length=max_measure_section_length,
                    )
                    prep_play_after = active_reset_sec.uid

                prep_section_kwargs = {
                    "name": f"prep_{prepared_label}",
                    "alignment": SectionAlignment.LEFT,
                }
                if prep_play_after is not None:
                    prep_section_kwargs["play_after"] = prep_play_after

                with dsl.section(**prep_section_kwargs) as prep_sec:
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
                        handle=readout_calibration_handle(ctrl.uid, prepared_label),
                    )
                    sec_targ = qop.measure(
                        targ,
                        handle=readout_calibration_handle(targ.uid, prepared_label),
                    )
                    #sec_ctrl.length = max_measure_section_length
                    #sec_targ.length = max_measure_section_length
                    qop.passive_reset(ctrl)
                    qop.passive_reset(targ)

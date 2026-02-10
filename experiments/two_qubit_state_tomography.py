"""This module defines the experiments for 2-qubit state tomography with RIP state preparation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from laboneq import workflow
from laboneq.dsl.enums import AcquisitionType, AveragingMode
from laboneq.simple import Experiment, SectionAlignment, dsl
from laboneq.workflow.tasks import compile_experiment, run_experiment

from analysis.two_qubit_state_tomography import analysis_workflow
from experiments.two_qubit_readout_calibration import (
    create_experiment as create_readout_calibration_experiment,
)
from experiments.two_qubit_tomography_common import (
    TOMOGRAPHY_SETTINGS,
    tomography_handle,
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
class TwoQStateTomographyExperimentOptions:
    """Options for 2Q state tomography experiment."""

    count: int = workflow.option_field(
        4096,
        description="Number of shots per tomography setting.",
    )
    acquisition_type: AcquisitionType = workflow.option_field(
        AcquisitionType.DISCRIMINATION,
        description="Acquire discrimination outcomes for state tomography.",
    )
    averaging_mode: AveragingMode = workflow.option_field(
        AveragingMode.SINGLE_SHOT,
        description="Use single-shot outcomes for state tomography counts.",
    )


@workflow.workflow_options
class TwoQStateTomographyWorkflowOptions:
    """Workflow options for 2Q state tomography."""

    do_analysis: bool = workflow.option_field(
        True,
        description="Whether to run tomography analysis workflow.",
    )
    do_readout_calibration: bool = workflow.option_field(
        True,
        description="Whether to run readout calibration before tomography.",
    )


@workflow.workflow(name="two_qubit_state_tomography")
def experiment_workflow(
    session: Session,
    qpu: QPU,
    ctrl: QuantumElements,
    targ: QuantumElements,
    bus: QuantumElements,
    bus_frequency: float,
    rip_amplitude: float,
    rip_length: float,
    rip_phase: float = np.pi / 2,
    readout_calibration_result=None,
    target_state=None,
    temporary_parameters: dict[str | tuple[str, str, str], dict | QuantumParameters]
    | None = None,
    options: TwoQStateTomographyWorkflowOptions | None = None,
) -> None:
    """Run 2Q tomography with RIP preparation and optional readout calibration."""
    options = (
        TwoQStateTomographyWorkflowOptions() if options is None else options
    )

    temp_qpu = temporary_qpu(qpu, temporary_parameters)
    ctrl = temporary_quantum_elements_from_qpu(temp_qpu, ctrl)
    targ = temporary_quantum_elements_from_qpu(temp_qpu, targ)
    bus = temporary_quantum_elements_from_qpu(temp_qpu, bus)

    calibration_result = readout_calibration_result
    with workflow.if_(
        options.do_readout_calibration and readout_calibration_result is None
    ):
        readout_cal_exp = create_readout_calibration_experiment(temp_qpu, ctrl, targ)
        compiled_readout_cal = compile_experiment(session, readout_cal_exp)
        calibration_result = run_experiment(session, compiled_readout_cal)

    exp = create_experiment(
        temp_qpu,
        ctrl,
        targ,
        bus,
        bus_frequency=bus_frequency,
        rip_amplitude=rip_amplitude,
        rip_length=rip_length,
        rip_phase=rip_phase,
    )
    compiled_exp = compile_experiment(session, exp)
    tomography_result = run_experiment(session, compiled_exp)

    analysis_result = None
    with workflow.if_(options.do_analysis):
        analysis_result = analysis_workflow(
            tomography_result=tomography_result,
            ctrl=ctrl,
            targ=targ,
            readout_calibration_result=calibration_result,
            target_state=target_state,
        )

    workflow.return_(
        {
            "tomography_result": tomography_result,
            "readout_calibration_result": calibration_result,
            "analysis_result": analysis_result,
        }
    )


def _apply_measurement_prerotation(qop, qubit, axis: str):
    """Apply pre-rotation so final Z-basis measurement corresponds to axis."""
    if axis == "X":
        return qop.ry(qubit, angle=-np.pi / 2)
    elif axis == "Y":
        return qop.rx(qubit, angle=np.pi / 2)
    elif axis == "Z":
        return None
    else:
        raise ValueError(f"Unsupported tomography axis: {axis!r}.")


@workflow.task
@dsl.qubit_experiment
def create_experiment(
    qpu: QPU,
    ctrl: QuantumElements,
    targ: QuantumElements,
    bus: QuantumElements,
    bus_frequency: float,
    rip_amplitude: float,
    rip_length: float,
    rip_phase: float = np.pi / 2,
    options: TwoQStateTomographyExperimentOptions | None = None,
) -> Experiment:
    """Create 2Q tomography experiment with RIP state preparation."""
    opts = TwoQStateTomographyExperimentOptions() if options is None else options
    ctrl = validation.validate_and_convert_single_qubit_sweeps(ctrl)
    targ = validation.validate_and_convert_single_qubit_sweeps(targ)
    bus = validation.validate_and_convert_single_qubit_sweeps(bus)

    qop = qpu.quantum_operations
    max_measure_section_length = qpu.measure_section_length([ctrl, targ])

    with dsl.acquire_loop_rt(
        count=opts.count,
        averaging_mode=opts.averaging_mode,
        acquisition_type=opts.acquisition_type,
        repetition_mode=opts.repetition_mode,
        repetition_time=opts.repetition_time,
        reset_oscillator_phase=opts.reset_oscillator_phase,
    ):
        # set_bus_frequency can only be called once per signal in one experiment
        qop.set_bus_frequency(bus, frequency=bus_frequency)

        for setting_label, (ctrl_axis, targ_axis) in TOMOGRAPHY_SETTINGS:
            with dsl.section(
                name=f"tomo_{setting_label}",
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

                prep_section_kwargs = {
                    "name": f"prep_{setting_label}",
                    "alignment": SectionAlignment.LEFT,
                }
                if prep_play_after is not None:
                    prep_section_kwargs["play_after"] = prep_play_after

                with dsl.section(
                    **prep_section_kwargs,
                ) as prep_sec:
                    with dsl.section(
                        name=f"prep_ctrl_g_{setting_label}",
                        alignment=SectionAlignment.LEFT,
                    ) as prep_ctrl_g:
                        qop.prepare_state(ctrl, state="g")

                    with dsl.section(
                        name=f"prep_targ_g_{setting_label}",
                        alignment=SectionAlignment.LEFT,
                        play_after=prep_ctrl_g.uid,
                    ) as prep_targ_g:
                        qop.prepare_state(targ, state="g")

                    # User-confirmed preparation: ctrl,targ both in |+>.
                    with dsl.section(
                        name=f"prep_ctrl_plus_{setting_label}",
                        alignment=SectionAlignment.LEFT,
                        play_after=prep_targ_g.uid,
                    ) as prep_ctrl_plus:
                        qop.y90(ctrl)

                    with dsl.section(
                        name=f"prep_targ_plus_{setting_label}",
                        alignment=SectionAlignment.LEFT,
                        play_after=prep_ctrl_plus.uid,
                    ):
                        qop.y90(targ)

                with dsl.section(
                    name=f"rip_{setting_label}",
                    alignment=SectionAlignment.LEFT,
                    play_after=prep_sec.uid,
                ) as rip_sec:
                    qop.rip(
                        bus,
                        amplitude=rip_amplitude,
                        phase=rip_phase,
                        length=rip_length,
                    )

                with dsl.section(
                    name=f"basis_{setting_label}",
                    alignment=SectionAlignment.LEFT,
                    play_after=rip_sec.uid,
                ) as basis_sec:
                    with dsl.section(
                        name=f"basis_ctrl_{setting_label}",
                        alignment=SectionAlignment.LEFT,
                    ) as basis_ctrl:
                        _apply_measurement_prerotation(qop, ctrl, ctrl_axis)

                    with dsl.section(
                        name=f"basis_targ_{setting_label}",
                        alignment=SectionAlignment.LEFT,
                        play_after=basis_ctrl.uid,
                    ):
                        _apply_measurement_prerotation(qop, targ, targ_axis)

                with dsl.section(
                    name=f"measure_{setting_label}",
                    alignment=SectionAlignment.LEFT,
                    play_after=basis_sec.uid,
                ):
                    sec_ctrl = qop.measure(
                        ctrl,
                        handle=tomography_handle(ctrl.uid, setting_label),
                    )
                    sec_targ = qop.measure(
                        targ,
                        handle=tomography_handle(targ.uid, setting_label),
                    )
                    sec_ctrl.length = max_measure_section_length
                    sec_targ.length = max_measure_section_length
                    qop.passive_reset(ctrl)
                    qop.passive_reset(targ)

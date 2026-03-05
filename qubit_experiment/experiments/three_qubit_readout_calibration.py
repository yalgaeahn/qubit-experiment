"""This module defines the experiments for 3-qubit computational-basis readout calibration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from laboneq import workflow
from laboneq.dsl.enums import AcquisitionType, AveragingMode
from laboneq.simple import Experiment, SectionAlignment, dsl
from laboneq.workflow.tasks import compile_experiment, run_experiment
from laboneq_applications.core import validation
from laboneq_applications.experiments.options import BaseExperimentOptions
from laboneq_applications.tasks.parameter_updating import (
    temporary_qpu,
    temporary_quantum_elements_from_qpu,
)

from experiments.three_qubit_tomography_common import (
    READOUT_CALIBRATION_STATES,
    readout_calibration_handle,
)

if TYPE_CHECKING:
    from laboneq.dsl.quantum import QuantumParameters
    from laboneq.dsl.quantum.qpu import QPU
    from laboneq.dsl.session import Session
    from laboneq_applications.typing import QuantumElements


@workflow.task_options(base_class=BaseExperimentOptions)
class ThreeQReadoutCalibrationExperimentOptions:
    """Options for 3Q readout calibration experiment."""

    count: int = workflow.option_field(
        4096,
        description="Number of shots per prepared basis state.",
    )
    acquisition_type: AcquisitionType = workflow.option_field(
        AcquisitionType.INTEGRATION,
        description="Acquire integrated complex IQ outcomes for assignment matrix calibration.",
    )
    averaging_mode: AveragingMode = workflow.option_field(
        AveragingMode.SINGLE_SHOT,
        description="Use single-shot integrated outcomes for assignment matrix calibration.",
    )


@workflow.workflow(name="three_qubit_readout_calibration")
def experiment_workflow(
    session: Session,
    qpu: QPU,
    qubits: QuantumElements,
    temporary_parameters: dict[str | tuple[str, str, str], dict | QuantumParameters]
    | None = None,
) -> None:
    """Run 3Q readout calibration (|000>,|001>,...,|111>)."""
    temp_qpu = temporary_qpu(qpu, temporary_parameters)
    qubits = temporary_quantum_elements_from_qpu(temp_qpu, qubits)

    exp = create_experiment(temp_qpu, qubits)
    compiled_exp = compile_experiment(session, exp)
    result = run_experiment(session, compiled_exp)
    workflow.return_(result)


@workflow.task
@dsl.qubit_experiment
def create_experiment(
    qpu: QPU,
    qubits: QuantumElements,
    options: ThreeQReadoutCalibrationExperimentOptions | None = None,
) -> Experiment:
    """Create 3Q readout calibration experiment."""
    opts = ThreeQReadoutCalibrationExperimentOptions() if options is None else options
    if AcquisitionType(opts.acquisition_type) != AcquisitionType.INTEGRATION:
        raise ValueError(
            "three_qubit_readout_calibration only supports AcquisitionType.INTEGRATION."
        )
    if AveragingMode(opts.averaging_mode) != AveragingMode.SINGLE_SHOT:
        raise ValueError(
            "three_qubit_readout_calibration only supports AveragingMode.SINGLE_SHOT."
        )

    q0, q1, q2 = _normalize_three_qubits(qubits)

    qop = qpu.quantum_operations
    max_measure_section_length = qop.measure_section_length([q0, q1, q2])

    with dsl.acquire_loop_rt(
        count=opts.count,
        averaging_mode=opts.averaging_mode,
        acquisition_type=opts.acquisition_type,
        repetition_mode=opts.repetition_mode,
        repetition_time=opts.repetition_time,
        reset_oscillator_phase=opts.reset_oscillator_phase,
    ):
        for prepared_label, (q0_state, q1_state, q2_state) in READOUT_CALIBRATION_STATES:
            with dsl.section(
                name=f"readout_cal_{prepared_label}",
                alignment=SectionAlignment.LEFT,
            ):
                prep_play_after = None
                if opts.active_reset:
                    active_reset_sec = qop.active_reset(
                        [q0, q1, q2],
                        active_reset_states=opts.active_reset_states,
                        number_resets=opts.active_reset_repetitions,
                        measure_section_length=max_measure_section_length,
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
                        name=f"prep_q0_{prepared_label}",
                        alignment=SectionAlignment.LEFT,
                    ) as prep_q0:
                        qop.prepare_state(q0, state=q0_state)

                    with dsl.section(
                        name=f"prep_q1_{prepared_label}",
                        alignment=SectionAlignment.LEFT,
                        play_after=prep_q0.uid,
                    ) as prep_q1:
                        qop.prepare_state(q1, state=q1_state)

                    with dsl.section(
                        name=f"prep_q2_{prepared_label}",
                        alignment=SectionAlignment.LEFT,
                        play_after=prep_q1.uid,
                    ):
                        qop.prepare_state(q2, state=q2_state)

                with dsl.section(
                    name=f"measure_{prepared_label}",
                    alignment=SectionAlignment.LEFT,
                    play_after=prep_sec.uid,
                ):
                    sec_q0 = qop.measure(
                        q0,
                        handle=readout_calibration_handle(q0.uid, prepared_label),
                    )
                    sec_q1 = qop.measure(
                        q1,
                        handle=readout_calibration_handle(q1.uid, prepared_label),
                    )
                    sec_q2 = qop.measure(
                        q2,
                        handle=readout_calibration_handle(q2.uid, prepared_label),
                    )
                    sec_q0.length = max_measure_section_length
                    sec_q1.length = max_measure_section_length
                    sec_q2.length = max_measure_section_length
                    qop.passive_reset(q0)
                    qop.passive_reset(q1)
                    qop.passive_reset(q2)


def _normalize_three_qubits(qubits: QuantumElements):
    """Validate qubits input and return exactly three single-qubit elements."""
    qlist = list(validation.validate_and_convert_qubits_sweeps(qubits))
    if len(qlist) != 3:
        raise ValueError(
            "three_qubit_readout_calibration expects exactly 3 qubits in `qubits`."
            f" Received {len(qlist)}."
        )
    q0 = validation.validate_and_convert_single_qubit_sweeps(qlist[0])
    q1 = validation.validate_and_convert_single_qubit_sweeps(qlist[1])
    q2 = validation.validate_and_convert_single_qubit_sweeps(qlist[2])
    return q0, q1, q2

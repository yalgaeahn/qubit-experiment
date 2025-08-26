""" This module defines the direct cr hamiltonian tomography experiment"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, cast

import numpy as np
from laboneq import workflow

from laboneq.simple import (
    AveragingMode,
    Experiment,
    SectionAlignment,
    SweepParameter,
    dsl,
)


#from laboneq.simple import AcquisitionType
#from laboneq.core.types.enums.section_alignment import SectionAlignment
#from laboneq.core.types.enums.acquisition_type import AcquisitionType #?? laboneq 자체에 같은 이름의 클래스 벌써 3개...

#from laboneq.dsl.parameter import SweepParameter
from laboneq.workflow.tasks import (
    compile_experiment,
    run_experiment,
)

from laboneq_applications.core import validation
from laboneq_applications.experiments.options import (
    TuneupExperimentOptions,
    TuneUpWorkflowOptions,
    BaseExperimentOptions
)
from laboneq_applications.tasks.parameter_updating import (
    temporary_qpu,
    temporary_quantum_elements_from_qpu,
    update_qubits,
)

if TYPE_CHECKING:
    from laboneq.dsl.quantum import QuantumParameters
    from laboneq.dsl.quantum.qpu import QPU
    from laboneq.dsl.session import Session
    from laboneq.dsl.quantum.quantum_element import QuantumElement

    from laboneq_applications.typing import QuantumElements, QubitSweepPoints
##########################################################################
from analysis.hamiltonian_tomography import analysis_workflow, HamiltonianTomographyAnalysisOptions
from options import DirectCRHamiltonianTomographyOptions
#######################EXPERIMENT####################################
@workflow.workflow(name="direct_cr_hamiltonian_tomography")
def experiment_workflow(
    session: Session,
    qpu: QPU,
    ctrl : QuantumElements, #현재로서는 targ 설정 못하고 고정
    targ : QuantumElements,
    amplitudes : QubitSweepPoints,
    lengths : QubitSweepPoints,
    temporary_parameters: dict[str | tuple[str, str, str], dict | QuantumParameters]
    | None = None,
    options: TuneUpWorkflowOptions | None = None,
) -> None:
    """ The CR drive amplitude calibration workflow.
     
    The workflow consists of the following steps:

    - [create_experiment]()
    - [compile_experiment]()
    - [run_experiment]()
    - [analysis_workflow]()
    - [update_qubits]()

    """
    temp_qpu = temporary_qpu(qpu, temporary_parameters)
    ctrl = temporary_quantum_elements_from_qpu(temp_qpu, ctrl)
    targ = temporary_quantum_elements_from_qpu(temp_qpu, targ)
    exp = create_experiment(
        temp_qpu, 
        ctrl,
        targ,
        amplitudes,
        lengths,
    )

    compiled_exp = compile_experiment(session, exp)
    result = run_experiment(session, compiled_exp)

    
    with workflow.if_(options.do_analysis):
        analysis_result = analysis_workflow(
            result=result,
            qubit=targ,
            lengths=lengths,
            amplitudes=amplitudes,
        )
        qubit_parameters = analysis_result.output
        with workflow.if_(options.update):
            update_qubits(qpu, qubit_parameters["new_parameter_values"])
    workflow.return_(result)
    
    
    

@workflow.task
@dsl.qubit_experiment
def create_experiment(
    qpu: QPU,
    ctrl: QuantumElement, 
    targ: QuantumElement,
    amplitudes: QubitSweepPoints,
    lengths : QubitSweepPoints,
    options : DirectCRHamiltonianTomographyOptions | None = None,
) -> Experiment:

    # Define the custom optiopns for the experiment
    opts = DirectCRHamiltonianTomographyOptions() if options is None else options
    ctrl, amplitudes = validation.validate_and_convert_single_qubit_sweeps(ctrl, amplitudes)
    ctrl, lengths = validation.validate_and_convert_single_qubit_sweeps(ctrl, lengths)
    
    if (
        opts.use_cal_traces
        and AveragingMode(opts.averaging_mode) == AveragingMode.SEQUENTIAL
    ):
        raise ValueError(
            "'AveragingMode.SEQUENTIAL' (or {AveragingMode.SEQUENTIAL}) cannot be used "
            "with calibration traces because the calibration traces are added "
            "outside the sweep."
        )



    
    
    ##### Define the sweep parameters ################################################
    amp_sweep_par = SweepParameter(values = np.asarray(amplitudes), axis_name="amp") 
    basis_sweep_par = SweepParameter(values=[0,1,2],axis_name="basis") 
    state_sweep_par = SweepParameter(values=[0,1], axis_name="state")
    length_sweep_par = SweepParameter(values = np.asarray(lengths), axis_name="length")
    ##################################################################################
    #max_measure_section_length = qpu.measure_section_length(ctrl) #for multiplexing
    qop = qpu.quantum_operations
    with dsl.acquire_loop_rt(
        name="acquire_loop",
        count=opts.count,
        averaging_mode=opts.averaging_mode,
        acquisition_type=opts.acquisition_type,
        repetition_mode=opts.repetition_mode,
        repetition_time=opts.repetition_time,
        reset_oscillator_phase=opts.reset_oscillator_phase,
    ):
        with dsl.sweep(
            name=f"amp_sweep_sec",
            parameter=amp_sweep_par,
            alignment=SectionAlignment.LEFT,
        ) as amplitude:
            with dsl.sweep(
                name=f"basis_sweep_sec",
                parameter=basis_sweep_par,
                alignment=SectionAlignment.LEFT,
            ) as basis:
                with dsl.sweep(
                    name=f"state_sweep_sec",
                    parameter=state_sweep_par,
                    alignment=SectionAlignment.LEFT,
                ) as state:
                    with dsl.sweep(
                        name="length_sweep_sec",
                        parameter=length_sweep_par,
                        alignment=SectionAlignment.LEFT,
                    ) as length:
                       
                        with dsl.match(name="ctrl_prep", sweep_parameter=state) as ctrl_prep:
                            with dsl.case(0):
                                qop.delay.omit_section(q=ctrl, time=ctrl.parameters.ge_drive_length)
                            with dsl.case(1):
                                qop.x180.omit_section(ctrl)
                        with dsl.section(name="main_cr_drive", on_system_grid=True, play_after=ctrl_prep.uid) as main_cr_drive:
                            qop.set_frequency.omit_section(q=ctrl, frequency=targ.parameters.resonance_frequency_ge) #안되면 외부에서 넣어주자
                            qop.direct_cr.omit_section(ctrl=ctrl, 
                                                       targ=targ, 
                                                       amplitude = amplitude, 
                                                       phase= 0.0, 
                                                       length=length, 
                                                       override_params = {'risefall_sigma_ratio': None, 'width': length -2*opts.risefall}) #pulse에는 
                            if opts.cancel:
                                qop.cr_cancel.omit_section(ctrl=ctrl,
                                                       targ=targ,
                                                       amplitude=0.0,
                                                       length=length,
                                                       override_params = {'risefall_sigma_ratio': None, 'width': length -2*opts.risefall}))


                        with dsl.match(name="targ_basis_prep", sweep_parameter=basis, play_after=main_cr_drive.uid) as targ_basis_prep:
                            with dsl.case(0): #X
                                qop.ry.omit_section(q=targ, angle=-np.pi/2)
                            with dsl.case(1): #Y
                                qop.rx.omit_section(q=targ, angle=np.pi/2)
                            with dsl.case(2): #Z
                                qop.delay.omit_section(q=targ, time=ctrl.parameters.ge_drive_length)

                        with dsl.section(name="measure", play_after=targ_basis_prep.uid) as measure:
                            qop.measure.omit_section(q=targ, handle=dsl.handles.result_handle(qubit_name=targ.uid))
                        with dsl.section(name="passive_reset", on_system_grid=True, play_after=measure.uid):
                            qop.passive_reset.omit_section(q=targ)
        
        if opts.use_cal_traces:               
            qop.calibration_traces.omit_section(
                qubits=targ,#only target is being measured
                states=opts.cal_states,
                active_reset=opts.active_reset,
                active_reset_states=opts.active_reset_states,
                active_reset_repetitions=opts.active_reset_repetitions,
                measure_section_length=None, # should be added for multiplex
            )           
                        
                                
    
        
      
        
    
        
        


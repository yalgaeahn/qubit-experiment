""" This module defines the direct cr hamiltonian tomography experiment"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from laboneq import workflow
from laboneq.simple import (
    AcquisitionType,
    AveragingMode,
    Experiment,
    SectionAlignment,
    SweepParameter,
    dsl,
)
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
from krisszi_core.contrib.jsahn.analysis import hamiltonian_tomography

#######################EXPERIMENT####################################



@workflow.task_options(base_class=BaseExperimentOptions)
class DirectCRHamiltonianTomographyOptions:
    """Base options for direct cr hamiltonian tomography experiment"""
    acquisition_type : AcquisitionType = workflow.option_field(
        AcquisitionType.INTEGRATION, description="Don't know why"
    )



@workflow.workflow(name="nondsl_direct_cr_hamiltonian_tomography")
def experiment_workflow(
    session: Session,
    qpu: QPU,
    ctrl : QuantumElement, #현재로서는 targ 설정 못하고 고정
    targ : QuantumElement,
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
    # with workflow.if_(options.do_analysis):
    #     analysis_workflow()
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
    #ctrl, amplitudes = validation.validate_and_convert_single_qubit_sweeps()
    #ctrl, lenghts = validation.validate_and_convert_single_qubit_sweeps()
    qop = qpu.quantum_operations
    
    ##### Define the sweep parameters ################################################
    amp_sweep_par = SweepParameter(values = amplitudes, axis_name="amp") 
    basis_sweep_par = SweepParameter(values=np.array([0,1,2]),axis_name="basis") # XYZ
    state_sweep_par = SweepParameter(values=np.array([0,1]),axis_name="state")  # g e
    length_sweep_par = SweepParameter(values = lengths, axis_name="length")
    ##################################################################################
    

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
                                qop.delay(q=ctrl, time=ctrl.parameters.ge_drive_length)
                            with dsl.case(1):
                                qop.x180(ctrl)
                                
                        with dsl.section(name="main_cr_drive", on_system_grid=True, play_after=ctrl_prep.uid) as main_cr_drive:
                            qop.set_frequency(q=ctrl, frequency=targ.parameters.resonance_frequency_ge) #안되면 외부에서 넣어주자
                            qop.direct_cr(ctrl=ctrl, targ=targ, amplitude = amplitude, phase=0.0, length=length) # 나중에 고치자
                        
                        with dsl.section(name="tomo_measure", on_system_grid=True, play_after=main_cr_drive.uid):
                            with dsl.match(name="targ_basis_prep", sweep_parameter=basis) as targ_basis_prep:
                                with dsl.case(0): #X
                                    qop.ry(q=targ, angle=-np.pi/2)
                                with dsl.case(1): #Y
                                    qop.rx(q=targ, angle=np.pi/2)
                                with dsl.case(2): #Z
                                    qop.delay(q=targ, time=ctrl.parameters.ge_drive_length)

                            qop.measure(q=targ, handle=dsl.handles.result_handle(targ.uid))
                        
                            qop.passive_reset(q=targ)
                            qop.passive_reset(q=ctrl)
                        
                                
    
        
      
        
    
        
        

    
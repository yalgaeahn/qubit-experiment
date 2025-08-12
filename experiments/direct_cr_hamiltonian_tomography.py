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

    from laboneq_applications.typing import QuantumElements, QubitSweepPoints
##########################################################################
from krisszi_core.contrib.jsahn.analysis import hamiltonian_tomography
from laboneq.dsl.experiment import AcquireLoopRt, Sweep, Match, Case
#######################EXPERIMENT####################################

@workflow.task_options(base_class=BaseExperimentOptions)
class DirectCRHamiltonianTomographyOptions:
    """Base options for direct cr hamiltonian tomography experiment"""
    acquisition_type : AcquisitionType = workflow.option_field(
        AcquisitionType.INTEGRATION, description="Don't know why"
    )



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


    
    
    
    ################################################################################################
    @workflow.task
    @dsl.qubit_experiment(context=False) # Declarative-Style DSL
    def create_experiment(
        qpu: QPU,
        ctrl: QuantumElements, 
        targ: QuantumElements,
        amplitudes: QubitSweepPoints,
        lengths : QubitSweepPoints,
        options : DirectCRHamiltonianTomographyOptions | None = None,
    ): #-> Experiment:

        # Define the custom optiopns for the experiment
        opts = DirectCRHamiltonianTomographyOptions() if options is None else options
        
        # ctrl, amplitudes = validation.validate_and_convert_qubits_sweeps(qubits=ctrl, sweep_points=amplitudes) # return list
        # targ, _amplitudes = validation.validate_and_convert_qubits_sweeps(qubits=targ, sweep_points=amplitudes) # return list
        
    
        
        #max_measure_section_length = qpu.measure_section_length(qubits) #matters when multipexing
        qop = qpu.quantum_operations
        
        acq_loop = AcquireLoopRt(
            uid="shots",
            count=opts.count,
            averaging_mode=opts.averaging_mode,
            acquisition_type=opts.acquisition_type,
            repetition_mode=opts.repetition_mode,
            repetition_time=opts.repetition_time,
            reset_oscillator_phase=opts.reset_oscillator_phase,
            on_system_grid=True,
        )
        
        ##### Define the sweep parameters & sweep loop section ################################################
        amp_sweep_par = SweepParameter(values = amplitudes, axis_name="amp") 
        basis_sweep_par = SweepParameter(values=np.array([0,1,2]),axis_name="basis") # XYZ
        state_sweep_par = SweepParameter(values=np.array([0,1]),axis_name="state")  # g e
        length_sweep_par = SweepParameter(values = lengths, axis_name="length")
        
        
        
        amp_sweep_loop = Sweep(parameters=amp_sweep_par)
        basis_sweep_loop = Sweep(parameters=basis_sweep_par)
        state_sweep_loop = Sweep(parameters=state_sweep_par)
        length_sweep_loop = Sweep(parameters=length_sweep_par)
        
        
        ##### Define the ctrl_prep section##################################################################
        ctrl_prep = Match(name="ctrl_prep", sweep_parameter=state_sweep_par, on_system_grid=True)
        # ctrl_prep.add(Case.from_section(section=qop.prepare_state(q=ctrl,state="g"),state=0))
        # ctrl_prep.add(Case.from_section(section=qop.prepare_state(q=ctrl,state="e"),state=1))
        ctrl_prep.add(Case.from_section(section=qop.delay(q=ctrl, time=ctrl.parameters.ge_drive_length), state=0)) 
        ctrl_prep.add(Case.from_section(section=qop.x180(q=ctrl), state=1)) 
        ##### Define the cr_drive_section###################################################################
        qop.set_frequency(q=ctrl,frequency=targ.parameters.resonance_frequency_ge)
        direct_cr = qop.direct_cr(ctrl=ctrl, targ=targ, amplitude = amp_sweep_par, phase=0.0, length=200e-9) # 나중에 고치자
        # Note that direct_cr itself is a section with a play() inside
        
        
        ##### Define the tomography section#####################################################################
        tomo = Match(name="tomo", sweep_parameter=basis_sweep_par, on_system_grid=True)
        x_case = Case.from_section(section=qop.ry(q=targ, angle=-np.pi/2), state=0) #should be Ry(-pi/2)
        y_case = Case.from_section(section=qop.rx(q=targ, angle=np.pi/2), state=1) #should be Rx(pi/2)
        z_case = Case.from_section(section=qop.delay(q=targ, time=targ.parameters.ge_drive_length), state=2) #should be identity
        tomo.add(x_case)
        tomo.add(y_case)
        tomo.add(z_case)
        ####################################################################################################        
        
        length_sweep_loop.add(ctrl_prep) #state sweep (Match) in ctrl_prep
        length_sweep_loop.add(direct_cr) #amp sweep in direct_cr 
        length_sweep_loop.add(tomo) #basis sweep in tomo
        length_sweep_loop.add(qop.measure(q=targ, handle=dsl.handles.result_handle(targ.uid)))
        
        state_sweep_loop.add(length_sweep_loop)
        basis_sweep_loop.add(state_sweep_loop)
        amp_sweep_loop.add(basis_sweep_loop)
        acq_loop.add(amp_sweep_loop)
        exp=Experiment()
        exp.add(acq_loop)
        return exp
        
        
        
    exp = create_experiment(qpu=temp_qpu, ctrl=ctrl,targ=targ,amplitudes=amplitudes,lengths=lengths)
    print(exp.signal_mapping_status)
    #exp.set_calibration(ctrl.calibration()) # Reference Object is not callable
    
    compiled_exp = compile_experiment(session, exp)
    result = run_experiment(session, compiled_exp)
    # with workflow.if_(options.do_analysis):
    #     analysis_workflow()
    workflow.return_(result)
    

    
"""2D plotting analysis workflow for direct CR Hamiltonian Tomography.

Works with experiments in `experiments/direct_cr_hamiltonian_tomography.py`.

This analysis provides:
- Raw I/Q heatmaps versus (CR amplitude, pulse length)
- Magnitude and phase heatmaps versus (CR amplitude, pulse length)

Notes:
- It expects the experiment to use `dsl.handles.result_handle(targ.uid)` for the
    measurement handle of the target qubit, as implemented in the experiment file.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import lmfit
import numpy as np
import uncertainties as unc
from laboneq import workflow

from laboneq_applications.analysis.calibration_traces_rotation import calculate_population_2d, calculate_population_1d

from laboneq_applications.analysis.options import (
    FitDataOptions,
    PlotPopulationOptions
)
from analysis.fitting_helpers import blochtrajectory_fit, find_oscillation_frequency_and_phase
from laboneq_applications.analysis.plotting_helpers import timestamped_title
#     plot_raw_complex_data_2d,
#     plot_signal_magnitude_and_phase_2d,
# )
from laboneq_applications.core.validation import (
    validate_and_convert_single_qubit_sweeps,
    validate_result,
)


if TYPE_CHECKING:
    import matplotlib as mpl
    from laboneq.workflow.tasks.run_experiment import RunExperimentResults
    from laboneq_applications.typing import QuantumElements, QubitSweepPoints
    from laboneq.workflow.tasks.run_experiment import RunExperimentResults
    from laboneq_applications.typing import QuantumElements, QubitSweepPoints
    from numpy.typing import ArrayLike

@workflow.workflow_options
class HamiltonianTomographyAnalysisOptions:
    """Options for CR Hamiltonian tomography plotting.
    Attributes:
    """
    #do_raw_data_plotting : bool = workflow.option_field(True, description="Whether to plot the raw complex I/Q heatmaps.")
    do_plotting : bool = workflow.option_field(True, description="Whether to plot the magnitude and phase heatmaps.")
    do_fitting : bool = workflow.option_field(True, description="Wheter to do Bloch trajectory fitting")

    normalize_vectors : bool = workflow.option_field(True, description="Where to do Bloch vector normalization (X,Y,Z)")


    use_cal_traces : bool = workflow.option_field(True, description="Whether to include calibration traces in the analysis.")
    cal_states : str | tuple = workflow.option_field("ge", description="The states prepared in the calibration traces")



@workflow.workflow(name="analysis_workflow")
def analysis_workflow(
    result: RunExperimentResults,
    qubit: QuantumElements,
    lengths: QubitSweepPoints,  #innermost sweep (CR length)
    amplitudes: QubitSweepPoints, #outermost sweep (CR amp)
    options: HamiltonianTomographyAnalysisOptions | None = None,
) -> None:
    """Plot direct-CR tomography data as 2D heatmaps.

    Arguments:
        result: Results from run_experiment
        qubits: Target qubit(s) whose readout handles exist in results
        sweep_points_1d: First sweep axis values 
        sweep_points_2d: Second sweep axis values 
        label_sweep_points_1d: Label for the first axis
        label_sweep_points_2d: Label for the second axis
        scaling_sweep_points_2d: Optional scaling for display (e.g., 1e9 to ns)
        options: Plotting options
    """
    #options = HamiltonianTomography2DAnalysisOptions() if options is None else options
 
   
    processed_data_dict = process_data(qubit, result, lengths, amplitudes)
    fit_results = fit_data(qubit,processed_data_dict, lengths, amplitudes) #dict with key: amp and value : dict
 
    #interaction_rates = extract_interaction_rates()

   
    with workflow.if_(options.do_plotting):
        figures = plot_trajectory(qubit, processed_data_dict, fit_results, lengths, amplitudes)
        #workflow.save_artifact(f"CR oscillation_{qubit.uid}", figures)
   


@workflow.task
def process_data(
    qubit: QuantumElements,
    result: RunExperimentResults,
    sweep_points_1d: QubitSweepPoints,#innermost sweep (CR length)
    sweep_points_2d: QubitSweepPoints,#outermost sweep (CR amp)
) -> dict | dict[str, None]:
   
    _q, sweep_points_1d = validate_and_convert_single_qubit_sweeps(qubit, sweep_points_1d)
    q, sweep_points_2d = validate_and_convert_single_qubit_sweeps(qubit, sweep_points_2d)
    validate_result(result)
    
    raw_data = result[f"{q.uid}/result"].data # ["amp", "basis", "state", "length"]

    calibration_traces = [result[f"{q.uid}/cal_trace/{s}"].data for s in ["g","e"]]

    processed_data_dict = {} # amplitude - state - basis - processed_data
    
    for a_i, amplitude in enumerate(sweep_points_2d):
        state_data= {"g" : None, "e" : None}
        for s_i, state in enumerate(state_data.keys()):     
            basis_data = {"x" : None, "y" : None, "z" : None}
            for b_i, basis in enumerate(basis_data.keys()):
                data = raw_data[a_i,b_i,s_i,:]
                processed_data = calculate_population_1d(
                    data,
                    sweep_points_1d,
                    calibration_traces,
                    do_pca=False)
                basis_data[basis] = processed_data
            state_data[state] = basis_data
        processed_data_dict[float(amplitude)] = state_data  
    return processed_data_dict

                
    

@workflow.task
def fit_data(
    qubit: QuantumElements,
    processed_data_dict: dict[str, dict[str, ArrayLike | dict]],
    sweep_points_1d: QubitSweepPoints,#innermost sweep (CR length)
    sweep_points_2d: QubitSweepPoints,#outermost sweep (CR amp)
    options : HamiltonianTomographyAnalysisOptions | None = None,
) -> dict | dict[str, lmfit.model.ModelResult]:
   
    opts = HamiltonianTomographyAnalysisOptions() if options is None else options
    _q, sweep_points_1d = validate_and_convert_single_qubit_sweeps(qubit, sweep_points_1d)
    q, sweep_points_2d = validate_and_convert_single_qubit_sweeps(qubit, sweep_points_2d)
    
    param_hints = dict(
        p_x=dict(),
        p_y=dict(),
        p_z=dict(),
        t_off=dict(min=0, max=np.inf),
        b=dict(value=1e-9, min=-1,max=1))

    fit_results = { }
    if not opts.do_fitting:
        return fit_results
    
    for a_i, amplitude in enumerate(sweep_points_2d):
        fit_results[float(amplitude)]={"g":{}, "e":{}}
        for s_i, state in enumerate(["g", "e"]):     

            swpts = processed_data_dict[float(amplitude)][state]['x']["sweep_points"] #length
            x_data=1-2*processed_data_dict[float(amplitude)][state]['x']["population"]
            y_data=1-2*processed_data_dict[float(amplitude)][state]['y']["population"]
            z_data=1-2*processed_data_dict[float(amplitude)][state]['z']["population"]
            magnitude = np.sqrt(x_data**2 + y_data**2 + z_data**2)
            #normalizing bloch vector
            
            if opts.normalize_vectors:
                x_data = x_data/magnitude
                y_data = y_data/magnitude
                z_data = z_data/magnitude
            try:
                init_param_set = generate_initial_guess(x=swpts, x_data=x_data, y_data=y_data, z_data=z_data) 

                fit_res = blochtrajectory_fit(
                    x=swpts,
                    x_data=x_data,
                    y_data=y_data,
                    z_data=z_data,
                    param_hints=param_hints,
                    init_param_set=init_param_set
                )#fit_res is list with 3 lmfit.model.ModelResult
                fit_results[float(amplitude)][state]=fit_res
                
            except ValueError as err:
                workflow.comment(f"Fit failed for {q.uid}: {err}.")

    return fit_results
        

@workflow.task
def extract_interaction_rates(
    qubit: QuantumElements,
    fit_results: dict,
    sweep_points_2d : QubitSweepPoints
    ):
    for amplitude in sweep_points_2d:
        g_val = fit_results[amplitude]['g'].best_values
        e_val = fit_results[amplitude]['e'].best_values


        



    pass

@workflow.task
def plot_trajectory(
    qubit: QuantumElements,
    processed_data_dict: dict[str, dict[str, ArrayLike]],
    fit_results: dict[str, lmfit.model.ModelResult] | None,
    sweep_points_1d: QubitSweepPoints,#innermost sweep (CR length)
    sweep_points_2d: QubitSweepPoints,#outermost sweep (CR amp)
    options: None = None
) -> dict[str, mpl.figure.Figure]:
    """Create the X, Y, Z trajectory fitting plot for each amplitude"""
    q = validate_and_convert_single_qubit_sweeps(qubit)
    _q, sweep_points_1d = validate_and_convert_single_qubit_sweeps(qubit, sweep_points_1d)
    q, sweep_points_2d = validate_and_convert_single_qubit_sweeps(qubit, sweep_points_2d)
    
    figures = {}
    
    eqr_temps = {
    "x": "(-pz * px + pz * px * cos(W * (x+t_off)) + W * py * sin(W * (x+t_off))) / W**2 + b",
    "y": "(pz * py - pz * py * cos(W * (x+t_off)) - W * px * sin(W * (x+t_off))) / W**2 + b",
    "z": "(pz**2 + (px**2 + py**2) * cos(W * (x+t_off))) / W**2 +b"
    }
    for axis, temp_eq in eqr_temps.items():
        eq = temp_eq
        eq = eq.replace("W", "sqrt(px**2 + py**2 + pz**2)")
        eqr_temps[axis]=eq
    
    x_model = lmfit.models.ExpressionModel(eqr_temps['x'], independent_vars=['x'])
    y_model = lmfit.models.ExpressionModel(eqr_temps['y'], independent_vars=['x'])
    z_model = lmfit.models.ExpressionModel(eqr_temps['z'], independent_vars=['x'])

    for amplitude in sweep_points_2d:
        x_model = lmfit.models.ExpressionModel(eqr_temps['x'], independent_vars=['x'])
        y_model = lmfit.models.ExpressionModel(eqr_temps['y'], independent_vars=['x'])
        z_model = lmfit.models.ExpressionModel(eqr_temps['z'], independent_vars=['x'])

        fig, (axx,axy,axz,axr) = plt.subplots(4,1, figsize=(15,15), sharex=True)
        fit_res = fit_results[float(amplitude)]

        g_params= fit_res['g'].params
        e_params = fit_res['e'].params

        swpts= sweep_points_1d
        swpts_dense = np.linspace(swpts[0],swpts[-1],1000)



        ctrl0_x_data = 1-2*processed_data_dict[float(amplitude)]['g']['x']["population"]
        ctrl0_y_data = 1-2*processed_data_dict[float(amplitude)]['g']['y']["population"]
        ctrl0_z_data = 1-2*processed_data_dict[float(amplitude)]['g']['z']["population"]
        ctrl1_x_data = 1-2*processed_data_dict[float(amplitude)]['e']['x']["population"]
        ctrl1_y_data = 1-2*processed_data_dict[float(amplitude)]['e']['y']["population"]
        ctrl1_z_data = 1-2*processed_data_dict[float(amplitude)]['e']['z']["population"]

        ctrl0_x = x_model.eval(params=g_params, x=swpts_dense )
        ctrl0_y = y_model.eval(params=g_params, x=swpts_dense )
        ctrl0_z = z_model.eval(params=g_params, x=swpts_dense )
        ctrl1_x = x_model.eval(params=e_params, x=swpts_dense )
        ctrl1_y = y_model.eval(params=e_params, x=swpts_dense )
        ctrl1_z = z_model.eval(params=e_params, x=swpts_dense )

        r0_traj = np.array([ctrl0_x,ctrl0_y,ctrl0_z]).T
        r1_traj = np.array([ctrl1_x,ctrl1_y,ctrl1_z]).T
        r0_traj_data = np.array([ctrl0_x_data, ctrl0_y_data, ctrl0_z_data]).T
        r1_traj_data = np.array([ctrl1_x_data, ctrl1_y_data, ctrl1_z_data]).T
        diff_data = r0_traj_data - r1_traj_data
        diff = r0_traj - r1_traj
        R_data = 0.5 * np.sum(diff_data**2, axis=1)
        R = 0.5 *np.sum(diff**2, axis=1)



        axx.scatter(swpts,ctrl0_x_data, color='blue', linewidths=1.0, label='ctrl in |0>' )
        axx.plot(swpts_dense, ctrl0_x, color='blue')
        axx.scatter(swpts,ctrl1_x_data, color='red', label='ctrl in |1>' )
        axx.plot(swpts_dense, ctrl1_x, color='red')  
        axx.set_ylabel('<X(t)>', fontsize = 20)
        axx.set_title('Pauli Expectation Value', fontsize = 20)
        axx.set_ylim(-1.0,1.0)

        axy.scatter(swpts,ctrl0_y_data, color='blue', linewidths=1.0, label='ctrl in |0>' )
        axy.plot(swpts_dense, ctrl0_y, color='blue')
        axy.scatter(swpts,ctrl1_y_data, color='red', label='ctrl in |1>' )
        axy.plot(swpts_dense, ctrl1_y, color='red')  
        axy.set_ylabel('<Y(t)>', fontsize = 20)
        axy.set_ylim(-1.0,1.0)

        axz.scatter(swpts,ctrl0_z_data, color='blue', linewidths=1.0, label='ctrl in |0>' )
        axz.plot(swpts_dense, ctrl0_z, color='blue')
        axz.scatter(swpts,ctrl1_z_data, color='red', label='ctrl in |1>' )
        axz.plot(swpts_dense, ctrl1_z, color='red')  
        axz.set_ylabel('Z<(t)>', fontsize = 20)
        axr.set_ylim(-1.0,1.0)
        ##########################################################
        axr.scatter(swpts, y=R_data, color = 'black')
        axr.plot(swpts_dense, R, color='black')
        axr.plot(swpts, np.linalg.norm(r0_traj_data, axis=1), color='blue')
        axr.plot(swpts, np.linalg.norm(r1_traj_data, axis=1), color='red')
        axr.set_ylim(0,1.2)
        
        figures[float(amplitude)]=fig
    return figures

   

def generate_initial_guess(x,x_data, y_data, z_data):
  
    omega_xyz = []

    for data in (x_data, y_data, z_data):
        ymin, ymax = np.percentile(data, [10, 90]) # should be fixed
        if ymax - ymin < 0.2:
            # oscillation amplitude might be almost zero,
            # then exclude from average because of lower SNR
            continue
        fft_freq, fft_phase = find_oscillation_frequency_and_phase(x,data) # this should be replaced 
        omega_xyz.append(fft_freq)
    if omega_xyz:
        
        omega = 2 * np.pi * np.average(omega_xyz)
    else:
        omega = 1e-3


    zmin, zmax = np.percentile(z_data, [10, 90])
    theta = np.arccos(np.sqrt((zmax - zmin) / 2))

    

    # The FFT might be up to 1/2 bin off
    df = 2 * np.pi / ((x[1] - x[0]) * len(x))

    print(f"GENERATED VALUES omega:{omega/(2*np.pi*1e6)} MHz,theta{theta} df:{df/(2*np.pi*1e6)} MHz")
    

    init_param_set=[]
    
    for omega_shifted in [omega, omega - df / 2, omega + df / 2]: #theta sweep should be added
        for theta in [theta-]
            for phi in np.linspace(-np.pi, np.pi, 21):
                init_param = lmfit.Parameters()
                init_param.set(
                    px=dict(value=omega_shifted * np.cos(theta) * np.cos(phi)),
                    py=dict(value=omega_shifted * np.cos(theta) * np.sin(phi)),
                    pz=dict(value=omega_shifted * np.sin(theta)),
                    t_off=dict(value=1e-8,min=0.0, max=np.inf),
                    b=dict(value=1e-9, min=-1,max=1))
                init_param_set.append(init_param)
            
    if omega < df:
        print()
        # empirical guess for low frequency case
        emp_param = lmfit.Parameters()
        emp_param.set(
            px=dict(value=omega), 
            py=dict(value=omega), 
            pz=dict(value=0),
            t_off=dict(value=0.0,min=0.0, max=np.inf),
            b=dict(value=1e-9, min=-1,max=1))
        init_param_set.append(emp_param)
    return init_param_set




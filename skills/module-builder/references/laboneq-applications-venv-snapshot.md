# laboneq_applications venv snapshot

- generated_utc: `2026-03-11 23:13:57Z`
- python: `/Users/yalgaeahn/Research/20_Projects/qubit-experiment/.venv/bin/python`
- package_version: `25.10.0`
- package_path: `/Users/yalgaeahn/Research/20_Projects/qubit-experiment/.venv/lib/python3.12/site-packages/laboneq_applications`

## Inventory
- experiments modules: `14`
- analysis modules: `17`
- contrib experiment modules: `11`
- contrib analysis modules: `9`

## Core experiment contracts
| module | workflow | create_experiment | temporary_qpu | temporary_quantum_elements | update_qpu | task_count |
| --- | --- | --- | --- | --- | --- | --- |
| laboneq_applications.experiments.amplitude_fine | experiment_workflow_x90(session, qpu, qubits, repetitions, temporary_parameters, options) | create_experiment(qpu, qubits, amplification_qop, repetitions, options) | yes | yes | yes | 1 |
| laboneq_applications.experiments.amplitude_rabi | experiment_workflow(session, qpu, qubits, amplitudes, temporary_parameters, options) | create_experiment(qpu, qubits, amplitudes, options) | yes | yes | yes | 1 |
| laboneq_applications.experiments.dispersive_shift | experiment_workflow(session, qpu, qubit, frequencies, states, temporary_parameters, options) | create_experiment(qpu, qubit, frequencies, states, options) | yes | yes | yes | 1 |
| laboneq_applications.experiments.drag_q_scaling | experiment_workflow(session, qpu, qubits, q_scalings, temporary_parameters, options) | create_experiment(qpu, qubits, q_scalings, options) | yes | yes | yes | 1 |
| laboneq_applications.experiments.echo | experiment_workflow(session, qpu, qubits, delays, temporary_parameters, options) | create_experiment(qpu, qubits, delays, options) | yes | yes | yes | 1 |
| laboneq_applications.experiments.iq_blobs | experiment_workflow(session, qpu, qubits, states, temporary_parameters, options) | create_experiment(qpu, qubits, states, options) | yes | yes | no | 1 |
| laboneq_applications.experiments.lifetime_measurement | experiment_workflow(session, qpu, qubits, delays, temporary_parameters, options) | create_experiment(qpu, qubits, delays, options) | yes | yes | yes | 1 |
| laboneq_applications.experiments.options | - | - | no | no | no | 0 |
| laboneq_applications.experiments.qubit_spectroscopy | experiment_workflow(session, qpu, qubits, frequencies, temporary_parameters, options) | create_experiment(qpu, qubits, frequencies, options) | yes | yes | yes | 1 |
| laboneq_applications.experiments.qubit_spectroscopy_amplitude | experiment_workflow(session, qpu, qubits, frequencies, amplitudes, temporary_parameters, options) | create_experiment(qpu, qubits, frequencies, amplitudes, options) | yes | yes | no | 1 |
| laboneq_applications.experiments.ramsey | experiment_workflow(session, qpu, qubits, delays, detunings, temporary_parameters, options) | create_experiment(qpu, qubits, delays, detunings, options) | yes | yes | yes | 1 |
| laboneq_applications.experiments.resonator_spectroscopy | experiment_workflow(session, qpu, qubit, frequencies, temporary_parameters, options) | create_experiment(qpu, qubit, frequencies, options) | yes | yes | yes | 1 |
| laboneq_applications.experiments.resonator_spectroscopy_amplitude | experiment_workflow(session, qpu, qubit, frequencies, amplitudes, temporary_parameters, options) | create_experiment(qpu, qubit, frequencies, amplitudes, options) | yes | yes | no | 1 |
| laboneq_applications.experiments.time_traces | experiment_workflow(session, qpu, qubits, states, temporary_parameters, options) | create_experiment(qpu, qubit, state, options) | yes | yes | yes | 1 |

## Core analysis contracts
| module | workflow | task_count | mentions_new_parameter_values |
| --- | --- | --- | --- |
| laboneq_applications.analysis.amplitude_fine | analysis_workflow(result, qubits, amplification_qop, repetitions, target_angle, phase_offset, parameter_to_update, options) | 5 | yes |
| laboneq_applications.analysis.amplitude_rabi | analysis_workflow(result, qubits, amplitudes, options) | 3 | yes |
| laboneq_applications.analysis.calibration_traces_rotation | - | 2 | no |
| laboneq_applications.analysis.dispersive_shift | analysis_workflow(result, qubit, frequencies, states, options) | 4 | yes |
| laboneq_applications.analysis.drag_q_scaling | analysis_workflow(result, qubits, q_scalings, options) | 4 | yes |
| laboneq_applications.analysis.echo | analysis_workflow(result, qubits, delays, options) | 4 | yes |
| laboneq_applications.analysis.fitting_helpers | - | 0 | no |
| laboneq_applications.analysis.iq_blobs | analysis_workflow(result, qubits, states, options) | 6 | no |
| laboneq_applications.analysis.lifetime_measurement | analysis_workflow(result, qubits, delays, options) | 3 | yes |
| laboneq_applications.analysis.options | - | 0 | no |
| laboneq_applications.analysis.plotting_helpers | - | 5 | no |
| laboneq_applications.analysis.qubit_spectroscopy | analysis_workflow(result, qubits, frequencies, options) | 4 | yes |
| laboneq_applications.analysis.ramsey | analysis_workflow(result, qubits, delays, detunings, options) | 3 | yes |
| laboneq_applications.analysis.resonator_spectroscopy | analysis_workflow(result, qubit, frequencies, options) | 5 | yes |
| laboneq_applications.analysis.resonator_spectroscopy_dc_bias | analysis_workflow(result, qubit, frequencies, voltages, options) | 6 | yes |
| laboneq_applications.analysis.spectroscopy_two_dimensional_plotting | analysis_workflow(result, qubits, sweep_points_1d, sweep_points_2d, label_sweep_points_1d, label_sweep_points_2d, scaling_sweep_points_2d, options) | 0 | no |
| laboneq_applications.analysis.time_traces | analysis_workflow(result, qubits, states, options) | 7 | yes |

## Options classes
| module | class | decorators | bases |
| --- | --- | --- | --- |
| laboneq_applications.experiments.options | BaseExperimentOptions | task_options | - |
| laboneq_applications.experiments.options | TuneupExperimentOptions | task_options | - |
| laboneq_applications.experiments.options | ResonatorSpectroscopyExperimentOptions | task_options | - |
| laboneq_applications.experiments.options | TuneUpWorkflowOptions | workflow_options | - |
| laboneq_applications.experiments.options | QubitSpectroscopyExperimentOptions | task_options | - |
| laboneq_applications.experiments.options | TWPASpectroscopyExperimentOptions | task_options | - |
| laboneq_applications.experiments.options | TWPATuneUpExperimentOptions | task_options | - |
| laboneq_applications.experiments.options | TWPATuneUpWorkflowOptions | workflow_options | - |
| laboneq_applications.analysis.options | DoFittingOption | task_options | - |
| laboneq_applications.analysis.options | FitDataOptions | task_options | - |
| laboneq_applications.analysis.options | ExtractQubitParametersTransitionOptions | task_options | - |
| laboneq_applications.analysis.options | BasePlottingOptions | task_options | - |
| laboneq_applications.analysis.options | PlotPopulationOptions | task_options | ExtractQubitParametersTransitionOptions, BasePlottingOptions |
| laboneq_applications.analysis.options | TuneUpAnalysisWorkflowOptions | workflow_options | - |

## Contrib modules
- contrib.experiments:
  - `laboneq_applications.contrib.experiments.amplitude_rabi_chevron`
  - `laboneq_applications.contrib.experiments.calibrate_cancellation`
  - `laboneq_applications.contrib.experiments.measure_gain_curve`
  - `laboneq_applications.contrib.experiments.measurement_qndness`
  - `laboneq_applications.contrib.experiments.scan_pump_parameters`
  - `laboneq_applications.contrib.experiments.signal_propagation_delay`
  - `laboneq_applications.contrib.experiments.single_qubit_randomized_benchmarking`
  - `laboneq_applications.contrib.experiments.spin_locking`
  - `laboneq_applications.contrib.experiments.time_rabi`
  - `laboneq_applications.contrib.experiments.time_rabi_chevron`
  - `laboneq_applications.contrib.experiments.twpa_spectroscopy`
- contrib.analysis:
  - `laboneq_applications.contrib.analysis.amplitude_rabi_chevron`
  - `laboneq_applications.contrib.analysis.calibrate_cancellation`
  - `laboneq_applications.contrib.analysis.measure_gain_curve`
  - `laboneq_applications.contrib.analysis.measurement_qndness`
  - `laboneq_applications.contrib.analysis.scan_pump_parameters`
  - `laboneq_applications.contrib.analysis.signal_propagation_delay`
  - `laboneq_applications.contrib.analysis.single_qubit_randomized_benchmarking`
  - `laboneq_applications.contrib.analysis.time_rabi`
  - `laboneq_applications.contrib.analysis.time_rabi_chevron`

## Canonical naming frequency
- `analysis_workflow`: `13` modules
- `experiment_workflow`: `12` modules
- `experiment_workflow_x90`: `1` modules

## New module guardrails inferred from snapshot
- Define workflow entrypoints with `@workflow.workflow`.
- Keep experiment modules centered on `experiment_workflow` + `create_experiment`.
- Keep runtime overrides on temporary copies via `temporary_qpu` and `temporary_quantum_elements_from_qpu`.
- For update-capable workflows, keep persistent updates on `analysis_results.output["new_parameter_values"]` + `update_qpu`.
- Use option classes based on `@task_options` / `@workflow_options` from applications patterns.

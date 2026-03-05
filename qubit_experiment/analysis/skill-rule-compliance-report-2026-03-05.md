# Skill Rule Compliance Report (experiments + analysis)

- generated: `2026-03-04T23:21:24.835249+00:00`
- scope: `experiments/*.py`, `analysis/*.py`
- basis: `skills/module-builder/SKILL.md` Contract Rules + Validation Checklist

## Summary
- total files reviewed: `92`
- syntax errors: `0`
- empty module files: `1`
- missing module docstring: `0`
- short module docstring (<2 non-empty lines): `30`
- deprecated `update_qubits` call sites: `0`
- `temporary_parameters` without `temporary_qpu` pattern: `0`
- experiment modules missing workflow decorator: `0`
- experiment modules missing `create_experiment`: `0`
- `create_experiment` missing task/qubit-experiment decorators: `1`
- `create_experiment` without explicit `ValueError` guard: `17`
- Python `if/for/while` in `@workflow.workflow` bodies: `6`
- forbidden container-method calls in workflow bodies: `0`
- possible OptionBuilder assignment-style writes in workflow bodies: `1`
- `workflow.comment/log/save_artifact` outside `@workflow.task`: `2`

## P0: Syntax Errors
- none

## P0: Empty Module Files
- `experiments/__init__.py:1` empty or comment-only module

## P1: Deprecated update_qubits Calls (code)
- none

## P1: Missing Module Docstring
- none

## P2: temporary_parameters Pattern Drift
- none

## P2: Experiment Structure Drift - Missing @workflow.workflow
- none

## P2: Experiment Structure Drift - Missing create_experiment
- none

## P2: create_experiment Missing Required Decorators
- `experiments/trajectory.py 02-12-14-632.py:176` create_experiment missing decorators: dsl.qubit_experiment

## P2: create_experiment Missing Explicit ValueError Guard
- `experiments/amplitude_rabi_chevron.py:125` create_experiment has no explicit ValueError guard
- `experiments/calibrate_cancellation.py:154` create_experiment has no explicit ValueError guard
- `experiments/ef_spectroscopy.py:139` create_experiment has no explicit ValueError guard
- `experiments/iq_cloud.py:117` create_experiment has no explicit ValueError guard
- `experiments/measure_gain_curve.py:156` create_experiment has no explicit ValueError guard
- `experiments/measurement_qndness.py:156` create_experiment has no explicit ValueError guard
- `experiments/qubit_gate_spectroscopy_amplitude.py:145` create_experiment has no explicit ValueError guard
- `experiments/qubit_spectroscopy.py:137` create_experiment has no explicit ValueError guard
- `experiments/qubit_spectroscopy_amplitude.py:146` create_experiment has no explicit ValueError guard
- `experiments/scan_pump_parameters.py:206` create_experiment has no explicit ValueError guard
- `experiments/signal_propagation_delay.py:120` create_experiment has no explicit ValueError guard
- `experiments/single_qubit_randomized_benchmarking.py:235` create_experiment has no explicit ValueError guard
- `experiments/spin_locking.py:145` create_experiment has no explicit ValueError guard
- `experiments/time_rabi.py:134` create_experiment has no explicit ValueError guard
- `experiments/time_rabi_chevron.py:125` create_experiment has no explicit ValueError guard
- `experiments/trajectory.py 02-12-14-632.py:176` create_experiment has no explicit ValueError guard
- `experiments/twpa_spectroscopy.py:122` create_experiment has no explicit ValueError guard

## P2: Workflow Body Uses Python Control Flow
- `analysis/drag_q_scaling.py:203` Python if in workflow function `analysis_workflow`
- `analysis/drag_q_scaling.py:206` Python if in workflow function `analysis_workflow`
- `analysis/drag_q_scaling.py:230` Python if in workflow function `analysis_workflow`
- `experiments/iq_time_trace.py:103` Python if in workflow function `experiment_workflow`
- `experiments/readout_mid_sweep.py:357` Python if in workflow function `experiment_workflow`
- `experiments/readout_mid_sweep.py:361` Python if in workflow function `experiment_workflow`

## P2: Workflow Body Uses Forbidden Reference Container Methods
- none

## P2: Option Assignment Style Writes in Workflow Body
- `experiments/two_qubit_state_tomography.py:267` possible OptionBuilder assignment `conv_analysis_options.do_plotting = ...` in workflow function `experiment_workflow`

## P2: workflow.comment/log/save_artifact Outside @workflow.task
- `analysis/iq_cloud.py:516` workflow.save_artifact(...) called outside @workflow.task in `_save_artifact_if_available`
- `experiments/ramsey.py:119` workflow.log(...) called outside @workflow.task in `_maybe_log_ignored_detunings`

## Info: Short Module Docstrings
- `analysis/fitting_helpers.py:4` short module docstring (1 non-empty line)
- `analysis/iq_cloud.py:1` short module docstring (1 non-empty line)
- `analysis/iq_time_trace.py:4` short module docstring (1 non-empty line)
- `analysis/plot_theme.py:1` short module docstring (1 non-empty line)
- `analysis/plotting_helpers.py:1` short module docstring (1 non-empty line)
- `analysis/readout_amplitude_sweep.py:1` short module docstring (1 non-empty line)
- `analysis/readout_frequency_sweep.py:1` short module docstring (1 non-empty line)
- `analysis/readout_integration_delay_sweep.py:1` short module docstring (1 non-empty line)
- `analysis/readout_length_sweep.py:1` short module docstring (1 non-empty line)
- `analysis/readout_mid_sweep.py:1` short module docstring (1 non-empty line)
- `analysis/readout_sweep_common.py:1` short module docstring (1 non-empty line)
- `analysis/residual_zz_echo.py:4` short module docstring (1 non-empty line)
- `analysis/three_qubit_state_tomography.py:1` short module docstring (1 non-empty line)
- `analysis/two_qubit_state_tomography.py:1` short module docstring (1 non-empty line)
- `experiments/direct_cr_hamiltonian_tomography.py:1` short module docstring (1 non-empty line)
- `experiments/iq_cloud.py:1` short module docstring (1 non-empty line)
- `experiments/iq_cloud_common.py:1` short module docstring (1 non-empty line)
- `experiments/iq_time_trace.py:4` short module docstring (1 non-empty line)
- `experiments/options.py:4` short module docstring (1 non-empty line)
- `experiments/readout_amplitude_sweep.py:1` short module docstring (1 non-empty line)
- `experiments/readout_frequency_sweep.py:1` short module docstring (1 non-empty line)
- `experiments/readout_integration_delay_sweep.py:1` short module docstring (1 non-empty line)
- `experiments/readout_length_sweep.py:1` short module docstring (1 non-empty line)
- `experiments/rip.py:4` short module docstring (1 non-empty line)
- `experiments/three_qubit_readout_calibration.py:1` short module docstring (1 non-empty line)
- `experiments/three_qubit_state_tomography.py:1` short module docstring (1 non-empty line)
- `experiments/three_qubit_tomography_common.py:1` short module docstring (1 non-empty line)
- `experiments/two_qubit_readout_calibration.py:1` short module docstring (1 non-empty line)
- `experiments/two_qubit_state_tomography.py:1` short module docstring (1 non-empty line)
- `experiments/two_qubit_tomography_common.py:1` short module docstring (1 non-empty line)

## Info: update_qubits Import-Only (no call)
- `experiments/ef_spectroscopy.py:1` update_qubits imported but never called
- `experiments/new_rip_echo.py:1` update_qubits imported but never called
- `experiments/rip.py:1` update_qubits imported but never called
- `experiments/rip2.py:1` update_qubits imported but never called
- `experiments/rip3.py:1` update_qubits imported but never called
- `experiments/rip4.py:1` update_qubits imported but never called
- `experiments/rip5.py:1` update_qubits imported but never called
- `experiments/rip_refocus.py:1` update_qubits imported but never called

## Info: update_qubits Mention in Docs/Comments Only
- `experiments/measurement_qndness.py:100` update_qubits mentioned in docs/comments only
- `experiments/qubit_spectroscopy.py:68` update_qubits mentioned in docs/comments only
- `experiments/ramsey.py:145` update_qubits mentioned in docs/comments only
- `experiments/time_rabi.py:67` update_qubits mentioned in docs/comments only
- `experiments/twpa_spectroscopy.py:65` update_qubits mentioned in docs/comments only

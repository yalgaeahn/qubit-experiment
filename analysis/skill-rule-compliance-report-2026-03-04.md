# Skill Rule Compliance Report (experiments + analysis)

- generated: `2026-03-04T00:43:59.926680+00:00`
- scope: `experiments/*.py`, `analysis/*.py`
- basis: `skills/laboneq-new-module-builder/SKILL.md` Contract Rules + Validation Checklist

## Summary
- total files reviewed: `90`
- empty module files: `2`
- missing module docstring: `0`
- short module docstring (<2 non-empty lines): `29`
- deprecated `update_qubits` call sites: `0`
- `temporary_parameters` without `temporary_qpu` pattern: `0`
- experiment modules missing workflow decorator: `0`
- experiment modules missing `create_experiment`: `1`
- `create_experiment` missing `@workflow.task`+`@dsl.qubit_experiment`: `1`
- `create_experiment` without explicit `ValueError` guard: `17`

## P0: Empty Module Files
- `experiments/__init__.py:1`
- `analysis/__init__.py:1`

## P1: Deprecated update_qubits Calls (code)
- none

## P1: Missing Module Docstring
- none

## P2: temporary_parameters Pattern Drift
- none

## P2: Experiment Structure Drift
- none

## P2: Experiment Missing create_experiment
- `experiments/readout_length_sweep.py:1`

## P2: create_experiment Missing Required Decorators
- `experiments/trajectory.py 02-12-14-632.py:176` decorators=['workflow.task']

## P3: create_experiment Without ValueError Guard (heuristic)
- `experiments/amplitude_rabi_chevron.py:125`
- `experiments/calibrate_cancellation.py:154`
- `experiments/ef_spectroscopy.py:139`
- `experiments/iq_cloud.py:117`
- `experiments/measure_gain_curve.py:156`
- `experiments/measurement_qndness.py:156`
- `experiments/qubit_gate_spectroscopy_amplitude.py:145`
- `experiments/qubit_spectroscopy.py:137`
- `experiments/qubit_spectroscopy_amplitude.py:146`
- `experiments/scan_pump_parameters.py:206`
- `experiments/signal_propagation_delay.py:120`
- `experiments/single_qubit_randomized_benchmarking.py:235`
- `experiments/spin_locking.py:145`
- `experiments/time_rabi.py:134`
- `experiments/time_rabi_chevron.py:125`
- `experiments/trajectory.py 02-12-14-632.py:176`
- `experiments/twpa_spectroscopy.py:122`

## P3: Short Module Docstring (<2 lines)
- `experiments/direct_cr_hamiltonian_tomography.py:1`
- `experiments/iq_cloud.py:1`
- `experiments/iq_cloud_common.py:1`
- `experiments/iq_time_trace.py:1`
- `experiments/options.py:1`
- `experiments/readout_amplitude_sweep.py:1`
- `experiments/readout_frequency_sweep.py:1`
- `experiments/readout_integration_delay_sweep.py:1`
- `experiments/readout_length_sweep.py:1`
- `experiments/rip.py:1`
- `experiments/three_qubit_readout_calibration.py:1`
- `experiments/three_qubit_state_tomography.py:1`
- `experiments/three_qubit_tomography_common.py:1`
- `experiments/two_qubit_readout_calibration.py:1`
- `experiments/two_qubit_state_tomography.py:1`
- `experiments/two_qubit_tomography_common.py:1`
- `analysis/fitting_helpers.py:1`
- `analysis/iq_cloud.py:1`
- `analysis/iq_time_trace.py:1`
- `analysis/readout_amplitude_sweep.py:1`
- `analysis/readout_frequency_sweep.py:1`
- `analysis/readout_integration_delay_sweep.py:1`
- `analysis/readout_length_sweep.py:1`
- `analysis/readout_mid_sweep.py:1`
- `analysis/readout_sweep_common.py:1`
- `analysis/residual_zz_echo.py:1`
- `analysis/three_qubit_state_tomography.py:1`
- `analysis/trajectory.py:1`
- `analysis/two_qubit_state_tomography.py:1`

## Info: update_qubits Import-Only (no call)
- `experiments/ef_spectroscopy.py:38`
- `experiments/new_rip_echo.py:44`
- `experiments/ramsey.py:51`
- `experiments/rip.py:40`
- `experiments/rip2.py:41`
- `experiments/rip3.py:41`
- `experiments/rip4.py:41`
- `experiments/rip5.py:44`
- `experiments/rip_refocus.py:41`

## Info: update_qubits Mention in Docs/Comments Only
- `experiments/measurement_qndness.py:100`
- `experiments/qubit_spectroscopy.py:68`
- `experiments/time_rabi.py:67`
- `experiments/twpa_spectroscopy.py:65`

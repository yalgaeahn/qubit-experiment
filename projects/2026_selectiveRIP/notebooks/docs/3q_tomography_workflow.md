# 3Q Quantum State Tomography Workflow

This guide shows the canonical product-state workflow for 3-qubit quantum state tomography with:

- 27 Pauli-product settings (`3^3`)
- 8 computational outcomes (`2^3`)
- readout-mitigated MLE using soft counts

## 1) Run 3Q readout calibration

```python
from qubit_experiment.experiments import three_qubit_readout_calibration

readout_cal_result = three_qubit_readout_calibration.experiment_workflow(
    session=session,
    qpu=qpu,
    qubits=[q0, q1, q2],
)
```

## 2) Run 3Q tomography workflow

```python
from qubit_experiment.analysis import threeq_qst as threeq_qst_analysis
from qubit_experiment.experiments import threeq_qst

opts = threeq_qst.experiment_workflow.options()
opts.do_readout_calibration(False)
opts.do_analysis(True)
opts.initial_state("+++")
opts.custom_prep(False)
opts.do_convergence_validation(True)
opts.convergence_repeats_per_state(1)
opts.convergence_suite_states(("000", "111", "+++", "---"))
opts.convergence_do_plotting(True)
opts.do_shot_sweep_convergence(True)
opts.shot_sweep_log2_values(tuple(range(3, 9)))
opts.shot_sweep_suite_states(("000", "111", "+++", "---"))
opts.shot_sweep_repeats_per_point(3)
opts.shot_sweep_do_plotting(True)

analysis_options = threeq_qst_analysis.analysis_workflow.options()
analysis_options.do_plotting(True)

threeq_result = threeq_qst.run_bundle(
    session=session,
    qpu=qpu,
    qubits=[q0, q1, q2],
    bus=bus,
    readout_calibration_result=readout_cal_result,
    options=opts,
    analysis_options=analysis_options,
)

analysis_result = threeq_result["analysis_result"]
convergence_report = threeq_result["convergence_report"]
shot_sweep_report = threeq_result["shot_sweep_report"]
```

If you need the raw experiment workflow output only, use:

```python
raw_result = threeq_qst.experiment_workflow(
    session=session,
    qpu=qpu,
    qubits=[q0, q1, q2],
    bus=bus,
    readout_calibration_result=readout_cal_result,
    options=opts,
)
```

## 3) Inspect outputs

`run_bundle(...)` returns a plain Python dictionary with:

- `tomography_result`
- `readout_calibration_result`
- `analysis_result`
- `convergence_report`
- `shot_sweep_report`
- `initial_state`
- `target_state_effective`
- `custom_prep`

Inside `analysis_result`:

- `assignment_matrix` (8x8)
- `tomography_counts` and `tomography_counts_hard`
- `rho_hat_real`, `rho_hat_imag` (8x8)
- `predicted_counts`
- `metrics` (`trace`, `purity`, `min_eigenvalue`, `fidelity_to_target`)
- `classification_diagnostics`
- `optimization_convergence`

Inside `convergence_report`:

- `suite_states`
- `repeats_per_state`
- `raw_run_records`
- `statistical_convergence`
- `main_run_optimization_convergence`

Inside `shot_sweep_report`:

- `suite_states`
- `shot_log2_values`
- `shot_counts`
- `repeats_per_point`
- `raw_run_records`
- `failed_runs`
- `validation_checks`
- `aggregated_stats`
- `final_summary`

## Notes

- `threeq_qst` is a product-state canonical module. It does not expose `use_rip`,
  `validation_mode`, or `used_rip`.
- `custom_prep=True` is reserved for a future preparation path and is not implemented
  in v1.
- Built-in convergence/shot-sweep defaults use the full 16-state product suite
  (`000..111` and `+++..---`). In notebooks, it is usually better to override these
  with a smaller suite for runtime control.
- Reconstruction uses soft outcomes from per-shot IQ posteriors.
- Hard outcomes are exported for diagnostics and comparison only.
- The run is informationally complete with all 27 settings.

# 3Q Quantum State Tomography Workflow

This guide shows the minimal workflow for 3-qubit quantum state tomography with:

- 27 Pauli-product settings (`3^3`)
- 8 computational outcomes (`2^3`)
- readout-mitigated MLE using soft counts

## 1) Run 3Q readout calibration

```python
from experiments import three_qubit_readout_calibration

readout_cal_result = three_qubit_readout_calibration.experiment_workflow(
    session=session,
    qpu=qpu,
    qubits=[q0, q1, q2],
)
```

## 2) Run 3Q tomography workflow

```python
from experiments import three_qubit_state_tomography

opts = three_qubit_state_tomography.experiment_workflow.options()
opts.do_readout_calibration = False
opts.do_analysis = True
opts.initial_state = "+++"
opts.use_rip = True

threeq_result = three_qubit_state_tomography.experiment_workflow(
    session=session,
    qpu=qpu,
    qubits=[q0, q1, q2],
    bus=bus,
    readout_calibration_result=readout_cal_result,
    target_state="ghz_plus",  # optional
    options=opts,
)
```

## 3) Inspect outputs

The workflow output dictionary includes:

- `tomography_result`
- `readout_calibration_result`
- `analysis_result`

Inside `analysis_result`:

- `assignment_matrix` (8x8)
- `tomography_counts` and `tomography_counts_hard`
- `rho_hat_real`, `rho_hat_imag` (8x8)
- `predicted_counts`
- `metrics` (`trace`, `purity`, `min_eigenvalue`, `fidelity_to_target`)
- `classification_diagnostics`
- `optimization_convergence`

## Notes

- Reconstruction uses soft outcomes from per-shot IQ posteriors.
- Hard outcomes are exported for diagnostics and comparison only.
- The run is informationally complete with all 27 settings.

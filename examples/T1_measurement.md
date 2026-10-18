# Single-shot T1 measurement

`qubit_experiment.experiments.T1_measurement` keeps the local lifetime sequence
and collects individual integrated IQ shots instead of hardware averages. Its
dedicated analysis provides:

- **Population data:** small blue markers labeled `data`, without per-point bars.
- **Fit:** a pure red exponential curve, with darker blue **95% confidence** and
  lighter blue **95% prediction** bands when fit covariance is available.
- **Calibration:** black dashed reference means and gray **±SEM** bands around
  population 0 and 1, combined under one `g/e reference ± SEM` legend entry.
- **Titles:** always include the local plot-generation timestamp.
- **Raw I/Q:** mean ± SEM error bars, with `SEM = sample SD / sqrt(shots)`.
- The original **unweighted** exponential fit, with offset fixed to zero unless
  overridden by fit hints. Plot styling does not change the fit weights.

Fit bands use `ModelResult.eval_uncertainty(sigma=0.95)`: Student-t-scaled local
covariance propagation for confidence, plus residual variance for prediction.
Both are approximate **pointwise** intervals, assuming independent normal
mean-data residuals with constant variance and fixed calibration. The prediction
target is **one new shot-averaged datum** collected with the same shot count and
conditions, not one individual IQ shot. Bands are omitted if fit uncertainty is
unavailable. They do not propagate calibration uncertainty or drift.

The calibration means define a single fixed IQ projection for all shots.
SEM describes uncertainty in the mean for independent shots; it does not include
reference-mean uncertainty, drift corrections, or readout bias. Reference bands
show the SEM of the projected or raw reference shots. For normalized population,
the projection treats calibration means as fixed; these bands do not propagate
calibration uncertainty into the T1 data or fit.
No individual normalized IQ value is clipped to [0, 1].

## Acquisition

Run only with an already configured session/QPU and calibrated pulse/readout
parameters. The delay grid below is an example, not a device recommendation.

```python
from pathlib import Path

import numpy as np
from laboneq import serializers
from laboneq_applications.tasks.parameter_updating import (
    temporary_qpu,
    temporary_quantum_elements_from_qpu,
)

from qubit_experiment.experiments import T1_measurement

qubit_uids = ["q0"]
delays = [np.linspace(0, 150e-6, 61)]  # one array per qubit when passing a list
transition = "ge"
temporary_parameters = None

options = T1_measurement.experiment_workflow.options()
options.count(1024)  # individual shots per delay AND per reference state
options.transition(transition)
options.cal_states(transition)  # use "ef" with transition="ef"
options.update(False)
options.do_plotting(True)
options.save_figures(True)
options.close_figures(False)

# Save the actual analysis context before optional QPU updates can change it.
# Use a new directory for each run.
run_dir = Path("/absolute/path/to/new-T1-run")
run_dir.mkdir(parents=True, exist_ok=False)
runtime_qpu = temporary_qpu.func(qpu, temporary_parameters)
runtime_qubits = temporary_quantum_elements_from_qpu.func(runtime_qpu, qubit_uids)
for qubit in runtime_qubits:
    serializers.save(qubit, run_dir / f"{qubit.uid}.json")
np.savez(
    run_dir / "context.npz",
    qubit_uids=np.array(qubit_uids),
    transition=np.array(transition),
    **{f"delays_{uid}": grid for uid, grid in zip(qubit_uids, delays)},
)

run = T1_measurement.experiment_workflow(
    session=session,
    qpu=qpu,
    qubits=qubit_uids,
    delays=delays,
    temporary_parameters=temporary_parameters,
    options=options,
).run()

raw_result = run.output  # original LabOne Q Results with all single shots
serializers.save(raw_result, run_dir / "raw.json")

analysis_result = run.tasks["analysis_workflow"]
parameters = analysis_result.output["new_parameter_values"]
statistics = analysis_result.output["processed_data"]["q0"]
population_figure = analysis_result.tasks["plot_population"].output["q0"]
raw_iq_figure = analysis_result.tasks["plot_raw_complex_data_1d"].output["q0"]
population_figure.savefig(run_dir / "T1_q0.png", dpi=180)
raw_iq_figure.savefig(run_dir / "Raw_data_q0.png", dpi=180)
```

For one UID string, pass a single 1D delay array instead. For several qubits,
pass a list of UIDs and a corresponding list of equal-length delay grids; the
grids may contain different delay values. Metadata identifies each qubit's delay
axis, including the case where the shot count equals the number of delay points.

Only `INTEGRATION`, `SINGLE_SHOT`, calibration rotation, and at least two shots
are supported. `use_cal_traces(False)`, `do_rotation(False)`, and `do_pca(True)`
raise `ValueError`. `transition` and the two ordered `cal_states` must match:
`"ge"`/`"ge"` or `"ef"`/`"ef"`.

`do_analysis(False)` returns raw results without analysis or updates.
`do_fitting(False)` still calculates statistics and plots data/reference bands.
The independent plot switches are `do_plotting`, `do_raw_data_plotting`, and
`do_qubit_population_plotting`. `save_figures` and `close_figures` retain the
usual LabOne Q behavior. `options.logstore([])` disables automatic workflow
execution logs if those records are not wanted.

## Reanalyze saved results

This does not connect to hardware or modify a QPU. Reload the saved qubit
snapshots rather than using possibly changed current QPU parameters.

```python
from pathlib import Path

import numpy as np
from laboneq import serializers
from qubit_experiment.analysis import T1_measurement as T1_analysis

run_dir = Path("/absolute/path/to/new-T1-run")
raw_result = serializers.load(run_dir / "raw.json")
with np.load(run_dir / "context.npz", allow_pickle=False) as context:
    qubit_uids = context["qubit_uids"].tolist()
    transition = context["transition"].item()
    delays = [context[f"delays_{uid}"].copy() for uid in qubit_uids]
qubits = [serializers.load(run_dir / f"{uid}.json") for uid in qubit_uids]

options = T1_analysis.analysis_workflow.options()
options.transition(transition)
options.cal_states(transition)
options.save_figures(False)
options.close_figures(False)
# Optional: override the default zero-offset constraint.
# options.fit_parameters_hints({"offset": {"value": 0.0, "vary": True}})

reanalyzed = T1_analysis.analysis_workflow(
    result=raw_result,
    qubits=qubits,
    delays=delays,
    options=options,
).run()
statistics = reanalyzed.output["processed_data"][qubit_uids[0]]
print(statistics["population"], statistics["population_sem"])
```

The example records the default analysis settings and transition. If custom
fit hints or other analysis overrides were used, preserve and reapply those
settings for the same fit.

## Processed statistics and updates

Each qubit's `processed_data` includes:

| Keys | Meaning |
| --- | --- |
| `sweep_points`, `num_shots`, `transition`, `cal_states` | Grid and acquisition context |
| `population`, `population_std`, `population_sem` | Per-delay projected mean, sample SD, SEM |
| `population_cal_traces`, `population_cal_traces_std`, `population_cal_traces_sem` | Reference statistics in the same projection |
| `data_raw`, `data_raw_cal_traces` | Complex IQ means |
| `raw_i`, `raw_q`, and their `_std` / `_sem` keys | Per-delay raw component statistics |
| `reference_i`, `reference_q`, and their `_std` / `_sem` keys | Raw reference component statistics |
| `uncertainty_method` | Explicit fixed-calibration SEM definition |

`old_parameter_values` / `new_parameter_values` use the existing `ge_T1` or
`ef_T1` keys. A successful positive finite T1 with a known fit uncertainty can
be applied with `options.update(True)`; otherwise that qubit's update mapping is
empty. A valid fit with unavailable covariance can still be drawn, with its
uncertainty explicitly marked unavailable. Pointwise SEM and T1 fit uncertainty
are different quantities.

Previously hardware-averaged results cannot supply these shot statistics and
are rejected. The existing `lifetime_measurement` modules remain available.

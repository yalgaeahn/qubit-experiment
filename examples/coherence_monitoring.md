# Overnight coherence monitoring

`qubit_experiment.experiments.coherence_tracking` measures T1, Ramsey T2*, and echo
T2 in that order. This is an acquisition workflow: run it only with an already
configured session and verified pulse/readout parameters. No hardware is accessed
by importing the module or building its options.

## Twelve-hour example

Here `session` and `qpu` are already configured. Adjust delay grids, detunings,
shot count, reset times, and the interval for the device. The grids below are
examples, not calibrated settings. Choose a new absolute directory for each run.

```python
import numpy as np
from qubit_experiment.experiments import coherence_tracking

options = coherence_tracking.experiment_workflow.options()
options.logstore([])  # Do not create automatic FolderStore execution-log folders.
options.tracked_parameters("coherence")
options.total_duration_s(12 * 3600)
options.interval_s(300)
options.track_frequency(True)
options.update(False)
options.continue_on_iteration_error(True)
options.do_plotting(True)
options.monitoring_directory("/absolute/path/to/new-overnight-run")
options.history_path("/absolute/path/to/coherence-history.jsonl")

output = coherence_tracking.experiment_workflow(
    session=session,
    qpu=qpu,
    qubits="q0",
    t1_delays=np.linspace(0, 150e-6, 61),
    t2_star_delays=np.linspace(0, 80e-6, 161),
    t2_delays=np.linspace(0, 150e-6, 61),
    t2_star_detunings=100e3,
    options=options,
).run().output
```

For several qubits pass UIDs as a list, one delay grid per qubit, and one detuning
per qubit (or a shared scalar). Calibration settings are forwarded from experiment
to population analysis and fitting. `options.do_pca(...)`, `do_rotation(...)`,
`do_fitting(...)`, and `fit_parameters_hints(...)` configure the common analysis.
`use_cal_traces(False)` enables the corresponding PCA fit defaults. The original
T1/echo offset models remain unchanged. Integration and spectroscopy acquisitions
are supported; sequential averaging with calibration traces is rejected up front.

## Frequency feedback

`track_frequency` defaults to `False`. When enabled it requires Ramsey analysis,
fitting, finite nonzero detunings, and `update=False`. It also records Ramsey
frequency even when `tracked_parameters="coherence"` is selected.

The initial runtime QPU is a copy with `temporary_parameters` applied. All three
measurements in an iteration use the same frequency. Accepted Ramsey estimates
are applied only to the copy, after that iteration, for the next iteration. The
input QPU and any parameter files remain unchanged. The returned
`final_frequencies` and `frequency_history` expose the separate tracking state.

An estimate is accepted only if its fit succeeded, its frequency uncertainty is
finite, its step from the current drive is at most half the absolute detuning,
and its standard error is at most one quarter of the absolute detuning. For
negative detuning the nearby signed Ramsey branch is selected. These guards do
not resolve arbitrary Ramsey aliases or rescue a lost excitation/readout signal.
Rejected estimates are recorded and the previous frequency is retained.

## Records and recovery

Long runs automatically create a unique directory under `monitoring/` beside the
history file if `monitoring_directory` is omitted. An explicit directory must not
already exist. A single pass can enable the same records with an explicit directory.
Relative `history_path` follows the existing active FolderStore/current-directory
resolution; an explicit monitoring directory is relative to the current directory.

- `manifest.json`: input/runtime qubit parameter snapshots, software versions,
  sweeps, detunings, and experiment/analysis options.
- `iteration_XXXXXX/<metric>/context.json` and `raw.json`: actual parameters and
  LabOne Q results, saved before analysis.
- `analysis.json`: processed population/calibration data, fit parameters, residuals,
  fit status, and per-qubit quality reasons.
- `events.jsonl`: run/iteration/metric starts and finishes, failures, raw-save
  checkpoints, missed slots, and frequency decisions.
- `status.json`: latest state and counts; `trend_<uid>.png`: periodically replaced
  trend with separate markers for quality warnings. Points are not joined across gaps.

The legacy history JSONL remains readable. New rows add run/iteration identity,
actual metric start time, drive frequency, and quality information. Missing fit
uncertainty is represented as `null` on disk and an unknown (`NaN`) standard error
in memory. Finite fitted values remain available even when quality is poor.
Nonpositive/nonfinite coherence times are retained in fit diagnostics, but are not
published as normal parameter values. Relative coherence-time error above 50%
produces a warning. No automatic lifetime-range or residual threshold is assumed.

To reanalyze, load raw results with `laboneq.serializers.load(path_to_raw_json)`;
use the saved context's qubit parameters, delays, detuning, and options with
`analysis.coherence_tracking.analyze_monitoring_metric_value`. This helper returns
parameter estimates, per-qubit quality, and diagnostics without modifying a QPU.
`FitDataT1Options` supplies the common analysis options for all three metrics.

A run without a terminal `run_finished` event or terminal status is incomplete.
Completed raw files and history rows can be read after interruption; the history
reader ignores malformed JSON lines. Runs do not automatically resume or reconnect.
Start a new directory after reviewing an interrupted run.

## Timing, failures, and returned output

Intervals specify start-to-start targets using a monotonic clock. Overdue slots
are skipped instead of triggering a burst of delayed measurements. No new metric
starts at or after the deadline. An in-flight compile/run may finish after it;
a hardware call that never returns cannot be forcibly terminated by this workflow.

By default, create/compile/run/analysis failures are logged and other metrics and
future iterations continue. Qubit-level analysis failures preserve other qubits'
results. Setting `continue_on_iteration_error(False)` completes the current suite
and skips later iterations after a failure. `KeyboardInterrupt`/`SystemExit` record
an interruption and propagate; no subsequent measurement starts. Required data or
history write failures stop the run. Plot errors are recorded and remain nonfatal.

Existing output keys (`analysis`, `iteration_summaries`, `tracked_parameters`) are
preserved. Additional keys are `monitoring_directory`, `run_id`, `metric_counts`,
`frequency_history`, and `final_frequencies`. Metric success/failure counts refer
to attempts; quality-warning counts refer to qubit fits. A partial suite is not
counted as a fully successful iteration. Run completion means the schedule ended,
not that every fit was valid. With `do_analysis=False`, raw recording still works
and `analysis` is `None`; frequency feedback is unavailable.

Validate a short hardware run before an unattended night. Software tests use
virtual clocks, synthetic IQ data, and offline compilation; they do not establish
hardware stability.

## Automatic workflow execution logs

Coherence tracking defaults to `logstore=[]`, so an activated global FolderStore
will not create a `*-coherence-tracking` execution-log directory, including when
validation fails. `options.logstore([])` explicitly keeps this disabled when an
older module is already loaded in a notebook kernel. Other experiment workflows
retain their own logging settings. Explicitly requested monitoring records and
history are separate from these automatic execution logs.

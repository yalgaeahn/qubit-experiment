# CW reflection resonator linewidth

The paired `qubit_experiment.experiments.resonator_linewidth_cw` and
`qubit_experiment.analysis.resonator_linewidth_cw` modules measure and fit one
qubit's **loaded readout mode** at a time. This is the hybridized mode when a
Purcell filter is present. They do not identify bare resonator losses, Qi, or Qc.

## Acquisition

```python
from copy import deepcopy
import numpy as np
from qubit_experiment.experiments import resonator_linewidth_cw

q = qpu["q2"]
parameters = deepcopy(q.parameters)
parameters.readout_range_out = -10
reference = 7.3452e9  # Expected low-power resonance, NOT the readout operating tone.
options = resonator_linewidth_cw.experiment_workflow.options()
options.count(2048)
options.integration_length(2e-6)
options.settling_time(5e-6)
options.update(False)
run = resonator_linewidth_cw.experiment_workflow(
    session=session, qpu=qpu, qubit=q.uid,
    frequencies=reference + np.linspace(-4e6, 4e6, 1601),
    amplitudes=np.array([0.000647098, 0.001294196, 0.002588392]),
    reference_frequency_hz=reference,
    temporary_parameters={q.uid: parameters}, options=options,
).run()
metrics = run.output["analysis"]
print(metrics["representative"], metrics["power_convergence"])
```

Amplitude is an absolute CW oscillator gain, independent of readout_amplitude.
The reported instrument port power is `readout_range_out + 20*log10(amplitude)`;
it is not chip input power. Frequency sweeps can be ascending or descending.
Only spectroscopy acquisition with cyclic averaging is accepted. The spectroscopy integrator receives an explicit length with no readout kernel,
with acquisition delay zero and pulse-based automatic muting disabled in CW mode.
This also avoids incorrectly requiring the device's optional long-readout (LRT)
feature for long spectroscopy integration.
No qubit state preparation is performed; the result reflects the natural state
population. Settling is explicit before every acquisition. Only one UID is
accepted per workflow; measure other qubits sequentially.

All runtime overrides use a temporary QPU. The input QPU and its operating
readout frequency are preserved. `update=True` is rejected. `do_analysis(False)`
returns the raw result and sweep metadata without accessing any analysis output.

## Offline reanalysis

```python
from laboneq.serializers import load
from qubit_experiment.analysis.resonator_linewidth_cw import analysis_workflow
raw = load("/absolute/path/to/acquisition.raw.json")
opts = analysis_workflow.options()
opts.geometry("reflection")
metrics = analysis_workflow(
    raw, q, frequencies, amplitudes, reference,
    options=opts,
).run().output
```

Use the recorded qubit parameters, frequency axis, amplitude axis and output
range. The expected raw shape is `(number_of_amplitudes, number_of_frequencies)`.
The returned `raw_complex` is unchanged. Individual failed fits have a status
and reasons; failure never becomes an invented numerical representative value.

## Model and limitations

[NPL MAT58 (2021), equation 26](https://eprintspublications.npl.co.uk/9304/)
models the complex response as a complex background plus complex pole, multiplied
by a line-delay factor. We use installed scikit-rf NLQFIT7 for initialization,
then a bounded seven-real-parameter complex least-squares refinement. Frequency
and linewidth are scaled to MHz. The number of NPL iterations is bounded to
avoid an unbounded convergence loop; the refinement checks convergence.

The saved electrical delay initializes de-embedding; a residual delay is fitted.
Reconstruction includes the NLQFIT7 delay term explicitly: installed scikit-rf
1.8's `fitted_s()` alone does not reconstruct that term. Centered frequency is
used to avoid a redundant absolute phase/delay offset.

The main linewidth must be resolved in all three windows, +/-1.5, +/-2 and +/-3
linewidths. Rank deficiency, Q statistical error above 5%, Q window variation
above 5%, and residual lag-one correlation above 0.3 reject a reliable single
pole. For structured backgrounds, a two-complex-pole model is compared on the
full supplied trace. It must improve BIC by more than 10, have identifiable
filter parameters, and pass the target-pole window checks. Narrow data cannot
identify a broad filter mode; supply a wider trace for this comparison. For the
two-pole window checks the filter contribution is constrained by the full-band
fit; full-band covariance retains correlations with that pole.

Only the **lowest two measured amplitudes** can establish a low-power plateau.
Both must pass; a noisy low-power trace is not silently skipped. Q must agree
within the larger of 5% and twice the combined statistical error; resonance
frequency must agree within the larger of 5% of linewidth and twice its combined
statistical error. Otherwise `representative` is `None`.

Statistical errors are conditional on the selected response model and a local
least-squares noise approximation. Power and fit-window variations are separate
checks, not a complete metrological uncertainty budget. A two-pole partial
fraction fit does not determine bare circuit couplings or loss rates. Overlapping
qubit-state resonances, multiple filter modes, nonlinear response and drift can
invalidate the model; retain the diagnostics instead of forcing a value.

Definitions:

- `q_loaded = fr_hz / linewidth_hz`
- `kappa_over_2pi_hz = linewidth_hz`
- `kappa_rad_s = 2*pi*linewidth_hz` (energy decay convention)
- `energy_decay_time_s = 1/kappa_rad_s`

## Independent KPS runner

`projects/2026_alltoall_kps/scripts/measure_resonator_linewidth_cw.py` loads the
latest saved KPS QPU and descriptor without using a notebook kernel. Its default
mode compiles offline; `--execute` acquires. Specify `--qubit`,
`--reference-frequency-hz` and optionally `--snapshot` for another qubit. The
runner exclusively controls the device: other acquisition programs must remain
idle. It saves raw JSON and NPZ before numerical analysis, checks power
convergence, then validates the candidate using longer integration and reverse
frequency direction. Cleanup explicitly stops the QA generator/CW and turns the
QA output off. Cleanup errors are recorded, not suppressed.

The low-power survey and fine scans are retained independently. The default
runner fits the local fine data; if the broad survey is needed to identify a
filter, reanalyze it with an appropriate local frequency range rather than
combining traces of different integration/count as equally weighted data.

The runner can resume a saved, unclipped local pilot with `--resume-from`. For
very weak power the fine scans use 16 us spectroscopy integration and per-power
shot counts (8192, 2048, 512 for the default ascending power list), preserving
each trace's settings. The module default remains 2 us. `--input-range-dbm`
sets receiver sensitivity separately from output power; input/output overrange
warnings cause the runner to preserve raw data and abort before fitting it.

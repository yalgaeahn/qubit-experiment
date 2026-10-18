# Q3 spectroscopy — 2026-10-18

Confirmed transition: approximately **4.81303 GHz**, q3 = qubits[2], SG2 / DEV12256.

Existing `qubit_experiment.experiments.qubit_spectroscopy` workflow was executed
from `scripts/find_q3_frequency.py`, independently of the notebook kernel.
QPU source: `qpu_parameters/20261018-152737_1.json`.
No notebook or persistent QPU parameters were changed. All acquisitions used
`update(False)` and temporary parameters; each script session disconnected.

## Evidence

- Survey: 4.0–5.0 GHz, two 500 MHz bands, 500 kHz spacing, 256 averages,
  -43.46 dBm nominal instrument drive output, 10 us pulse.
- Confirmation: -49.46 dBm, 10 us, 10 kHz spacing, 1024 averages.
- Forward center: 4.813035239 GHz, fit-only standard error 0.732 kHz.
- Reverse center: 4.813029113 GHz, fit-only standard error 0.708 kHz.
- Repeat difference: 6.13 kHz. Use a rounded 4.81303 GHz starting value;
  the fit errors do not include drift, calibration, or model systematics.
- FWHM: 93.84 kHz forward, 89.05 kHz reverse.
- Drive-off control: no statistically supported line (delta BIC = -18.98
  against a linear complex baseline); drive-on delta BIC > 970.
- Readout: 7.555232 GHz, amplitude 0.4, output range -10 dBm,
  pulse/integration 1 us, integration delay 123 ns, unchanged from snapshot.

## Files and reproduction

Raw NPZ/CSV, request JSON, individual figures, and complete workflow logs are saved.
`analyze_scans.py` reuses the prior Q2 complex Lorentzian + linear baseline fit.
`plot_confirmation.py` rebuilds `q3_spectroscopy_confirmed.png`.
Coarse fits at their minimum linewidth bound identify scan placement only;
final values use the resolved precise scans. No coherent-control calibration was done.
`survey.json` records an initial failed offline LO configuration, with no acquisition.

Example repeat from the repository root (a fresh output label is required):

```sh
.venv/bin/python projects/2026_alltoall_kps/scripts/find_q3_frequency.py \
  --execute --start 4.8122e9 --stop 4.8138e9 --points 161 \
  --count 1024 --power -49.46 --label repeat
```

`--off` runs a drive-off control. Omitting `--execute` compiles offline only.
The script loads the latest snapshot and records its exact path for each scan.

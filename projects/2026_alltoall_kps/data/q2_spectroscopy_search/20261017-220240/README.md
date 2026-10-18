# Q2 spectroscopy search - 2026-10-17

Confirmed transition: approximately **4.68013 GHz**.

The 4.6793-4.6809 GHz fine scans used 20 kHz spacing and 2048 averages
at -49.46 dBm nominal SG1 output. A separate reverse scan at -43.46 dBm
confirmed the same transition with a broader line. A drive-off control
had no statistically supported Lorentzian in the original 8 MHz window.
No drive setting exceeded -43.46 dBm, below the user-approved -30 dBm ceiling.

## Selected operating parameters

- Qubit/channel: q2 / SG1, DEV12256, existing notebook session
- resonance_frequency_ge: 4.68013e9 Hz (rounded to 10 kHz)
- drive_lo_frequency: 4.6e9 Hz
- drive_range: +10 dBm
- spectroscopy_length: 10e-6 s
- spectroscopy_amplitude: 0.0010641430182243164 (-49.46 dBm nominal)
- readout_resonator_frequency: 7.345129641e9 Hz
- readout_lo_frequency: 7.4e9 Hz
- readout_range_out: +5 dBm
- readout_amplitude: 0.001294196 (-52.76 dBm nominal)
- readout_length / integration_length: 2e-6 s / 2e-6 s
- readout_integration_delay: 244e-9 s

## Evidence

Fine forward low-power center: 4.68013765 GHz, model standard error 4.58 kHz.
Fine reverse low-power center: 4.68012969 GHz, model standard error 3.25 kHz.
Higher-power reverse center: 4.68013628 GHz, model standard error 4.01 kHz.
These are fit-only statistical errors and do not include instrument or model systematics.
Low-power FWHM: approximately 158-180 kHz; at -43.46 dBm: 260 kHz.
The low-power fine-scan centers differ by 8.0 kHz.

The original magnitude-only fit selected an isolated point. The confirmed
line is supported by both complex quadratures, repeated measurements,
a drive-off control, and the higher-power reverse sweep.
Only spectroscopy was measured; coherent-control calibration remains to be done.

## Files

Every scan has raw NPZ/CSV data, a JSON parameter manifest, and PNG plots.
workflow_logs contains the actual experiment, compiled experiment, and acquired results.
analysis_summary.json contains complex Lorentzian fits with a linear complex baseline.
selected_parameters.json contains the parameter update applied through update_qpu.
analyze_scans.py reproduces the per-scan fits from the NPZ files.
measurement_setup.py and executed_scan_requests.py record the executed orchestration.
The final scan returned to -49.46 dBm. The existing notebook session remains connected;
no new hardware session was created. The helper kernel client has been closed.

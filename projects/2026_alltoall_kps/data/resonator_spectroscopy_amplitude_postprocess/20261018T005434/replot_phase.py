"""Replot saved spectroscopy data using the existing dispersive-shift delay convention."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from laboneq.serializers import load
from qubit_experiment.analysis.dispersive_shift import _phase_delay_params

SOURCE = Path('/Users/yalgaeahn/Library/CloudStorage/SynologyDrive-KAIST/ALLTOALL_KPS_2026/experiment_store/20261018/20261018T005434-resonator-spectroscopy-amplitude')
OUTPUT = Path(__file__).resolve().parent

def main():
    qubit = load(SOURCE / 'temporary-quantum-elements-from-qpu.output.json')
    results = load(SOURCE / 'run-experiment.output.json')
    acquired = results.acquired_results[f'{qubit.uid}/result']
    assert list(acquired.axis_name) == [f'amplitudes_{qubit.uid}', f'frequencies_{qubit.uid}']
    amplitudes, frequencies = (np.asarray(a, dtype=float) for a in acquired.axis)
    raw = np.asarray(acquired.data)
    assert raw.shape == (amplitudes.size, frequencies.size)
    assert np.all(np.isfinite(raw))
    tau, f_ref, applied, source = _phase_delay_params(qubit, frequencies, None)
    if not applied:
        raise ValueError('No recorded electrical delay available; correction aborted.')
    # Same correction as calculate_dispersive_shift_metrics: +2*pi*(f-f_ref)*tau.
    correction = 2 * np.pi * (frequencies - f_ref) * tau
    phase_unwrapped = np.unwrap(np.angle(raw), axis=-1)
    phase_corrected_unwrapped = phase_unwrapped + correction[None, :]
    corrected = raw * np.exp(1j * correction[None, :])
    np.testing.assert_allclose(np.abs(corrected), np.abs(raw), rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(np.exp(1j * phase_corrected_unwrapped), np.exp(1j * np.angle(corrected)), atol=1e-12)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharex=True, sharey=True, layout='constrained')
    for ax, phase, title in zip(axes, (np.angle(raw), np.angle(corrected)), ('Before correction', 'After electrical delay correction')):
        mesh = ax.pcolormesh(frequencies / 1e9, amplitudes, phase, shading='auto', cmap='twilight', vmin=-np.pi, vmax=np.pi, rasterized=True)
        ax.set_title(title)
        ax.set_xlabel('Readout frequency (GHz)')
    axes[0].set_ylabel('Readout amplitude (a.u.)')
    fig.colorbar(mesh, ax=axes, label='Wrapped phase (rad)', ticks=[-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    fig.suptitle(f'{qubit.uid}: pulsed resonator spectroscopy | delay = {tau*1e9:.4f} ns\n20261018T005434 | zero-amplitude row contains noise')
    fig.savefig(OUTPUT / 'phase_delay_comparison.png', dpi=180)
    plt.close(fig)
    np.savez_compressed(OUTPUT / 'phase_corrected.npz', frequencies_hz=frequencies, amplitudes=amplitudes, raw_complex=raw, corrected_complex=corrected, phase_corrected_unwrapped_rad=phase_corrected_unwrapped, tau_s=tau, f_ref_hz=f_ref)
    metadata = dict(source_run=str(SOURCE), qubit=qubit.uid, shape=list(raw.shape), tau_s=tau, f_ref_hz=f_ref, delay_source=source, correction='phase + 2*pi*(f-f_ref)*tau; same convention as analysis/dispersive_shift.py', display='wrapped phase, [-pi, pi]', checks=['finite complex data', 'axes and shape', 'magnitude preserved', 'wrapped and unwrapped correction agree'])
    (OUTPUT / 'provenance.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2))

if __name__ == '__main__':
    main()

from copy import deepcopy as _kps_deepcopy
from datetime import datetime as _KpsDatetime
from zoneinfo import ZoneInfo as _KpsZoneInfo
from pathlib import Path as _KpsPath
import json as _kps_json
import math as _kps_math
import numpy as _kps_np
import matplotlib.pyplot as _kps_plt
from scipy.ndimage import gaussian_filter1d as _kps_smooth
from laboneq.workflow.logbook import FolderStore as _KpsFolderStore
from qubit_experiment.experiments import qubit_spectroscopy as _kps_spec

assert str(PROJECT_ROOT).endswith('/projects/2026_alltoall_kps')
assert session._connection_state.connected and not session._connection_state.emulated
assert [(inst.uid,inst.address) for inst in setup.instruments]==[('SHFQC_DEV12256','DEV12256')]
_kps_target=next(item for item in qpu.groups.qubits if item.uid=='q2')
assert _kps_target.signals['drive']=='q2/drive'
KPS_SEARCH_DIR=PROJECT_ROOT/'data'/'q2_spectroscopy_search'/_KpsDatetime.now(_KpsZoneInfo('Asia/Seoul')).strftime('%Y%m%d-%H%M%S')
KPS_SEARCH_DIR.mkdir(parents=True,exist_ok=False)
KPS_SEARCH_RESULTS=[]
_kps_source_parameters=_kps_deepcopy(_kps_target.parameters)
(KPS_SEARCH_DIR/'limits.json').write_text(_kps_json.dumps({
 'drive_output_limit_dbm':-30.0,'saturation_duration_s':10e-6,
 'target':'q2','drive_channel':'SG1','instrument':'DEV12256',
 'note':'Existing connected notebook session; temporary parameters; no QPU update',
},indent=2)+'\n')

def _kps_run_scan(label, frequencies_hz, power_dbm, count=512, integration_delay_s=244e-9, readout_frequency_hz=None):
    frequency=_kps_np.asarray(frequencies_hz,dtype=float)
    p=_kps_deepcopy(_kps_source_parameters)
    p.drive_lo_frequency=4.6e9
    p.spectroscopy_length=10e-6
    p.spectroscopy_amplitude=0.0 if power_dbm is None else 10**((float(power_dbm)-p.drive_range)/20)
    p.readout_integration_delay=float(integration_delay_s)
    if readout_frequency_hz is not None:
        p.readout_resonator_frequency=float(readout_frequency_hz)
    assert power_dbm is None or power_dbm<=-30.0
    assert p.spectroscopy_length==10e-6
    assert 0<=p.spectroscopy_amplitude<=1
    assert _kps_np.max(_kps_np.abs(frequency-p.drive_lo_frequency))<=500e6
    assert 4.3e9<=frequency.min() and frequency.max()<=5.0e9
    assert p.readout_range_out==5 and p.readout_amplitude<=0.001294197
    options=_kps_spec.experiment_workflow.options()
    options.update(False)
    options.do_analysis(False)
    options.count(int(count))
    options.logstore(_KpsFolderStore(KPS_SEARCH_DIR/'workflow_logs',save_mode='raise'))
    manifest={
      'label':label,'power_dbm':power_dbm,'saturation_s':p.spectroscopy_length,
      'drive_range_dbm':p.drive_range,'drive_lo_hz':p.drive_lo_frequency,
      'drive_amplitude':p.spectroscopy_amplitude,'averages':int(count),
      'frequency_start_hz':float(frequency.min()),'frequency_stop_hz':float(frequency.max()),
      'frequency_points':int(frequency.size),'readout_frequency_hz':p.readout_resonator_frequency,
      'readout_amplitude':p.readout_amplitude,'readout_range_dbm':p.readout_range_out,
      'integration_delay_s':p.readout_integration_delay,
      'started_at':_KpsDatetime.now(_KpsZoneInfo('Asia/Seoul')).isoformat(),
    }
    (KPS_SEARCH_DIR/f'{label}.json').write_text(_kps_json.dumps(manifest,indent=2)+'\n')
    print('KPS_SCAN_START',_kps_json.dumps(manifest),flush=True)
    run=_kps_spec.experiment_workflow(session=session,qpu=qpu,qubits='q2',frequencies=frequency,temporary_parameters={'q2':p},options=options).run()
    acquired=run.output['q2/result']
    z=_kps_np.asarray(acquired.data,dtype=complex)
    assert z.shape==frequency.shape and _kps_np.all(_kps_np.isfinite(z))
    _kps_np.savez(KPS_SEARCH_DIR/f'{label}.npz',frequency_hz=frequency,signal=z)
    _kps_np.savetxt(KPS_SEARCH_DIR/f'{label}.csv',_kps_np.column_stack((frequency,z.real,z.imag,_kps_np.abs(z))),delimiter=',',header='frequency_hz,I,Q,magnitude',comments='')
    manifest['finished_at']=_KpsDatetime.now(_KpsZoneInfo('Asia/Seoul')).isoformat()
    manifest['signal_mean_I']=float(z.real.mean())
    manifest['signal_mean_Q']=float(z.imag.mean())
    manifest['point_noise_I']=float(_kps_np.std(_kps_np.diff(z.real))/_kps_np.sqrt(2))
    manifest['point_noise_Q']=float(_kps_np.std(_kps_np.diff(z.imag))/_kps_np.sqrt(2))
    (KPS_SEARCH_DIR/f'{label}.json').write_text(_kps_json.dumps(manifest,indent=2)+'\n')
    KPS_SEARCH_RESULTS.append({'metadata':manifest,'frequency_hz':frequency,'signal':z,'workflow':run})
    fig,axes=_kps_plt.subplots(3,1,figsize=(10,8),sharex=True)
    for ax,values,name in zip(axes,(z.real,z.imag,_kps_np.abs(z)),('I','Q','Magnitude')):
      ax.plot(frequency/1e9,values,'.',ms=3,alpha=0.65)
      ax.plot(frequency/1e9,_kps_smooth(values,1.0),lw=1.3)
      ax.set_ylabel(name)
      ax.grid(alpha=0.25)
    axes[0].set_title(f'Q2 {label}: {power_dbm} dBm, 10 us, delay {integration_delay_s*1e9:g} ns, N={count}')
    axes[-1].set_xlabel('Drive frequency (GHz)')
    fig.tight_layout()
    fig.savefig(KPS_SEARCH_DIR/f'{label}.png',dpi=160)
    _kps_plt.close(fig)
    print('KPS_SCAN_DONE',_kps_json.dumps({'label':label,'data_dir':str(KPS_SEARCH_DIR),'mean_I':manifest['signal_mean_I'],'mean_Q':manifest['signal_mean_Q'],'noise_I':manifest['point_noise_I'],'noise_Q':manifest['point_noise_Q']}),flush=True)
    return KPS_SEARCH_RESULTS[-1]
print('KPS_SETUP_READY',str(KPS_SEARCH_DIR),flush=True)

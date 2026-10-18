import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize, least_squares
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

folder=Path(__file__).resolve().parent
results=[]
for path in sorted(folder.glob('*.npz')):
    data=np.load(path)
    f=data['frequency_hz']
    z=data['signal']
    order=np.argsort(f)
    f=f[order]
    z=z[order]
    x=(f-np.mean(f))/1e6
    design0=np.column_stack([np.ones(x.size),x])
    coeff0=np.linalg.lstsq(design0,z,rcond=None)[0]
    z0=design0@coeff0
    sse0=float(np.sum(np.abs(z-z0)**2))
    def fit_at(center,width):
        shape=1/(1+((x-center)/width)**2)
        design=np.column_stack([np.ones(x.size),x,shape])
        coeff=np.linalg.lstsq(design,z,rcond=None)[0]
        model=design@coeff
        return float(np.sum(np.abs(z-model)**2)),coeff,model
    span=float(np.ptp(x))
    step=float(np.median(np.diff(x)))
    seeds=[]
    for center in np.linspace(x.min()+step,x.max()-step,41):
        for width in [max(step,0.1),max(2*step,0.3),max(3*step,0.8),span/6]:
            score,_,_=fit_at(center,width)
            seeds.append((score,center,width))
    candidates=[]
    for _,center,width in sorted(seeds)[:5]:
        fit=minimize(lambda a:fit_at(*a)[0],[center,width],method='Nelder-Mead',bounds=[(float(x.min()),float(x.max())),(step,span/2)],options={'maxiter':600})
        candidates.append((fit.fun,*fit.x))
    _,center,width=min(candidates)
    sse,coeff,model=fit_at(center,width)
    def model_full(a):
        return a[0]+1j*a[1]+(a[2]+1j*a[3])*x+(a[4]+1j*a[5])/(1+((x-a[6])/a[7])**2)
    def residual_full(a):
        residual=model_full(a)-z
        return np.r_[residual.real,residual.imag]
    initial=[coeff[0].real,coeff[0].imag,coeff[1].real,coeff[1].imag,coeff[2].real,coeff[2].imag,center,width]
    full=least_squares(residual_full,initial,bounds=([-np.inf]*6+[x.min(),step],[np.inf]*6+[x.max(),span/2]),max_nfev=1000)
    center,width=full.x[6:8]
    sse=float(np.sum(full.fun**2))
    coeff=np.array([full.x[0]+1j*full.x[1],full.x[2]+1j*full.x[3],full.x[4]+1j*full.x[5]])
    model=model_full(full.x)
    covariance=np.linalg.pinv(full.jac.T@full.jac)*sse/max(2*x.size-8,1)
    center_stderr_hz=float(np.sqrt(max(covariance[6,6],0))*1e6)
    sigma=np.std(np.diff(z))/np.sqrt(2)
    improvement=2*len(x)*np.log(max(sse0,1e-30)/max(sse,1e-30))-4*np.log(2*len(x))
    record={'width_at_lower_bound':bool(width <= 1.01*step),'label':path.stem,'center_hz':float(np.mean(f)+center*1e6),'center_stderr_hz':center_stderr_hz,'hwhm_hz':float(width*1e6),'complex_amplitude':float(abs(coeff[2])),'point_noise_complex':float(sigma),'amplitude_over_point_noise':float(abs(coeff[2])/max(sigma,1e-30)),'delta_BIC_vs_linear':float(improvement)}
    results.append(record)
    fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True)
    for ax,raw,pred,quantity in zip(axes,[z.real,z.imag,np.abs(z)],[model.real,model.imag,np.abs(model)],['I','Q','Magnitude']):
        ax.plot(f/1e9,raw,'.',ms=3,alpha=0.65,label='raw')
        ax.plot(f/1e9,gaussian_filter1d(raw,1),'k-',lw=1,alpha=.7,label='smoothed')
        ax.plot(f/1e9,pred,'r-',lw=1,label='candidate Lorentzian')
        ax.set_ylabel(quantity)
        ax.grid(alpha=.2)
    axes[0].set_title(f'{path.stem}: candidate {record["center_hz"]/1e9:.7f} GHz, delta BIC {improvement:.1f}')
    axes[0].legend()
    axes[-1].set_xlabel('Drive frequency (GHz)')
    fig.tight_layout()
    fig.savefig(folder/f'{path.stem}_fit.png',dpi=160)
    plt.close(fig)
(folder/'analysis_summary.json').write_text(json.dumps(results,indent=2)+'\n')
for result in results:
    print(json.dumps(result))

#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, os, platform
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize
from scipy.stats import qmc
import mne

from empirical_csd_lock import multitaper_csd, chronological_two_block_indices, normalize_whittle_score, FREQS_HZ
from statistical_lock import complex_whittle_score
from model_frequency_lock import M1Params, M2Params, projected_model_csd

SEED = 97
SOBOL_M = 5
POLISH_STARTS = 4
POLISH_MAXITER = 120
REL_FLOOR = 1e-6

def _load_L20(forward_dir: Path):
    U20 = np.load(forward_dir / 'leadfield_U20.npy')
    df = pd.read_csv(forward_dir / 'leadfield_DK68_64x68.csv', index_col=0)
    L = df.to_numpy(float)
    if U20.shape != (64,20) or L.shape != (64,68):
        raise RuntimeError(f'Frozen forward shapes wrong: U20={U20.shape}, L={L.shape}')
    if np.max(np.abs(U20.T @ U20 - np.eye(20))) >= 1e-10:
        raise RuntimeError('Frozen U20 orthogonality failed')
    return U20, U20.T @ L

def _decode(model: str, z: np.ndarray, amp_center: float, noise_center: float):
    i=0
    G=float(z[i]); i+=1
    v=float(z[i]); i+=1
    pE=float(z[i]); i+=1
    source_scale=amp_center * (10.0 ** float(z[i])); i+=1
    sensor_floor=noise_center * (10.0 ** float(z[i])); i+=1
    m1=m2=None
    if model=='M1':
        tau=float(z[i]); gE=float(z[i+1]); gI=float(z[i+2])
        m1=M1Params(tau,gE,gI)
    elif model=='M2':
        ta,tb=sorted((float(z[i]),float(z[i+1]))); i+=2
        m2=M2Params(ta,tb,*map(float,z[i:i+4]))
    return dict(G_N=G, velocity_m_s=v, pE=pE, source_scale=source_scale,
                sensor_floor=sensor_floor, m1=m1, m2=m2)

def _bounds(model: str):
    b=[(50.0,185.0),(3.0,12.0),(1e-4,1-1e-4),(-4.0,4.0),(-8.0,2.0)]
    if model=='M1':
        b += [(0.03,30.0),(-0.5,0.5),(-0.5,0.5)]
    elif model=='M2':
        b += [(0.03,30.0),(0.03,30.0)] + [(-0.5,0.5)]*4
    return b

def _amp_centers(empirical_csd, network_dir, L20, model):
    escale=float(np.median(np.real(np.trace(empirical_csd,axis1=1,axis2=2))/empirical_csd.shape[1]))
    escale=max(escale,np.finfo(float).tiny)
    kwargs={}
    if model=='M1': kwargs['m1']=M1Params(1.0,0.0,0.0)
    if model=='M2': kwargs['m2']=M2Params(0.1,10.0,0.0,0.0,0.0,0.0)
    base=projected_model_csd(network_dir,L20,FREQS_HZ,model,117.5,7.5,0.5,1.0,escale*1e-8,**kwargs)
    bscale=float(np.median(np.real(np.trace(base,axis1=1,axis2=2))/base.shape[1]))
    bscale=max(bscale,np.finfo(float).tiny)
    return math.sqrt(escale/bscale), escale

def fit_block(empirical_csd, dof, model, network_dir, L20):
    amp_center,noise_center=_amp_centers(empirical_csd,network_dir,L20,model)
    bounds=_bounds(model)
    def objective(z):
        try:
            p=_decode(model,z,amp_center,noise_center)
            S=projected_model_csd(network_dir,L20,FREQS_HZ,model,**p)
            score=complex_whittle_score(empirical_csd,S,dof=dof,rel_floor=REL_FLOOR)
            val=-normalize_whittle_score(score,dof)
            return val if np.isfinite(val) else 1e100
        except Exception:
            return 1e100
    sampler=qmc.Sobol(d=len(bounds),scramble=True,seed=SEED)
    unit=sampler.random_base2(m=SOBOL_M)
    cand=qmc.scale(unit,[a for a,b in bounds],[b for a,b in bounds])
    vals=np.array([objective(x) for x in cand])
    order=np.argsort(vals)[:POLISH_STARTS]
    best=None
    for k in order:
        res=minimize(objective,cand[k],method='L-BFGS-B',bounds=bounds,
                     options={'maxiter':POLISH_MAXITER,'ftol':1e-9,'gtol':1e-6,'maxls':30})
        if best is None or res.fun < best.fun: best=res
    if best is None or not np.isfinite(best.fun) or best.fun>=1e99:
        raise RuntimeError(f'optimizer failed for {model}')
    pars=_decode(model,best.x,amp_center,noise_center)
    serial={k:v for k,v in pars.items() if k not in ('m1','m2')}
    if pars['m1'] is not None: serial['m1']=pars['m1'].__dict__
    if pars['m2'] is not None: serial['m2']=pars['m2'].__dict__
    return pars,serial,float(-best.fun),dict(success=bool(best.success),message=str(best.message),nit=int(best.nit),nfev=int(best.nfev))

def score_holdout(empirical_csd,dof,model,pars,network_dir,L20):
    S=projected_model_csd(network_dir,L20,FREQS_HZ,model,**pars)
    return normalize_whittle_score(complex_whittle_score(empirical_csd,S,dof=dof,rel_floor=REL_FLOOR),dof)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('epochs_fif',type=Path)
    ap.add_argument('--forward-dir',type=Path,required=True)
    ap.add_argument('--network-dir',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--models',nargs='+',default=['M2','M3'])
    a=ap.parse_args()
    U20,L20=_load_L20(a.forward_dir)
    ep=mne.read_epochs(a.epochs_fif,preload=True,verbose='error')
    frozen_channels=[x.strip() for x in Path(__file__).with_name('channels_64.txt').read_text().splitlines() if x.strip()]
    if ep.ch_names != frozen_channels:
        raise RuntimeError(f'Clean-epoch channel order drift: got {ep.ch_names}')
    # Bad channels were already interpolated during frozen preprocessing but
    # deliberately remain flagged in info["bads"] for provenance. Select the
    # frozen names explicitly so interpolated channels are retained for the
    # 64-channel forward operator.
    x=ep.get_data(picks=frozen_channels)
    if x.shape[1:] != (64,1000): raise RuntimeError(f'Expected clean epochs [n,64,1000], got {x.shape}')
    A,B=chronological_two_block_indices(len(ep))
    blocks={}
    for name,idx in [('A',A),('B',B)]:
        f,S,nu=multitaper_csd(x[idx],sfreq=float(ep.info['sfreq']),project=U20)
        if not np.array_equal(f,FREQS_HZ): raise RuntimeError('frequency lock drift')
        blocks[name]=(S,nu)
    result={'seed':SEED,'n_epochs':int(len(ep)),'split_A':int(len(A)),'split_B':int(len(B)),
            'optimizer':{'sobol_candidates':2**SOBOL_M,'polish_starts':POLISH_STARTS,'polish_maxiter':POLISH_MAXITER,'method':'L-BFGS-B','rel_floor':REL_FLOOR},
            'environment':{
                'platform':platform.platform(),
                'python':platform.python_version(),
                'numpy':np.__version__,
                'scipy':scipy.__version__,
                'mne':mne.__version__,
                'thread_env':{k:os.environ.get(k) for k in (
                    'OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','OPENBLAS_CORETYPE',
                    'MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS',
                    'OMP_DYNAMIC','PYTHONHASHSEED')},
                'runner_image':{
                    'ImageOS':os.environ.get('ImageOS'),
                    'ImageVersion':os.environ.get('ImageVersion'),
                    'RUNNER_OS':os.environ.get('RUNNER_OS'),
                    'RUNNER_ARCH':os.environ.get('RUNNER_ARCH')},
            },
            'models':{}}
    for model in [m.upper() for m in a.models]:
        dirs=[]
        for train_name,test_name in [('A','B'),('B','A')]:
            trainS,trainNu=blocks[train_name]; testS,testNu=blocks[test_name]
            pars,serial,train_score,opt=fit_block(trainS,trainNu,model,a.network_dir,L20)
            held=score_holdout(testS,testNu,model,pars,a.network_dir,L20)
            dirs.append({'train':train_name,'test':test_name,'parameters':serial,'train_score_normalized':train_score,'heldout_score_normalized':held,'optimizer':opt})
        result['models'][model]={'directions':dirs,'cv_elpd':float(np.mean([d['heldout_score_normalized'] for d in dirs]))}
    if 'M3' in result['models'] and 'M2' in result['models']:
        result['delta_elpd_M3_minus_M2']=result['models']['M3']['cv_elpd']-result['models']['M2']['cv_elpd']
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()

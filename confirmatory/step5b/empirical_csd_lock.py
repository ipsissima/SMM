#!/usr/bin/env python3
"""Frozen empirical CSD estimator for SMM Step 5B, completed pre-EEG."""
from __future__ import annotations
import numpy as np
from scipy.signal.windows import dpss
SEED=97; SFREQ=250.0; EPOCH_S=4.0; N_SAMPLES=1000; NW=3.0; KMAX=5
FREQS_HZ=np.arange(1.0,41.0,1.0)
def _validate(x,sfreq):
    x=np.asarray(x,float)
    if x.ndim!=3: raise ValueError('data must be [epoch,channel,time]')
    if abs(float(sfreq)-SFREQ)>1e-9: raise ValueError(f'frozen CSD requires {SFREQ} Hz')
    if x.shape[-1]!=N_SAMPLES: raise ValueError(f'frozen CSD requires {N_SAMPLES} samples per 4-s epoch')
    return x
def multitaper_csd(data,sfreq=SFREQ,project=None):
    x=_validate(data,sfreq); x=x-x.mean(axis=-1,keepdims=True)
    if project is not None:
        P=np.asarray(project)
        if P.ndim!=2 or P.shape[0]!=x.shape[1]: raise ValueError('project must have shape [channel,p]')
        x=np.einsum('ecT,cp->epT',x,P,optimize=True)
    n=x.shape[-1]; tapers=dpss(n,NW=NW,Kmax=KMAX,sym=False,norm=2)
    taper_power=np.sum(tapers*tapers,axis=1)
    X=np.fft.rfft(x[:,None,:,:]*tapers[None,:,None,:],axis=-1)
    fft_freq=np.fft.rfftfreq(n,d=1.0/sfreq)
    idx=np.array([int(np.argmin(np.abs(fft_freq-f))) for f in FREQS_HZ])
    if not np.allclose(fft_freq[idx],FREQS_HZ,atol=1e-12): raise RuntimeError('integer-Hz targets are not exact FFT bins')
    X=X[...,idx]/np.sqrt(taper_power)[None,:,None,None]
    scale=2.0/float(sfreq)
    csd=scale*np.einsum('ekcf,ekdf->fcd',X,X.conj(),optimize=True)/(x.shape[0]*KMAX)
    csd=(csd+np.swapaxes(csd.conj(),-1,-2))/2.0
    dof=np.full(len(FREQS_HZ),2.0*KMAX*x.shape[0],float)
    return FREQS_HZ.copy(),csd,dof
def chronological_two_block_indices(n_epochs:int):
    if n_epochs<30: raise ValueError('primary lock requires >=30 clean epochs')
    cut=n_epochs//2
    return np.arange(0,cut),np.arange(cut,n_epochs)
def normalize_whittle_score(score:float,dof)->float:
    den=float(np.sum(np.asarray(dof,float)))
    if den<=0: raise ValueError('non-positive total DOF')
    return float(score)/den

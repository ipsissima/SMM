#!/usr/bin/env python3
"""SMM Step 8A: source-pinned EDNM somatic voltage-current audit.
Reference: Saetra & Mori (2024), doi:10.1371/journal.pcbi.1012114;
martejulie/electrodiffusive-network-model, commit
826c211c70d9027a4acb02202ca91a60de79a15d, src/ednm/model.py
(lines 138-212, 412-449, 456-481, 524-545, 595-669, 946-974,
1004-1005, 1053-1173) and solver.py (237-309).

No archive is bundled. No new full EDNM simulation is claimed.
Usage: python ednm_voltage_audit.py baseline
       python ednm_voltage_audit.py compare STIM/data.npz CONTROL/data.npz --node 1 --out currents.csv
Requires NumPy/SciPy. node is zero-based and MUST be unstimulated in both runs.
Assumes unmodified somatic neuronal kinetics and neuronal soma-dendrite flux,
constant membrane-area density/capacitance and every native integration step
saved with source layout. If these assumptions fail, the closure residual
cannot be interpreted as a solver error. Not a clamp or causal intervention.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.special import expit
from scipy.optimize import brentq

F = 96.48
PSI = 8.314 * 309.14 / F
CM = 3e-6
GAMMA = 1.714683e3
DXL = 6.67e-2
THETA_N = 350.
LAMBDA_N = 3.2
Z = np.array([1., 1., -1., 2.])
D = np.array([1.33e-5, 1.96e-5, 2.03e-5, .71e-5])
UPS_N = np.array([1., 1., 1., .01])
G_NA_LEAK = .246e-4
G_K_LEAK = .245e-4
G_CL_LEAK = 1e-4
G_NA = 300e-4
G_DR = 150e-4
RHO_N = 1.87e-4


def exprel(x):
    # x/expm1(x), with continuous extension at zero.
    x = np.asarray(x, dtype=float)
    out = np.ones_like(x)
    np.divide(x, np.expm1(x), out=out, where=x != 0.)
    return out


def m_inf(v):
    a = 1280. * exprel(-(v + 46.9)/4.)
    b = 1400. * exprel((v + 19.9)/5.)
    return a/(a+b)


def pump(na_i, k_e):
    """Net outward electrical current, mA/cm^2 (3 Na out, 2 K in)."""
    return F * RHO_N * expit((na_i-25.)/3.) * expit(k_e-3.5)


def baseline():
    na = 18.736726421059462
    k = 3.543992191091376
    vm = -66.95238678754431 - 9.404342525113351e-05
    h = .9993107753531558
    n = .0003052605509823481
    dk, dek = .10172636, .75364144
    dv = -.11109374
    gk = G_K_LEAK + G_DR*n
    gna = G_NA_LEAK + G_NA*m_inf(vm)**2*h
    gtot = gk + gna + G_CL_LEAK
    dp = pump(na,k+dk)-pump(na,k)
    dkcurrent = -gk*dek
    frozen_forcing = lambda a: pump(na,k+a)-pump(na,k)-gk*PSI*np.log1p(a/k)
    result = dict(reference='published initial soma, not archived trajectory',
                  delta_K_mM=dk, archived_delta_EK_mV=dek, archived_delta_Vm_mV=dv,
                  baseline_Vm_mV=vm, resting_gK_S_cm2=float(gk),
                  resting_total_frozen_g_S_cm2=float(gtot),
                  pump_baseline_uA_cm2=float(1000*pump(na,k)),
                  pump_delta_fixed_Na_uA_cm2=float(1000*dp),
                  K_reversal_only_delta_uA_cm2=float(1000*dkcurrent),
                  ratio_pump_to_K=float(dp/(-dkcurrent)),
                  net_direct_outward_uA_cm2=float(1000*(dp+dkcurrent)),
                  frozen_local_equilibrium_delta_Vm_mV=float(-(dp+dkcurrent)/gtot),
                  K_only_equilibrium_delta_Vm_mV=float(-dkcurrent/gtot),
                  delta_Na_for_zero_direct_forcing_mM=float(brentq(lambda x:pump(na+x,k+dk)-pump(na,k)+dkcurrent,-2,2)),
                  delta_Na_for_observed_Vm_in_frozen_model_mM=float(brentq(lambda x:pump(na+x,k+dk)-pump(na,k)+dkcurrent+gtot*dv,-2,2)),
                  frozen_direct_forcing_nonzero_root_delta_K_mM=float(brentq(frozen_forcing,2,30)),
                  root_note='isolated K/pump forcing crossover; NOT a biological coupling threshold')
    return result


def read_run(path, node):
    with np.load(path,allow_pickle=False) as data:
        t, alpha, c, phi, ss = [np.asarray(data[k],dtype=float) for k in ('t','alpha','c','phi','ss')]
        dt = float(np.asarray(data['dt']).reshape(()))
    N = phi.shape[-1]
    expected = [(len(t),2,2,N),(len(t),2,4,3,N),(len(t),2,3,N),(len(t),6,N)]
    if [alpha.shape,c.shape,phi.shape,ss.shape] != expected:
        raise ValueError(f'{path}: source layout mismatch. Expected {expected}; got {[alpha.shape,c.shape,phi.shape,ss.shape]}')
    if len(t)<2 or not np.isfinite(dt) or dt <= 0 or not np.allclose(np.diff(t),dt,rtol=1e-7,atol=1e-10):
        raise ValueError(f'{path}: not a complete native uniform time grid; cannot check discrete charge closure')
    if not 0<=node<N:
        raise ValueError('node out of range')
    if not all(np.isfinite(x).all() for x in (alpha,c,phi,ss)):
        raise ValueError('nonfinite native state')
    if np.any(c[:,0,:3,[0,2],node]<=0):
        raise ValueError('nonpositive somatic neuronal or extracellular Na/K/Cl')
    return t,dt,alpha[...,node],c[...,node],phi[...,node],ss[...,node]


def currents(run):
    t,dt,alpha,c,phi,ss=run
    vm=phi[:,0,0]-phi[:,0,2]
    # Current n evaluated with present reversal potentials and membrane voltage,
    # but old gating variables and old pump state (source IMEX update).
    ei=PSI/Z[:3]*np.log(c[1:,0,:3,2]/c[1:,0,:3,0])
    driving=vm[1:,None]-ei
    out={
        'Na_leak':G_NA_LEAK*driving[:,0],
        'Na_gated':G_NA*m_inf(vm[:-1])**2*ss[:-1,0]*driving[:,0],
        'K_leak':G_K_LEAK*driving[:,1],
        'K_DR':G_DR*ss[:-1,1]*driving[:,1],
        'Cl_leak':G_CL_LEAK*driving[:,2],
        'NaK_pump':pump(c[:-1,0,0,0],c[:-1,0,1,2]),
    }
    # KCC2, NKCC1 and the source's 2Na/Ca exchanger are charge-neutral
    # directly (they may affect future concentrations and currents).
    # Soma -> dendrite axial ionic currents; source uses previous soma alpha.
    cs=c[1:,0,:,0]; cd=c[1:,1,:,0]
    dv=(phi[1:,1,0]-phi[1:,0,0])[:,None]
    coef=THETA_N*D*UPS_N/(LAMBDA_N**2*DXL)
    geom=F*alpha[:-1,0,0,None]/(DXL*GAMMA)
    diff=-coef*(cd-cs)
    migration=-coef*Z/(2*PSI)*(cs+cd)*dv
    for i,s in enumerate(['Na','K','Cl','Ca']):
        out[f'axial_{s}_diff']=geom[:,0]*Z[i]*diff[:,i]
        out[f'axial_{s}_drift']=geom[:,0]*Z[i]*migration[:,i]
    total=np.sum(np.array(list(out.values())),axis=0)
    residual=CM*np.diff(vm)/dt+total
    return vm, out, residual


def compare(stim,control,node,outpath):
    a=read_run(stim,node);b=read_run(control,node)
    t,dt=a[:2]
    if a[1]!=b[1] or a[0].shape!=b[0].shape or not np.allclose(a[0],b[0],rtol=0,atol=1e-10):
        raise ValueError('paired runs must use identical native times')
    va,ia,ra=currents(a);vb,ib,rb=currents(b)
    delta={k:ia[k]-ib[k] for k in ia}
    accum={k:-dt/CM*np.cumsum(v) for k,v in delta.items()}
    dv=va-vb
    predicted=np.sum(np.array(list(accum.values())),axis=0)
    err=dv[1:]-dv[0]-predicted
    header=['t_s','delta_Vm_mV','delta_K_e_mM','delta_EK_mV','closure_error_mV']
    header+=['dI_'+k+'_uA_cm2' for k in delta]+['integrated_'+k+'_mV' for k in delta]
    dke=a[3][:,0,1,2]-b[3][:,0,1,2]
    def ek(x): return PSI*np.log(x[3][:,0,1,2]/x[3][:,0,1,0])
    dek=ek(a)-ek(b)
    outpath=Path(outpath);outpath.parent.mkdir(parents=True,exist_ok=True)
    with outpath.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(header)
        for i in range(len(t)-1):
            w.writerow([t[i+1],dv[i+1],dke[i+1],dek[i+1],err[i]]+[1000*v[i] for v in delta.values()]+[v[i] for v in accum.values()])
    j=int(np.argmin(np.abs(t[1:]-.4)))
    result=dict(node=node,reference='paired actual native trajectories; outward-positive sign',
                n_steps=len(t)-1,dt_s=dt,
                absolute_current_closure_stim_max_uA_cm2=float(1000*np.max(np.abs(ra))),
                absolute_current_closure_control_max_uA_cm2=float(1000*np.max(np.abs(rb))),
                paired_voltage_closure_max_mV=float(np.max(np.abs(err))),
                paired_voltage_peak_mV=float(np.max(np.abs(dv))),
                nearest_0p4s=dict(t_s=float(t[j+1]),delta_Vm_mV=float(dv[j+1]),delta_K_e_mM=float(dke[j+1]),delta_EK_mV=float(dek[j+1]),
                    reconstructed_delta_Vm_mV=float(dv[0]+predicted[j]),
                    current_differences_uA_cm2={k:float(1000*v[j]) for k,v in delta.items()},
                    integrated_contributions_mV={k:float(v[j]) for k,v in accum.items()}),
                note='Current accounting along observed trajectories is not an independent counterfactual. Check convergence and absolute/paired charge closure before interpreting; unmodeled direct drive invalidates the audit.')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='cmd',required=True)
    sub.add_parser('baseline')
    q=sub.add_parser('compare');q.add_argument('stim');q.add_argument('control');q.add_argument('--node',type=int,default=1);q.add_argument('--out',default='currents.csv')
    a=p.parse_args();r=baseline() if a.cmd=='baseline' else compare(a.stim,a.control,a.node,a.out)
    print(json.dumps(r,indent=2))

if __name__=='__main__': main()

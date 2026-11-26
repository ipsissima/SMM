"""Matched seven-site passive/EDNM comparison and ideal-design Fisher diagnostic.
Source: integrated 34-page manuscript eqs 5-6; original EDNM graph artifact
15131038799 / 15132425485 and graph run_full_source_graph.py.
Assumptions: EDNM rest fractions 0.2/0.4, source-reconciled passive rates,
physical source flux gamma_m*abs(j_stim). No neuronal feedback in passive model.
"""
import json, zipfile, io
from pathlib import Path
import numpy as np
from scipy.linalg import expm, expm_frechet, eigh
from scipy.optimize import minimize_scalar
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
DATA=ROOT/"inputs" if (ROOT/"inputs").exists() else ROOT.parent
N=7; PE=.2; PA=.4
KE=1.29629; KA=.52863; KN=.232; DE=1.05270e-9; DG=2.29581e-10
GAMMA=1714.683; J=1e-4; DOSE=GAMMA*J
ON=.08; OFF=.4; TMAX=.8; SIGMA=.001
W=np.zeros((N,N))
for i in range(N-1): W[i,i+1]=W[i+1,i]=1
C=np.zeros_like(W)
for i,j,v in [(0,4,.8),(1,5,.5),(2,6,.6)]: C[i,j]=C[j,i]=v
lap=lambda W: np.diag(W.sum(1))-W
L=lap(W); E=lap(C); I=np.eye(N)
B=np.vstack((I/PE,np.zeros((N,N))))
def generator(d=5,mu=1,g=1,de=DE,dg=DG,ke=KE,ka=KA,kn=KN,pe=PE,pa=PA):
    d2=(d*1e-6)**2
    return np.block([[(-(ke+kn)*I-de*L/d2)/pe,ka*I/pe],[ke*I/pa,(-ka*I-mu*dg*(L+g*E)/d2)/pa]])
def stepmat(A,t):
    if t<=0:return np.zeros_like(B)
    return np.linalg.solve(A,(expm(A*t)-np.eye(2*N))@B)
def pulse(A,t,on=ON,off=OFF):
    return stepmat(A,t-on)-stepmat(A,t-off)
def response(A,t,on=ON,off=OFF):
    # Weighted symmetric eigendecomposition; stable for long or stiff grids.
    scale=np.r_[np.full(N,np.sqrt(KE*PE)),np.full(N,np.sqrt(KA*PA))]
    Z=scale[:,None]*A/scale[None,:]
    if not np.allclose(Z,Z.T,atol=1e-8): raise ValueError('not symmetric')
    l,V=eigh(Z); w=V.T@(scale*B[:,0]); v=V[:N,:]/scale[:N,None]
    tt=np.maximum(np.asarray(t)-on,0)[:,None]
    z=np.expm1(tt*l)/l
    if np.isfinite(off):
        tt2=np.maximum(np.asarray(t)-off,0)[:,None]
        z-=np.expm1(tt2*l)/l
    return DOSE*(z*w)@v.T

def load(name):
    with zipfile.ZipFile(DATA/name) as z:
        s=json.loads(z.read('summary.json'))
        def rd(label):
            with np.load(io.BytesIO(z.read(label+'/data.npz'))) as q:return {k:q[k] for k in q.files}
        a=rd('baseline'); b=rd('stimulated')
    return s,a,b

def stat(k1,k2):
    return dict(relative_whole_field_L2=float(np.linalg.norm(k2-k1)/np.linalg.norm(k1)),peak_abs_delta_mM=float(np.max(abs(k2-k1))),peak_path_mM=float(np.max(k1)),peak_relative_to_peak=float(np.max(abs(k2-k1))/np.max(k1)),site6_peak_abs_delta_mM=float(np.max(abs(k2[:,6]-k1[:,6]))))
sp,ap,bp=load('EDNM7_GRAPH_FINE_path.zip'); sc,ac,bc=load('EDNM7_GRAPH_FINE_shortcut.zip')
assert np.allclose(np.array(sp['graph_glial']),W) and np.allclose(np.array(sc['graph_glial']),W+C)
t=ap['t']; kp=bp['c'][:,0,1,2,:]-ap['c'][:,0,1,2,:]; kc=bc['c'][:,0,1,2,:]-ac['c'][:,0,1,2,:]
assert np.allclose(t,ac['t'])
# Source applies at t>.08 and t<.4; state is stored after stepping at label t.
# Use native physical interval exact discrete stimulus via zero-order hold at t_j.
def discrete(A):
    dt=float(ap['dt']); Q=expm(A*dt); V=DOSE*stepmat(A,dt)[:,0]
    x=np.zeros(2*N); out=[x[:N].copy()]
    for ti in t[1:]:
        x=Q@x+(V if ti>ON and ti<OFF else 0)
        out.append(x[:N].copy())
    return np.array(out)
pr=discrete(generator(g=0)); sr=discrete(generator(g=1))
# For alternative mechanism-scale comparison use concentration-only EDNM
# diffusion coefficients times volume fractions; EDNM electrical migration omitted.
ED_DE=PE*(1.96e-5/1.6**2)*1e-4; ED_DG=PA*(1.96e-5/3.2**2)*1e-4
pr2=discrete(generator(g=0,de=ED_DE,dg=ED_DG)); sr2=discrete(generator(g=1,de=ED_DE,dg=ED_DG))
# Fit one common amplitude only (does not affect relative statistic).
ampfit=float(np.sum(pr*kp)/np.sum(pr*pr))
results={'parameters':dict(phi_e=PE,phi_a=PA,k_e=KE,k_a=KA,k_N=KN,D_e=DE,D_g=DG,d_um=5,source_mM_tissue_per_s=DOSE,t_on=ON,t_off=OFF),'matched':{'full_EDNM':stat(kp,kc),'passive_source_reconciled':stat(pr,sr),'passive_EDNM_diffusion_only':stat(pr2,sr2),'amplitude_fit_to_full_path':ampfit,'relative_path_mismatch_after_amplitude_fit':float(np.linalg.norm(ampfit*pr-kp)/np.linalg.norm(kp)),'full_vs_passive_contrast_ratio':float(stat(pr,sr)['relative_whole_field_L2']/stat(kp,kc)['relative_whole_field_L2'])},'EDNM_diffusion_only_de_dg': [ED_DE,ED_DG]}
# Mobility window, exact same graph and matched pulse/times; contrast at source-reconciled point.
dvals=np.geomspace(2,1000,75); mvals=np.geomspace(1e-4,1e5,82)
matched_t=np.linspace(.081,.8,100); long_t=np.array([.5,1,3,10])
def ratio(d,mu,t,on,off):
    x=response(generator(d,mu,0),t,on,off); y=response(generator(d,mu,1),t,on,off)
    return np.linalg.norm(y-x)/np.linalg.norm(x)
window=np.array([[ratio(d,m,matched_t,ON,OFF) for m in mvals] for d in dvals]); longwin=np.array([[ratio(d,m,long_t,0,np.inf) for m in mvals] for d in dvals]);
for name,times,on,off,win in [('matched',matched_t,ON,OFF,window),('long_step',long_t,0,np.inf,longwin)]:
    rr=minimize_scalar(lambda ld:-ratio(np.exp(ld),1,times,on,off),bounds=(np.log(2),np.log(1000)),method='bounded')
    opt=np.unravel_index(np.argmax(win),win.shape)
    results[name+'_window']=dict(optimal_spacing_at_mu1_um=float(np.exp(rr.x)),contrast_at_optimum=float(-rr.fun),contrast_at_5um_mu1=float(ratio(5,1,times,on,off)),global_grid_optimum=dict(d_um=float(dvals[opt[0]]),mu=float(mvals[opt[1]]),contrast=float(win[opt])))
# Ideal source-to-all sensors experiment under same illustrative noise, same source amplitude,
# 8 declared (new) times spanning 0.25-3 s. 7 separate sources.
times=np.array([.25,.5,.75,1,1.5,2,2.5,3.]);
def prediction(theta, spacing=5, select_src=np.arange(N),select_obs=np.arange(N)):
    g,scale_g,scale_e,scale_ke,scale_ka,scale_kn=theta
    A=generator(spacing,scale_g,g,de=DE*scale_e,ke=KE*scale_ke,ka=KA*scale_ka,kn=KN*scale_kn)
    return np.array([DOSE*pulse(A,ti)[:N,:][np.ix_(select_obs,select_src)] for ti in times]).ravel()
def information(spacing=5,src=np.arange(N),obs=np.arange(N)):
    p=np.ones(6); eye=np.eye(6); h=1e-4
    jac=np.column_stack([(prediction(p+h*e,spacing,src,obs)-prediction(p-h*e,spacing,src,obs))/(2*h*SIGMA) for e in eye]);
    x=jac[:,0]; X=jac[:,1:];
    # independent finite-difference convergence
    h2=h/2; j2=np.column_stack([(prediction(p+h2*e,spacing,src,obs)-prediction(p-h2*e,spacing,src,obs))/(2*h2*SIGMA) for e in eye]);
    def eff(a):return float(np.linalg.norm(x-a@np.linalg.lstsq(a,x,rcond=None)[0])**2)
    known=float(x@x); scale=eff(X[:,:1]); alln=eff(X);
    return dict(known=known,unknown_glial_scale=scale,all_five_passive_nuisances=alln,CRLB_SD_known=float(1/np.sqrt(known)),nominal_N_3sigma_known=float(9/known),nominal_N_3sigma_all_nuisances=float(9/alln),jacobian_rel_step_error=float(np.linalg.norm(j2-jac)/np.linalg.norm(jac)),N_measurements=len(x),jacobian_singular_values=np.linalg.svd(jac,compute_uv=False).tolist())
results['ideal_design']=dict(source_flux_mM_tissue_s=DOSE,source_duration_s=OFF-ON,noise_mM=SIGMA,times_s=times.tolist(),full7_at5um=information(),one_source_at5um=information(src=np.array([0])),full7_at_matched_optimum=information(results['matched_window']['optimal_spacing_at_mu1_um']),full7_at_long_step_optimum=information(results['long_step_window']['optimal_spacing_at_mu1_um']))
# Exact recovery conditioning at known parameters for real Laplace s;
# derivative R -> weighted operator = -G/(ka*ke)? use Q dHinv Q etc.
# dHinv=-Hinv dH Hinv; dQ=Q (dM) Q/(ka ke), dM=dHinv.
# operator Frobenius relative norm bound from spectral norm exact on symmetric directions.
def recovery(d,s):
    d2=(d*1e-6)**2; G=DG*(L+E)/d2
    P=(PE*s+KE+KN)*I+DE*L/d2; Q=(PA*s+KA)*I+G
    H=np.linalg.inv(P-KA*KE*np.linalg.inv(Q)); A=Q@np.linalg.inv(H)
    # derivative dG= - A dH A^T/(ka ke)
    rel=float(np.linalg.norm(A,2)**2/(KA*KE)*np.linalg.norm(H,'fro')/np.linalg.norm(G,'fro'))
    a=float(np.sum(G*L)/np.sum(L*L)); residual=float(np.linalg.norm(G-a*L)/np.linalg.norm(G))
    return dict(s_s_inverse=s,d_um=d,relative_Frobenius_worst_case_gain=rel,best_scaled_path_relative_operator_residual=residual,best_scaled_path_coefficient_s_inverse=a)
results['recovery_conditioning']=[recovery(d,s) for d in [5,results['matched_window']['optimal_spacing_at_mu1_um']] for s in [.25,1,4]]
# Exact direct electrical stimulus footprint at onset/offset in original two-site records.
_,b,a=load('EDNM_SECONDS_native_convergence.zip'); tt=b['t']; de=a['phi'][:,0,2,1]-b['phi'][:,0,2,1]; dk=a['c'][:,0,1,2,1]-b['c'][:,0,1,2,1]; dv=(a['phi'][:,0,0,1]-a['phi'][:,0,2,1])-(b['phi'][:,0,0,1]-b['phi'][:,0,2,1]);
results['onset_offset']=[dict(t=float(tt[i]),delta_phi_e_mV=float(de[i]),delta_ke_mM=float(dk[i]),delta_vm_mV=float(dv[i])) for anchor in [.05,.55] for i in range(np.argmin(abs(tt-anchor))-2,np.argmin(abs(tt-anchor))+4)]
# Source EDNM axial flux at all added shortcut edges: actual diffusion vs migration along full trajectories.
# This is descriptive, not effective scalar calibration: glial K flux has cross-ion/electrical dependencies.
D=1.96e-5; lam=3.2; dxu=5e-4; psi=8.314*309.14/96.48
edges=[]
for i,j,w in [(0,4,.8),(1,5,.5),(2,6,.6)]:
    def terms(q):
        c=q['c'][1:,0,1,1,:]; p=q['phi'][1:,0,1,:]; alpha=q['alpha'][:-1,0,1,i]; coef=alpha*D*w/(lam**2*dxu**2)
        return -coef*(c[:,j]-c[:,i]),-coef*(c[:,i]+c[:,j])*(p[:,j]-p[:,i])/(2*psi)
    d0,m0=terms(ac); d1,m1=terms(bc); dd=d1-d0; dm=m1-m0
    edges.append(dict(edge=[i,j],weight=w,diffusive_L2=float(np.linalg.norm(dd)),migration_L2=float(np.linalg.norm(dm)),net_L2=float(np.linalg.norm(dd+dm)),net_over_diffusion=float(np.linalg.norm(dd+dm)/np.linalg.norm(dd)),migration_diffusion_correlation=float(np.corrcoef(dd,dm)[0,1])))
results['actual_glial_shortcut_flux_decomposition']=edges
(ROOT/'results.json').write_text(json.dumps(results,indent=2)+'\n')
np.savez_compressed(ROOT/'window_data.npz',spacing_um=dvals,mobility=mvals,matched=window,long_step=longwin)
fig,ax=plt.subplots(1,2,figsize=(11,4.7),constrained_layout=True)
for axx,ww,title,rr in [(ax[0],window,'Matched 0.8 s pulse',results['matched_window']),(ax[1],longwin,'Original-style 0.5–10 s step',results['long_step_window'])]:
    p=axx.pcolormesh(mvals,dvals,ww,shading='auto',cmap='viridis'); fig.colorbar(p,ax=axx,label='Relative whole-field contrast')
    axx.set(xscale='log',yscale='log',xlabel='Glial mobility multiplier',ylabel='Site spacing (µm)',title=title)
    axx.plot(1,5,'wo',mec='k',label='Reference reduced point (5 µm, µ=1)'); axx.plot(1,rr['optimal_spacing_at_mu1_um'],'w^',mec='k',label='Optimum at µ=1'); axx.legend(fontsize=7,loc='upper left')
fig.savefig(ROOT/'matched_windows.png',dpi=180); plt.close(fig)
fig,ax=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
for a,k,r,site,title in [(ax[0],kp,pr,0,'Source site'),(ax[1],kp,pr,6,'Distal site')]:
    a.plot(t,k[:,site],label='Full EDNM');a.plot(t,r[:,site],label='Reduced passive');a.plot(t,ampfit*r[:,site],ls='--',label='Reduced, common amplitude fit');a.set(title=title,xlabel='Time (s)',ylabel='Δ extracellular K (mM)');a.legend(fontsize=8)
fig.savefig(ROOT/'matched_waveforms.png',dpi=180);plt.close(fig)
print(json.dumps(results,indent=2))

"""Finite, noise-scaled forward discrimination of a scaled or arbitrary weighted shared path.
No real biological inference: data generated from the passive alternative with three added edges.
Fit restricted models to all 7 source x 7 sensor x 8 time measurements.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import json
import numpy as np
from scipy.linalg import expm,eigh
from scipy.optimize import least_squares
R=Path(__file__).parent
p=json.loads((R/'results.json').read_text()); q=p['parameters'];N=7; I=np.eye(N)
pe,pa,ke,ka,kn,de,dg=[q[x] for x in ['phi_e','phi_a','k_e','k_a','k_N','D_e','D_g']]
T=np.array(p['ideal_design']['times_s']); on=q['t_on'];off=q['t_off'];u=q['source_mM_tissue_per_s'];noise=p['ideal_design']['noise_mM']
Lparts=[]
for i in range(N-1):
    a=I[i]-I[i+1];Lparts.append(np.outer(a,a))
Lparts=np.array(Lparts); L=Lparts.sum(axis=0); C=np.zeros_like(L)
for i,j,w in [(0,4,.8),(1,5,.5),(2,6,.6)]:
    a=I[i]-I[j];C+=w*np.outer(a,a)
# support-free reconstruction (known topology in synthetic input, not used by restricted fit)
def predict(d,G, scale=(1,1,1,1)):
    se,ske,ska,skn=scale; rke,rka,rkn=ke*ske,ka*ska,kn*skn;d2=(d*1e-6)**2
    A=np.block([[(-(rke+rkn)*I-de*se*L/d2)/pe,rka*I/pe],[rke*I/pa,(-rka*I-dg*G/d2)/pa]])
    w=np.r_[np.full(N,np.sqrt(rke*pe)),np.full(N,np.sqrt(rka*pa))]
    lam,V=eigh(w[:,None]*A/w[None,:]); assert np.max(lam)<0
    S=V[:N,:]/w[:N,None]; B=V[:N,:].T*w[:N][None,:]/pe
    ret=[]
    for t in T:
        f=(np.expm1(lam*max(0,t-on))-np.expm1(lam*max(0,t-off)))/lam
        ret.append(u*(S*f)@B)
    return np.array(ret)

def fit(d):
    y=predict(d,L+C)
    out={}
    for mode in ['scaled_path','weighted_shared_support','weighted_shared_support_plus_local_nuisances']:
        n=1 if mode=='scaled_path' else 6 if mode=='weighted_shared_support' else 10
        def calculate(logx):
            x=np.exp(logx)
            G=x[0]*L if n==1 else np.einsum('i,ijk->jk',x[:6],Lparts)
            sc=x[6:] if n==10 else (1,1,1,1)
            return predict(d,G,sc)
        def residual(logx): return ((calculate(logx)-y)/noise).ravel()
        # all paths positive; very wide 0.01--100 multipliers, no measured uncertainty set.
        rng=np.random.default_rng(20261126)
        starts=[np.zeros(n),np.full(n,.5),np.full(n,-.5)]+[rng.uniform(-1.5,1.5,size=n) for _ in range(7)]
        opts=[least_squares(residual,x,bounds=(np.log(.01),np.log(100)),xtol=1e-10,ftol=1e-10,gtol=1e-10,max_nfev=250) for x in starts]
        opt=min(opts,key=lambda o:o.cost)
        delta=calculate(opt.x)-y; r=float(np.linalg.norm(delta)/noise)
        out[mode]=dict(success=bool(opt.success),nfev=opt.nfev,multistart_cost_range=[float(min(o.cost for o in opts)),float(max(o.cost for o in opts))],parameters=np.exp(opt.x).tolist(),relative_residual=float(np.linalg.norm(delta)/np.linalg.norm(y)),whitened_norm=r,squared_whitened_distance=r*r,max_abs_difference_uM=float(np.max(abs(delta))*1000),nominal_N_for_3_sigma_separation=9/(r*r) if r else None)
    return out
results={'description':'Synthetic alternative L+chords versus fitted shared-path family; finite forward separation, not experimental power. All 392 observations independent with sigma=1 uM; one pulse at each site; nuisance bounds illustrative.', 'fits':{str(d):fit(d) for d in [5.,p['matched_window']['optimal_spacing_at_mu1_um'],p['long_step_window']['optimal_spacing_at_mu1_um']]}}
(R/'shared_path_results.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))

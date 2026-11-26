"""Phase V: nonlinear forward test of the frozen Phase IV shared-support rival.
Original EDNM, conservative symmetric edge volumes, native local kinetics.
No claim of age-calibrated physiology. Pulse traces are not used in refitting.
"""
import os,json,contextlib,re
from pathlib import Path
import numpy as np
from ednm import model,solver
from scipy.linalg import norm
from scipy.sparse.linalg import splu
from scipy.sparse import coo_matrix,csr_matrix
from ednm.aot_residual import residual
from ednm_graph_adapter import correction
N=16; DT=float(os.getenv("DT",".001")); D=float(os.getenv("D_UM","42")); MU=float(os.getenv("MU",".179881685")); SRC=int(os.getenv("SOURCE","0"))
rates=[float(q) for q in os.getenv("RATES",".03,3,30").split(",")]
root=Path("output");root.mkdir(exist_ok=True)
m=model.Model(N,D*1e-4,0,0); n=m.N_unknowns_tot
offset=np.arange(N)*m.N_unknowns_unit
JC=[m.jc_s[:,:,None]+offset,m.jc_d[:,:,None]+offset]
JP=[m.jphi_s[:,None]+offset,m.jphi_d[:,None]+offset]
JA=[m.jalpha_s[:,None]+offset,m.jalpha_d[:,None]+offset]
a0=np.array([m.alpha_sn_init,m.alpha_sg_init,m.alpha_se_init]); a1=np.array([m.alpha_dn_init,m.alpha_dg_init,m.alpha_de_init])
c0=np.array([[getattr(m,k+"_s"+r+"_init") for r in "nge"] for k in ["Na","K","Cl","Ca"]]);c1=np.array([[getattr(m,k+"_d"+r+"_init") for r in "nge"] for k in ["Na","K","Cl","Ca"]])
p0=np.array([m.phi_sn_init,m.phi_sg_init,m.phi_se_init]);p1=np.array([m.phi_dn_init,m.phi_dg_init,m.phi_de_init])
g0=np.array([m.h_init,m.n_init,m.s_init,m.c_init,m.q_init,m.z_init]);ap=np.zeros((2,N));sp=np.zeros((2,N))
def pack(a,c,p):
 x=np.zeros(n)
 for l in [0,1]:x[JA[l]]=a[l][:2];x[JC[l]]=c[l];x[JP[l]]=p[l]
 return x
def split(x):
 a=[np.vstack([x[z],1-x[z].sum(axis=0)]) for z in JA]
 return a,[x[z] for z in JC],[x[z] for z in JP]
def r(x,y,g):
 a,c,p=split(x);b,d,z=split(y)
 return residual(DT,N,m.dxu,0,0,a[0],a[1],c[0],c[1],p[0],p[1],b[0],b[1],d[0],d[1],z[0],z[1],g.reshape(6,N),0.,ap,0.,0,sp)
def gate(x,g):
 a,c,p=split(x)
 return solver.gating_variables(m,DT,g.reshape(6,N),c,p).ravel()
x=pack([a0,a1],[c0,c1],[p0,p1]);g=g0.ravel()
def derivative(fun,z,eps=2.5e-6):
 f=fun(z);out=np.zeros((len(f),len(z)))
 for j in range(len(z)):
  h=eps*max(abs(z[j]),.01);w=z.copy();w[j]+=h;u=z.copy();u[j]-=h;out[:,j]=(fun(w)-fun(u))/(2*h)
 assert np.isfinite(out).all()
 return csr_matrix(out)
J=derivative(lambda y:r(y,x,g),x)
P=derivative(lambda y:r(x,y,g),x);RG=derivative(lambda y:r(x,x,y),g)
GX=derivative(lambda y:gate(y,g),x);GG=derivative(lambda y:gate(x,y),g)
# The graph is homogeneous at the frozen rest state, so derivatives of axial
# flux with respect to previous volumes vanish exactly at that state.
a,c,v=split(x);Z=np.zeros((N,N));I=np.eye(N)
xy=np.array([(i,j) for i in range(4) for j in range(4)]);dd=np.abs(xy[:,None]-xy[None,:]);d2=(np.minimum(dd,4-dd)**2).sum(2)
E=[(i,j,1.) for i in range(N) for j in range(i+1,N) if d2[i,j]==1]
A=E+[(i,j,.5) for i,j in [(0,7),(1,12),(2,13),(3,4),(5,10),(6,9),(8,15),(11,14)]]
B=E+[(i,j,.5) for i,j in [(0,13),(1,6),(2,7),(3,12),(4,9),(5,10),(8,15),(11,14)]]
def W(edges,weights=None):
 w=Z.copy()
 for k,(i,j,s) in enumerate(edges):w[i,j]=w[j,i]=s*(1. if weights is None else weights[k])
 return w
we=W(E);wa=W(A)*MU;wb=W(B)*MU
assert np.allclose(wa.sum(0),wb.sum(0))
assert np.allclose(np.linalg.eigvalsh(np.diag(wa.sum(0))-wa),np.linalg.eigvalsh(np.diag(wb.sum(0))-wb))
old=json.loads((Path(__file__).with_name("phase4_ednm_tangent_results.json")).read_text())
# Only the 42 um cases have a preexisting frozen rival at the same scale.
prior=next((a for a in old["results"] if a["spacing_um"]==D and abs(a["mu"]-MU)<1e-8),None)
graphs={"A":(wa,we),"B":(wb,we)}
if prior:
 w=np.asarray(prior["params"]);assert len(w)==2*len(E)
 graphs["R"]=(W(E,w[:len(E)])*MU,W(E,w[len(E):]))
S=np.zeros(n);S[JC[0][1,2,SRC]]=DT/m.gamma_m;S[JC[0][2,2,SRC]]=DT/m.gamma_m
def jac(wg,ee):
 return J+sum((correction(m,wg,ee,DT,a[l],c[l],v[l],layer)[1] for l,layer in enumerate(["soma","dendrite"])),csr_matrix((n,n)))
O=np.zeros((2*N,n));O[np.arange(N),JC[0][1,2]]=1;O[N+np.arange(N),JP[0][0]]=1;O[N+np.arange(N),JP[0][2]]=-1
def tangent(wg,ee,t):
 lu=splu(jac(wg,ee).tocsc());xx=np.zeros(n);gg=np.zeros(6*N);out=np.zeros((len(t),2*N))
 for j,tt in enumerate(t):
  if j==0:continue
  q=1. if round(.08/DT)<=round(tt/DT)<round(.28/DT) else 0.
  xx=lu.solve(-(P@xx+RG@gg+S*q));gg=GX@xx+GG@gg;out[j]=O@xx
 return out
def run(label,wg,ee,q):
 out=root/label;out.mkdir(exist_ok=True);solver.graph_glial=wg;solver.graph_ecs=ee
 mm=model.Model(N,D*1e-4,0,0)
 protocol=dict(j_stim=0.,stim_start=.08,stim_end=.28,N_stim=1,spike_train=np.zeros((2,N)),kcl_rate_tissue=q,kcl_site=SRC)
 with (out/'solver.log').open('w') as f,contextlib.redirect_stdout(f):solver.solve_system(mm,str(out)+'/',.8,protocol)
 with np.load(out/'data.npz') as data:
  t,alpha,c,phi,ss=[data[k] for k in ('t','alpha','c','phi','ss')]
  assert all(np.isfinite(a).all() for a in (t,alpha,c,phi,ss))
  fr=np.concatenate([alpha,1-alpha.sum(axis=2,keepdims=True)],axis=2)
  assert np.min(fr)>0 and np.min(c[:,:,:3,:,:])>0 and np.min(c)>=-1e-8
  inv=np.einsum('tldu,tlidu->ti',fr,c)
  nsteps=np.clip(np.rint(t/DT).astype(int)-round(.08/DT)+1,0,round(.28/DT)-round(.08/DT))
  expected=inv[0]+(nsteps*DT*q)[:,None]*np.array([0.,1.,1.,0.])[None,:]
  err=np.max(abs(inv-expected)/np.maximum(abs(expected),1e-12),axis=0)
  rows=[(int(i),float(e)) for i,e in re.findall(r'N_iter:\s*(\d+),\s*rsd:\s*([0-9.eE+-]+)',(out/'solver.log').read_text())]
  assert rows
  bad=sum(i>=10 and e>1e-12 for i,e in rows)
  if bad or max(err)>=1e-7:raise RuntimeError((label,bad,err))
  k=c[:,0,1,2,:];v=phi[:,0,0,:]-phi[:,0,2,:]
  signals=np.concatenate([k,v],axis=1)
  diag=dict(max_relative_mass_error=err.tolist(),max_newton=max(i for i,_ in rows),unconverged=bad,min_volume=float(fr.min()),max_vm=float(v.max()),spikes=int(np.count_nonzero(data['t_AP'])))
  np.savez_compressed(out/'observables.npz',t=t,signals=signals)
 (out/'data.npz').unlink()
 print('RUN_COMPLETE',label,diag,flush=True)
 return t,signals,diag
def compare(x,y,ind=slice(None)):
 # x candidate, y reference; report K and Vm separately.
 z=[]
 for k in range(2):
  a=x[:,k*N:(k+1)*N][:,ind];b=y[:,k*N:(k+1)*N][:,ind]
  den=float(norm(b));assert den>0
  z.append(dict(relative=float(norm(a-b)/den),max_abs=float(np.max(abs(a-b))),l2=float(norm(a-b))))
 return z
base={};pred={};diags={};responses={}
for label,(wg,ee) in graphs.items():
 t,b,di=run(label+'_base',wg,ee,0.)
 base[label]=b;diags[label+'_base']=di;pred[label]=tangent(wg,ee,t)
 for q in rates:
  tt,y,dg=run(label+'_q'+str(q),wg,ee,q);assert np.allclose(tt,t,rtol=0,atol=1e-11)
  responses[(label,q)]=y-b;diags[label+'_q'+str(q)]=dg
# Compare at 1 ms and also a 10-ms sampling grid; detection design uses 8 fixed independent time samples.
idx=[int(np.argmin(abs(t-z))) for z in [.10,.15,.20,.25,.30,.40,.60,.80]]
assert len(set(idx))==8
distal=np.array([j for j in range(N) if j!=SRC])
results=[]
for q in rates:
 b=responses['B',q];a=responses['A',q]
 z=dict(q=q,topology=compare(a,b),distal_topology=compare(a,b,distal),
        source_peak_K=float(b[:,SRC].max()),tangent_error_B=compare(q*pred['B'],b),
        tangent_error_A=compare(q*pred['A'],a),
        tangent_contrast_error=compare(q*(pred['A']-pred['B']),a-b))
 for c in ['A','R']:
  if c not in graphs:continue
  v=responses[c,q];e=v[idx]-b[idx];z[c+'_squared_distance_8_samples']=[float(np.sum(e[:,:N]**2)),float(np.sum(e[:,N:]**2))]
  z[c+'_distal_squared_distance_8_samples']=[float(np.sum(e[:,:N][:,distal]**2)),float(np.sum(e[:,N:][:,distal]**2))]
  if c=='R':z['frozen_rival']=compare(v,b);z['distal_rival']=compare(v,b,distal);z['tangent_error_R']=compare(q*pred[c],v)
 results.append(z)
out=dict(scope='Nonlinear native-state EDNM, not age-calibrated, neutral KCl pulse; R if available was fitted to v0.4 transfer training frequencies only; no fitting to these time series',d_um=D,mu=MU,dt=DT,source=SRC,volumes='symmetric new edges, original convention subtracted',pulse=[.08,.28],prior_rival_job=prior['job_id'] if prior else None,samples=t[idx].tolist(),results=results,audits=diags)
(root/'result.json').write_text(json.dumps(out,indent=2,allow_nan=False))
np.savez_compressed(root/'responses.npz',t=t,**{f'{k[0]}_q{k[1]}':v for k,v in responses.items()},**{f'tangent_{k}':v for k,v in pred.items()})
print('PHASE5_NONLINEAR '+json.dumps(out,allow_nan=False),flush=True)

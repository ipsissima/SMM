"""Frozen small-signal EDNM transfer; uses the actual pinned multionic residual.
Not a nonlinear simulation or a developmental calibration. All inputs synthetic.
"""
import os,json
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix,coo_matrix
from scipy.sparse.linalg import splu
from scipy.optimize import minimize
from ednm import model,solver
from ednm.aot_residual import residual
from ednm.aot_jacobian import jacobian
from ednm_graph_adapter import correction,closed_path
N=16; DT=float(os.getenv("DT",".001")); D=float(os.getenv("D_UM","42")); MU=float(os.getenv("MU",".179881685"))
m=model.Model(N,D*1e-4,0,0); n=m.N_unknowns_tot; q=6*N
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
base=r(x,x,g);gbase=gate(x,g);print("base",np.max(abs(base)),np.max(abs(gbase-g)),flush=True)
# Frozen native state, not assumed a developmental steady state.
# All derivative matrices are of actual residual and gating update; central FD
# of previous and gating arguments. Current native analytic Jacobian checked
# against a random directional difference below.
a,c,v=split(x)
i,j,val=jacobian(DT,N,m.dxu,0,0,a[0],a[1],c[0],c[1],v[0],v[1],a[0],a[1],c[0],c[1],v[0],v[1],g.reshape(6,N),0.,ap,sp)
J=coo_matrix((val,(i.astype(int),j.astype(int))),shape=(n,n)).tocsr()
def derivative(fun,z,eps=1e-5):
 f=fun(z);out=np.zeros((len(f),len(z)))
 for j in range(len(z)):
  h=eps*max(abs(z[j]),.01);w=z.copy();w[j]+=h;u=z.copy();u[j]-=h;out[:,j]=(fun(w)-fun(u))/(2*h)
 if not np.isfinite(out).all():raise ValueError("nonfinite derivative")
 return csr_matrix(out)
P=derivative(lambda y:r(x,y,g),x);RG=derivative(lambda y:r(x,x,y),g)
GX=derivative(lambda y:gate(y,g),x);GG=derivative(lambda y:gate(x,y),g);gd=GG.diagonal()
assert np.linalg.norm((GG-csr_matrix(np.diag(gd))).data)<1e-8
rng=np.random.default_rng(20261126);u=rng.normal(size=n)*np.maximum(abs(x),.01);h=1e-6
f=(r(x+h*u,x,g)-r(x-h*u,x,g))/(2*h);err=np.linalg.norm(f-J@u)/np.linalg.norm(f);print("current_derivative_error",err,flush=True)
# Native AOT current derivative is an implementation comparator, not assumed exact.
# Use centered differentiation of the actual native residual, with a refinement
# and independent directional check; preserve the failed AOT run separately.
Jfd1=derivative(lambda y:r(y,x,g),x,eps=1e-5)
Jfd2=derivative(lambda y:r(y,x,g),x,eps=2.5e-6)
fd_refine=np.linalg.norm((Jfd1-Jfd2).data)/np.linalg.norm(Jfd2.data)
fd_direction=np.linalg.norm(f-Jfd2@u)/np.linalg.norm(f)
print("current_fd_refinement",fd_refine,"current_fd_direction",fd_direction,flush=True)
if fd_refine>1e-4 or fd_direction>1e-4:raise ValueError("current residual derivative refinement failed")
J=Jfd2
# remove original path by the verified SMM-I correction; add graph edges linearly.
Z=np.zeros((N,N))
def corr(Wg,We):
 return sum((correction(m,Wg,We,DT,a[l],c[l],v[l],layer)[1] for l,layer in enumerate(["soma","dendrite"])),csr_matrix((n,n)))
C0=corr(Z,Z);J0=J+C0
xy=np.array([(i,j) for i in range(4) for j in range(4)]); dd=np.abs(xy[:,None]-xy[None,:]);d2=(np.minimum(dd,4-dd)**2).sum(2)
E=[(i,j,1.) for i in range(N) for j in range(i+1,N) if d2[i,j]==1]
A=E+[(i,j,.5) for i,j in [(0,7),(1,12),(2,13),(3,4),(5,10),(6,9),(8,15),(11,14)]]
B=E+[(i,j,.5) for i,j in [(0,13),(1,6),(2,7),(3,12),(4,9),(5,10),(8,15),(11,14)]]
def edge(e,g):
 i,j,w=e;W=Z.copy();W[i,j]=W[j,i]=w*(MU if g else 1.)
 return (corr(W,Z) if g else corr(Z,W))-C0
eg=[edge(e,True) for e in E]; ee=[edge(e,False) for e in E];da=[edge(e,True) for e in A];db=[edge(e,True) for e in B]
def combine(es,w):return sum((v*b for v,b in zip(w,es)),csr_matrix((n,n)))
EA=combine(ee,np.ones(len(ee))); JA0=J0+EA+combine(da,np.ones(len(da)));JB0=J0+EA+combine(db,np.ones(len(db)))
# positive KCl influx into somatic ECS residual, neutral charge.
S=np.zeros((n,N));S[JC[0][1,2],np.arange(N)]=DT/m.gamma_m;S[JC[0][2,2],np.arange(N)]=DT/m.gamma_m
O=np.zeros((2*N,n));O[np.arange(N),JC[0][1,2]]=1;O[N+np.arange(N),JP[0][0]]=1;O[N+np.arange(N),JP[0][2]]=-1
train=[.2,1.,5.];real=[.1,.5,2.,10.];imag=[1j*s for s in real];all_s=train+real+imag
# Residual sign is R_current x + R_previous xprev + R_g gprev + S input =0.
# Eliminate the lagged gate state gprev with r=exp(-s*dt).
lag={}
for s in all_s:
 z=np.exp(-s*DT);lag[s]=z*P+RG@csr_matrix(np.diag(z/(1-z*gd)))@GX
def solve(Js,s,grad=False):
 mat=(Js+lag[s]).tocsc();lu=splu(mat);X=lu.solve(-S.astype(mat.dtype));H=O@X
 if not grad:return H
 Y=lu.solve(O.T.astype(mat.dtype),trans='T').T
 return H,X,Y
truth={s:solve(JB0,s) for s in all_s};alt={s:solve(JA0,s) for s in all_s}
def errs(p,t,ss):
 return [float(np.linalg.norm(np.stack([p[s][k*N:(k+1)*N] for s in ss])-np.stack([t[s][k*N:(k+1)*N] for s in ss]))/np.linalg.norm(np.stack([t[s][k*N:(k+1)*N] for s in ss]))) for k in range(2)]
sc={s:np.repeat([np.linalg.norm(truth[s][:N]),np.linalg.norm(truth[s][N:])],N)[:,None] for s in train}
es=eg+ee
def fg(v):
 w=np.exp(v);JJ=J0+combine(es,w);loss=0.;grad=np.zeros(len(es))
 for s in train:
  H,X,Y=solve(JJ,s,True);D=(H-truth[s])/sc[s];loss+=.5*np.sum(D**2);W=D/sc[s]
  for k,e in enumerate(es):grad[k]-=w[k]*np.sum(W*(Y@(e@X)))
 return float(loss),grad
v0=np.zeros(len(es));delta=rng.normal(size=len(es));delta/=np.linalg.norm(delta);f0,gf=fg(v0);fd=(fg(v0+1e-5*delta)[0]-fg(v0-1e-5*delta)[0])/2e-5
gerr=abs(fd-gf@delta)/max(abs(fd),abs(gf@delta),1e-12);print("fit_gradient_error",gerr,flush=True);assert gerr<1e-3
bounds=[(np.log(.05),np.log(20.))]*len(eg)+[(np.log(.3),np.log(3.))]*len(ee)
fits=[minimize(fg,v,method="L-BFGS-B",jac=True,bounds=bounds,options={"maxiter":250,"ftol":1e-14,"gtol":1e-9,"maxls":35}) for v in [v0,rng.normal(0,.08,len(es))]]
best=min(fits,key=lambda t:t.fun);w=np.exp(best.x);JJ=J0+combine(es,w);pred={s:solve(JJ,s) for s in all_s}
lo,hi=np.array(bounds).T;pg=float(np.max(abs(best.x-np.clip(best.x-best.jac,lo,hi))))
out=dict(scope="Frozen native EDNM discrete tangent, full ionic and gate equations, not nonlinear or age-calibrated; fixed local physiology, fitted glial and ECS shared weights only",spacing_um=D,mu=MU,dt=DT,initial_residual=float(max(abs(base))),gate_step_drift=float(max(abs(gbase-g))),native_aot_jacobian_direction_error=float(err),current_fd_refinement=float(fd_refine),current_fd_direction=float(fd_direction),fit_gradient_error=float(gerr),train=errs(pred,truth,train),hold_real=errs(pred,truth,real),hold_imag=errs(pred,truth,imag),AB_real=errs(alt,truth,real),AB_imag=errs(alt,truth,imag),converged=[bool(f.success) for f in fits],messages=[str(f.message) for f in fits],iterations=[int(f.nit) for f in fits],losses=[float(f.fun) for f in fits],projected_gradient=pg,params=w.tolist(),observables=["ECS K mM","neuronal Vm mV"])
Path("output").mkdir(exist_ok=True);Path("output/result.json").write_text(json.dumps(out,indent=2));print("SMMII_EDNM "+json.dumps(out),flush=True)

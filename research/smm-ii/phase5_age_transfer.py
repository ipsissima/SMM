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
# Age-specific mobility / spacing, native local physiology held fixed.
# This is a band-limited forced-response covariance, not a claim of
# stationary spontaneous activity or spike correlations.
assert D==42 and MU==1
SG=.03240968736;SE=.1321300159;GM=3.335314393e6
tri=np.triu_indices(N,1);Pi=np.eye(N)-np.ones((N,N))/N
Q=np.diag(1+.8*np.exp(-d2[:,0]/.8));Q*=N/np.trace(Q)
Qb=Pi@Q@Pi;Qb*=N/np.trace(Qb)
qs={'all':Q,'balanced':Qb}
S=np.zeros((n,N));S[JC[0][1,2],np.arange(N)]=DT/m.gamma_m;S[JC[0][2,2],np.arange(N)]=DT/m.gamma_m
O=np.zeros((2*N,n));O[np.arange(N),JC[0][1,2]]=1;O[N+np.arange(N),JP[0][0]]=1;O[N+np.arange(N),JP[0][2]]=-1
EG=combine(eg,np.ones(len(eg)));GA=combine(da,np.ones(len(da)));GB=combine(db,np.ones(len(db)))
def Jscale(d,mu,G):
 return J0+(42/d)**2*(EA+mu*G)
def lag(s):
 z=np.exp(-s*DT);return z*P+RG@csr_matrix(np.diag(z/(1-z*gd)))@GX
def H(Js,s,L=None):
 M=(Js+(lag(s) if L is None else L)).tocsc()
 lu=splu(M);X=lu.solve(-S.astype(M.dtype));return O@X
# Directly verify scale transform at the SMM-I 30 um peak.
m.dxu=30e-4
Jdirect=derivative(lambda y:r(y,x,g),x,eps=2.5e-6)
def weights(edges):
 out=Z.copy()
 for i,j,w in edges:out[i,j]=out[j,i]=w
 return out
Jdirect=Jdirect+corr(weights(B)*.179881685,weights(E))
Jscaled=Jscale(30,.179881685,GB)
errs=[float(np.linalg.norm(H(Jdirect,1j*w)-H(Jscaled,1j*w))/np.linalg.norm(H(Jdirect,1j*w))) for w in [.1,1,10]]
m.dxu=42e-4
print("SCALE_CHECK",errs,flush=True);assert max(errs)<1e-5
# Discrete forced harmonic solutions at frequencies with OU-shaped weights.
# This finite band avoids the conserved uniform mode at zero frequency.
band=np.geomspace(.05,50,65)
lags=[lag(1j*w) for w in band]
def quadrature(w):
 b=np.diff(w);a=np.r_[b[0]/2,(b[:-1]+b[1:])/2,b[-1]/2]
 a*=.4/(1+(.2*w)**2);return a/a.sum()
def cov(HH,q,w):
 return np.einsum('f,fij,jk,flk->il',quadrature(w),HH,q,HH.conj(),optimize=True).real
def co(C):
 C=(C+C.T)/2;sd=np.sqrt(np.diag(C));assert np.min(sd)>0
 assert np.linalg.eigvalsh(C).min()>-1e-7*np.linalg.norm(C)
 return (C/np.outer(sd,sd))[tri]
def metrics(ha,hb,q,grid):
 cA,cB=cov(ha,q,grid),cov(hb,q,grid)
 a,b=co(cA),co(cB)
 cD=cov(ha-hb,q,grid)
 return dict(corr_A=a.tolist(),corr_B=b.tolist(),corr_L2=float(np.linalg.norm(a-b)),
             corr_relative=float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-15)),
             transfer_rel_energy=float(np.sqrt(np.trace(cD)/np.trace(cB))),
             rms_B=float(np.sqrt(np.trace(cB)/N)),rms_difference=float(np.sqrt(np.trace(cD)/N)))
age=json.loads(Path(__file__).with_name('phase5_age_inputs.json').read_text())
mode=int(os.environ.get("MODE","1")); rows=[]
for v in age['rows']:
 day,d,xi=v[0],v[1],v[2+mode]
 if xi is None:continue
 mu=xi*GM*(d*1e-6)**2/SG
 ell=1e6*np.sqrt(1/(GM*(1/(SG*mu)+1/SE)))
 JAs,JBs=Jscale(d,mu,GA),Jscale(d,mu,GB)
 ha=np.stack([H(JAs,1j*w,L) for w,L in zip(band,lags)])
 hb=np.stack([H(JBs,1j*w,L) for w,L in zip(band,lags)])
 assert np.isfinite(ha).all() and np.isfinite(hb).all()
 o=dict(day=day,d_um=d,xi=xi,mu=mu,length_um=ell,lambda_two=(ell/d)**2,metrics={})
 for key,q0 in qs.items():
  o['metrics'][key]={}
  for obs,k in [('K',0),('Vm',1)]:
   a,b=ha[:,k*N:(k+1)*N],hb[:,k*N:(k+1)*N]
   fine=metrics(a,b,q0,band);coarse=metrics(a[::2],b[::2],q0,band[::2])
   fine['coarse_corr_L2']=coarse['corr_L2']
   fine['coarse_relative_energy']=coarse['transfer_rel_energy']
   fine['coarse_corr_diff_L2']=float(np.linalg.norm(np.array(fine['corr_A'])-coarse['corr_A']))
   o['metrics'][key][obs]=fine
 rows.append(o)
 print("AGE "+json.dumps({k:v for k,v in o.items() if k!="metrics"}|{"metrics":{k:{a:{q:p for q,p in z.items() if not q.startswith('corr_A') and not q.startswith('corr_B')} for a,z in v.items()} for k,v in o['metrics'].items()}}),flush=True)
Path('output').mkdir(exist_ok=True)
result=dict(scope='Native EDNM frozen tangent, conditional age-varying axial mobility and spacing only, finite-band forced-response covariance, no firing or anatomical calibration',mode=mode,mode_label=age['columns'][2+mode],dt=DT,grid=band.tolist(),tau_s=.2,scale_check=errs,fd_refinement=float(fd_refine),fd_direction=float(fd_direction),rows=rows)
Path('output/age_results.json').write_text(json.dumps(result,allow_nan=False))
# Exact 120-edge covariance vectors and metadata; compact logs for connector transfer.
for row in rows:print("AGE_VECTOR "+json.dumps(row,separators=(',',':')),flush=True)
print("AGE_COMPLETE "+json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)

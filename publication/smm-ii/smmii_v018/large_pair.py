"""SMM-II v0.18: prospective 3D paired astrocyte impedance assay.
All graph draws are synthetic. Zhong PA resistance anchors are used as output
scale, not independent validation or measurements of isolated membrane shunt.
The fixed-source normalized operator and native EDNM calculations are unchanged.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,csv,json,time
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg, LinearOperator, expm_multiply
from scipy.fft import rfftn,irfftn
from concurrent.futures import ProcessPoolExecutor
HERE=Path(__file__).resolve().parent
REF=np.loadtxt(HERE/'ca1_reference.csv',delimiter=',',skiprows=1)
DAYS=[7,9,10,11,12,15,21]; P11=3; P21=6
SG=.03240968736; SE=.1321300159; GM=3.335314393e6
# Fig. 5e2 PA, not the rare residual VRA population. Interpolated P7/P9/P10
# is a planning convention; P10--12 and later have plateau 9.7 MOhm.
# P10/11/12 use same measured group, not individual age measurements.
R_TARGET=np.interp(DAYS,[2.5,10,21],[18.4,9.7,9.7])
I_FIXED=200.; V_MATCH=2.
# Nine patch positions allow axial distances 1,2,3 and diagonal equal-position controls.
OFFSETS=np.array([[0,0,0],[1,0,0],[2,0,0],[3,0,0],[0,1,0],[0,2,0],[0,3,0],[0,0,1],[0,0,2],[0,0,3]])
PAIRS=np.array([(i,j) for i in range(len(OFFSETS)) for j in range(i+1,len(OFFSETS)) if 1<=np.abs(OFFSETS[i]-OFFSETS[j]).sum()<=3])
DIST=np.abs(OFFSETS[PAIRS[:,0]]-OFFSETS[PAIRS[:,1]]).sum(1)
# Separate axial separation from Manhattan path-length. Axial class removes the
# confound of mixing different Euclidean separations across the three shells.
AXIAL=(np.count_nonzero(OFFSETS[PAIRS[:,0]]-OFFSETS[PAIRS[:,1]],axis=1)==1)

class Lattice:
 def __init__(self,n):
  self.n=n;self.N=n**3;self.shape=(n,)*3;self.ids=np.arange(self.N).reshape(self.shape)
  self.nbr=np.array([np.roll(self.ids,-1,axis=a).ravel() for a in range(3)])
  self.rows=np.tile(np.arange(self.N),3); self.cols=self.nbr.ravel()
  self.points=np.ravel_multi_index((OFFSETS+n//2).T,self.shape)
  self.e=self.lap(np.ones((3,self.N)))
  l=2-2*np.cos(2*np.pi*np.arange(n)/n); self.ev=l[:,None,None]+l[None,:,None]+l[None,None,:]
  self.ef=self.ev[:,:,:n//2+1]
 def lap(self,w):
  w=np.asarray(w).reshape(-1); A=sparse.coo_matrix((np.r_[w,w],(np.r_[self.rows,self.cols],np.r_[self.cols,self.rows])),shape=(self.N,self.N)).tocsr()
  return sparse.diags(np.asarray(A.sum(1)).ravel())-A
 def weights(self,rng,amp):
  # Each plaquette adds +t to its two parallel a-edges and -t to its
  # two parallel b-edges. Superpose independent faces; all degrees stay six.
  w=np.ones((3,*self.shape)); scale=.225*amp
  for a,b in [(0,1),(0,2),(1,2)]:
   f=rng.uniform(-scale,scale,self.shape)
   w[a]+=f+np.roll(f,1,axis=b);w[b]-=f+np.roll(f,1,axis=a)
  w=w.reshape(3,self.N);deg=sum(w[a]+np.roll(w[a].reshape(self.shape),1,axis=a).ravel() for a in range(3))
  assert np.max(abs(deg-6))<1e-12 and w.min()>=.1-1e-12
  return w
 def homdiag(self,xi,alpha,bd):
  if bd=='clamped':h=1/(1+xi*self.ev)
  else:h=1/(1+xi/(1+alpha*xi)*self.ev)
  return h.mean()
 def transfer(self,L,xi,alpha,bd,tol=1e-10):
  N=self.N; pts=self.points
  if bd=='clamped':
   mat=sparse.eye(N,format='csr')+xi*L
   p=1/(1+xi*self.ef)
   def prec(v):return irfftn(rfftn(v.reshape(self.shape))*p,s=self.shape).ravel()
   M=LinearOperator((N,N),matvec=prec,dtype=float)
  else:
   beta=1/alpha; Le=self.e; a=1+xi*self.ef;b=1+beta*self.ef; den=a*b-1;den[0,0,0]=1
   def mv(v):
    x,y=v[:N],v[N:];return np.r_[x-y+xi*(L@x),y-x+beta*(Le@y)]
   mat=LinearOperator((2*N,2*N),matvec=mv,dtype=float)
   def prec(v):
    x=rfftn(v[:N].reshape(self.shape)); y=rfftn(v[N:].reshape(self.shape))
    p=(b*x+y)/den;q=(x+a*y)/den;p[0,0,0]=(x[0,0,0]-y[0,0,0])/4;q[0,0,0]=-p[0,0,0]
    return np.r_[irfftn(p,s=self.shape).ravel(),irfftn(q,s=self.shape).ravel()]
   M=LinearOperator((2*N,2*N),matvec=prec,dtype=float)
  H=np.zeros((len(pts),len(pts))); residual=0;iters=0
  if bd=='clamped':M=None
  for j,pnt in enumerate(pts):
   rhs=np.zeros(mat.shape[0]);rhs[pnt]=1
   if bd!='clamped':rhs[N+pnt]=-1
   count=[0]
   def cb(x):count[0]+=1
   x,info=cg(mat,rhs,M=M,rtol=tol,atol=0,maxiter=300,callback=cb)
   err=np.linalg.norm(mat@x-rhs)/np.linalg.norm(rhs); residual=max(residual,err);iters=max(iters,count[0]);assert info==0 and err<max(tol*10,1e-9),(info,err)
   H[:,j]=x[pts] if bd=='clamped' else x[pts]-x[N+pts]
  symmetry=np.max(abs(H-H.T));assert symmetry<1e-8
  return H,dict(residual=residual,symmetry=symmetry,iterations=iters)
 def tracer(self,L,xi,exposure):
  # Clamped donor fluorescence. Markov semigroup with absorbing donor row.
  # Relative dye time and permeability are scenario parameters, not calibrated kinetics.
  from scipy.stats import poisson
  pts=self.points;N=self.N; k=len(pts);j=np.arange(k);X=np.zeros((N,k));X[pts,j]=1
  rate=float(L.diagonal().max());u=rate*xi*exposure;P=sparse.eye(N,format='csr')-L/rate
  w=np.exp(-u);out=w*X[pts,:];K=int(poisson.ppf(1-1e-13,u))
  for i in range(1,K+1):
   X=P@X;X[pts,j]=1;w*=u/i;out+=w*X[pts,:]
  assert poisson.sf(K,u)<2e-13 and out.min()>-1e-10 and out.max()<1+1e-10
  return out

def job(args):
 n,g,amp,bd,age_idx,tracer=args; lat=Lattice(n);rng=np.random.default_rng(2026112800+g+n*1000)
 W=lat.weights(rng,amp);L=lat.lap(W); Wmis=W*np.exp(.5*rng.normal(size=W.shape)-.125); Lmis=lat.lap(Wmis)
 rows=[];rng_noise=np.random.default_rng(2026112801+g+n*1000)
 for ai in age_idx:
  age=DAYS[ai];day,d,xi,*_=REF[REF[:,0]==age][0];alpha=GM*(d*1e-6)**2/SE
  H,checks=lat.transfer(L,xi,alpha,bd)
  z={}
  if tracer and age in (11,21):
   for mode,l in [('ideal',L),('mismatch',Lmis)]:
    T=lat.tracer(l,xi,1.)
    if mode=='mismatch':T+=.02*rng_noise.normal(size=T.shape)
    z[mode]=T.tolist()
  rows.append(dict(day=age,H=H.tolist(),homogeneous_diagonal=lat.homdiag(xi,alpha,bd),checks=checks,dye=z))
 return dict(graph=g,weight_min=float(W.min()),weight_max=float(W.max()),weight_sd=float(W.std()),rows=rows)

def run(n,ng,amp,bd,age_idx,with_tracer,workers,start=0):
 t=time.monotonic();args=[(n,g,amp,bd,age_idx,with_tracer) for g in range(start,start+ng)]
 with ProcessPoolExecutor(max_workers=workers) as pool:
  out=[]
  for o in pool.map(job,args):
   out.append(o)
   if len(out)%8==0:print(n,ng,amp,bd,len(out),'elapsed',round(time.monotonic()-t,1),flush=True)
 return dict(n=n,sites=n**3,graphs=ng,amplitude=amp,boundary=bd,age_idx=age_idx,records=out)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--n',type=int,default=31);p.add_argument('--ng',type=int,default=64);p.add_argument('--amp',type=float,default=1);p.add_argument('--boundary',choices=['clamped','differential'],default='clamped');p.add_argument('--ages',nargs='+',type=int,default=DAYS);p.add_argument('--tracer',action='store_true');p.add_argument('--workers',type=int,default=4);p.add_argument('--start',type=int,default=0);a=p.parse_args()
 # Check uniform-lattice inversion and independent dense reference on a tiny graph.
 tiny=Lattice(7);xi=.4084605045;alpha=.031
 for bd in ['clamped','differential']:
  H,chk=tiny.transfer(tiny.e,xi,alpha,bd);assert max(abs(np.diag(H)-tiny.homdiag(xi,alpha,bd)))<1e-10
 ans=run(a.n,a.ng,a.amp,a.boundary,[DAYS.index(d) for d in a.ages],a.tracer,a.workers,a.start)
 path=HERE/f'large_{a.n}_{a.ng}_{a.amp:g}_{a.boundary}_{"-".join(map(str,a.ages))}{'-start'+str(a.start) if a.start else ''}.json';path.write_text(json.dumps(ans,separators=(',',':')));print('wrote',path,flush=True)

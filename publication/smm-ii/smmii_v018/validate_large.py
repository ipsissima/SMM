"""Independent dense small-graph and tracer checks; refinement at full size."""
from large_pair import *
from scipy.linalg import null_space
lat=Lattice(7);rng=np.random.default_rng(7726);L=lat.lap(lat.weights(rng,1));N=lat.N;U=null_space(np.ones((1,N)));I=np.eye(N);Ic=np.eye(N-1);le=lat.e.toarray();lg=L.toarray();pts=lat.points
out={}
for xi,alpha in [(.4084605045,.031),(2.2127505358,.045)]:
 for bd in ['clamped','differential']:
  if bd=='clamped':exact=np.linalg.solve(I+xi*lg,I)
  else:
   A=np.linalg.solve(xi*(U.T@lg@U),Ic)+alpha*np.linalg.solve(U.T@le@U,Ic)
   exact=I-U@np.linalg.solve(Ic+A,Ic)@U.T
  H,check=lat.transfer(L,xi,alpha,bd)
  ref=exact[np.ix_(pts,pts)];err=np.linalg.norm(H-ref)/np.linalg.norm(ref);assert err<1e-9
  out[f'dense_{xi}_{bd}']=float(err)
 T=lat.tracer(L,xi,1.);T2=np.zeros_like(T)
 for j,i in enumerate(pts):
  A=L.tolil();A[i,:]=0;A=A.tocsr();u=np.zeros(N);u[i]=1;T2[:,j]=expm_multiply(-xi*A,u,traceA=-xi*A.diagonal().sum())[pts]
 e=np.max(abs(T-T2));assert e<1e-11;out[f'tracer_{xi}']=float(e)
lat=Lattice(31);L=lat.lap(lat.weights(np.random.default_rng(2026112800+31*1000),1))
for xi,alpha in [(.4084605045,.031),(2.2127505358,.045)]:
 for bd in ['clamped','differential']:
  a,c=lat.transfer(L,xi,alpha,bd,1e-10);b,d=lat.transfer(L,xi,alpha,bd,1e-12);err=np.linalg.norm(a-b)/np.linalg.norm(b);assert err<1e-9;out[f'refined_{xi}_{bd}']=float(err)
Path(HERE/'large_validation.json').write_text(json.dumps(out,indent=2)+'\n');print(out)

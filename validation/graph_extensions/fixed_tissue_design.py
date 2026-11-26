"""Fixed physical tissue, conservative finite-volume passive SMM diagnostic.

Synthetic, one-dimensional, sealed domain; not a CA1 reconstruction.
Seven coarse cells of 37 um, fine mesh subdivisions r odd. Chord integrated
conductance fixed in physical units, Dg*w/d, between coarse-cell centers;
edge flux divided by fine control volume h. All injections deliver equal
integrated source mass per unit cross-sectional area. No re-scaling of
the underlying tissue between observation designs. Independent equal sensor
noise is illustrative, not calibrated. Profile Fisher information is local.
"""
import itertools, json, os
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import least_squares

PE,PA=.2,.4
KE,KA,KN=1.29629,.52863,.232
DE,DG=1.05270e-9,2.29581e-10
DOSE=.1714683
D=37e-6
ON,OFF=.08,.4
T=np.array([.25,.5,.75,1.,1.5,2.,2.5,3.])
SIGMA=.001
MU0=.17988168474408803
MUB=(.11900244408271873,.3107547903908609)
CHORDS=[(0,4,.8),(1,5,.5),(2,6,.6)]
OUT=Path(os.environ.get("SMM_FIXED_OUTPUT","output"));OUT.mkdir(parents=True,exist_ok=True)

def setup(r):
    n=7*r; h=D/r
    B=np.eye(n)[:-1]-np.eye(n)[1:]
    L=B.T@B/h**2
    C=np.zeros((n,n))
    centers=np.arange(7)*r+r//2
    for i,j,w in CHORDS:
        a=np.eye(n)[centers[i]]-np.eye(n)[centers[j]]
        C+=w*np.outer(a,a)/(D*h)
    coarse=np.zeros((7,n))
    for i in range(7):coarse[i,i*r:(i+1)*r]=1/r
    # Point-like means a single fine cell. Within each mesh all sources have
    # the same integrated mass DOSE*D. Point widths change across mesh levels;
    # only block-average convergence is interpreted as a fixed-protocol limit.
    wide=np.eye(n)[centers]
    dense=np.eye(n)[np.arange(7)]
    return n,h,L,C,coarse,wide,dense

def predict(r,theta,src_kind,obs_kind,model=None):
    n,h,L,C,coarse,wide,dense=setup(r);I=np.eye(n)
    g,mu,se,ske,ska,skn=theta
    ke,ka,kn=KE*ske,KA*ska,KN*skn
    G=mu*DG*(L+g*C) if model is None else DG*model
    A=np.block([[(-(ke+kn)*I-DE*se*L)/PE,ka*I/PE],
                [ke*I/PA,(-ka*I-G)/PA]])
    S={"block":coarse,"wide":wide,"dense":dense,"all":I}
    src=S[src_kind]; obs=S[obs_kind]
    # positive time integrator in weighted symmetric coordinates
    w=np.r_[np.full(n,np.sqrt(ke*PE)),np.full(n,np.sqrt(ka*PA))]
    Z=w[:,None]*A/w[None,:]
    assert np.allclose(Z,Z.T,rtol=1e-11,atol=1e-7)
    lam,V=eigh(Z); assert lam[-1]<0
    F=obs@(V[:n,:]/w[:n,None])
    # r * src.T gives equal integrated flux D times DOSE for each source
    B=V[:n,:].T@(w[:n,None]*r*src.T/PE)
    k=np.stack([DOSE*(F*((np.expm1(lam*max(t-ON,0))-np.expm1(lam*max(t-OFF,0)))/lam))@B for t in T])
    assert np.isfinite(k).all()
    return k

def metrics(r,theta,src,obs):
    theta=np.array(theta,float);base=predict(r,theta,src,obs)
    t0=theta.copy();t0[0]=0.;rival=predict(r,t0,src,obs)
    def jac(h):
        cols=[]
        for i in range(6):
            d=np.zeros(6);d[i]=h*max(abs(theta[i]),.1)
            cols.append(((predict(r,theta+d,src,obs)-predict(r,theta-d,src,obs))/(2*d[i]*SIGMA)).ravel())
        return np.column_stack(cols)
    J=jac(1e-4);J2=jac(5e-5);x=J2[:,0];X=J2[:,1:]
    e=x-X@np.linalg.lstsq(X,x,rcond=None)[0]
    return dict(r=r,h_um=37/r,domain_um=259.,source=src,sensor=obs,theta=theta.tolist(),
                n_measurements=int(base.size),known_I=float(x@x),profile_I=float(e@e),
                relative_K=float(np.linalg.norm(base-rival)/np.linalg.norm(rival)),
                max_K_uM=float(1000*np.max(abs(base-rival))),
                derivative_convergence=float(np.linalg.norm(J2-J)/np.linalg.norm(J2)),
                smallest_singular_value=float(np.linalg.svd(J2,compute_uv=False)[-1]))

rows=[]
# Independent reconstruction, not a claim to reproduce the referee's exact mesh.
for r in [1,3,7,15,21]:
    for mu in [1.,MU0]:
        rows.append(metrics(r,[1,mu,1,1,1,1],"block","block"))
# Fixed tissue and mesh, equal count and physical support width for point-like
# sources and sensors; compare coverage of the candidate conduits.
for mu in [1.,MU0]:
    for s,o in [("wide","wide"),("dense","dense"),("wide","all"),("dense","wide"),("wide","dense")]:
        rows.append(metrics(7,[1,mu,1,1,1,1],s,o))
for row in rows: print("FIXED_TISSUE_RESULT",json.dumps(row),flush=True)
# A structured robustness box, NOT a probabilistic uncertainty distribution.
# Mu endpoints are conditional empirical conductivity ratios; the remaining
# multipliers are illustrative ranges, not empirical confidence bounds.
# Passive mu need not equal electrodiffusive electrical mobility quantitatively.
unc=[]
for mu,se,ske,ska,skn in itertools.product(MUB,[.7,1.3],[.5,2.],[.5,2.],[.5,2.]):
    unc.append(metrics(7,[1,mu,se,ske,ska,skn],"wide","wide"))
unc.append(metrics(7,[1,MU0,1,1,1,1],"wide","wide"))
for row in unc:print("FIXED_TISSUE_ROBUSTNESS",json.dumps(row),flush=True)
result=dict(description=__doc__,baseline_rows=rows,robustness_rows=unc,
            robustness_summary={k:{"min":float(min(x[k] for x in unc)),"max":float(max(x[k] for x in unc)),"median":float(np.median([x[k] for x in unc]))} for k in ["profile_I","relative_K","max_K_uM"]})
(OUT/"fixed_tissue_results.json").write_text(json.dumps(result,indent=2)+"\n")
print("FIXED_TISSUE_SUMMARY",json.dumps(result["robustness_summary"]),flush=True)

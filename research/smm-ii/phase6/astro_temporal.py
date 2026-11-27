"""Exploratory temporal replication of calcium coactivation beyond fixed margins.
The original full-record conMat is NOT used as a predictor. No neuronal outcome,
causal inference or direct gap-junction measurement. Recorded ROI set is
post-selected by authors; conditional rather than prospective validation.
"""
import io,json,hashlib,urllib.request
from pathlib import Path
import numpy as np
from scipy.io import loadmat
root=Path("output");root.mkdir(exist_ok=True)
rng=np.random.default_rng(20261127)
source="e68a8970c6e3db32f4f30f51bfc0b2e79aea5c78"
def walk(x,path=""):
    if hasattr(x,"_fieldnames"):x={k:getattr(x,k) for k in x._fieldnames}
    if isinstance(x,dict):
        if "evtBegin" in x and "nAstros" in x:yield path,x
        else:
            for k,v in x.items():yield from walk(v,path+"/"+k)
    elif isinstance(x,(list,tuple)) or isinstance(x,np.ndarray) and x.dtype==object:
        for i,v in enumerate(np.asarray(x,dtype=object).flat):yield from walk(v,path+"/"+str(i))
def curveball(rows,steps):
    nr=len(rows)
    for _ in range(steps):
        a,b=rng.choice(nr,2,replace=False)
        x,y=rows[a],rows[b];common=x&y
        aa=x-y;bb=y-x
        if aa and bb:
            pool=np.array(sorted(aa|bb));rng.shuffle(pool)
            rows[a]=common|set(pool[:len(aa)])
            rows[b]=common|set(pool[len(aa):])
def sample(X,N):
    nr,nc=X.shape;tri=np.triu_indices(nc,1)
    rows=[set(np.where(x)[0]) for x in X]
    curveball(rows,100*nr)
    out=[]
    for i in range(N):
        curveball(rows,10*nr)
        Y=np.zeros_like(X)
        for j,r in enumerate(rows):Y[j,list(r)]=1
        if i in (0,N-1):
            assert np.array_equal(Y.sum(0),X.sum(0))
            assert np.array_equal(Y.sum(1),X.sum(1))
        out.append((Y.T@Y)[tri])
    return np.asarray(out,dtype=float)
def cos(x,y):
    den=np.linalg.norm(x)*np.linalg.norm(y)
    return float(x@y/den) if den>1e-12 else None
def cor(x,y):return cos(x-x.mean(),y-y.mean())
out=[]
for fn in ["allCA1.mat","allCTX.mat"]:
    b=urllib.request.urlopen(f"https://raw.githubusercontent.com/louzonca/AstroNet/{source}/Data/{fn}",timeout=40).read(2_000_000)
    (root/fn).write_bytes(b)
    m=loadmat(io.BytesIO(b),simplify_cells=True)
    for path,x in walk(m):
        n=int(x["nAstros"]);time=np.asarray(x["timeVec"]).ravel();T=len(time)
        assert np.allclose(np.diff(time),0.5)
        ev=x["evtBegin"]
        assert len(ev)==n and len(np.asarray(x["allActiveAstros"]).ravel())==n
        starts=[np.asarray(v,dtype=int).ravel() for v in ev]
        assert all(np.all((a>=1)&(a<=T)) for a in starts)
        for seconds in [5,2.5,10]:
            frames=int(seconds/.5);nb=T//frames;half=nb//2
            rec=dict(file=fn,path=path,name=x["name"],n_cells=n,n_frames=T,seconds=seconds,bins=[half,nb-half],events=sum(map(len,starts)))
            if min(half,nb-half)<8:
                rec.update(status="too_short");out.append(rec);continue
            Z=np.zeros((nb,n),dtype=np.int64)
            for i,a in enumerate(starts):
                k=(a-1)//frames;k=k[k<nb];Z[k,i]=1
            X,Y=Z[:half],Z[half:]
            tri=np.triu_indices(n,1)
            A=(X.T@X)[tri].astype(float);B=(Y.T@Y)[tri].astype(float)
            S=sample(X,256);R=sample(Y,512)
            mu=S.mean(0);nu=R[:256].mean(0)
            a=A-mu;b=B-nu
            # Primary held-out statistic: cosine of excess pair counts.
            # Approximate fixed-margin null: train direction frozen; held-out
            # margins fixed; last 256 MCMC draws excluded from null-mean fit.
            stat=cos(a,b)
            null=np.array([v for r in R[256:] if (v:=cos(a,r-nu)) is not None])
            base=float(np.sum(b*b))
            pred=np.clip(nu+.5*(len(Y)/len(X))*a,0,np.minimum(Y.sum(0)[tri[0]],Y.sum(0)[tri[1]]))
            gain=1-float(np.sum((B-pred)**2))/base if base>1e-12 else None
            rec.update(status="ok",raw_correlation=cor(A,B),residual_cosine=stat,
                       conditional_p_approx=float((1+np.sum(null>=stat))/(1+len(null))) if stat is not None and len(null) else None,
                       fixed_shrinkage_mse_gain=gain,train_active=int((X.sum(0)>0).sum()),test_active=int((Y.sum(0)>0).sum()),
                       train_coincidences=int(A.sum()),test_coincidences=int(B.sum()),null_mean_cosine=float(null.mean()) if len(null) else None,
                       source_sha256=hashlib.sha256(urllib.request.urlopen(f"https://raw.githubusercontent.com/louzonca/AstroNet/{source}/Data/{fn}").read()).hexdigest() if False else hashlib.sha256((root/fn).read_bytes()).hexdigest())
            out.append(rec)
            print("TEMPORAL_RECORD "+json.dumps(rec,allow_nan=False),flush=True)
summ={}
for sec in [5,2.5,10]:
    a=[r for r in out if r["seconds"]==sec and r["status"]=="ok"]
    d={"n":len(a)}
    for key in ["raw_correlation","residual_cosine","fixed_shrinkage_mse_gain"]:
        v=np.array([r[key] for r in a if r[key] is not None])
        d[key]={"n":len(v),"median":float(np.median(v)),"mean":float(v.mean()),"positive":int((v>0).sum())} if len(v) else None
    d["p_approx_le_0.05"]=sum(r["conditional_p_approx"] is not None and r["conditional_p_approx"]<=.05 for r in a)
    summ[str(sec)]=d
(root/"astro_temporal_results.json").write_text(json.dumps({"source_commit":source,"rows":out,"summary":summ},indent=2,allow_nan=False))
print("TEMPORAL_SUMMARY "+json.dumps(summ,allow_nan=False),flush=True)

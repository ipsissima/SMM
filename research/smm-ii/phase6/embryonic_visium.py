"""Phase VI exploratory cross-embryo prediction, fixed before reading GSE240715.
No measured glial graph, no causal or longitudinal inference. Counts and barcode
positions are publisher observations. Outcomes are RNA marker scores, not cell
numbers. Hold out each of four embryos in turn; no test outcome used in tuning.
"""
import gzip,hashlib,io,json,os,tarfile,urllib.request
from pathlib import Path
import h5py,numpy as np,pandas as pd
from scipy import sparse
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
ROOT=Path("output");ROOT.mkdir(exist_ok=True)
URL="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE240nnn/GSE240715/suppl/GSE240715_RAW.tar"
SAMPLES={"GSM7708218":"Control02","GSM7708219":"Control05","GSM7708220":"Control06","GSM7708221":"Control09"}
PANELS={
 "cycle":["Mki67","Top2a","Cdk1","Ccnb1","Ccna2","Ube2c","Cenpf","Birc5"],
 "neuron":["Tubb3","Dcx","Neurod1","Neurod2","Rbfox3","Stmn2","Elavl3"],
 "radial":["Sox2","Nes","Pax6","Hes1","Fabp7","Vim","Slc1a3","Hes5"],
 "intermediate":["Eomes","Ascl1","Neurog2","Neurod4","Dlx1","Dlx2"],
 "territory":["Foxg1","Emx1","Emx2","Nkx2-1","Gsx2","Lhx2","Pax3","Pax7"],
 "vascular":["Pecam1","Kdr","Cldn5","Rgs5","Pdgfrb"],
 "guidance":["Sema3a","Sema6a","Plxna2","Plxna4","Dcc","Ntn1","Reln","Robo1","Slit2"],
 "notch":["Notch1","Notch2","Dll1","Jag1","Hey1"],
 "connexin":["Gja1","Gjc1","Gjb2","Gjb6"],
 "purinergic":["P2ry1","Panx1","Itpr1","Itpr2","Plcb1","Cacna1c"]
}
def retrieve(url,cap):
    with urllib.request.urlopen(urllib.request.Request(url,headers={"User-Agent":"SMM-II public scientific data reanalysis"}),timeout=50) as r:
        b=r.read(cap+1)
    if len(b)>cap:raise ValueError("size cap")
    return b
raw=ROOT/"GSE240715_RAW.tar"
if not raw.exists():
    try:raw.write_bytes(retrieve(URL,60*1024**2))
    except Exception as e:
        print("TAR_ERROR",repr(e),flush=True)
        with tarfile.open(raw,"w") as out:
            for gsm,sid in SAMPLES.items():
                for suffix in ["filtered_feature_bc_matrix.h5","tissue_positions_list.csv.gz"]:
                    name=f"{gsm}_{sid}_{suffix}"
                    b=retrieve(f"https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM7708nnn/{gsm}/suppl/{name}",20*1024**2)
                    info=tarfile.TarInfo(name);info.size=len(b);out.addfile(info,io.BytesIO(b))
source_sha=hashlib.sha256(raw.read_bytes()).hexdigest()
samples=[]; audit=[]; genes=None
with tarfile.open(raw) as t:
    inv=[(m.name,m.size) for m in t.getmembers() if m.isfile()]
    print("TAR_MEMBERS",json.dumps(inv),flush=True)
    for gsm,sid in SAMPLES.items():
        def member(suffix):
            matches=[m for m in t.getmembers() if m.isfile() and gsm in m.name and m.name.endswith(suffix)]
            assert len(matches)==1,(gsm,suffix,matches)
            return t.extractfile(matches[0]).read()
        h=member("filtered_feature_bc_matrix.h5");pos=member("tissue_positions_list.csv.gz")
        with h5py.File(io.BytesIO(h),"r") as f:
            m=f["matrix"];g=m["features"]["name"][:].astype(str);bc=m["barcodes"][:].astype(str)
            assert len(set(g))==len(g),"duplicate gene symbols"
            X=sparse.csc_matrix((m["data"][:],m["indices"][:],m["indptr"][:]),shape=tuple(m["shape"][:])).T.tocsr()
        if genes is None:genes=g
        assert np.array_equal(genes,g)
        p=pd.read_csv(io.BytesIO(gzip.decompress(pos)),header=None,names=["barcode","in_tissue","row","col","pixel_y","pixel_x"]).set_index("barcode")
        assert set(bc)<=set(p.index)
        p=p.loc[bc];mask=p.in_tissue.astype(int).values==1
        X=X[mask];p=p.iloc[np.flatnonzero(mask)]
        rawtotal=np.asarray(X.sum(1)).ravel();keep=rawtotal>=500
        X=X[keep];p=p.iloc[np.flatnonzero(keep)]
        # Size factor excludes all prespecified panels to reduce target/predictor
        # denominator coupling; no outcome in geometry or input selection.
        gi={v:i for i,v in enumerate(genes)}
        drop=set(sum(PANELS.values(),[]));other=[i for i,k in enumerate(genes) if k not in drop]
        total=np.asarray(X[:,other].sum(1)).ravel();assert min(total)>0
        expr=(sparse.diags(1e4/total)@X).tocsr();expr.data=np.log1p(expr.data)
        coords=p[["row","col"]].to_numpy(dtype=float)
        x=np.column_stack((coords[:,1]/2,coords[:,0]*np.sqrt(3)/2))
        D=cdist(x,x); A=(abs(D-1)<1e-7).astype(float);assert np.allclose(A,A.T)
        W=A+np.eye(len(A)); W/=W.sum(1)[:,None]
        z=(x-x.mean(0))/np.maximum(x.std(0),1e-9)
        def columns(names):return expr[:,[gi[k] for k in names if k in gi]].toarray()
        available={k:[a for a in v if a in gi] for k,v in PANELS.items()}
        assert all(len(v)>=2 for v in available.values())
        scores={k:columns(v).mean(1) for k,v in available.items()}
        neural=(columns(["Foxg1"])[:,0]>0)&((columns(["Sox2","Nes","Pax6","Eomes","Dlx2","Neurog2","Ascl1","Slc1a3","Emx1","Emx2"])>0).sum(1)>=2)
        assert neural.sum()>=40,(gsm,neural.sum())
        local=np.column_stack([columns(available[k]) for k in ["radial","intermediate","territory","vascular","guidance","notch","purinergic"]])
        geo=np.column_stack([z,z*z,z[:,0]*z[:,1],np.log1p(total),A.sum(1)])
        c=columns(["Gja1","Gjc1","Gjb2","Gjb6"])
        assert "Gja1" in gi and (c[:,0]>0).sum()>10
        a=dict(gsm=gsm,sid=sid,p=p,X=X,expr=expr,local=local,geo=geo,W=W,A=A,c=c,scores=scores,neural=neural,total=total)
        samples.append(a)
        audit.append(dict(gsm=gsm,sid=sid,n=len(p),neural_n=int(neural.sum()),n_genes=len(g),h5_sha=hashlib.sha256(h).hexdigest(),pos_sha=hashlib.sha256(pos).hexdigest(),isolates=int((A.sum(1)==0).sum()),mean_total=float(total.mean()),Gja1_detect=float((c[:,0]>0).mean()),mean_scores={k:float(v.mean()) for k,v in scores.items()}))
print("AUDIT",json.dumps(audit),flush=True)
# Graph features are weighted averages of measured non-outcome markers only.
# They are expression-weighted spatial proxies, NOT inferred conductance.
def walk(a,Z,steps=(1,2,4)):
    ans=[];V=Z
    for i in range(1,max(steps)+1):
        V=a@V
        if i in steps:ans.append(V)
    return np.column_stack(ans)
def wgraph(s,g):
    v=.25+np.maximum(g,0)
    A=s["A"]*np.sqrt(v[:,None]*v[None,:])+np.diag(v)
    return A/A.sum(1)[:,None]
def feature(s,pc,control_idx):
    base=np.column_stack([s["local"],pc])
    ordinary=np.column_stack([s["geo"],base,walk(s["W"],base),s["c"],walk(s["W"],s["c"])])
    nogap=np.column_stack([s["geo"],base,walk(s["W"],base)])
    localgap=np.column_stack([nogap,s["c"]])
    # Equal feature count for all three graph variants. No outcome values.
    payload=np.column_stack([s["scores"][k] for k in ["radial","intermediate","territory","vascular","guidance","notch","purinergic"]]+[pc[:,:5]])
    def weighted(g):return walk(wgraph(s,g),payload)-walk(s["W"],payload)
    result={"base":nogap,"local":localgap,"spatial":ordinary,
            "gja1_weighted":np.column_stack([ordinary,weighted(s["c"][:,0])]),
            "radial_weighted":np.column_stack([ordinary,weighted(s["scores"]["radial"])])}
    # The three nearest expression-matched non-panel genes are negative controls;
    # no claims about their biological inertness.
    for i,ix in enumerate(control_idx):
        g=s["expr"][:,ix].toarray().ravel()
        result["matched_"+str(i)]=np.column_stack([ordinary,weighted(g)])
    return result
alphas=[1.,10.,100.,1000.]
def matrices(train,hold):
    tr=[samples[i] for i in train]
    # Unsupervised PCA fit on training embryos only, excluding all panels.
    allowed=np.array([i for i,g in enumerate(genes) if g not in drop and not g.startswith(("mt-","Rpl","Rps"))])
    m=sparse.vstack([s["expr"][:,allowed] for s in tr]).tocsr()
    mean=np.asarray(m.mean(0)).ravel()
    var=np.asarray(m.power(2).mean(0)).ravel()-mean*mean
    ok=np.flatnonzero((mean>.025)&(var>0))
    ix=allowed[ok[np.argsort(var[ok])[-300:]]]
    pca=PCA(n_components=15,random_state=20261127).fit(sparse.vstack([s["expr"][:,ix] for s in tr]).toarray())
    gja=np.concatenate([s["c"][:,0] for s in tr]);target=[gja.mean(),(gja>0).mean()]
    det=np.asarray((m>0).mean(0)).ravel();dist=(np.log1p(mean)-np.log1p(target[0]))**2+(det-target[1])**2
    # select expressed matched genes by training-only moments; deterministic
    idx=allowed[np.argsort(dist)[:3]]
    out={i:feature(samples[i],pca.transform(samples[i]["expr"][:,ix].toarray()),idx) for i in list(train)+[hold]}
    return out,genes[idx].tolist()
def eval_predict(train,hold,kind,alpha,endpoint,subset,mat):
    def mask(i):return samples[i]["neural"] if subset=="neural" else np.ones(len(samples[i]["p"]),dtype=bool)
    X=np.vstack([mat[i][kind][mask(i)] for i in train]);y=np.concatenate([samples[i]["scores"][endpoint][mask(i)] for i in train])
    # Equal embryo weight even if numbers of spots differ.
    weights=np.concatenate([np.full(mask(i).sum(),len(X)/(len(train)*mask(i).sum())) for i in train])
    scale=StandardScaler().fit(X,sample_weight=weights)
    model=Ridge(alpha=alpha).fit(scale.transform(X),y,sample_weight=weights)
    xt=mat[hold][kind][mask(hold)];yt=samples[hold]["scores"][endpoint][mask(hold)];pred=model.predict(scale.transform(xt))
    return yt,pred
def summary(y,p):
    mse=float(np.mean((y-p)**2));v=float(np.var(y));return dict(n=len(y),mse=mse,R2=float(1-mse/v) if v>0 else None,cor=float(np.corrcoef(y,p)[0,1]),bias=float(np.mean(p-y)))
# Precompute all train/test feature transforms before fitting outcomes.
cache={}
for hold in range(4):
    tr=tuple(i for i in range(4) if i!=hold)
    cache[(tr,hold)]=matrices(tr,hold)
    for val in tr:
        sub=tuple(i for i in tr if i!=val)
        cache[(sub,val)]=matrices(sub,val)
out=[];predictions=[]
for hold in range(4):
    tr=tuple(i for i in range(4) if i!=hold);ma,ctrl=cache[(tr,hold)]
    for subset in ["neural","all"]:
        for endpoint in ["cycle","neuron"]:
            row=dict(heldout=samples[hold]["gsm"],subset=subset,endpoint=endpoint,controls=ctrl,models={})
            for kind in ma[hold]:
                cv=[]
                for alpha in alphas:
                    losses=[]
                    for val in tr:
                        sub=tuple(i for i in tr if i!=val);m,_=cache[(sub,val)]
                        y,p=eval_predict(sub,val,kind,alpha,endpoint,subset,m);losses.append(float(np.mean((y-p)**2)))
                    cv.append(np.mean(losses))
                alpha=alphas[int(np.argmin(cv))]
                y,p=eval_predict(tr,hold,kind,alpha,endpoint,subset,ma)
                row["models"][kind]=summary(y,p)|{"alpha":alpha,"inner_losses":[float(v) for v in cv]}
                mask=samples[hold]["neural"] if subset=="neural" else np.ones(len(samples[hold]["p"]),bool)
                predictions.extend(dict(heldout=samples[hold]["gsm"],subset=subset,endpoint=endpoint,model=kind,barcode=str(b),observed=float(o),prediction=float(v)) for b,o,v in zip(samples[hold]["p"].index[mask],y,p))
            out.append(row);print("FOLD",json.dumps(row,allow_nan=False),flush=True)
aggregates=[]
for subset in ["neural","all"]:
    for endpoint in ["cycle","neuron"]:
        r=[x for x in out if x["subset"]==subset and x["endpoint"]==endpoint]
        m={k:float(np.mean([v["models"][k]["mse"] for v in r])) for k in r[0]["models"]}
        a={"subset":subset,"endpoint":endpoint,"equal_embryo_MSE":m,"increment_vs_spatial":{k:float(1-m[k]/m["spatial"]) for k in m if k not in ["base","local","spatial"]},"spatial_vs_base":float(1-m["spatial"]/m["base"]),"local_vs_base":float(1-m["local"]/m["base"]),"spatial_vs_local":float(1-m["spatial"]/m["local"]),"fold_gain_gja1":[float(1-v["models"]["gja1_weighted"]["mse"]/v["models"]["spatial"]["mse"]) for v in r]}
        aggregates.append(a);print("AGG",json.dumps(a),flush=True)
res=dict(scope=__doc__,source=URL,source_sha256=source_sha,available_panels=available,audit=audit,folds=out,aggregate=aggregates,alphas=alphas)
(ROOT/"visium_results.json").write_text(json.dumps(res,indent=2,allow_nan=False))
pd.DataFrame(predictions).to_csv(ROOT/"heldout_predictions.csv",index=False)
for s in samples:
    p=s["p"].copy();p["neural_subset"]=s["neural"]
    for k,v in s["scores"].items():p["score_"+k]=v
    p.to_csv(ROOT/(s["gsm"]+"_spot_scores.csv"))
print("COMPLETE",json.dumps({"source_sha256":source_sha,"panels":available,"audit":audit,"aggregate":aggregates}),flush=True)

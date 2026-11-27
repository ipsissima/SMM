"""Inventory the original AstroNet functional-network arrays and test their topology.
This is an adult functional-network analysis, not developmental conductance.
"""
import json,hashlib,io,urllib.request
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy.sparse.csgraph import connected_components
root=Path('output');root.mkdir(exist_ok=True)
sha='e68a8970c6e3db32f4f30f51bfc0b2e79aea5c78'
def norm(v):
    if hasattr(v,'_fieldnames'):return {k:norm(getattr(v,k)) for k in v._fieldnames}
    if isinstance(v,dict):return {k:norm(x) for k,x in v.items()}
    if isinstance(v,np.ndarray) and v.dtype==object:return [norm(x) for x in v.flat]
    if isinstance(v,(list,tuple)):return [norm(x) for x in v]
    return v
def encode(x):
    if isinstance(x,np.ndarray):return encode(x.tolist())
    if isinstance(x,np.generic):return encode(x.item())
    if isinstance(x,dict):return {k:encode(v) for k,v in x.items()}
    if isinstance(x,list):return [encode(v) for v in x]
    if isinstance(x,float) and not np.isfinite(x):return None
    return x
def describe(x):
    if isinstance(x,np.ndarray):
        d={'shape':x.shape,'dtype':str(x.dtype)}
        if x.dtype.kind in 'biuf' and x.size:
            a=x.astype(float);d.update(finite=int(np.isfinite(a).sum()),min=float(np.nanmin(a)),max=float(np.nanmax(a)))
        return d
    if isinstance(x,list):return {'list_length':len(x),'first':describe(x[0]) if x else None}
    if isinstance(x,dict):return {'fields':list(x)}
    return encode(x)
def walk(x,path=''):
    if isinstance(x,dict):
        if any(k in x for k in ['nAstros','conMat','meanNodeDeg','meanEvtFreq']):
            yield path,x
        else:
            for k,v in x.items():yield from walk(v,path+'/'+k)
    elif isinstance(x,list):
        for i,v in enumerate(x):yield from walk(v,path+'/'+str(i))
out={}
for fn in ['allCA1.mat','allCTX.mat']:
    url=f'https://raw.githubusercontent.com/louzonca/AstroNet/{sha}/Data/{fn}'
    b=urllib.request.urlopen(url,timeout=40).read(2_000_000)
    (root/fn).write_bytes(b)
    m=loadmat(io.BytesIO(b),simplify_cells=True)
    data=norm({k:v for k,v in m.items() if not k.startswith('__')})
    records=[]
    for path,x in walk(data):
        schema={k:describe(v) for k,v in x.items()}
        rec={'path':path,'schema':schema}
        for k in ['name','nAstros','meanEvtFreq','meanNodeDeg','varNodeDeg','numHighConnNodes','propHighConnNodes','allActiveAstros','GlobalEventsTime']:
            if k in x:rec[k]=encode(x[k])
        if 'conMat' in x:
            try:
                A=np.asarray(x['conMat'],dtype=float)
                valid=A.ndim==2 and A.shape[0]==A.shape[1] and A.shape[0]>1 and np.isfinite(A).all()
                rec['adjacency_shape']=list(A.shape)
                if valid:
                    rec.update(nonnegative=bool((A>=0).all()),symmetric=bool(np.allclose(A,A.T)),diagonal=np.diag(A).tolist())
                    if rec['nonnegative'] and rec['symmetric']:
                        A=A.copy();np.fill_diagonal(A,0);degree=A.sum(1);L=np.diag(degree)-A
                        e=np.linalg.eigvalsh(L);n=len(A);m=int(np.count_nonzero(np.triu(A>0,1)))
                        cc,labels=connected_components(A,directed=False)
                        rec['graph']={'n':n,'m':m,'components':cc,'component_sizes':np.bincount(labels).tolist(),'mean_strength':float(degree.mean()),'weighted_degree_cv':float(degree.std()/degree.mean()) if degree.mean()>0 else None,'lambda2':float(e[1]),'lambda_max':float(e[-1]),'eigenvalues':e.tolist(),'density':2*m/(n*(n-1))}
                        rec['adjacency']=A.tolist()
            except Exception as e:rec['matrix_error']=repr(e)
        records.append(rec)
        print('ASTRO_RECORD '+json.dumps(encode({k:v for k,v in rec.items() if k!='adjacency'})),flush=True)
    out[fn]={'sha256':hashlib.sha256(b).hexdigest(),'n_records':len(records),'records':records,'source_commit':sha}
    (root/(fn+'.json')).write_text(json.dumps(encode(data),allow_nan=False))
(root/'astro_graph_audit.json').write_text(json.dumps(encode(out),allow_nan=False,indent=2))
print('ASTRO_SUMMARY '+json.dumps({k:{'n_records':v['n_records'],'graphs':sum('graph' in x for x in v['records']),'fields':sorted(set(y for x in v['records'] for y in x['schema']))} for k,v in out.items()}),flush=True)

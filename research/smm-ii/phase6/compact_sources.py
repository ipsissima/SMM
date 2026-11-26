"""Phase VI compact-source follow-up. Keep actual arrays and field provenance."""
import urllib.request,json,hashlib,io,gzip,zipfile,re
from pathlib import Path
import numpy as np
from scipy.io import loadmat,whosmat
root=Path('output');root.mkdir(exist_ok=True)
def get(u,cap=80*1024**2):
    with urllib.request.urlopen(urllib.request.Request(u,headers={'User-Agent':'Mozilla/5.0','Accept':'*/*'}),timeout=35) as r:
        b=r.read(cap+1)
        if len(b)>cap:raise ValueError("size cap")
        return b
def conv(v):
    if hasattr(v,'_fieldnames'):return {k:conv(getattr(v,k)) for k in v._fieldnames}
    if isinstance(v,dict):return {k:conv(x) for k,x in v.items()}
    if isinstance(v,np.ndarray):return conv(v.tolist())
    if isinstance(v,(list,tuple)):return [conv(x) for x in v]
    if isinstance(v,np.generic):return conv(v.item())
    if isinstance(v,float) and not np.isfinite(v):return None
    if isinstance(v,complex):return [v.real,v.imag]
    return v
for fn in ['allCA1.mat','allCTX.mat']:
    try:
        u='https://raw.githubusercontent.com/louzonca/AstroNet/e68a8970c6e3db32f4f30f51bfc0b2e79aea5c78/Data/'+fn
        b=get(u);(root/fn).write_bytes(b)
        d=conv({k:v for k,v in loadmat(io.BytesIO(b),simplify_cells=True).items() if not k.startswith('__')})
        (root/(fn+'.json')).write_text(json.dumps(d,allow_nan=False))
        def walk(v,path=''):
            if isinstance(v,dict):
                print('ASTRO_FIELDS '+json.dumps([fn,path,list(v)]),flush=True)
                for k,x in v.items():
                    if k in ['name','nAstros','meanNodeDeg','varNodeDeg','numHighConnNodes','propHighConnNodes','meanEvtFreq','meanDeg','conMat','allActiveAstros']:
                        print('ASTRO_VALUE '+json.dumps([fn,path,k,x])[:60000],flush=True)
                    elif isinstance(x,dict) or isinstance(x,list) and x and isinstance(x[0],dict):walk(x,path+'/'+k)
            elif isinstance(v,list):
                for i,x in enumerate(v):
                    if isinstance(x,(dict,list)):walk(x,path+'/'+str(i))
        print('ASTRO_INFO '+json.dumps([fn,hashlib.sha256(b).hexdigest(),whosmat(io.BytesIO(b))]),flush=True);walk(d)
    except Exception as e:print('ASTRO_ERROR',fn,repr(e),flush=True)
base='https://p4-brain-lineage.cells.ucsc.edu/'
try:
    d=json.loads(get(base+'dataset.json',2*1024**2))
    candidates=[a for a in d.get('datasets',[]) if 'p4-brain-lineage' in a.get('name','') or ('lineage' in a.get('shortLabel','').lower() and ('brain' in a.get('shortLabel','').lower() or 'forebrain' in a.get('shortLabel','').lower()))]
    print('LINEAGE_CANDIDATES '+json.dumps(candidates),flush=True)
    for a in candidates:
        n=a['name'];u=base+n.strip('/')+'/dataset.json'
        cfg=json.loads(get(u,3*1024**2));(root/('cfg_'+n.replace('/','_')+'.json')).write_text(json.dumps(cfg))
        print('LINEAGE_CFG '+json.dumps({k:v for k,v in cfg.items() if k not in ['metaFields','markers','geneOffsets']})[:12000],flush=True)
        sub=cfg.get('datasets',[]) or [{'name':n}]
        for z in sub:
            sn=z['name'];path=sn if sn.startswith(n) else n+'/'+sn
            prefix=base+path.strip('/')+'/'
            try:
                conf=json.loads(get(prefix+'dataset.json',3*1024**2))
                print('LINEAGE_SUB '+json.dumps({"path":path,"count":conf.get('sampleCount'),"metaFields":conf.get('metaFields'),"coords":conf.get('coords')})[:18000],flush=True)
                (root/('cfg_'+path.replace('/','_')+'.json')).write_text(json.dumps(conf))
                for f in ['meta.tsv.gz','meta.tsv']:
                    try:
                        b=get(prefix+f,35*1024**2)
                        t=gzip.decompress(b) if f.endswith('.gz') else b
                        header=t.splitlines()[0].decode()
                        print('LINEAGE_META '+json.dumps([path,f,hashlib.sha256(b).hexdigest(),len(t.splitlines())-1,header,t.splitlines()[1][:1000].decode()]),flush=True)
                        (root/(path.replace('/','_')+'_'+f)).write_bytes(b);break
                    except Exception as e:print('LINEAGE_META_ERROR',path,f,repr(e),flush=True)
            except Exception as e:print('LINEAGE_SUB_ERROR',path,repr(e),flush=True)
except Exception as e:print('LINEAGE_ERROR',repr(e),flush=True)
for id,fn,cap in [(2967585,'events.zip',75*1024**2),(2967588,'README.md',1*1024**2),(2967584,'ramping.zip',2*1024**2)]:
    for u in [f'https://datadryad.org/stash/downloads/file_stream/{id}',f'https://datadryad.org/api/v2/files/{id}/download']:
        try:
            b=get(u,cap)
            if fn.endswith('.zip'):
                with zipfile.ZipFile(io.BytesIO(b)) as z:
                    mem=[[i.filename,i.file_size] for i in z.infolist()]
                    heads={}
                    for i in z.infolist():
                        if i.filename.lower().endswith(('.csv','.tsv','.txt')):
                            with z.open(i) as r:heads[i.filename]=r.readline(3000).decode('utf-8','replace')
                    print('CAHILL_CONTENT '+json.dumps([fn,mem,heads])[:30000],flush=True)
            else:print('CAHILL_README '+b[:200].decode('utf-8','replace'),flush=True)
            (root/fn).write_bytes(b);print('CAHILL_OK '+json.dumps([fn,u,len(b),hashlib.sha256(b).hexdigest()]),flush=True);break
        except Exception as e:print('CAHILL_FAIL',fn,u,repr(e),flush=True)

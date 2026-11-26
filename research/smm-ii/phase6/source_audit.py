"""Bounded public-data audit; do not infer cell identity or conductance from unknown fields."""
import json,hashlib,urllib.request,urllib.parse,zipfile,io,xml.etree.ElementTree as ET,re,concurrent.futures
from pathlib import Path
import numpy as np
from scipy.io import loadmat,whosmat
root=Path("output");root.mkdir(exist_ok=True)
def get(url,cap=80*1024**2):
    with urllib.request.urlopen(urllib.request.Request(url,headers={"User-Agent":"SMM-II public research data audit","Accept":"*/*"}),timeout=35) as r:
        b=r.read(cap+1)
        if len(b)>cap:raise ValueError("size cap")
        return b,r.geturl()
def save(name,b):
    (root/name).write_bytes(b)
    return {"name":name,"size":len(b),"sha256":hashlib.sha256(b).hexdigest()}
def astro():
    out=[];sha="e68a8970c6e3db32f4f30f51bfc0b2e79aea5c78"
    for fn in ["allCA1.mat","allCTX.mat"]:
        u=f"https://raw.githubusercontent.com/louzonca/AstroNet/{sha}/Data/{fn}"
        b,final=get(u);a=save(fn,b);d=loadmat(io.BytesIO(b),simplify_cells=True)
        def structure(v,depth=0):
            if isinstance(v,dict):return {k:structure(x,depth+1) for k,x in v.items()} if depth<5 else list(v)
            if isinstance(v,(tuple,list)):return {"type":type(v).__name__,"n":len(v),"first":[structure(x,depth+1) for x in v[:2]]} if depth<4 else {"n":len(v)}
            if isinstance(v,np.ndarray):
                rec={"shape":v.shape,"dtype":str(v.dtype)}
                if v.dtype.kind in "biufc" and v.size:rec.update(min=float(np.nanmin(v.real)),max=float(np.nanmax(v.real)),finite=int(np.isfinite(v).sum()))
                elif v.size and depth<4:rec["first"]=structure(v.flat[0],depth+1)
                return rec
            return str(v)[:200]
        a.update(url=final,variables=whosmat(io.BytesIO(b)),schema={k:structure(v) for k,v in d.items() if not k.startswith("__")})
        out.append(a)
        # Export processed values losslessly for compact numeric fields only.
        def conv(v):
            if isinstance(v,dict):return {k:conv(w) for k,w in v.items()}
            if isinstance(v,np.ndarray):return conv(v.tolist())
            if isinstance(v,list):return [conv(w) for w in v]
            if isinstance(v,np.generic):return conv(v.item())
            if isinstance(v,float) and not np.isfinite(v):return None
            if isinstance(v,complex):return {"real":v.real,"imag":v.imag}
            return v
        o=conv({k:v for k,v in d.items() if not k.startswith("__")})
        (root/(fn+".json")).write_text(json.dumps(o,allow_nan=False))
        print("ASTRO_SCHEMA "+json.dumps(a),flush=True)
    return out
def lineage():
    base="https://p4-brain-lineage.cells.ucsc.edu/"
    out=[]
    paths=["dataset.json","datasets.json","index.html","meta.tsv","meta.tsv.gz","dataset.json?","merfish/dataset.json","scRNA/dataset.json","MERFISH/dataset.json"]
    for path in paths:
        try:
            b,f=get(base+path,12*1024**2)
            a={"path":path,"url":f,"size":len(b)}
            if path.endswith(".json") or path.endswith(".json?"):
                d=json.loads(b);a["json_preview"]=json.dumps(d)[:6000]
                save("lineage_"+path.replace("/","_").replace("?",""),b)
            elif path.endswith(".tsv") or path.endswith(".gz"):
                txt=gzip.decompress(b) if path.endswith(".gz") else b
                a["header"]=txt.splitlines()[0][:3000].decode("utf-8","replace");a["n_lines"]=len(txt.splitlines())
                save("lineage_"+path,b)
            else:a["html_preview"]=b[:3000].decode("utf-8","replace")
        except Exception as e:a={"path":path,"error":repr(e)}
        out.append(a);print("LINEAGE "+json.dumps(a),flush=True)
    return out
def dryad():
    u="https://datadryad.org/api/v2/datasets/"+urllib.parse.quote("doi:10.5061/dryad.83bk3jb0j",safe="")
    d=json.loads(get(u,4*1024**2)[0]);save("cahill_dryad_dataset.json",json.dumps(d).encode())
    links=d.get("_links",{});v=links.get("stash:version",{}).get("href")
    if not v:
        q=links.get("stash:versions",{}).get("href")
        vv=json.loads(get(urllib.parse.urljoin(u,q),4*1024**2)[0]);v=vv["_embedded"]["stash:versions"][-1]["_links"]["self"]["href"]
    version=json.loads(get(urllib.parse.urljoin(u,v),4*1024**2)[0]);save("cahill_dryad_version.json",json.dumps(version).encode())
    files=version["_links"]["stash:files"]["href"];ff=json.loads(get(urllib.parse.urljoin(u,files),4*1024**2)[0])
    save("cahill_dryad_files.json",json.dumps(ff).encode())
    out=[]
    for f in ff["_embedded"]["stash:files"]:
        path=f.get("path");print("DRYAD_FILE "+json.dumps({"path":path,"size":f.get("size"),"links":f.get("_links")}),flush=True)
        if path not in ["events.zip","README.md","ramping.zip"]:continue
        try:
            h=f["_links"]["stash:download"]["href"];b,url=get(urllib.parse.urljoin(u,h),80*1024**2)
            a=save("cahill_"+path,b);a["url"]=url
            if path.endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(b)) as z:
                    a["members"]=[[i.filename,i.file_size] for i in z.infolist()]
                    # Data schemas for all modest delimited files; no assumptions about provenance.
                    a["headers"]={}
                    for n in z.namelist():
                        if n.lower().endswith((".csv",".tsv",".txt")):
                            with z.open(n) as r:a["headers"][n]=r.readline(10000).decode("utf-8","replace")
            print("CAHILL "+json.dumps(a)[:25000],flush=True)
        except Exception as e:a={"path":path,"error":repr(e)}
        out.append(a)
    return out
def malmersjo():
    u="https://www.ebi.ac.uk/europepmc/webservices/rest/PMC3631687/fullTextXML"
    b,url=get(u,5*1024**2);r=ET.fromstring(b);a=save("malmersjo.xml",b)
    sections=[]
    for p in r.findall(".//body//p"):
        t="".join(p.itertext())
        if any(x in t.lower() for x in ["brdu-positive","octanol","proliferation","small-world","scale-free"]):
            if len(t)<6000:sections.append(t)
    a["relevant_paragraphs"]=sections
    a["supplementary"]=[{"href":x.attrib,"text":"".join(x.itertext())[:200]} for x in r.findall(".//supplementary-material")]
    print("MALMERSJO "+json.dumps(a)[:30000],flush=True);return a
def em():
    out=[]
    for code in ["9891","3785","6638","4157","2013"]:
        try:
            b,u=get("https://wklink.org/"+code,2*1024**2);a={"code":code,"url":u,"size":len(b),"title":re.findall(r"<title[^>]*>(.*?)</title>",b.decode("utf-8","replace"),re.S|re.I)[:1]}
        except Exception as e:a={"code":code,"error":repr(e)}
        print("GOUR "+json.dumps(a),flush=True);out.append(a)
    return out
out={}
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
    futures={ex.submit(fn):k for k,fn in [("astro",astro),("lineage",lineage),("dryad",dryad),("malmersjo",malmersjo),("gour",em)]}
    for f in concurrent.futures.as_completed(futures):
        k=futures[f]
        try:out[k]=f.result()
        except Exception as e:out[k]={"error":repr(e)};print("AUDIT_ERROR",k,repr(e),flush=True)
(root/"audit.json").write_text(json.dumps(out,indent=2))
print("AUDIT_COMPLETE "+json.dumps({k:{"error":v["error"]} if isinstance(v,dict) and "error" in v else "completed" for k,v in out.items()}),flush=True)

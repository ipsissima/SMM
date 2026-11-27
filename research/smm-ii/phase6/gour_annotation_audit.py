"""Inspect only published Gour skeleton annotations; do not download EM volumes."""
from pathlib import Path
import json,hashlib,re
from webknossos import Annotation
root=Path("output");root.mkdir(exist_ok=True)
ids={"P5":"600591190100008900bd2d37","P7":"60057f030100007d00bd2d08","P9":"60057f6d0100008900bd2d0b","P14":"6005801d0100007f00bd2d11","P28":"600580920100007d00bd2d17"}
rows=[]
for age,aid in ids.items():
    try:
        ann=Annotation.download("https://webknossos.org/annotations/"+aid,skip_volume_data=True,retry_count=1)
        p=root/(age+".nml")
        ann.skeleton.save(p)
        names=[str(t.name) for t in ann.skeleton.flattened_trees()]
        group_names=[str(g.name) for g in ann.skeleton.flattened_groups()]
        r=dict(age=age,id=aid,status="downloaded",dataset=ann.skeleton.dataset_name,n_trees=len(names),n_nodes=sum(len(t.nodes) for t in ann.skeleton.flattened_trees()),groups=group_names,glial_tree_names=[n for n in names if re.search("astro|glia|ependym",n,re.I)],glial_group_names=[g for g in group_names if re.search("astro|glia|ependym",g,re.I)],sample_tree_names=names[:30],sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    except Exception as e:r=dict(age=age,id=aid,status="failed",error=repr(e))
    rows.append(r);print("EM_RECORD "+json.dumps(r),flush=True)
(root/"em_annotation_audit.json").write_text(json.dumps(rows,indent=2))

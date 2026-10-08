#!/usr/bin/env python3
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
SRC=ROOT/'upstream_sources'
OUT=ROOT/'network_frozen'
OUT.mkdir(exist_ok=True)

labels=pd.read_csv(SRC/'strucLabels_ctx.csv',header=None).iloc[0].astype(str).tolist()
Wraw=pd.read_csv(SRC/'strucMatrix_ctx.csv',header=None).to_numpy(float)
if len(labels)!=68 or Wraw.shape!=(68,68):
    raise RuntimeError((len(labels),Wraw.shape))

lambda_max=173.53663723019415
W=np.round(Wraw/lambda_max,6)
Wdf=pd.DataFrame(W,index=labels,columns=labels)
Wpath=OUT/'W_ENIGMA_DK68_spectral_normalized.csv'
Wdf.to_csv(Wpath)

cent={}
for line in (SRC/'connectivity_68_QL20120814_centres.txt').read_text().splitlines():
    line=line.strip()
    if not line:
        continue
    p=line.split()
    if len(p)!=4:
        raise RuntimeError(f'Bad center line: {line}')
    raw,x,y,z=p
    if raw.endswith('_L'):
        lab='L_'+raw[:-2]
    elif raw.endswith('_R'):
        lab='R_'+raw[:-2]
    else:
        raise RuntimeError(raw)
    cent[lab]=(float(x),float(y),float(z))
if set(cent)!=set(labels):
    raise RuntimeError('center labels do not match ENIGMA labels')
C=np.array([cent[x] for x in labels],float)
Cdf=pd.DataFrame(C,index=labels,columns=['x_mm','y_mm','z_mm'])
Cpath=OUT/'DK68_centers_proxy_mm.csv'
Cdf.to_csv(Cpath)

rows=[]
for i in range(68):
    for j in range(i+1,68):
        if W[i,j]!=0.0:
            d=float(np.linalg.norm(C[i]-C[j]))
            rows.append({
                'i':i,'j':j,
                'W_spectral_normalized':float(W[i,j]),
                'euclidean_center_distance_mm':round(d,3),
                'source_label':labels[i],
                'target_label':labels[j],
            })
Epath=OUT/'DK68_edges_with_delay_proxy.csv'
pd.DataFrame(rows,columns=['i','j','W_spectral_normalized','euclidean_center_distance_mm','source_label','target_label']).to_csv(Epath,index=False)

expected={
 'W_ENIGMA_DK68_spectral_normalized.csv':'9f83f8ea5ba38aa68094739395db225e2e53c1b8541fbe8d2a9a6295dc2bd210',
 'DK68_centers_proxy_mm.csv':'5857bc9473c84574e2105a464b2475676ba2fe1c349500088d2bb6308ac0384a',
 'DK68_edges_with_delay_proxy.csv':'16604f620233b3f19ff6f1917f7f62589e8b3b4ea343bf8790b38e6871b72ff1',
}
for name,want in expected.items():
    p=OUT/name
    got=hashlib.sha256(p.read_bytes()).hexdigest()
    print(name,got)
    if got!=want:
        raise RuntimeError(f'hash mismatch for {name}: {got} != {want}')
print('NETWORK_HASH_GATE=PASS')

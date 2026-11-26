"""Unmodified EDNM nonlinear amplitude test; no SMM fitting."""
from pathlib import Path
import json, sys, numpy as np
sys.path.insert(0,'validation')
from ednm_stimulus_challenge import one
out=Path('output/dose_response')
base,_=one(out/'baseline',0.)
t=base['t'];kb=base['c'][:,0,1,2,:]
rows={}
for label,amp in [('low',-3e-5),('original',-3e-4),('high',-9e-4)]:
    d,drift=one(out/label,amp)
    assert np.allclose(t,d['t'])
    dke=d['c'][:,0,1,2,:]-kb
    v=d['phi']-base['phi']
    rows[label]={'j_stim':amp,'peak_local_dKe':float(np.max(dke[:,0])),
                 'peak_neighbor_dKe':float(np.max(dke[:,1])),
                 'max_abs_voltage_diff_mV':float(np.max(np.abs(v))),
                 'relative_inventory_drift':drift}
    np.savetxt(out/f'{label}_dke.csv',np.column_stack([t,dke]),delimiter=',',header='time_seconds,site0_delta_Ke,site1_delta_Ke',comments='')
# if linear, original/low amplitude ~10, high/low amplitude ~30
for label,factor in [('original',10.),('high',30.)]:
    rows[label]['peak_local_nonlin_ratio_to_low']=rows[label]['peak_local_dKe']/(factor*rows['low']['peak_local_dKe'])
    rows[label]['peak_neighbor_nonlin_ratio_to_low']=rows[label]['peak_neighbor_dKe']/(factor*rows['low']['peak_neighbor_dKe'])
obj={'scope':'Full original source three-dose challenge; original K ion transfer and all other model parameters fixed',
     'reference_commit':'826c211c70d9027a4acb02202ca91a60de79a15d',
     'dose_results':rows,'caveat':'Physiological scaling and long-time model reduction not established'}
(out/'summary.json').write_text(json.dumps(obj,indent=2)+'\n');print(json.dumps(obj,indent=2))

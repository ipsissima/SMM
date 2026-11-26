"""Bounded charge-neutral K stimulus on unmodified EDNM published solver.

This is a paired two-cell forward-model check only: it is NOT SMM
validation, mechanistic parameter identification, or the original paper's
published multiunit protocol. Parameter j_stim is in source native units.
"""
from pathlib import Path
import contextlib, json, numpy as np
from ednm import model,solver

def inventory(d):
    a=d['alpha']; c=d['c']
    alpha=np.concatenate([a,1-a.sum(axis=2,keepdims=True)],axis=2)
    return np.einsum('tldu,tlidu->ti',alpha,c)

def one(out,stim):
    out.mkdir(parents=True,exist_ok=True)
    m=model.Model(2,5e-4,0,0)
    proto={'j_stim':stim,'stim_start':.0005,'stim_end':.0045,
           'N_stim':1,'spike_train':np.zeros((2,2))}
    with (out/'solver.log').open('w') as f,contextlib.redirect_stdout(f):
        solver.solve_system(m,str(out)+'/',.006,proto)
    d=np.load(out/'data.npz')
    assert np.isfinite(d['c']).all()
    inv=inventory(d)
    return d,{ion:float((inv[-1,i]-inv[0,i])/inv[0,i])
              for i,ion in enumerate(['Na','K','Cl','Ca'])}

def main():
    root=Path('output/pulse_test')
    baseline,base_drift=one(root/'baseline',0.)
    # Negative source j transfers K from neuronal to ECS.
    stimulated,stim_drift=one(root/'stimulated',-3e-4)
    t=baseline['t']; assert np.allclose(t,stimulated['t'])
    # soma extracellular K, layer 0, ion 1, ECS 2, units 0 and 1
    k_base=baseline['c'][:,0,1,2,:]
    k_stim=stimulated['c'][:,0,1,2,:]
    delta=k_stim-k_base
    np.savetxt(root/'potassium_delta.csv',
               np.column_stack([t,delta]),
               delimiter=',',header='time_seconds,site_0_dKe_mM,site_1_dKe_mM',comments='')
    result={'scope':'Original full EDNM solver charge-neutral K perturbation',
            'ref':'826c211c70d9027a4acb02202ca91a60de79a15d',
            'duration_s':.006,'pulse_start_s':.0005,'pulse_end_s':.0045,
            'j_stim_original_source_units':-3e-4,
            'delta_Ke_mM_at_last_sample_site0':float(delta[-1,0]),
            'delta_Ke_mM_at_last_sample_site1':float(delta[-1,1]),
            'max_abs_delta_ke_mM':list(map(float,np.max(np.abs(delta),axis=0))),
            'max_abs_relative_inventory_drift_baseline':base_drift,
            'max_abs_relative_inventory_drift_stimulated':stim_drift,
            'samples':len(t),'caveat':'Sub-second response does not test slow glial modes, EEG, or physiological identifiability'}
    (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__': main()

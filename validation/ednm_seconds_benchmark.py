"""Original EDNM full-physics, seconds-scale potassium transport challenge.

The ORIGINAL EDNM physical equations remain unchanged for 'native'.
Only the numerical step size in source solver is varied by CI, after
explicitly checking this source edit; variants selectively change the
interunit glial Nernst-Planck mobility coefficient. All results are
in-silico, NOT biological validation or unique cellular attribution.

Protocol is fixed in advance of inspecting the output:
2 units separated by 5 um, closed boundaries, synapses off,
charge-neutral neuronal->extracellular K transfer on unit 0,
j_stim=-1e-4 in original source units, t in [0.05,0.55] s.
Full comparison time horizon: 2.0 s.
"""
import contextlib
import json
import os
from pathlib import Path
import re
import numpy as np
from ednm import model,solver

DT = float(os.environ.get("EDNM_LONG_DT", "0.002"))
DURATION = float(os.environ.get("EDNM_LONG_DURATION","2.0"))
STIMULUS = -1e-4
START, END = .05, .55
assert DURATION > END
assert 0 < DT <= .005
OUT = Path(os.environ.get("EDNM_LONG_OUT","longout"))
OUT.mkdir(parents=True,exist_ok=True)
VARIANT = os.environ.get("EDNM_LONG_VARIANT","native")

def total_ion_mass(d):
    alpha = d["alpha"]
    fractions = np.concatenate([alpha,1.-alpha.sum(axis=2,keepdims=True)],axis=2)
    return np.einsum("tldu,tlidu->ti",fractions,d["c"])

def solver_log_checks(p):
    txt=p.read_text()
    rows=[(int(i),float(e)) for i,e in re.findall(
        r"N_iter:\s*(\d+),\s*rsd:\s*([0-9.eE+-]+)",txt)]
    if not rows:
        raise RuntimeError("Solver Newton iteration log is empty")
    nonconverged=[(i,e) for i,e in rows if i>=10 and e>1e-12]
    return {"num_residual_records":len(rows),"max_recorded_residual":max(e for _,e in rows),
            "nonconverged_timestep_count":len(nonconverged),
            "last_nonconverged":nonconverged[-3:],
            "max_newton_iterations":max(i for i,e in rows)}

def run_case(label,stim):
    target=OUT/label
    target.mkdir(parents=True,exist_ok=True)
    m=model.Model(2,5e-4,0,0)
    protocol={"j_stim":stim,"stim_start":START,"stim_end":END,
              "N_stim":1,"spike_train":np.zeros((2,2),dtype=float)}
    log=target/"solver.log"
    with log.open("w") as f,contextlib.redirect_stdout(f):
        solver.solve_system(m,str(target)+"/",DURATION,protocol)
    d=np.load(target/"data.npz")
    for key in ("t","alpha","c","phi","ss"):
        assert np.isfinite(d[key]).all(), key
    if np.any(d["c"] < -1e-8) or np.any(d["c"][:,:,0:3,:,:] <= 0):
        raise RuntimeError("Unphysical ion concentrations: substantially negative ions or nonpositive Na/K/Cl")
    ions=total_ion_mass(d)
    diff=ions-ions[0]
    worst=np.max(np.abs(diff)/np.maximum(np.abs(ions[0]),1e-12),axis=0)
    checks=solver_log_checks(log)
    checks.update({"max_relative_ion_inventory_drifts":dict(zip(
        ("Na","K","Cl","Ca"),map(float,worst))),
        "max_ion_inventory_error":float(np.max(np.abs(diff)))})
    if checks["nonconverged_timestep_count"]>0:
        raise RuntimeError("Unconverged nonlinear timesteps: "+str(checks["last_nonconverged"]))
    if max(worst)>1e-7:
        raise RuntimeError("Ion inventory conservation failure: "+str(worst))
    return d,checks

def main():
    base,bcheck=run_case("baseline",0.)
    case,ccheck=run_case("stimulated",STIMULUS)
    t=base["t"]
    if not np.allclose(t,case["t"],rtol=0,atol=1e-12):
        raise RuntimeError("Baseline and stimulated time arrays differ")
    kE0=case["c"][:,0,1,2,:]-base["c"][:,0,1,2,:]
    kEd=case["c"][:,1,1,2,:]-base["c"][:,1,1,2,:]
    kG=case["c"][:,0,1,1,:]-base["c"][:,0,1,1,:]
    Vm0=case["phi"][:,0,0,:]-case["phi"][:,0,2,:]
    Vmb=base["phi"][:,0,0,:]-base["phi"][:,0,2,:]
    dVm=Vm0-Vmb
    dVm_d=(case["phi"][:,1,0,:]-case["phi"][:,1,2,:])-(base["phi"][:,1,0,:]-base["phi"][:,1,2,:])
    idx=np.where(t>=0)[0]
    tt=t[idx]
    signal=np.column_stack([tt,kE0[idx],kEd[idx],kG[idx],dVm[idx],dVm_d[idx]])
    colnames=["time_s"]+[f"{v}_site{i}" for v in
        ("dKe_soma_mM","dKe_dend_mM","dKg_soma_mM","dVm_soma_mV","dVm_dend_mV") for i in (0,1)]
    np.savetxt(OUT/"signals.csv",signal,delimiter=",",header=",".join(colnames),comments="")
    # Save only the paired-difference signal at prespecified anchors.
    anchors=[.1,.25,.5,1.0,1.5,2.0]
    obs=[]
    for a in anchors:
        if a<=float(tt[-1])+1e-9:
            j=int(np.argmin(np.abs(tt-a)))
            obs.append({"requested_t_s":a,"actual_t_s":float(tt[j]),
                        "dKe_mM":list(map(float,kE0[idx[j]])),
                        "dVm_soma_mV":list(map(float,dVm[idx[j]])),
                        "dKg_mM":list(map(float,kG[idx[j]]))})
    res={"model":"Original Saetra-Mori full EDNM (modified integrator step only; glial mobility intervention for variant)",
         "source_commit":"826c211c70d9027a4acb02202ca91a60de79a15d",
         "variant":VARIANT, "dt_s":DT, "end_s":DURATION,
         "stimulus_protocol":{"ion":"K","type":"charge-neutral neuron-to-ECS","j_stim_original_units":STIMULUS,
                              "t_start_s":START,"t_end_s":END,
                              "N_units":2,"grid_spacing_cm":5e-4,"synapses":0,"bc":0},
         "samples":len(t),
         "baseline":bcheck,"stimulated":ccheck,
         "peak_positive_dKe_soma_mM":list(map(float,np.max(kE0[idx],axis=0))),
         "peak_absolute_dVm_soma_mV":list(map(float,np.max(np.abs(dVm[idx]),axis=0))),
         "final_dKe_soma_mM":list(map(float,kE0[idx[-1]])),
         "anchor_observations":obs,
         "warning":"The model outputs membrane potential, not a validated neuronal firing rate. Do not interpret these as biological measured responses."}
    (OUT/"summary.json").write_text(json.dumps(res,indent=2)+"\n")
    print(json.dumps(res,indent=2))
if __name__=="__main__":
    main()

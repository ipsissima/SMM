"""Seven-site original full EDNM source-model graph validation, synthetic edges.

Only the Python Newton residual/Jacobian assembly is augmented with exact
interunit Nernst-Planck graph differences. Original compiled ionic kinetics,
membrane conductances, synaptic parameters, water/charge and ion laws are used.
The glial graph shortcut is synthetic and NOT anatomical or physiological proof.
"""
import contextlib,json,os
from pathlib import Path
import numpy as np
from ednm import model,solver
from ednm_graph_adapter import closed_path

VARIANT=os.environ.get("SMM_GRAPH_VARIANT","path")
ROOT=Path(os.environ.get("SMM_GRAPH_OUTPUT","output/ednm7"))
ROOT.mkdir(parents=True,exist_ok=True)
N=7
E=closed_path(N)
G=E.copy()
if VARIANT=="shortcut":
    for u,v,w in [(0,4,.8),(1,5,.5),(2,6,.6)]:
        G[u,v]=G[v,u]=w
elif VARIANT=="glial_off":
    G[:]=0.
elif VARIANT!="path":
    raise ValueError("Unknown variant "+VARIANT)
solver.graph_glial=G
solver.graph_ecs=E

def run(label,amplitude):
    out=ROOT/label
    out.mkdir(parents=True,exist_ok=True)
    m=model.Model(N,5e-4,0,0)
    duration=.8
    protocol={"j_stim":amplitude,"stim_start":.08,
              "stim_end":.4,"N_stim":1,
              "spike_train":np.zeros((2,N))}
    with (out/"solver.log").open("w") as f,contextlib.redirect_stdout(f):
        solver.solve_system(m,str(out)+"/",duration,protocol)
    d=np.load(out/"data.npz")
    for key in ("t","alpha","c","phi","ss"):
        assert np.all(np.isfinite(d[key])),key
    a=d["alpha"]
    fractions=np.concatenate([a,1-a.sum(axis=2,keepdims=True)],axis=2)
    mass=np.einsum("tldu,tlidu->ti",fractions,d["c"])
    drift=np.max(abs(mass-mass[0])/np.maximum(abs(mass[0]),1e-12),axis=0)
    assert np.max(drift)<1e-7,drift
    assert np.min(d["c"])>-1e-8
    return d,list(map(float,drift))

a,drift_a=run("baseline",0.)
b,drift_b=run("stimulated",-1e-4)
assert np.allclose(a["t"],b["t"])
deltaK=b["c"][:,0,1,2,:]-a["c"][:,0,1,2,:]
deltaVm=(b["phi"][:,0,0,:]-b["phi"][:,0,2,:])-(a["phi"][:,0,0,:]-a["phi"][:,0,2,:])
np.savetxt(ROOT/"response.csv",np.column_stack([a["t"],deltaK,deltaVm]),
           delimiter=",",header="time_s,"+",".join("dKe_"+str(j) for j in range(N))+","+
           ",".join("dVm_"+str(j) for j in range(N)),comments="")
data={
"status":"FULL PUBLISHED EDNM GRAPH AUGMENTATION SIMULATION; not anatomical validation",
"pinned_source_commit":"826c211c70d9027a4acb02202ca91a60de79a15d",
"variant":VARIANT,"n":N,"T_s":.8,"stimulus_s":[.08,.4],
"graph_glial":G.tolist(),"graph_ecs":E.tolist(),
"max_abs_neighbor_K_mM":float(np.max(abs(deltaK[:,1:]))),
"max_abs_neighbor_voltage_mV":float(np.max(abs(deltaVm[:,1:]))),
"max_abs_siteK_mM":list(map(float,np.max(abs(deltaK),axis=0))),
"max_abs_siteVm_mV":list(map(float,np.max(abs(deltaVm),axis=0))),
"Na_K_Cl_Ca_drift_baseline":drift_a,"Na_K_Cl_Ca_drift_stimulated":drift_b,
"caveats":["Synthetic graph geometry","No anatomical edge constraints",
"Only 0.8-second test; graph-transport effect may remain tiny",
"One source site and one stimulus type, no held-out model competition",
"Numerical source solver still needs 2-timestep convergence check"]}
(ROOT/"summary.json").write_text(json.dumps(data,indent=2)+"\n")
print(json.dumps(data,indent=2))

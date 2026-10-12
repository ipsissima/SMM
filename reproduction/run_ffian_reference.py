#!/usr/bin/env python3
"""Independent SMALL source-level run of the pinned ffian FEniCS R1/R2 parents.

This runs the ORIGINAL upstream Model/Solver classes and records K_e directly.
It does not claim to regenerate the historical SMM calibration or R1/R2 errors.
"""
import csv, json, math, os, subprocess, sys, time
from pathlib import Path
import dolfin as df
from ffian import zero_flow_model, flow_model

OUT=Path("repro_outputs/ffian")
OUT.mkdir(parents=True,exist_ok=True)
L=3.0e-4
N=int(os.environ.get("SMM_FENICS_N","21"))
DT=float(os.environ.get("SMM_FENICS_DT","0.02"))
TEND=float(os.environ.get("SMM_FENICS_TEND","0.8"))
J=float(os.environ.get("SMM_FENICS_J","2.5e-8"))
STIM_START=float(os.environ.get("SMM_FENICS_START","0.1"))
STIM_END=float(os.environ.get("SMM_FENICS_END","0.4"))
xs=[0.25*L,0.5*L,0.75*L]

def run(version):
    t0=time.time()
    mesh=df.IntervalMesh(N,0,L)
    t=df.Constant(0.)
    if version=="M0":
        model=zero_flow_model.Model(mesh,L,t,J,STIM_START,STIM_END)
        solver=zero_flow_model.Solver(model,DT,TEND)
        ke_index=3
    else:
        model=flow_model.Model(version,mesh,L,t,J,STIM_START,STIM_END,"constant")
        solver=flow_model.Solver(model,DT,TEND)
        ke_index=4
    rows=[]
    for step in range(int(round(TEND/DT))):
        t.assign(float(t)+DT)
        solver.PDE_solver.solve()
        solver.w_.assign(solver.w)
        components=solver.w.split(deepcopy=True)
        ke=components[ke_index]
        pts=[float(ke(df.Point(x))) for x in xs]
        if not all(math.isfinite(v) and v>0 for v in pts):
            raise RuntimeError("nonphysical extracellular K")
        rows.append({"version":version,"step":step+1,"time_s":float(t),
                     "Ke_75um_mM":pts[0],"Ke_center_mM":pts[1],
                     "Ke_225um_mM":pts[2]})
    return rows,{"version":version,"steps":len(rows),"seconds":time.time()-t0,
                 "peak_center_deltaK_mM":max(r["Ke_center_mM"]-float(model.K_e_init) for r in rows),
                 "baseline_K_mM":float(model.K_e_init)}

all_rows=[]
summaries=[]
for version in ("M0","M3"):
    print("BEGIN_UPSTREAM",version,flush=True)
    r,s=run(version)
    all_rows.extend(r);summaries.append(s)
    print("END_UPSTREAM",version,s,flush=True)
with (OUT/"reference_parent_K_tseries.csv").open("w") as f:
    w=csv.DictWriter(f,fieldnames=list(all_rows[0]))
    w.writeheader();w.writerows(all_rows)
record={"classification":"NEW SOURCE-LEVEL PINNED FENICS SIMULATION, NOT historical SMM Step3B verification",
        "upstream_source":"martejulie/fluid-flow-in-astrocyte-networks",
        "upstream_commit":subprocess.check_output(["git","-C","ffian-parent","rev-parse","HEAD"],text=True).strip(),
        "n":N,"dt_s":DT,"tend_s":TEND,"stim_start_s":STIM_START,"stim_end_s":STIM_END,
        "j_in_mol_m2_s":J,"samples":summaries}
(OUT/"source_reproduction.json").write_text(json.dumps(record,indent=2)+"\n")
print("SOURCE_REPRODUCTION_COMPLETED",json.dumps(record))

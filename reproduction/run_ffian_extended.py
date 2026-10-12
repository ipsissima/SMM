#!/usr/bin/env python3
"""New explicit-amplitude runs of authentic ffian parent models.

Original ffian Model and Solver implementations; no retuning of their physics.
Sampling all 31 finite-element node-center positions at 0.1 s intervals.
"""
from pathlib import Path
import csv,json,time,subprocess
import dolfin as df

OUT=Path("repro_outputs/extended")
OUT.mkdir(parents=True,exist_ok=True)
from ffian import zero_flow_model,flow_model
L=3e-4;N=31;DT=.01;TEND=4.;T0=.1;T1=2.1
CASES=[("M0",2.5e-8),("M3",2.5e-8),("M3",5e-8),("M3",1e-7)]
xs=[L*(i+.5)/N for i in range(N)]
all_rows=[];reports=[]

for ver,j in CASES:
    print("START",ver,j,flush=True)
    st=time.time()
    mesh=df.IntervalMesh(N,0.,L)
    tt=df.Constant(0.)
    if ver=="M0":
        m=zero_flow_model.Model(mesh,L,tt,j,T0,T1)
        sv=zero_flow_model.Solver(m,DT,TEND)
        idx=3
    else:
        m=flow_model.Model(ver,mesh,L,tt,j,T0,T1,"constant")
        sv=flow_model.Solver(m,DT,TEND)
        idx=4
    baseline=float(m.K_e_init)
    maxcenter=-1e6
    for istep in range(1,int(round(TEND/DT))+1):
        tt.assign(istep*DT)
        sv.PDE_solver.solve()
        sv.w_.assign(sv.w)
        if istep%10==0 or istep==1:
            cs=sv.w.split(deepcopy=True)
            k=cs[idx]
            for ix,x in enumerate(xs):
                ke=float(k(df.Point(float(x))))
                all_rows.append(dict(model=ver,j_in=j,n=N,dt_s=DT,t_s=round(istep*DT,8),
                    x_m=x,Ke_mM=ke,deltaKe_mM=ke-baseline))
                if ix==N//2:maxcenter=max(maxcenter,ke-baseline)
    reports.append(dict(model=ver,j_in=j,steps=int(round(TEND/DT)),N=N,dt_s=DT,
        peak_sampled_center_deltaK_mM=maxcenter,elapsed_wall_s=time.time()-st,
        reference_baseline_Ke_mM=baseline,stim_start_s=T0,stim_end_s=T1))
    print("COMPLETE",ver,j,reports[-1],flush=True)
with (OUT/"parent_spatiotemporal.csv").open("w") as f:
    w=csv.DictWriter(f,fieldnames=list(all_rows[0]))
    w.writeheader();w.writerows(all_rows)
(OUT/"extended_summary.json").write_text(json.dumps({
  "classification":"NEW ORIGINAL FEniCS M0/M3 runs at 3 amplitudes, not old SMM Step3B source reproduction",
  "upstream_commit":subprocess.check_output(["git","-C","ffian-parent","rev-parse","HEAD"],text=True).strip(),
  "reference_runs":reports},indent=2)+"\n")

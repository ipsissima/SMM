"""Reproduce the final held-out generalization figure from exact archived logs."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1]
def records(name):
    return json.loads((root/"reproducibility"/name).read_text())["records"]
data=records("final_empirical_fit_records.json")+records("final_native_and_refinement_records.json")
def fit(job,tag):
    r=next(r for r in data if r["job_id"]==job)
    return next(e["value"] for e in r["events"] if e["tag"]==tag)
lo=fit(140121454401,"EMPIRICAL_RIVAL_FIT")
hi=fit(140121454362,"EMPIRICAL_RIVAL_FIT")
native=fit(140219131058,"JOINT_RIVAL_FIT")
glial=fit(140098938402,"JOINT_RIVAL_FIT")
series=[
    [("Weight-matched",lo["controls"]["total_matched"]),
     ("12 weights: cap 0.932",lo),("12 weights: cap 3.108",hi)],
    [("Weight-matched",native["controls"]["total_matched"]),
     ("6 glial weights",glial),("12 weights (budget stop)",native)]
]
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,
                     "axes.spines.top":False,"axes.spines.right":False})
fig,axes=plt.subplots(2,2,figsize=(8.3,5.8),constrained_layout=True)
colors=["#9b9b9b","#25768e","#a84d31"]
for row,(label,models) in enumerate(zip(["Reduced mobility","Native mobility"],series)):
    for col,(field,unit) in enumerate([("K_RMS_mM","K RMS (mM)"),("Vm_RMS_mV","Membrane voltage RMS (mV)")]):
        ax=axes[row,col]; x=np.arange(3); peak=0
        for i,(name,v) in enumerate(models):
            vals=[v[key][field] for key in ("train","train2","held_out")]
            peak=max(peak,max(vals))
            ax.bar(x+(i-1)*.23,vals,width=.21,label=name,color=colors[i])
        ax.axvline(1.5,color="#5b5b5b",lw=.75,ls="--")
        ax.set_xticks(x,["Train 0","Train 2","Held-out 4"])
        ax.set_ylabel(unit); ax.set_ylim(0,peak*1.42)
        ax.grid(axis="y",color="#dedede",lw=.6); ax.set_axisbelow(True)
        ax.set_title(f"{chr(65+row*2+col)}   {label}",loc="left",fontweight="bold",fontsize=10)
        if col==1: ax.legend(loc="upper right",frameon=False,fontsize=7.3)
fig.savefig(root/"paper/nonlinear_rival_generalization.png",dpi=300,bbox_inches="tight")
plt.close(fig)

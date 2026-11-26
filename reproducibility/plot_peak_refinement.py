"""Generate the peak-refinement figure from archived, inspected EDNM summaries."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
root=Path(__file__).resolve().parent.parent
data=json.loads((root/"reproducibility/peak_refinement_results.json").read_text())
fig,ax=plt.subplots(1,2,figsize=(10.8,4.0),layout="constrained")
colors=["#3D6B8F","#214A69","#7295AA","#BC8353","#993E33","#C6A35D"]
groups={}
for row in data["rows"]:
    groups.setdefault((row["mode"],row["glial_axial_migration_multiplier"]),[]).append(row)
keys=sorted(groups,key=lambda k:(k[0]!="migration_only",k[1]))
for i,key in enumerate(keys):
    a=sorted(groups[key],key=lambda r:r["spacing_um"])
    d=np.array([r["spacing_um"] for r in a])
    R=np.array([r["whole_field_relative_K_contrast"] for r in a])
    ell=a[0]["frozen_electrical_mode"]["frozen_two_conductor_length_um"]
    j=np.argmax(R)
    mode,mu=key
    label=("Migration " if mode=="migration_only" else "Joint flux ")+f"{mu:.3g}"
    style="--" if mode=="migration_only" else "-"
    for p,x,y in [(ax[0],d,100*R),(ax[1],d/ell,R/R[j])]:
        p.plot(x,y,style,marker="o",markersize=3,linewidth=1.5,color=colors[i],label=label)
        p.scatter(x[j],y[j],s=44,marker="D",facecolor=colors[i],edgecolor="white",linewidth=.6,zorder=5)
ax[0].set(xlabel="Intrinsic graph scale (µm)",ylabel="Relative K contrast (%)",title="A  Electrical-scale refinement")
ax[1].set(xlabel="Intrinsic scale / electrical length",ylabel="Contrast / sampled maximum",title="B  Normalized sensitivity windows")
ax[1].axvline(.7,color=".5",lw=.8,ls=":",zorder=0)
for p in ax:
    p.spines[["top","right"]].set_visible(False)
    p.grid(alpha=.15)
ax[0].legend(fontsize=7.5,loc="best",frameon=False)
fig.savefig(root/"paper/electrical_peak_refinement.png",dpi=220,bbox_inches="tight")

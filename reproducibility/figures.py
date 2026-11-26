import io, zipfile, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
R=Path(__file__).parent; P=R.parent
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':160,'savefig.dpi':220})
# Diagram: graph supports, not a measured anatomical embedding.
fig,ax=plt.subplots(figsize=(7.4,3.1)); ax.set_xlim(-.5,6.5);ax.set_ylim(-.35,2.65);ax.axis('off')
for y,label,color in [(0,'Extracellular', '#46637d'),(1,'Astroglial', '#157f76'),(2,'Neuronal', '#8e5365')]:
    ax.text(-.45,y+.17,label,color=color,weight='bold',fontsize=10)
    for i in range(7): ax.scatter(i,y,s=175,color='white',edgecolor=color,zorder=3,linewidth=1.5);ax.text(i,y,str(i),ha='center',va='center',fontsize=8,zorder=4)
    if y!=2:
        for i in range(6):ax.plot([i,i+1],[y,y],c=color,lw=1.3,zorder=1)
    else:
        for i,j in [(0,1),(1,3),(2,4),(3,5),(5,6)]:ax.add_patch(FancyArrowPatch((i+.13,y),(j-.13,y),arrowstyle='-|>',mutation_scale=10,lw=1,color=color,connectionstyle='arc3,rad=-.18'))
for i,j,w,rad in [(0,4,.8,-.25),(1,5,.5,-.21),(2,6,.6,-.17)]:
    ax.add_patch(FancyArrowPatch((i,1),(j,1),arrowstyle='-',connectionstyle=f'arc3,rad={rad}',lw=1.6,linestyle='--',color='#157f76',zorder=2)); ax.text((i+j)/2,1+abs(rad)*(j-i)/2+.08,str(w),ha='center',fontsize=8,color='#157f76')
for i in range(7):ax.plot([i,i],[.12,.88],ls=':',lw=.8,c='.6');ax.plot([i,i],[1.12,1.88],ls=':',lw=.8,c='.6')
ax.text(3,-.25,'Site index; vertical links denote local physiological exchange / feedback',ha='center',fontsize=9)
fig.tight_layout();fig.savefig(R/'architecture.png',bbox_inches='tight');plt.close(fig)
# Current decomposition and clamp.
def get(zpath, name='data.npz'):
 with zipfile.ZipFile(zpath) as z:
  out=[]
  for l in ['baseline','stimulated']:
   with np.load(io.BytesIO(z.read(l+'/'+name))) as d:out.append({k:d[k] for k in d.files})
 return out
fig,ax=plt.subplots(1,2,figsize=(7.4,3.2),layout='constrained')
for z,label,col in [('EDNM_SECONDS_native_convergence.zip','Native','#344e72'),('EDNM_STEP8_pump_k_clamp_1ms.zip','Pump K sensitivity clamped','#c37438')]:
 b,a=get(P/z); t=a['t']; v=(a['phi'][:,0,0,1]-a['phi'][:,0,2,1])-(b['phi'][:,0,0,1]-b['phi'][:,0,2,1]);ax[0].plot(t,v,label=label,c=col,lw=1.4)
ax[0].axhline(0,c='.5',lw=.7); ax[0].axvspan(.05,.55,alpha=.09,color='gray');ax[0].set(xlabel='Time (s)',ylabel='Neighboring $\\Delta V_m$ (mV)',title='A  Counterfactual intervention');ax[0].legend(frameon=False,fontsize=8)
a=json.loads((P/'smm_referee3/voltage_audit/EDNM_SECONDS_native.json').read_text())['nearest_0p4s']['integrated_contributions_mV'];vals=[a['NaK_pump'],a['K_leak']+a['K_DR'],a['Cl_leak'],sum(v for k,v in a.items() if k.startswith('axial')),a['Na_leak']+a['Na_gated']]; labels=['Na/K pump','K channels','Cl leak','Axial ions','Na channels'];ax[1].barh(labels[::-1],vals[::-1],color=['#c37438' if x>0 else '#344e72' for x in vals[::-1]]); ax[1].axvline(0,c='.5',lw=.7); ax[1].set(xlabel='Integrated contribution (mV)',title='B  Current balance at 0.4 s')
fig.savefig(R/'pump.png');plt.close(fig)
# Published computational benchmark at original dt, plot thin/decimated trajectories with maxima preserved in bins for spikes.
fig,ax=plt.subplots(2,1,figsize=(7.4,4.8),sharex=True,layout='constrained')
for s,label,col in [('weak','Native coupling','#344e72'),('strong','2000-fold glial mobility','#c37438')]:
 with zipfile.ZipFile(P/f'EDNM_STEP8_{s}_50us.zip') as z:
  with np.load(io.BytesIO(z.read('stimulated/observable.npz'))) as d:t=d['t'];v=d['vm'][:,0]; k=d['ke'][:,0]-d['ke'][0,0]
 ax[0].plot(t[::20],k[::20],lw=1.3,c=col,label=label)
 ax[1].plot(t[::5],v[::5],lw=.55,c=col,alpha=.85,label=label)
ax[0].set(ylabel='Somatic $\\Delta[K^+]_e$ (mM)',title='A  Potassium accumulation');ax[0].legend(frameon=False);ax[1].set(xlabel='Time (s)',ylabel='Somatic $V_m$ (mV)',title='B  Neuronal voltage');fig.savefig(R/'high_load.png');plt.close(fig)
# Information and shared-support discrimination at matched design.
r=json.loads((P/'smm_referee3/results.json').read_text());s=json.loads((P/'smm_referee3/shared_path_results.json').read_text());ds=[5,14.759877679228888,36.980111000660955]; keys=['full7_at5um','full7_at_matched_optimum','full7_at_long_step_optimum'];fig,ax=plt.subplots(1,2,figsize=(7.4,3.2),layout='constrained');x=np.arange(3)
for j,key,col,lab in [(0,'known','#344e72','Known kinetics'),(1,'all_five_passive_nuisances','#c37438','Five nuisances')]:ax[0].bar(x+(j-.5)*.31,[r['ideal_design'][k][key] for k in keys],width=.31,color=col,label=lab)
ax[0].set(yscale='log',ylabel='Fisher information',title='A  Local identification',xticks=x,xticklabels=['5','14.76','36.98'],xlabel='Site spacing (µm)');ax[0].legend(frameon=False,fontsize=8)
for j,key,col,lab in [(0,'scaled_path','#344e72','Scaled path'),(1,'weighted_shared_support','#157f76','Free path weights'),(2,'weighted_shared_support_plus_local_nuisances','#c37438','Free weights + nuisances')]:ax[1].bar(x+(j-1)*.24,[s['fits'][str(float(d))][key]['squared_whitened_distance'] for d in ds],width=.23,color=col,label=lab)
ax[1].set(yscale='log',ylabel='Squared whitened distance',title='B  Finite discrimination',xticks=x,xticklabels=['5','14.76','36.98'],xlabel='Site spacing (µm)');ax[1].legend(frameon=False,fontsize=7.5);fig.savefig(R/'information.png');plt.close(fig)

"""Phase V postprocessing: exact archived correlation vectors; illustrative occupancy law."""
import json, math, pathlib
import numpy as np
from scipy.special import expit, xlogy
ROOT=pathlib.Path(__file__).parent
def kl(a,b):return float(np.sum(xlogy(a,a/b)+xlogy(1-a,(1-a)/(1-b))))
dist=np.array([math.hypot(min(abs(i%4-j%4),4-abs(i%4-j%4)),min(abs(i//4-j//4),4-abs(i//4-j//4))) for i in range(16) for j in range(i+1,16)])
t=np.arange(2.,42.001,.25); mid=(t[:-1]+t[1:])/2
def develop(c,closed,prune=1):
    p=np.full(120,.08); hist=[p.copy()]
    for a,row in zip(mid,c):
        h=.18*(1-expit((a-25)/2)) if closed else .18
        b=h*np.exp(-dist/1.5)*expit(6*(row-.4))
        d=h*(.15+.45*expit(-6*(row-.4))+prune*.2*np.exp(-((a-15)/5)**2))
        p=b/(b+d)+(p-b/(b+d))*np.exp(-(b+d)*.25)
        hist.append(p.copy())
    return np.array(hist)
out=[]
for f in sorted(ROOT.glob('age_*.json')):
    d=json.loads(f.read_text()); s=d['summary']; rows=d['rows']; ages=[r['day'] for r in rows]
    assert len(ages)==len(set(ages))
    for inp in ['all','balanced']:
        for obs in ['K','Vm']:
            ca=np.array([r['metrics'][inp][obs]['corr_A'] for r in rows]); cb=np.array([r['metrics'][inp][obs]['corr_B'] for r in rows])
            assert ca.shape==(len(ages),120) and np.max(np.abs(ca))<=1.000001
            A=np.array([np.interp(mid,ages,ca[:,i]) for i in range(120)]).T
            B=np.array([np.interp(mid,ages,cb[:,i]) for i in range(120)]).T
            peaks={k:max(rows,key=lambda r:r['metrics'][inp][obs][k])['day'] for k in ['corr_L2','transfer_rel_energy','rms_difference']}
            for closed in [True,False]:
                pa=develop(A,closed); pr=np.linalg.norm(develop(A,closed,2)[-1]-develop(A,closed,0)[-1])
                def evaluate(a,b):
                    alt=A.copy(); mask=(mid>=a)&(mid<b);alt[mask]=B[mask]; pb=develop(alt,closed)
                    dif=pb-pa; re=np.linalg.norm(dif[int(round((b-2)/.25))]);end=np.linalg.norm(dif[-1]);k=min(kl(pa[-1],pb[-1]),kl(pb[-1],pa[-1]))
                    return dict(start=a,end=b,L2=float(end),restore_L2=float(re),retention=float(end/re) if re else None,KL=k,ratio_pruning=float(end/pr),N95=math.ceil(1.62/k) if k>0 else None)
                timing=[evaluate(a,a+5) for a in range(2,38)]
                out.append(dict(mode=s['mode'],dt=s['dt'],input=inp,observable=obs,closed=closed,peaks=peaks,history=evaluate(2,20),best=max(timing,key=lambda x:x['L2']),P7=evaluate(7,12),P9=evaluate(9,14),P15=evaluate(15,20),P19=evaluate(19,24),timing=timing,pruning_L2=float(pr)))
(ROOT/'structural_summary.json').write_text(json.dumps(out,indent=2))
for r in out:
    print('STRUCT5 '+json.dumps({k:v for k,v in r.items() if k!='timing'}))

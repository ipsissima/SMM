"""Whole-network held-out summary. Report fixed-current and locally matched
voltage residuals. The same data fitted only on training graphs; no age selection.
"""
from large_pair import *
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def analyze(y,local,dye=None):
 # Input y is (graphs,pairs); local is (graphs,pairs,2).
 # Position fixed effects learnt exclusively on train networks. Equal cluster sizes.
 G,M=y.shape;tr=np.arange(G)<G//2;te=~tr
 def cen(x):return x-x[tr].mean(axis=0)
 yy=cen(y).ravel();b=cen(local).reshape(G*M,-1);t=np.repeat(tr,M);e=~t
 def predict(X):
  z=(X-X[t].mean(0))/np.maximum(X[t].std(0),1e-15);z=np.c_[np.ones(len(z)),z];coef=np.linalg.lstsq(z[t],yy[t],rcond=1e-10)[0];return z@coef
 base=predict(b);r=(yy-base)[e].reshape(-1,M);a=dict(rms=float(np.sqrt(np.mean(r*r))),unadjusted_rms=float(np.sqrt(np.mean(yy[e]**2))),mean_transfer=float(y[te].mean()),pairs=M,graphs=int(te.sum()))
 if dye is not None:
  for name,z in dye.items():
   z=cen(z).reshape(G*M,-1);p=predict(np.c_[b,z])-base;w=p/max(np.sqrt(np.mean(p[t]**2)),1e-15);ww=w[e].reshape(-1,M);score=ww*r;w2=float(np.mean(ww**2));gvar=float(score.mean(1).var(ddof=1));within=float(score.var(axis=1,ddof=1).mean());a[name]=dict(score=float(score.mean()),between=gvar,within=within,w2=w2,graph_w2=float(np.mean(ww.mean(1)**2)),increment_RMS=float(np.sqrt(np.mean(p[e]**2))),augmented_RMSE=float(np.sqrt(np.mean((r-p[e].reshape(-1,M))**2))))
 return a

def summarize(x):
 G=x['graphs'];m=G//2;ages=[DAYS[i] for i in x['age_idx']];rows=[]
 Hs=np.array([[r['H'] for r in g['rows']] for g in x['records']]);hom=np.array([r['homogeneous_diagonal'] for r in x['records'][0]['rows']]);diag=np.diagonal(Hs,axis1=2,axis2=3);mean_diag=diag[:m].mean(axis=(0,2));test_diag=diag[m:].mean(axis=(0,2));p=ages.index(11);r=R_TARGET[[DAYS.index(a) for a in ages]];gscale=1000*mean_diag/r;fixed_g=gscale[p];p1,p2=PAIRS.T
 for i,age in enumerate(ages):
  h=Hs[:,i];d=diag[:,i];local=np.stack([d[:,p1],d[:,p2]],axis=-1);v=h[:,p1,p2];vrel=.5*v*(1/local[:,:,0]+1/local[:,:,1]);locnorm=local/mean_diag[i];dyes={}
  if x['records'][0]['rows'][i]['dye']:
   for key in x['records'][0]['rows'][i]['dye']:
    tt=np.array([g['rows'][i]['dye'][key] for g in x['records']]);dyes[key]=np.stack([tt[:,p1,p2],tt[:,p2,p1]],axis=-1)
  groups={}
  for cls in ['axial','all']:
   for dist in [1,2,3]:
    sel=(DIST==dist)&(AXIAL if cls=='axial' else True);z={k:v[:,sel] for k,v in dyes.items()};loc=locnorm[:,sel,:];raw=v[:,sel];rel=vrel[:,sel]
    # V=I(pA)/g(nS) H, with g fitted per age or once at P11.
    # Equalized 2 mV local deflection (direction-averaged) gives mV=2*ratio.
    groups[f'{cls}_{dist}']={
      'relative':analyze(rel,loc,z),
      'matched_2mV':analyze(V_MATCH*rel,loc,z),
      'calibrated_200pA':analyze(I_FIXED/gscale[i]*raw,loc,z),
      'P11_anchor_fixed_shunt_200pA':analyze(I_FIXED/fixed_g*raw,loc,z),
      'Rin6mature_200pA':analyze(I_FIXED/gscale[i]*raw*(6/9.7 if age==21 else 1),loc,z)}
  rows.append(dict(day=age,R_target_MOhm=float(r[i]),R_test_MOhm=float(1000*test_diag[i]/gscale[i]),R_fixed_anchor_MOhm=float(1000*test_diag[i]/fixed_g),g_age_fit_nS=float(gscale[i]),g_P11_anchor_nS=float(fixed_g),R_homogeneous_anchor_MOhm=float(1000*hom[i]/fixed_g),local_200pA_mean_mV=float(I_FIXED*test_diag[i]/gscale[i]),local_equalized_current_pA=float(1000*V_MATCH/r[i]),groups=groups))
 meta={k:v for k,v in x.items() if k!='records'};meta.update(rows=rows,weight_SD_mean=float(np.mean([g['weight_sd'] for g in x['records']])),max_linear_residual=max(r['checks']['residual'] for g in x['records'] for r in g['rows']),max_symmetry_error=max(r['checks']['symmetry'] for g in x['records'] for r in g['rows']),max_CG_iterations=max(r['checks']['iterations'] for g in x['records'] for r in g['rows']))
 return meta

def powers(s):
 rows=s['rows'];A=next(r for r in rows if r['day']==11);B=next(r for r in rows if r['day']==21);out=[]
 z=(norm.ppf(.975)+norm.ppf(.8))**2
 for group in A['groups']:
  for key in ['matched_2mV','calibrated_200pA','Rin6mature_200pA']:
   a=A['groups'][group][key];b=B['groups'][group][key]
   for tracer in ['ideal','mismatch']:
    if tracer not in a:continue
    x,y=a[tracer],b[tracer];M=a['pairs']
    for k in sorted(set([min(M,4),M])):
     for sd in [.02,.05,.1]:
      for common in [0.,.05]:
       def var(u):return u['between']+u['within']*(1/k-1/M)+sd**2*u['w2']/k+common**2*(u['graph_w2']+(u['w2']-u['graph_w2'])*M/(M-1)*(1/k-1/M))
       va,vb=var(x),var(y);delta=x['score']-y['score'];out.append(dict(group=group,readout=key,tracer=tracer,pairs=k,noise_mV=sd,common_mV=common,P11_score_mV=x['score'],P21_score_mV=y['score'],delta_mV=delta,n_age_comparison=int(np.ceil(z*(va+vb)/delta**2)) if abs(delta)>1e-12 else None,n_P11_geometry=int(np.ceil(z*va/x['score']**2)) if x['score']>0 else None,n_P21_geometry=int(np.ceil(z*vb/y['score']**2)) if y['score']>0 else None))
 return out

if __name__=='__main__':
 files=sorted(HERE.glob('large_*_*.json'));s=[]
 for f in files:
  x=json.loads(f.read_text());
  if 'records' not in x or x['graphs']<16:continue
  z=summarize(x);z['power']=powers(z);s.append(z)
 out=HERE/'large_summary.json';out.write_text(json.dumps(s,indent=2))
 for a in s:
  print(a['n'],a['graphs'],a['boundary'],a['amplitude'],'err',a['max_linear_residual'],'weight SD',a['weight_SD_mean'])
  for r in a['rows']:
   print(r['day'],'Rfit/fixed',round(r['R_test_MOhm'],3),round(r['R_fixed_anchor_MOhm'],3),'g',round(r['g_age_fit_nS'],3),'axial relative %', [round(100*r['groups'][f'axial_{d}']['relative']['rms'],4) for d in [1,2,3]],'200pA mV',[round(r['groups'][f'axial_{d}']['calibrated_200pA']['rms'],5) for d in [1,2,3]])
  for p in a['power']:
   if p['group'].startswith('axial') and p['readout']=='matched_2mV' and p['tracer']=='mismatch' and p['pairs']==4 and p['noise_mV']==.05 and p['common_mV']==0:print(p)
 # CSV for reproduction and independent review (include failed directional cases).
 flat=[];power=[]
 for a in s:
  meta={k:a[k] for k in ['n','sites','graphs','amplitude','boundary']}
  for r in a['rows']:
   for group,metrics in r['groups'].items():
    for obs,val in metrics.items():flat.append(meta|{'day':r['day'],'group':group,'readout':obs,'R_target_MOhm':r['R_target_MOhm'],'R_test_MOhm':r['R_test_MOhm'],'R_fixed_anchor_MOhm':r['R_fixed_anchor_MOhm'],'g_age_fit_nS':r['g_age_fit_nS'],'rms':val['rms'],'mean_transfer':val['mean_transfer'],'pairs':val['pairs'],'mismatch_score':val.get('mismatch',{}).get('score'),'mismatch_RMSE':val.get('mismatch',{}).get('augmented_RMSE')})
  power.extend(meta|p for p in a['power'])
 for name,rows in [('large_summary.csv',flat),('large_power.csv',power)]:
  if rows:
   with (HERE/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

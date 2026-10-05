#!/usr/bin/env python3
"""Frozen preprocessing for ds005385 SMM analysis.

Fail-closed: no auto-correction of unexpected channel sets, sampling rates,
EDF amplitude scaling, or excessive artifacts.
"""
from __future__ import annotations
from pathlib import Path
import argparse, json
import numpy as np
from packaging.version import Version

SEED=97
CHANNELS=[x.strip() for x in Path(__file__).with_name('channels_64.txt').read_text().splitlines() if x.strip()]
ARTIFACT_LABELS={'muscle artifact','eye blink','heart beat','line noise','channel noise'}

def robust_mad(x):
    x=np.asarray(x,float); med=np.median(x)
    return 1.4826*np.median(np.abs(x-med))

def amplitude_sanity(raw):
    data_uV=raw.get_data(picks='eeg')*1e6
    rms=np.sqrt(np.mean(data_uV**2,axis=1))
    median_rms=float(np.median(rms))
    p999=float(np.percentile(np.abs(data_uV),99.9))
    ok=(0.1<=median_rms<=500.0) and (p999<=5000.0)
    return {'median_channel_rms_uV':median_rms,'abs_p999_uV':p999,'pass':bool(ok)}

def preprocess(edf:Path,out_dir:Path):
    import mne
    from pyprep.find_noisy_channels import NoisyChannels
    from mne.preprocessing import ICA
    from mne_icalabel import label_components

    if Version(mne.__version__)<Version('1.13.2'):
        raise RuntimeError(f'Locked confirmatory preprocessing requires MNE >=1.13.2; found {mne.__version__}')
    if 'fsaverage_1005' not in mne.channels.get_builtin_montages():
        raise RuntimeError('fsaverage_1005 unavailable; do not substitute another montage.')

    out_dir.mkdir(parents=True,exist_ok=True)
    raw=mne.io.read_raw_edf(edf,preload=True,verbose='error')
    if raw.ch_names!=CHANNELS:
        raise RuntimeError('Channel names/order differ from frozen 64-channel ds005385 list.')
    if abs(float(raw.info['sfreq'])-1000.0)>1e-6:
        raise RuntimeError(f'Expected 1000 Hz raw sampling; got {raw.info["sfreq"]}')
    if raw.n_times/raw.info['sfreq']<180.0:
        raise RuntimeError('Recording shorter than 180 s before frozen 2-s edge crop.')

    montage=mne.channels.make_standard_montage('fsaverage_1005')
    raw.set_montage(montage,on_missing='raise')
    raw.crop(tmin=2.0,tmax=raw.times[-1]-2.0,include_tmax=False)
    raw.resample(250.0,npad='auto')

    amp=amplitude_sanity(raw)
    if not amp['pass']:
        raise RuntimeError(f'EDF amplitude sanity check failed: {amp}. Dataset warns EDF physical min/max may be invalid; no guessed rescaling is permitted.')

    prep=raw.copy().filter(1.0,45.0,fir_design='firwin',phase='zero',verbose='error')
    nc=NoisyChannels(prep,do_detrend=True,random_state=SEED,ransac=True,correlation=True)
    nc.find_all_bads(ransac=True,correlation=True)
    bads=sorted(nc.get_bads())
    if len(bads)>10:
        raise RuntimeError(f'Primary QC failed: {len(bads)} globally bad channels > 10.')
    raw.info['bads']=bads

    ica_raw=raw.copy().filter(1.0,100.0,fir_design='firwin',phase='zero',verbose='error')
    ica_raw.set_eeg_reference('average',projection=False,verbose='error')
    n_good=len(mne.pick_types(ica_raw.info,eeg=True,exclude='bads'))
    n_components=max(2,n_good-1)
    ica=ICA(n_components=n_components,method='infomax',fit_params={'extended':True},
            random_state=SEED,max_iter='auto')
    ica.fit(ica_raw,picks='eeg',reject_by_annotation=True,verbose='error')
    labels=label_components(ica_raw,ica,method='iclabel')
    component_labels=list(labels['labels'])
    component_probs=np.asarray(labels['y_pred_proba'],float)
    exclude=[i for i,(lab,p) in enumerate(zip(component_labels,component_probs))
             if lab in ARTIFACT_LABELS and p>=0.80]
    ica.exclude=exclude
    fraction_removed=len(exclude)/max(1,len(component_labels))
    ica_qc_flag=fraction_removed>0.20

    clean=raw.copy().filter(0.5,45.0,fir_design='firwin',phase='zero',verbose='error')
    clean.set_eeg_reference('average',projection=False,verbose='error')
    ica.apply(clean,exclude=exclude,verbose='error')
    clean.interpolate_bads(reset_bads=False,verbose='error')
    clean.set_eeg_reference('average',projection=False,verbose='error')

    epochs=mne.make_fixed_length_epochs(clean,duration=4.0,overlap=0.0,preload=True,
                                        reject_by_annotation=True,verbose='error')
    x_uV=epochs.get_data(picks='eeg')*1e6
    ptp=np.ptp(x_uV,axis=-1)
    hard=(ptp.max(axis=1)>250.0)|((ptp>150.0).mean(axis=1)>0.10)
    med_ptp=np.median(ptp,axis=1)
    hf=epochs.copy().filter(30.0,45.0,fir_design='firwin',phase='zero',verbose='error').get_data(picks='eeg')*1e6
    med_hf_rms=np.median(np.sqrt(np.mean(hf**2,axis=-1)),axis=1)
    robust=np.zeros(len(epochs),bool)
    for metric in (med_ptp,med_hf_rms):
        med=np.median(metric); mad=robust_mad(metric)
        if mad>0: robust|=metric>(med+6.0*mad)
    keep=~(hard|robust)

    n_clean=int(keep.sum())
    if n_clean<30:
        raise RuntimeError(f'Primary QC failed: only {n_clean} clean 4-s epochs (<30 / <120 s).')

    epochs_clean=epochs[keep]
    epochs_clean.save(out_dir/'clean-epo.fif',overwrite=True,verbose='error')
    ica.save(out_dir/'ica.fif',overwrite=True)

    qc={
      'input_edf':str(edf),'seed':SEED,'amplitude':amp,'bad_channels':bads,
      'n_bad_channels':len(bads),'ica_labels':component_labels,
      'ica_max_class_probability':component_probs.tolist(),
      'ica_excluded_components':exclude,'ica_fraction_removed':fraction_removed,
      'ica_qc_flag_fraction_gt_0.20':bool(ica_qc_flag),
      'n_epochs_total':int(len(epochs)),'n_epochs_clean':n_clean,
      'clean_seconds':4*n_clean,
      'rejected_epoch_indices':np.where(~keep)[0].astype(int).tolist()
    }
    (out_dir/'qc.json').write_text(json.dumps(qc,indent=2),encoding='utf-8')
    print(json.dumps(qc,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('edf',type=Path)
    ap.add_argument('--out-dir',type=Path,required=True)
    args=ap.parse_args()
    preprocess(args.edf,args.out_dir)

#!/usr/bin/env python3
"""Audited same-cohort POST-OPEN amended inference; frozen statistics are imported verbatim.

This program does NOT modify the scientific/fit code and never claims that an
amended result was the untouched frozen primary holdout PASS/FAIL verdict.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'confirmatory' / 'step5b'))
import aggregate_confirmatory as frozen  # noqa: E402

ORIGINAL_RUN = 37752719639
DATASET_COMMIT = '6a558cd5852503e66df9ac7fdac2f3a7f4ed5f12'
EXPECTED_STRUCTURAL = {
    'sub-206': ('QC_1_45_AMPLITUDE_P999_EXCEEDS_5000_UV',
                '9a25d6079631d57c1f21c0c8a8cd3ff4f9d29e58824ef598c54a40291c4263c4',
                113258985187),
    'sub-230': ('RAW_DURATION_BELOW_180_S',
                'cd11d262a93de7847828d3cb8fb75b7a5b5481b767b555aca35cb4f6a185e40b',
                113262946359),
    'sub-425': ('QC_1_45_AMPLITUDE_RMS_AND_P999_EXCEED_LIMITS',
                '7d9cc41b0e7e8c4d2a7a9630da1c2af0c83a7a38eace92e8452d771d06818fb8',
                113229660519),
}
FILENAME_PATTERN = re.compile(r'(sub-[0-9]{3})(\.excluded|\.ineligible)?\.json\Z')


def locate_amended(input_dir: Path):
    """Exactly 565 uniquely named records; strict ineligible suffix accounting."""
    if not input_dir.is_dir():
        raise RuntimeError(f'Missing amended input directory: {input_dir}')
    found = {}
    for path in input_dir.rglob('sub-*.json'):
        match = FILENAME_PATTERN.fullmatch(path.name)
        if match is None:
            raise RuntimeError(f'Unexpected amended subject filename: {path}')
        subject, extension = match.groups()
        if subject not in frozen.EXPECTED:
            raise RuntimeError(f'OUT_OF_COHORT_OR_DEVELOPMENT_CONTAMINATION: {path}')
        if subject in found:
            raise RuntimeError(f'Duplicate subject {subject}: {found[subject][0]} vs {path}')
        record_type = {None:'fit','.excluded':'qc_excluded','.ineligible':'structural_ineligible'}[extension]
        found[subject] = (path, record_type)
    extra = sorted(set(found)-set(frozen.EXPECTED))
    missing = sorted(set(frozen.EXPECTED)-set(found))
    if extra or missing:
        raise RuntimeError(f'565 assignment accounting invalid: missing={missing}, extra={extra}')
    assert len(found) == 565
    actual_structural = {s for s,(p,t) in found.items() if t == 'structural_ineligible'}
    if actual_structural != set(EXPECTED_STRUCTURAL):
        raise RuntimeError(f'Post-open structural record set changed: {actual_structural}')
    if found['sub-075'][1] == 'structural_ineligible':
        raise RuntimeError('sub-075 must not be reclassified as structurally ineligible')
    return {s:found[s] for s in frozen.EXPECTED}


def validate_structural(subject: str, data: dict) -> str:
    expected_code, sha, job = EXPECTED_STRUCTURAL[subject]
    required = {
        'subject':subject,
        'analysis_classification':'POST_OPEN_AMENDED',
        'record_type':'post_open_structural_ineligible',
        'included':False,
        'model_fit_performed':False,
        'reason_code':expected_code,
        'edf_sha256':sha,
        'original_holdout_run_id':ORIGINAL_RUN,
        'original_preprocess_job_id':job,
        'source_dataset_commit':DATASET_COMMIT,
        'primary_condition':'ses-1 / EyesClosed / acq-pre',
    }
    for key, value in required.items():
        if data.get(key) != value:
            raise RuntimeError(f'{subject} incompatible structural provenance: {key}')
    msg = data.get('original_failure_message')
    if not isinstance(msg,str) or not msg:
        raise RuntimeError(f'{subject} missing original failure text')
    if any(k in data for k in ('delta_elpd_M3_minus_M2','models','n_epochs','elpd')):
        raise RuntimeError(f'{subject}: structural record contains model/score data')
    if msg.startswith('Primary QC failed:'):
        raise RuntimeError(f'{subject}: do not relabel a structural stop as frozen QC')
    return expected_code


def validate_source_manifest(root: Path, records: dict) -> dict:
    """Optional runner-side artifact-source gate written *before* aggregate stats."""
    p = root / 'artifact_sources.json'
    if not p.is_file():
        raise RuntimeError('Missing source-run provenance manifest; no silent input mixing')
    m = json.loads(p.read_text())
    required = {
        'original_preprocess_run':ORIGINAL_RUN,
        'original_batch_b_fit_run':ORIGINAL_RUN,
        'amended_preprocess_subject':'sub-075',
        'amended_preprocess_input_sha256':'bc2cf91a253a490342d849f39baba530bbe4a58b5c2ea86fb78db08e037fcb02',
        'scientific_base_commit':'20fab272edf33d2513470c60ac9398754535cacb',
        'analysis_classification':'POST_OPEN_AMENDED',
    }
    for key,value in required.items():
        if m.get(key) != value:
            raise RuntimeError(f'Artifact-source drift: {key}')
    sourced = m.get('sourced_fits', {})
    if set(sourced) != set(frozen.EXPECTED)-set(EXPECTED_STRUCTURAL):
        raise RuntimeError('Artifact-source subject set does not equal required 562')
    for s, (path, kind) in records.items():
        if kind == 'structural_ineligible':
            continue
        group = 'original_B' if 232 <= int(s[4:]) <= 419 else 'amended_A_C'
        if sourced[s] != group:
            raise RuntimeError(f'{s}: wrong fit source group {sourced[s]} != {group}')
    return m


def run(input_dir: Path, out_dir: Path):
    profile, opt = frozen.load_profile()
    records = locate_amended(input_dir)
    source_manifest = validate_source_manifest(input_dir, records)

    subject_rows, optimizer_rows, provenance = [], [], []
    included_deltas, frozen_qc_reasons, structural_reasons = [], [], []
    for subject, (path, kind) in records.items():
        d = json.loads(path.read_text(encoding='utf-8'))
        provenance.append({
            'subject':subject, 'record_type':kind,
            'input_json_sha256':frozen.sha256(path),
            'path':str(path),
            'source_group':source_manifest['sourced_fits'].get(subject,'not_fitted'),
        })
        common = {
            'subject':subject,
            'record_type':kind,
            'analysis_classification':'POST_OPEN_AMENDED',
            'primary_included':0,
            'reason':'',
            'n_epochs':'', 'split_A':'', 'split_B':'',
            'M2_cv_elpd':'', 'M3_cv_elpd':'',
            'delta_elpd_M3_minus_M2':'',
            'M3_better':'',
            'max_embedding_score_error':'',
            'min_M2_minus_M3_training_gap':'',
            'transfer_identity_max_abs_error':'',
        }
        if kind == 'structural_ineligible':
            code = validate_structural(subject,d)
            structural_reasons.append(code)
            common['reason'] = f'POST_OPEN_STRUCTURAL_INELIGIBILITY: {code}'
        elif kind == 'qc_excluded':
            reason = frozen.validate_exclusion(subject,d)
            frozen_qc_reasons.append(reason)
            common['reason'] = reason
        else:
            frozen.validate_fit(subject,d,profile,opt)
            m2=float(d['models']['M2']['cv_elpd'])
            m3=float(d['models']['M3']['cv_elpd'])
            delta=float(d['delta_elpd_M3_minus_M2'])
            included_deltas.append(delta)
            nested=d['nestedness']
            common.update({
                'primary_included':1, 'n_epochs':int(d['n_epochs']),
                'split_A':int(d['split_A']), 'split_B':int(d['split_B']),
                'M2_cv_elpd':m2, 'M3_cv_elpd':m3,
                'delta_elpd_M3_minus_M2':delta,
                'M3_better':int(delta > 0.0),
                'max_embedding_score_error':max(float(z['embedding_score_error']) for z in nested),
                'min_M2_minus_M3_training_gap':min(float(z['M2_minus_M3_training_gap']) for z in nested),
                'transfer_identity_max_abs_error':float(d['transfer_identity_max_abs_error']),
            })
            for model in ('M2','M3'):
                for direction in d['models'][model]['directions']:
                    o=direction['optimizer']
                    optimizer_rows.append({
                        'subject':subject,'model':model,
                        'train':direction['train'],'test':direction['test'],
                        'success':bool(o['success']), 'message':str(o['message']),
                        'nit':int(o['nit']), 'nfev':int(o['nfev']),
                        'train_score_normalized':float(direction['train_score_normalized']),
                        'heldout_score_normalized':float(direction['heldout_score_normalized'])
                    })
        subject_rows.append(common)

    x=np.asarray(included_deltas,dtype=float)
    if len(x) < 1:
        raise RuntimeError('No valid included fits')
    if len(subject_rows) != 565 or len(included_deltas)+len(frozen_qc_reasons)+len(structural_reasons) != 565:
        raise RuntimeError('Amended subject accounting invariant failed')
    if len(optimizer_rows) != len(x)*4 or not all(z['success'] for z in optimizer_rows):
        raise RuntimeError('Amended N1 optimizer accounting invariant failed')

    # Invoke the EXACT pre-existing frozen statistical functions, unchanged.
    mean=float(x.mean())
    ci_low,ci_high=frozen.bootstrap_mean_ci(x)
    p_flip,n_extreme=frozen.sign_flip_pvalue(x)
    meets_frozen_numerical_criterion=bool(ci_low > 0.0 and p_flip < 0.05)
    max_embedding=max(float(z['max_embedding_score_error']) for z in subject_rows if z['primary_included'])
    min_gap=min(float(z['min_M2_minus_M3_training_gap']) for z in subject_rows if z['primary_included'])
    max_transfer=max(float(z['transfer_identity_max_abs_error']) for z in subject_rows if z['primary_included'])
    checks=(max_embedding <= float(profile['nested_training_tolerance']) and
            min_gap >= -float(profile['nested_training_tolerance']) and
            max_transfer <= float(profile['transfer_identity_tolerance']))
    if not checks:
        raise RuntimeError('Exact M3-in-M2 nesting integrity failed')

    result={
        'phase':'step5b_same_dataset_amended_holdout',
        'analysis_classification':'POST_OPEN_AMENDED',
        'untouched_frozen_primary_verdict':'NOT_EVALUABLE',
        'amendment_lock':'amended/step5b/AMENDMENT_LOCK_2026-10-08.md',
        'original_frozen_head_sha':'20fab272edf33d2513470c60ac9398754535cacb',
        'original_holdout_run_id':ORIGINAL_RUN,
        'n_holdout_assigned':565,
        'n_fit_included_amended':len(x),
        'n_frozen_qc_excluded':len(frozen_qc_reasons),
        'n_structural_ineligible_post_open':len(structural_reasons),
        'structural_reasons':dict(Counter(structural_reasons)),
        'frozen_qc_reasons':dict(Counter(frozen_qc_reasons)),
        'contrast':'ELPD(M3)-ELPD(M2)',
        'mean_delta_elpd_M3_minus_M2':mean,
        'median_delta_elpd_M3_minus_M2':float(np.median(x)),
        'sd_delta_elpd_M3_minus_M2':float(np.std(x,ddof=1)) if len(x)>1 else 0.0,
        'bootstrap':{'seed':frozen.SEED,'resamples':frozen.N_BOOT,'method':'subject bootstrap percentile 95% CI, frozen function',
                     'ci_low':ci_low,'ci_high':ci_high},
        'sign_flip':{'seed':frozen.SEED,'permutations':frozen.N_FLIP,'alternative':'mean DeltaELPD >0',
                     'extreme_or_equal_count':n_extreme,'plus_one_correction':True,'p_value':p_flip},
        'n_M3_better':int(np.count_nonzero(x>0.0)),
        'fraction_M3_better':float(np.mean(x>0.0)),
        'optimizer_calls':len(optimizer_rows),
        'all_optimizers_success':True,
        'max_embedding_score_error':max_embedding,
        'minimum_M2_minus_M3_training_gap':min_gap,
        'max_transfer_identity_abs_error':max_transfer,
        'all_nesting_checks_pass':checks,
        'frozen_optimizer':opt,
        'numerical_profile':profile,
        'unchanged_inferential_rule':'CI_low > 0 AND one-sided sign-flip p < 0.05',
        'amended_meets_frozen_numerical_success_criterion':meets_frozen_numerical_criterion,
        'NOT_a_frozen_confirmatory_PASS':True,
        'science_caveat':'Post-holdout amendments to representation and ineligible-subject accounting; not an untouched primary confirmatory analysis.',
        'source_manifest':source_manifest,
        'inputs':provenance,
    }
    out_dir.mkdir(parents=True,exist_ok=True)
    with (out_dir/'amended_subject_results.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(subject_rows[0]))
        w.writeheader(); w.writerows(subject_rows)
    with (out_dir/'amended_optimizer_diagnostics.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(optimizer_rows[0]))
        w.writeheader(); w.writerows(optimizer_rows)
    (out_dir/'amended_primary_result.json').write_text(json.dumps(result,indent=2)+chr(10),encoding='utf-8')
    verdict='MEETS FROZEN NUMERICAL CRITERION' if meets_frozen_numerical_criterion else 'DOES NOT MEET FROZEN NUMERICAL CRITERION'
    message=[
        '# Step 5B amended same-cohort result — NOT untouched frozen confirmation',
        '',
        '**Original frozen-primary verdict: NOT EVALUABLE.**',
        f'**Post-open amended analysis: {verdict}.**',
        '',
        'This is the same cohort with an explicit post-open one-to-one channel-order amendment for sub-075 and',
        'three structural-ineligibility records for sub-206, sub-230 and sub-425; these are not frozen Primary QC exclusions.',
        '',
        '- Holdout assignments: 565/565 accounted for',
        f'- Amended included fits: {len(x)}',
        f'- Ordinary frozen-QC exclusions: {len(frozen_qc_reasons)}',
        f'- Post-open structural ineligible: {len(structural_reasons)}',
        f'- Mean DeltaELPD(M3-M2): {mean:.9f}',
        f'- 95% subject bootstrap CI: [{ci_low:.9f}, {ci_high:.9f}]',
        f'- One-sided paired sign-flip p: {p_flip:.8g}',
        f'- M3 > M2: {int(np.count_nonzero(x>0.0))}/{len(x)}',
        f'- All M2/M3 N1 optimizer and nesting validations passed: {checks}',
        '',
        'The frozen numerical criterion was NOT adjusted after examining M2/M3 outcomes.',
        'Do not cite this as a PASS of the original frozen primary confirmatory experiment.',
    ]
    (out_dir/'amended_primary_result.md').write_text(chr(10).join(message)+chr(10),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('inputs','source_manifest','numerical_profile')},indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('input_dir',type=Path)
    parser.add_argument('--out-dir',type=Path,required=True)
    opt=parser.parse_args()
    run(opt.input_dir,opt.out_dir)

#!/usr/bin/env python3
"""Synthetic-only integration tests: amended subject accounting with untouched frozen stats."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile

import aggregate_amended as amended

frozen = amended.frozen


def fake_fit(s, profile, optimizer):
    directions = [{
        'train':'A','test':'B','optimizer':{'success':True,'message':'synthetic','nit':1,'nfev':1},
        'train_score_normalized':1.0,'heldout_score_normalized':1.0,
    },{
        'train':'B','test':'A','optimizer':{'success':True,'message':'synthetic','nit':1,'nfev':1},
        'train_score_normalized':1.0,'heldout_score_normalized':1.0,
    }]
    threads = {k:'1' for k in frozen.THREAD_KEYS}
    threads.update({'OMP_DYNAMIC':'FALSE','PYTHONHASHSEED':'0',
                    'OPENBLAS_CORETYPE':profile['openblas_coretype']})
    return {
        'subject':s,'seed':int(profile['seed']), 'optimizer':optimizer,
        'environment':{'numpy':'2.3.5','scipy':'1.17.0','mne':'1.13.2',
                       'thread_env':threads},
        'n_epochs':49,'split_A':24,'split_B':25,
        'models':{
            'M2':{'cv_elpd':1.0,'directions':directions},
            'M3':{'cv_elpd':1.01,'directions':directions},
        },
        'delta_elpd_M3_minus_M2':0.01,
        'transfer_identity_max_abs_error':0.0,
        'nestedness':[{'embedding_score_error':0.0,
                       'M2_minus_M3_training_gap':0.0}] * 2,
    }


def main():
    profile, opt = frozen.load_profile()
    with tempfile.TemporaryDirectory() as temp:
        d=Path(temp)
        sourced={}
        for s in frozen.EXPECTED:
            if s in amended.EXPECTED_STRUCTURAL:
                source=amended.ROOT/'amended/step5b/ineligible'/f'{s}.ineligible.json'
                data=json.loads(source.read_text())
                (d/f'{s}.ineligible.json').write_text(json.dumps(data))
                continue
            sourced[s]='original_B' if 232 <= int(s[4:]) <= 419 else 'amended_A_C'
            if s in ('sub-048','sub-427'):
                data={'subject':s,'included':False,
                      'reason':'Primary QC failed: synthetic-only test exclusion.'}
                p=d/f'{s}.excluded.json'
            else:
                data=fake_fit(s,profile,opt)
                p=d/f'{s}.json'
            p.write_text(json.dumps(data))
        manifest={
            'original_preprocess_run':amended.ORIGINAL_RUN,
            'original_batch_b_fit_run':amended.ORIGINAL_RUN,
            'amended_preprocess_subject':'sub-075',
            'amended_preprocess_input_sha256':'bc2cf91a253a490342d849f39baba530bbe4a58b5c2ea86fb78db08e037fcb02',
            'scientific_base_commit':'20fab272edf33d2513470c60ac9398754535cacb',
            'analysis_classification':'POST_OPEN_AMENDED',
            'sourced_fits':sourced,
        }
        (d/'artifact_sources.json').write_text(json.dumps(manifest))
        result=amended.run(d,d/'output')
        assert result['n_holdout_assigned']==565
        assert result['n_fit_included_amended']==560
        assert result['n_frozen_qc_excluded']==2
        assert result['n_structural_ineligible_post_open']==3
        assert result['untouched_frozen_primary_verdict']=='NOT_EVALUABLE'
        assert result['amended_meets_frozen_numerical_success_criterion'] is True
        assert result['bootstrap']['resamples']==10_000
        assert result['sign_flip']['permutations']==100_000
        assert result['all_nesting_checks_pass'] is True
        assert len(result['inputs'])==565
        assert len((d/'output'/'amended_subject_results.csv').read_text().splitlines())==566
        assert not (d/'output'/'confirmatory_primary_result.json').exists()
        # Fail-closed if any structural record is removed or renamed.
        target=d/'sub-230.ineligible.json'
        target.rename(d/'sub-230.ineligible.json.off')
        try:
            amended.locate_amended(d)
        except RuntimeError as exc:
            assert 'missing' in str(exc)
        else:
            raise AssertionError('Missing structural record was accepted')
        (d/'sub-230.ineligible.json.off').rename(target)
        bad=json.loads(target.read_text())
        bad['reason_code']='Primary QC failed: forged claim'
        try:
            amended.validate_structural('sub-230',bad)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Forged structural reason was accepted')
        # Frozen numeric decisions must be exactly the same functions.
        assert amended.frozen.bootstrap_mean_ci is frozen.bootstrap_mean_ci
        assert amended.frozen.sign_flip_pvalue is frozen.sign_flip_pvalue
        print('PASS_SYNTHETIC_AMENDED_565_ACCOUNTING')
        print('PASS_SYNTHETIC_FROZEN_N1_VALIDATION_AND_INFERENCE_REUSE')
        print('PASS_SYNTHETIC_STRUCTURAL_EXCLUSION_AND_MISSING_RECORD_FAIL_CLOSED')


if __name__ == '__main__':
    main()

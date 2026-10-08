# Step 5B definitive pipeline - reproduction entry point

This directory contains the canonical implementation of the definitive SMM empirical test. Historical wave/telegraph SMM code elsewhere in the repository is retained for genealogy and backward reproducibility but is not the primary model tested here.

## Current information barrier

- Development: historically exposed `sub-001..043`
- Primary metadata-clean development set: 38
- QC-passed deterministic fit set: 34
- Confirmatory holdout: `sub-044..608` (565 subjects)
- Holdout gate: `HOLDOUT_GATE.json`
- The holdout workflow is manual-only and cannot execute subject jobs while the gate is `CLOSED`.

## Canonical code

- `preprocess_ds005385.py` - frozen EEG preprocessing/QC
- `empirical_csd_lock.py` - frozen empirical CSD construction
- `model_frequency_lock.py` - M0/M1/M2/M3 frequency-domain models
- `statistical_lock.py` - complex Whittle score
- `fit_cv_subject.py` - frozen two-block subject-level fitting
- `reconstruct_network_frozen.py` - deterministic DK68 network reconstruction
- `aggregate_development.py` - descriptive 34-subject development aggregation
- `aggregate_confirmatory.py` - frozen confirmatory inference with complete 565-assignment accounting and frozen-QC inclusion
- `subject_split.csv` - complete 608-subject role/QC map

## Scientific locks

Read in this order:

1. pre-EEG lock bundle / `FREEZE_COMMIT.txt`;
2. `DEVELOPMENT_NUMERICAL_FREEZE_2026-10-06.md`;
3. `RESULTS_PROVENANCE_MANIFEST.md`;
4. `HOLDOUT_GATE.json`.

## Forward model

The canonical forward model is not rebuilt during fitting. It is downloaded from canonical run `37376954431` and hash-gated before use.

Required hashes include:

- `leadfield_DK68_64x68.csv`: `86afbddbd637ef4239b5755ef3853aafb62890ac989340fd093482ad7c9853b2`
- `leadfield_U20.npy`: `17eb1218c6cfb8b3e27ed51283e91d3e7068261a9c7595b6da2b2e358e3b8891`

See `RESULTS_PROVENANCE_MANIFEST.md` for the complete list.

## Development workflow

P1 and P2 are archived failed numerical-robustness probes. N1, the normalized nested-comparator method, passed the predeclared Haswell/Sandybridge/Zen robustness gate in canonical run `37599448753`. The numerical profile is now **FROZEN**.

The definitive post-freeze development workflow `.github/workflows/step5b-development-final.yml` completed in canonical run `37621942072`:

1. pinned package versions and one-thread execution were verified;
2. the frozen network and canonical forward operator were hash-gated;
3. all 34 frozen-QC-passed development recordings were fit with the frozen N1 M2/M3 two-block CV procedure;
4. 136/136 selected model/direction fits passed;
5. all exact M3-in-M2 nesting checks passed;
6. the complete aggregate was produced and permanently recorded.

The descriptive development aggregate is `confirmatory/step5b/development_summary.json`; the sign-off is `confirmatory/step5b/DEVELOPMENT_SIGNOFF_2026-10-08.md`. Development performance did not alter any scientific setting or confirmatory criterion.

## Numerical freeze

The final production profile is N1:

- seed 97;
- 512 Sobol candidates;
- 16 Sobol polish starts;
- normalized unit-cube coordinates;
- smooth ordered logarithmic M2 time constants over unchanged 0.03-30 s bounds;
- exact fitted-M3 point retained as a feasible M2 anchor;
- L-BFGS-B maxiter 300;
- ftol \(10^{-11}\);
- gtol \(10^{-7}\);
- maxls 50;
- relative CSD/model eigenvalue floor \(10^{-6}\);
- pinned NumPy/SciPy/MNE/pandas;
- one BLAS/OpenMP thread;
- production OpenBLAS kernel Haswell.

The permanent final-freeze record is `confirmatory/step5b/FINAL_NUMERICAL_FREEZE_TEMPLATE.md`. The sign or magnitude of DeltaELPD(M3-M2) was never used to choose the numerical profile.

## Opening the holdout

Do not edit the model when opening the holdout.

Opening requires a provenance-only commit to `HOLDOUT_GATE.json` that:

- changes `status` from `CLOSED` to `OPEN`;
- records the final deterministic development run ID;
- records the development aggregate artifact ID;
- records its SHA256/digest;
- retains the numerical-freeze commit.

No scientific setting may change in the same commit.

The primary workflow is:

`.github/workflows/step5b-confirmatory-holdout.yml`

It is `workflow_dispatch` only and also requires the explicit manual acknowledgement `OPEN-HOLDOUT`.

## Confirmatory workflow

When legitimately opened, the workflow:

1. partitions `sub-044..608` into three fixed job matrices solely to respect GitHub's matrix-size limit;
2. downloads each primary-condition EDF from the frozen OpenNeuro snapshot and verifies its git-annex SHA256 and byte size;
3. applies frozen preprocessing/QC;
4. runs the same deterministic M2/M3 fit;
5. accounts for exactly 565 holdout assignments, representing frozen-QC failures explicitly rather than replacing them;
6. computes subject fit contrasts only for recordings passing the frozen primary QC/inclusion rules;
7. applies the frozen bootstrap/sign-flip rule to the frozen-QC-included paired contrasts;
8. emits the primary PASS/FAIL result plus a complete 565-subject accounting table.

Reusable jobs:

- `.github/workflows/step5b-reusable-preprocess-one.yml`
- `.github/workflows/step5b-reusable-fit-one.yml`

## Frozen primary inference

For subject (i),

[
\Delta_i=ELPD_i(M_3)-ELPD_i(M_2).
]

Specific SMM predictive success requires:

- 10,000-resample subject bootstrap 95% CI lower bound > 0;
- one-sided paired 100,000-permutation sign-flip (p<0.05).

Seed = 97.

The executable implementation is `aggregate_confirmatory.py`.

## Permanent results

After each completed stage, transient Actions artifacts must be condensed into permanent repository files:

- subject result CSV;
- optimizer diagnostics CSV;
- aggregate JSON;
- aggregate Markdown summary;
- run/artifact IDs and digests in `RESULTS_PROVENANCE_MANIFEST.md`;
- figure source data used by the manuscript.

Heavy FIF/forward binaries do not need to be committed to Git, provided their provenance and hashes are permanent.

## Manuscript

Pre-holdout manuscript architecture:

`../../manuscript/DEFINITIVE_SMM_MANUSCRIPT_PREHOLDOUT.md`

Figure/table plan:

`../../manuscript/FIGURE_TABLE_PLAN.md`

Primary-result figure generator:

`../../manuscript/scripts/generate_primary_figures.py`

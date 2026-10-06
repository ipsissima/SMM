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

Workflow:

Optimizer robustness P1 currently uses:

`.github/workflows/step5b-development-smoke.yml`

The definitive 34-subject post-freeze development matrix is already prebuilt separately at:

`.github/workflows/step5b-development-final.yml`

It remains inert while `NUMERICAL_PROFILE.json` is `PENDING`. Once P1 or the predetermined P2 passes and the final numerical profile is frozen, that matrix must:

1. verifies pinned package versions;
2. forces single-thread numerical execution;
3. reconstructs/hash-gates the network;
4. downloads/hash-gates the canonical forward operator;
5. downloads each frozen QC-passed development epoch artifact;
6. runs M2/M3 two-block CV;
7. requires successful optimization in all directions;
8. uploads one fit JSON per subject;
9. aggregates all 34 fits in-run.

The aggregate artifact is:

`SMM_STEP5B_DEVELOPMENT_AGGREGATE`

The aggregate is descriptive only. It intentionally does not compute the confirmatory p-value or bootstrap decision.

## Numerical freeze

The frozen fit configuration is:

- seed 97;
- 32 Sobol candidates;
- 4 L-BFGS-B polish starts;
- maxiter 120;
- ftol (10^{-9});
- gtol (10^{-6});
- maxls 30;
- relative CSD/model eigenvalue floor (10^{-6});
- pinned NumPy/SciPy/MNE/pandas;
- one BLAS/OpenMP thread.

The single-thread requirement was added after development exposed cross-run M2 local-minimum variability. Three independent reproducibility runs then returned bit-identical scores and optimizer diagnostics.

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

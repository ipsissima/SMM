# Step 5B development numerical freeze

**Date:** 2026-10-06  
**Status:** FROZEN AFTER DEVELOPMENT NUMERICAL REPRODUCIBILITY TEST, BEFORE CONFIRMATORY HOLDOUT  
**Scope:** implementation-level numerical settings only

## Reason for this freeze

The pre-data scientific model, primary endpoint, M2 comparator, development/holdout split,
frequency range, exclusion rules, and success criterion were already frozen before EEG.

During development, repeated fits of the same QC-passed subject (sub-001) produced different
M2 optima across GitHub Actions runs while M3 was much more stable. The model code, empirical
epochs, forward operator, network, and statistical scoring code were verified identical across
those runs. This isolated a numerical reproducibility problem in the optimization environment,
not a scientific-model discrepancy.

The protocol explicitly permits one post-development freeze of optimizer/numerical choices.

## Reproducibility test

Workflow run: 37483948147  
Commit: 5ecc084d4d193629657052c9f69448b8e0327850

Three independent GitHub-hosted runners executed the same sub-001 M2/M3 two-block CV fit with:

- ubuntu-24.04
- Python 3.13
- NumPy 2.3.5
- SciPy 1.17.0
- pandas 2.2.3
- MNE 1.13.2
- OMP_NUM_THREADS=1
- OPENBLAS_NUM_THREADS=1
- MKL_NUM_THREADS=1
- VECLIB_MAXIMUM_THREADS=1
- NUMEXPR_NUM_THREADS=1
- OMP_DYNAMIC=FALSE
- PYTHONHASHSEED=0

All three runs returned exactly:

- M2 CV ELPD = 534.8237844922861
- M3 CV ELPD = 534.7535949015517
- Delta ELPD (M3-M2) = -0.07018959073445785

All three also returned identical optimizer diagnostics:

- M2 direction A: nit=29, nfev=504
- M2 direction B: nit=52, nfev=816
- M3 direction A: nit=34, nfev=378
- M3 direction B: nit=34, nfev=282

Every local optimization reported successful L-BFGS-B convergence.

## Frozen numerical implementation

The following fit-driver settings remain unchanged from the pre-data driver:

- random seed = 97
- Sobol exponent m = 5, hence 32 initial candidates
- 4 polish starts
- optimizer = L-BFGS-B
- maxiter = 120
- ftol = 1e-9
- gtol = 1e-6
- maxls = 30
- CSD/model eigenvalue relative floor = 1e-6

The only new implementation freeze is deterministic single-thread numerical execution using
the environment variables listed above and ubuntu-24.04 runners.

No biological parameter, model equation, parameter bound, M2 flexibility, primary outcome,
frequency range, QC criterion, subject split, or success rule was changed.

## Consequence

All 34 QC-passed primary development subjects are to be fit under this environment.
After those development fits are complete and audited, no further optimizer/numerical
changes are permitted before opening sub-044..608.

The confirmatory holdout remains unopened at the time of this freeze.

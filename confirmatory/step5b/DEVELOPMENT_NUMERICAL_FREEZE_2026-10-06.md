# Step 5B development numerical freeze

**Date:** 2026-10-06  
**Status:** SUPERSEDED DURING DEVELOPMENT - CROSS-RUNNER-IMAGE REPRODUCIBILITY FAILURE; HOLDOUT REMAINS CLOSED  
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


---

## Supersession record - 2026-10-06

The single-thread freeze above passed three independent runners in workflow run `37483948147`, but all three used GitHub runner image:

`ubuntu-24.04 / 20260927.320.1`.

The subsequent 34-subject run `37495554793` used the updated GitHub runner image:

`ubuntu-24.04 / 20261004.327.1`.

Its first completed `sub-001` fit, under the same pinned Python packages, the same one-thread environment, the same model code, the same input epochs, the same forward model and the same reconstructed network, returned:

- M2 CV ELPD = `534.8734362017726`
- M3 CV ELPD = `534.7059328161354`
- Delta ELPD(M3-M2) = `-0.16750338563724654`

rather than the three-run gate value:

- M2 CV ELPD = `534.8237844922861`
- M3 CV ELPD = `534.7535949015517`
- Delta ELPD(M3-M2) = `-0.07018959073445785`.

Direct commit comparison confirmed that `fit_cv_subject.py`, `model_frequency_lock.py`, `statistical_lock.py`, `empirical_csd_lock.py`, `reconstruct_network_frozen.py`, `requirements-gate.txt`, and `channels_64.txt` were byte-identical between the reproducibility run and the new run.

Therefore the prior freeze did **not** establish cross-runner-image optimizer robustness. It is superseded before any confirmatory holdout data are opened.

This does not authorize any scientific-model change. The only permitted next operation is a development-only optimizer robustness study based on training-objective convergence/reproducibility, followed by a new final numerical freeze and a complete rerun of the 34 QC-passed development subjects.

The holdout remains unopened.

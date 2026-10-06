# Step 5B development sign-off template

**This document is intentionally incomplete until the canonical deterministic 34-subject run finishes.**

The confirmatory gate may be opened only after every item below is resolved from the canonical aggregate.

## 0. Final numerical profile

- [ ] `NUMERICAL_PROFILE.json` status is `FROZEN`.
- [ ] The passing N1 robustness run and aggregate are permanently recorded; P1/P2 failures remain in provenance.
- [ ] `FINAL_NUMERICAL_FREEZE_TEMPLATE.md` has been completed into the final freeze record.
- [ ] The fit driver consumes the frozen versioned profile.
- [ ] No optimizer profile was selected using the sign or magnitude of DeltaELPD(M3-M2).

## A. Completeness

- [ ] Exactly 34 expected QC-passed development subjects are present.
- [ ] No subject `sub-044` or later appears in any input or aggregate.
- [ ] All 34 subject JSON files pass the frozen schema checks.
- [ ] The aggregate subject set equals the frozen `development_primary` PASS rows in `subject_split.csv`.

Expected subjects:

`001,002,003,004,005,006,007,010,011,014,015,016,017,018,019,020,021,022,023,024,025,028,029,030,031,032,033,034,035,036,038,039,040,042`

## B. Numerical integrity

- [ ] Seed = 97 for all subjects.
- [ ] Every fit exactly matches the final `NUMERICAL_PROFILE.json` produced by the passing N1 nested-comparator robustness rule.
- [ ] Final profile is exactly N1: 512 Sobol candidates / 16 Sobol polish starts / exact M3-in-M2 anchor / normalized coordinates.
- [ ] Production kernel is the predeclared `OPENBLAS_CORETYPE=Haswell`, after cross-kernel robustness has passed on Haswell/Sandybridge/Zen.
- [ ] N1 unit-cube L-BFGS-B is used for all fits, with the exact M3-in-M2 raw anchor retained for M2.
- [ ] maxiter = 300, ftol = 1e-11, gtol = 1e-7, maxls = 50.
- [ ] relative CSD floor = (10^{-6}).
- [ ] Single-thread BLAS/OpenMP environment used with pinned NumPy/SciPy/MNE versions.
- [ ] All selected model/direction optimizer results (34 subjects x 2 models x 2 CV directions) satisfy the N1 success schema.
- [ ] For every subject and both training blocks, embedded M3-as-M2 score error <= 1e-8.
- [ ] For every subject and both training blocks, training score M2 >= training score M3 - 1e-8.
- [ ] No nonfinite score or parameter appears.
- [ ] No implementation-level pathology remains that would justify another numerical change.

If the last item fails, do **not** open the holdout. Any permitted numerical correction must be documented and the complete development run repeated before sign-off.

## C. Scientific invariants

Confirm that development performance caused no change to:

- [ ] biological mechanism;
- [ ] K-to-QIF mapping;
- [ ] glial topology;
- [ ] primary condition;
- [ ] development/holdout boundary;
- [ ] M2 comparator strength;
- [ ] primary M3-vs-M2 contrast;
- [ ] 1-40 Hz primary frequency range;
- [ ] success criterion;
- [ ] exclusion/QC thresholds.

## D. Development result record

Fill after aggregation:

- Canonical run ID:
- Canonical commit SHA:
- Aggregate artifact ID:
- Aggregate artifact digest/SHA256:
- Mean development DeltaELPD:
- Median development DeltaELPD:
- SD:
- IQR:
- Range:
- Subjects with M3 > M2:
- Optimizer calls successful:

These are descriptive development quantities only. No development p-value is required or allowed to substitute for the confirmatory test.

## E. Permanent repository record

Before opening holdout:

- [ ] Commit `development_subject_results.csv`.
- [ ] Commit `development_optimizer_diagnostics.csv`.
- [ ] Commit `development_summary.json`.
- [ ] Commit `development_summary.md`.
- [ ] Update `RESULTS_PROVENANCE_MANIFEST.md` with run/artifact/digest.
- [ ] Record the commit SHA containing those permanent results.
- [ ] Verify static integrity CI passes on that commit.

## F. Holdout opening commit

Only after A-E pass:

- [ ] Update `HOLDOUT_GATE.json` in a provenance-only commit.
- [ ] Set `status` to `OPEN`.
- [ ] Fill development run/artifact/digest fields.
- [ ] Make **no model/code/scientific changes in the same commit**.
- [ ] Manually dispatch `step5b-confirmatory-holdout.yml` with acknowledgement `OPEN-HOLDOUT`.

## Sign-off verdict

**PENDING**

Allowed final values:

- `PASS - HOLDOUT MAY OPEN`
- `FAIL - DEVELOPMENT NUMERICAL/IMPLEMENTATION ISSUE REMAINS`

# Step 5B lock-to-executable compliance audit

**Date:** 2026-10-06
**Scope:** frozen primary confirmatory analysis only
**Holdout status:** CLOSED
**Verdict:** PASS WITH DOCUMENTED IMPLEMENTATION ADDENDA; NO PRIMARY SCIENTIFIC DIVERGENCE FOUND

This audit compares the pre-EEG scientific lock against the current executable
pipeline before any confirmatory holdout EEG is opened.

## 1. Subject boundary and primary condition — PASS

Frozen lock:

- development: sub-001..043;
- holdout: sub-044..608;
- primary condition: ses-1 / EyesClosed / acq-pre.

Executable pipeline:

- subject_split.csv preserves the same boundary;
- reusable holdout preprocessing rejects subjects outside 044..608 before data download;
- the holdout workflow constructs exactly sub-044..608;
- the EDF filename is fixed to ses-1_task-EyesClosed_acq-pre.

No subject can move across the development/holdout boundary based on signal
results.

## 2. Forward model and observable subspace — PASS

Frozen lock requires:

- fsaverage template head;
- fsaverage_1005 64-channel montage;
- fixed-normal cortical source orientation;
- DK68 regional aggregation;
- explicit average reference;
- U20 from the forward operator, not data PCA.

The canonical forward artifact is pinned to run 37376954431 and verified
before every fit by cryptographic hashes, including:

- leadfield_DK68_64x68.csv
  86afbddbd637ef4239b5755ef3853aafb62890ac989340fd093482ad7c9853b2;
- leadfield_U20.npy
  17eb1218c6cfb8b3e27ed51283e91d3e7068261a9c7595b6da2b2e358e3b8891.

No EEG enters construction of U20.

## 3. Preprocessing — PASS WITH ONE ALREADY-DOCUMENTED IMPLEMENTATION AMENDMENT

The executable preprocessing matches the frozen procedure for:

- exact 64-channel list/order;
- 1000-Hz input;
- >=180-s raw duration;
- 2-s crop at both ends;
- resampling to 250 Hz;
- PyPREP seed 97 with RANSAC/correlation;
- failure for >10 global bad channels;
- 1–100-Hz extended-Infomax ICA, seed 97;
- ICLabel 0.9.0;
- removal only of the frozen artifact classes at probability >=0.80;
- 0.5–45-Hz analysis stream;
- average reference;
- bad-channel interpolation with provenance retained;
- 4-s non-overlapping epochs;
- hard rejection at 250 uV max PTP or >10% channels above 150 uV;
- robust 6-MAD PTP/high-frequency rejection;
- >=30 clean epochs / >=120 s inclusion.

### Amplitude-sanity amendment

The lock originally described amplitude sanity after conversion to microvolts.
Development exposed the documented ds005385 combination of large DC offsets and
invalid EDF physical min/max metadata. The numerical thresholds were not
changed. Their application was moved to the already-prespecified 1–45-Hz QC
stream rather than the uncentered DC-coupled raw signal.

This amendment is already recorded in the provenance manifest and development
history. It is an implementation-location correction, not a threshold
relaxation or result-dependent rescaling rule.

### ICA >20% flag

The lock says that >20% IC removal produces a QC flag. The executable code
does exactly that through ica_qc_flag_fraction_gt_0.20; it does not silently
change the removal threshold or automatically redefine primary inclusion.

## 4. Empirical primary endpoint — PASS

Frozen lock:

- multivariate CSD;
- 1–40 Hz;
- 1-Hz bins;
- frozen U20 projection;
- chronological two-block prediction.

Executable CSD:

- 250-Hz sampling;
- 4-s / 1000-sample epochs;
- DPSS NW=3, Kmax=5;
- exact integer frequencies 1..40 Hz;
- identical U20 projection;
- first chronological half A and second half B;
- fit A -> score B and fit B -> score A.

The old 4.65% threshold, critical scale and band-selection logic do not enter
the executable primary score.

## 5. Predictive likelihood — PASS

The executable score implements the frozen complex multivariate
Whittle/complex-Wishart form:

-sum_f nu_f [ log|S_M(f)| + tr(S_M(f)^-1 S_emp(f)) ]

with the same relative eigenvalue regularization applied to model and empirical
CSDs.

Per-direction held-out scores are normalized by total spectral degrees of
freedom before the two directions are averaged. This normalization is common to
M2 and M3 and does not change the paired model ordering.

## 6. Comparator architecture — PASS

The primary executable comparison is M3 versus M2, exactly the frozen decisive
contrast.

M2 remains a flexible stable generic two-state slow controller with:

- tau1,tau2 in [0.03,30] s;
- four independent E/I slow-control gains in [-0.5,0.5].

M3 retains the upstream physiologically constrained K-homeostasis transfer.
The biological K-to-QIF coefficients and astroglial homeostatic constants are
not fit per subject from EEG.

The current primary workflow does not need M0 or M1 to decide the frozen
SMM-specific success criterion. They remain part of the adversarial hierarchy
and may be reported as frozen secondary comparator analyses. The manuscript
must not call M3>M0 alone SMM-specific success.

## 7. Shared nuisance/network fitting — PASS

The fit driver enforces the frozen bounds including:

- G_N in [50,185];
- propagation velocity in [3,12] m/s;
- positive source and sensor-noise scales;
- a bounded E/I stochastic-source mixture parameter.

No subject-level fit re-estimates the upstream SMM biological constants.

## 8. Numerical optimizer — DEVELOPMENT-ONLY FREEZE IN PROGRESS

The lock explicitly permits one post-development implementation-level freeze of
optimizer choice/starts/stopping/tolerance.

The original narrow profile proved runner-image sensitive and was superseded
before holdout opening.

The predeclared escalation rule is now:

- P1: 256 Sobol / 8 polish starts;
- if P1 fails, P2: 512 Sobol / 16 polish starts;
- no intermediate profile;
- profile choice cannot use DeltaELPD sign or magnitude.

P1 failed its predeclared M2 training-basin recovery rule on all three tested
OpenBLAS kernels. P2 run 37503947541 is in progress.

NUMERICAL_PROFILE.json remains PENDING; therefore the final 34-subject
workflow and holdout cannot validly run yet.

## 9. Confirmatory group inference — PASS, EXECUTABLE DETAILS COMPLETED PRE-HOLDOUT

Frozen scientific rule:

- Delta_i = ELPD_i(M3)-ELPD_i(M2);
- mean paired subject contrast;
- 10,000-resample 95% bootstrap CI with lower bound >0;
- one-sided paired 100,000-permutation sign-flip p<0.05;
- seed 97;
- both conditions required.

The executable implementation, fixed before holdout signal access, specifies:

- percentile bootstrap interval;
- NumPy generator seed 97;
- sign flips of the paired subject contrasts;
- plus-one Monte-Carlo p-value correction;
- success iff CI_low>0 AND p<0.05.

These are inferential implementation details, not changes to the scientific
success criterion.

## 10. Holdout QC accounting — PASS

The holdout contains 565 assigned subjects.

The frozen preprocessing also contains signal-dependent primary inclusion rules.
Therefore the executable pipeline:

- accounts for all 565 assignments;
- represents frozen-QC failures explicitly;
- does not replace excluded subjects;
- performs paired M3-vs-M2 inference on the recordings satisfying the frozen
  primary QC/inclusion criteria.

Unexpected infrastructure/data-integrity failures still fail closed rather than
being relabeled as QC exclusions.

## 11. Replication-order textual inconsistency in the lock — RESOLVED WITHOUT AFFECTING PRIMARY

The pre-EEG lock contains two replication-order descriptions.

Section 3 gives an explicit ordered list:

1. session-1 EyesOpen pre;
2. session-1 EyesClosed post;
3. session-1 EyesOpen post;
4. session-2 EyesClosed pre;
5. remaining session-2 conditions.

Section 11 later summarizes session-1 EyesOpen pre, session-2 EyesClosed pre
where available, then post conditions.

These two passages are internally inconsistent.

For reproducibility, the controlling interpretation is the earlier Section 3
language because it explicitly says “The replication order is fixed as” and
provides a numbered sequence. REPLICATION_AND_ROBUSTNESS_PLAN.md already uses
that numbered Section-3 order.

This resolution concerns only post-primary replication sequencing and cannot
change the primary holdout result.

## 12. Information barrier — PASS

Before this audit:

- no confirmatory holdout workflow has run;
- HOLDOUT_GATE.json is CLOSED;
- NUMERICAL_PROFILE.json is PENDING;
- optimizer robustness uses development sub-001 only;
- the canonical development EEG artifact run contains only development
  subjects;
- a separate holdout-leakage audit records the same boundary.

## Final audit verdict

**PASS WITH DOCUMENTED ADDENDA.**

No discrepancy found in this audit changes:

- the biological mechanism;
- K-to-QIF map;
- astroglial topology;
- primary condition;
- subject boundary;
- M2 strength;
- primary M3-vs-M2 contrast;
- 1–40-Hz endpoint;
- QC thresholds;
- predictive likelihood;
- or confirmatory success criterion.

The only currently unresolved prerequisite is the final development numerical
profile. P2 must pass its predeclared robustness rule before the final
34-subject rerun can begin.

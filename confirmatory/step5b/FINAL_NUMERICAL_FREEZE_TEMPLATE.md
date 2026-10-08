# Step 5B final numerical freeze

**Status: FROZEN FOR FINAL DEVELOPMENT AND CONFIRMATORY HOLDOUT**

**Canonical N1 robustness run:** `37599448753`  
**Canonical N1 commit:** `8ec566fd8f6e82d838a4c041db2a81d81ece4227`  
**Frozen-profile commit:** `901739e232bedd29785dbae5a416cb8fd346f388`  
**Holdout:** CLOSED at the time of this freeze.

The confirmatory holdout remained closed throughout P1, P2 and N1 numerical-method development. P1 and P2 failed their predeclared robustness criteria. N1 passed the predeclared nested-comparator and cross-kernel criteria; the sign and magnitude of DeltaELPD(M3-M2) were audit-only and were not used to select N1.

## Selected method

- Profile: **N1**
- Search coordinates: normalized unit cube
- M2 time constants: ordered logarithmic parameterization over the unchanged 0.03-30 s physical domain
- Exact comparator safeguard: fitted M3 solution embedded exactly as a feasible M2 candidate
- Sobol candidates: 512
- L-BFGS-B Sobol polish starts: 16
- Exact nested anchor: retained feasible candidate plus deterministic local start
- maxiter: 300
- ftol: 1e-11
- gtol: 1e-7
- maxls: 50
- CSD/model relative floor: 1e-6
- seed: 97
- production OpenBLAS kernel: Haswell
- thread policy: one BLAS/OpenMP thread

## Robustness provenance

- Robustness run ID: `37599448753`
- Aggregate artifact: `SMM_STEP5B_OPT_N1_AGGREGATE`
- Aggregate artifact ID: `11480529099`
- Aggregate artifact digest: `sha256:31bddc93ebd2030c63e54fde00b345cb0bc50a41727230aeaa23fb161d680152`
- Kernels tested: Haswell, Sandybridge, Zen

## Frozen N1 pass criteria

- Analytic transfer-identity pass: True
- Max transfer identity error: 1.023e-18
- Embedded M3-as-M2 score equality pass: True
- Max embedded score error: 1.137e-13
- M2 >= contained M3 training invariant pass: True
- Minimum M2-M3 training gap: 0.00430180409501
- Training-score cross-kernel range pass: True
- Per-model CV cross-kernel range pass: True
- Training-score ranges: `{"M2_A": 3.964970483139041e-05, "M2_B": 0.0002054554164487854, "M3_A": 0.0, "M3_B": 1.1368683772161603e-13}`
- CV-score ranges: `{"M2": 0.00015412882703458308, "M3": 4.0055283534456976e-08}`
- Finite/bounds audit: True

## Selection firewall

**The sign and magnitude of DeltaELPD(M3-M2) were not N1 selection criteria.**

## Scientific invariants

N1 changes no biological mechanism, K-to-QIF mapping, astroglial topology or constants, neuronal equations, M2 flexibility or physical bounds, M3 constraints, structural network, EEG forward model, primary condition, preprocessing/QC threshold, 1-40-Hz endpoint, likelihood, development/holdout split, primary M3-vs-M2 contrast, or confirmatory success criterion.

After this freeze, the only admissible development operation was a complete fresh 34-subject rerun under this exact profile, followed by permanent aggregation/sign-off before any holdout opening.

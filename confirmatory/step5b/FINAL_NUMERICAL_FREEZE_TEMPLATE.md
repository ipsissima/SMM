# Step 5B final numerical freeze

**Status:** PENDING N1 ROBUSTNESS VERDICT
**Holdout:** CLOSED
**Scientific model:** unchanged from the pre-EEG lock

P1 and P2 both failed their predeclared numerical-robustness criteria. This
template is therefore now controlled only by
`N1_NESTED_OPTIMIZER_PROTOCOL.md`.

## Selection rule

N1 may be frozen only if all predeclared numerical conditions pass:

- exact analytic M3-in-M2 transfer identity;
- exact embedded M3-as-M2 score equality;
- M2 training score never below its contained M3 solution;
- recovery of all historical training basins;
- cross-kernel training-objective reproducibility;
- per-model CV numerical reproducibility;
- finite outputs and unchanged physical-bound compliance.

The sign or magnitude of DeltaELPD(M3-M2) is not a selection criterion.

## Final profile

- Profile: **PENDING N1**
- Robustness run ID: **PENDING**
- Search coordinates: normalized unit cube
- M2 ordered time constants: logarithmic parameterization over unchanged 0.03–30 s bounds
- Sobol candidates: 512
- Sobol polish starts: 16
- Exact M3-in-M2 raw anchor: required
- Exact M3-in-M2 local-polish start: required
- L-BFGS-B maxiter: 300
- ftol: 1e-11
- gtol: 1e-7
- maxls: 50
- CSD/model relative floor: 1e-6
- seed: 97
- production OpenBLAS kernel: Haswell
- thread policy: one BLAS/OpenMP thread

## Robustness evidence

- Kernels required: Haswell, Sandybridge, Zen
- Aggregate artifact/digest: **PENDING**
- Transfer-identity max error: **PENDING**
- Embedded-score max error: **PENDING**
- Minimum M2-M3 training nesting gap: **PENDING**
- Training-score ranges: **PENDING**
- Per-model CV ELPD ranges: **PENDING**
- Historical best training basins recovered: **PENDING**
- Finite/bounds audit: **PENDING**
- Verdict: **PENDING**

## Activation sequence

When and only when N1 returns PASS:

1. generate the completed freeze document directly from the N1 aggregate;
2. commit that completed freeze record without opening the holdout;
3. use that completed-freeze commit SHA as `final_freeze_commit`;
4. generate and commit `NUMERICAL_PROFILE.json` with status `FROZEN`;
5. rerun all 34 QC-passed development subjects from scratch using the exact
   N1 production fitter;
6. require the M3-in-M2 nesting invariant on both training blocks of every
   development subject;
7. aggregate and sign off the complete 34-subject result;
8. only then create a provenance-only holdout-opening commit.

If N1 fails, no numerical profile is frozen and the holdout remains closed.
A new numerical-method protocol must be written before further testing.

No biological parameter, equation, comparator bound, frequency range, QC
threshold, subject split, likelihood, or confirmatory success criterion may be
altered by this process.

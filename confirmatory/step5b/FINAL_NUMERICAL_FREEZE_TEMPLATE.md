# Step 5B final numerical freeze

**Status:** PENDING ROBUSTNESS VERDICT  
**Holdout:** CLOSED  
**Scientific model:** unchanged from the pre-EEG lock

This document is intentionally committed before the optimizer-robustness result is
known. It will be completed only from the predeclared P1/P2 decision rule in
`OPTIMIZER_ROBUSTNESS_DECISION_RULE.md`.

## Selection rule

The numerical profile is selected only by:

- successful optimizer termination;
- recovery of already observed best training basins;
- cross-kernel training-objective reproducibility;
- per-model CV numerical reproducibility;
- finite outputs and frozen-bound compliance.

The sign or magnitude of DeltaELPD(M3-M2) is not a selection criterion.

## Final profile

- Profile: **PENDING (P1 or P2 only)**
- Robustness run ID: **PENDING**
- Sobol candidates: **PENDING**
- Polish starts: **PENDING**
- L-BFGS-B maxiter: 120
- ftol: 1e-9
- gtol: 1e-6
- maxls: 30
- CSD/model relative floor: 1e-6
- seed: 97
- production OpenBLAS kernel: Haswell
- thread policy: one BLAS/OpenMP thread

## Robustness evidence

- Kernels required: Haswell, Sandybridge, Zen
- Aggregate artifact/digest: **PENDING**
- Training-score ranges: **PENDING**
- Per-model CV ELPD ranges: **PENDING**
- All historical best training basins recovered: **PENDING**
- All optimizer calls successful: **PENDING**
- Finite/bounds audit: **PENDING**
- Verdict: **PENDING**

## Activation sequence

When and only when the robustness verdict is PASS:

1. complete this document from the robustness aggregate;
2. commit it without opening the holdout;
3. use that completed-document commit SHA as the
   `final_freeze_commit` recorded in `NUMERICAL_PROFILE.json`;
4. change `NUMERICAL_PROFILE.json` from `PENDING` to `FROZEN`, recording
   the passing profile and robustness run;
5. the versioned fit driver automatically consumes the frozen profile;
6. rerun all 34 QC-passed development subjects from scratch using
   `.github/workflows/step5b-development-final.yml`;
7. sign off the complete 34-subject aggregate;
8. only then create a provenance-only holdout-opening commit.

If P1 fails, this document remains pending while the predetermined P2 runs. If
P2 fails, no numerical profile is frozen and the holdout remains closed.

No biological parameter, equation, comparator, frequency range, QC threshold,
subject split, or confirmatory success criterion may be altered by this process.

# Step 5B new numerical-method protocol after P1/P2 failure

**Date frozen:** 2026-10-06
**Scope:** development-only numerical implementation
**Holdout:** CLOSED
**Scientific model:** unchanged

## Why a genuinely new method is required

P1 (256 Sobol / 8 L-BFGS-B starts) failed the predeclared optimizer-robustness rule.

P2 (512 Sobol / 16 L-BFGS-B starts) also failed: the Zen arm completed the full P2 search but M2 on train block B did not recover a training basin already known to exist.

Therefore the failure is not solved by simply adding more starts to the same parameterization.

A structural fact gives a stronger numerical criterion:

M2 contains the linearized M3 slow-feedback transfer exactly.

The executable model already exposes this identity through m2_exact_m3_params().

Therefore, for identical shared nuisance/network parameters and a given training block, the feasible M2 class contains the corresponding M3 solution. A correctly optimized M2 must consequently satisfy, up to numerical tolerance,

    train_score(M2) >= train_score(M3).

If this invariant fails, M2 is underoptimized and the primary M3-vs-M2 comparison is not admissible.

## Method N1: normalized nested-comparator optimizer

N1 is the only profile tested under this new protocol before any further numerical redesign.

### A. Smooth normalized search coordinates

Optimization is performed on a unit hypercube.

Shared parameters are mapped bijectively to the already frozen physical bounds:

- G_N: [50,185], linear map;
- velocity: [3,12] m/s, linear map;
- pE: [1e-4,1-1e-4], linear map;
- source-scale exponent: [-4,4], linear map;
- sensor-floor exponent: [-8,2], linear map.

For M2, the two slow time constants are no longer represented by two physical-time variables followed by a sort operation.

Instead:

- u_tau1 in [0,1] maps logarithmically to tau1 in [0.03,30] s;
- u_gap in [0,1] maps logarithmically to tau2 in [tau1,30] s.

This preserves exactly the frozen physical domain 0.03 <= tau1 <= tau2 <= 30 s while removing the nondifferentiable sorting fold and greatly reducing scale anisotropy.

The four M2 gains remain exact linear maps to [-0.5,0.5].

This is an optimizer coordinate change only. It does not alter the model or any physical bound.

### B. M3 search

For each chronological training block:

- seed 97;
- 512 scrambled Sobol candidates in normalized coordinates;
- 16 best candidates polished by L-BFGS-B;
- unit-cube bounds;
- maxiter 300;
- ftol 1e-11;
- gtol 1e-7;
- maxls 50.

Every finite raw Sobol candidate remains in the final candidate pool; successful local polishes are added rather than replacing their starts. The best finite candidate is retained.

### C. Exact M3-in-M2 anchor

After fitting M3 on a training block:

1. keep the fitted shared parameters G_N, velocity, pE, source scale and sensor floor;
2. replace the M3 slow subsystem by m2_exact_m3_params();
3. encode that physical point into the normalized M2 coordinates;
4. evaluate it directly before any M2 local optimization.

The executable transfer identity must satisfy a frequency-domain equality check over 1-40 Hz before the optimizer runs.

The direct embedded-M3 M2 training score must equal the M3 training score within 1e-8 normalized score.

The unpolished embedded point remains a valid candidate even if a local optimizer starting there moves to a worse point.

### D. M2 search

For each training block:

- the same 512 Sobol candidates;
- the 16 best Sobol candidates polished by L-BFGS-B;
- the exact embedded-M3 point added as an additional deterministic start;
- the unpolished embedded-M3 point itself retained in the final candidate pool;
- same normalized coordinates and stopping settings as M3.

The selected M2 solution is the best finite training solution among all raw/polished candidates.

This guarantees that M2 cannot be accepted with a training score below the M3 solution that it mathematically contains.

## Cross-kernel robustness probe

N1 is tested on development subject sub-001 only, on:

- Haswell;
- Sandybridge;
- Zen.

Each run uses the same frozen epochs, forward artifact, network, seed and package versions.

No holdout EEG is accessed.

## N1 PASS criteria

All conditions must hold.

1. The analytic M3-to-M2 transfer embedding agrees over 1-40 Hz to max absolute complex error <= 1e-12.
2. On every kernel and both training blocks, the directly embedded M3 point scored as M2 differs from the M3 training score by <= 1e-8.
3. On every kernel and both directions:
       optimized train_score(M2) >= optimized train_score(M3) - 1e-8.
4. All four model/direction fits on every kernel recover the historical best training targets within 0.0005 normalized score.
5. For each model/direction, the range of optimized training scores across the three kernels is <= 0.001.
6. For each model, the range of two-direction CV ELPD across kernels is <= 0.005.
7. All retained scores are finite and all decoded parameters satisfy the unchanged frozen physical bounds.
8. The final candidate pool retains every finite raw Sobol candidate as well as every successful local polish and, for M2, the raw exact-embedding candidate. A selected local-polish result must report success; a selected raw feasible candidate is valid without a local-optimizer success flag because its objective is evaluated directly.

The sign and magnitude of held-out DeltaELPD(M3-M2) are logged but are not used in any PASS criterion.

## If N1 passes

N1 becomes the final numerical method.

The production fit driver is updated to use the exact same normalized/nested procedure.

Then:

1. rerun all 34 QC-passed development subjects from scratch;
2. require the nesting invariant for M2 versus M3 on every training block;
3. aggregate all development results;
4. complete development sign-off;
5. open the holdout only through the existing provenance gate.

## If N1 fails

Do not open the holdout.

Do not increase Sobol candidates or local starts ad hoc.

Inspect the failed criterion and write a new explicit numerical-method protocol before further testing. In particular, a failure of the exact embedding equality would indicate an implementation/model inconsistency rather than an optimizer problem.

## Scientific firewall

N1 changes only numerical coordinates, search strategy and stopping settings.

It does not change:

- biological mechanism;
- K-to-QIF mapping;
- astroglial parameters or topology;
- neuronal equations;
- M2 flexibility or bounds;
- M3 constraints;
- primary EEG condition;
- preprocessing/QC;
- forward model;
- frequency range;
- likelihood;
- subject split;
- primary M3-vs-M2 contrast;
- confirmatory success criterion.

The holdout remains unopened.

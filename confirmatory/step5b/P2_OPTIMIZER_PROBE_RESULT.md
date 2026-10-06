# Step 5B P2 optimizer robustness result

**Date:** 2026-10-06
**Decision:** P2 FAIL
**Holdout:** CLOSED
**Next protocol:** N1 normalized nested-comparator optimizer

## Provenance

- P2 workflow run: 37503947541
- P2 profile: 512 Sobol candidates / 16 L-BFGS-B polish starts
- Kernels: Haswell, Sandybridge, Zen
- Decisive completed arm: Zen
- Zen job ID: 112407717464

## Decisive failure

The Zen arm completed the full P2 broad search successfully. Infrastructure,
forward hashes, network reconstruction, package versions and data loading all
passed.

The subsequent predeclared basin-recovery assertion failed for M2 on train block
B:

- M2 train A = 535.9428288002408; historical target 535.8336527151650 -> PASS
- M2 train B = 534.0061847445946; historical target 534.0933137283831 -> FAIL
- M3 train A = 535.9409257004763; historical target 535.8430963300440 -> PASS
- M3 train B = 534.0944535060165; historical target 533.8829279760863 -> PASS

For audit only:

- M2 CV ELPD = 534.9055700819081
- M3 CV ELPD = 534.9323858063995
- DeltaELPD(M3-M2) = +0.02681572449137093

The positive Delta did not enter the P2 verdict.

## Structural numerical diagnosis

The decisive anomaly is stronger than simple failure to recover a historical
number.

The executable model defines M2 so that it contains the linearized M3
slow-feedback transfer exactly via m2_exact_m3_params().

Yet on train block B the P2 optimizer returned

    M3 train = 534.0944535060165
    M2 train = 534.0061847445946.

A flexible comparator that mathematically contains the M3 solution cannot have a
true optimum below that contained solution. Therefore this is direct evidence
of M2 underoptimization.

This observation motivates the next numerical method independently of the sign
of held-out DeltaELPD.

## Consequence

P2 cannot become the final numerical profile.

The remaining P2 kernel jobs may complete for audit, but cannot reverse the
verdict.

No additional brute-force Sobol/start escalation is authorized under the old
parameterization.

The next method is frozen in N1_NESTED_OPTIMIZER_PROTOCOL.md and changes only:

- optimizer coordinates;
- local-search numerical settings;
- explicit exploitation of the exact M3-in-M2 feasible embedding.

No scientific model, model bound, comparator flexibility, endpoint, split or
success criterion changes.

The holdout remains unopened.

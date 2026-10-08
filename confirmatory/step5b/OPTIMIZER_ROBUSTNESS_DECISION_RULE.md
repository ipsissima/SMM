# Step 5B optimizer robustness decision rule

**Date frozen:** 2026-10-06  
**Scope:** development-only numerical implementation  
**Holdout status:** CLOSED

This rule is written before the current broad-search probe is interpreted.

## Why a new rule is necessary

The original optimizer profile (32 Sobol candidates, four L-BFGS-B polish starts) produced materially different sub-001 optima across GitHub runner-image releases despite:

- byte-identical scientific code;
- the same frozen clean epochs;
- the same forward model;
- the same reconstructed network;
- pinned Python package versions;
- one-thread execution.

Therefore numerical stability, not scientific-model performance, must determine the final optimizer profile.

## Historical training-objective targets

Across the two incompatible sub-001 executions, retain the best observed **training** objective for each model/direction:

| Model | Train block | Best observed normalized training score |
|---|---|---:|
| M2 | A | 535.8336527151650 |
| M2 | B | 534.0933137283831 |
| M3 | A | 535.8430963300440 |
| M3 | B | 533.8829279760863 |

These targets are used only to detect whether a candidate optimizer fails to recover a basin already known to exist.

The sign or magnitude of held-out DeltaELPD is **not** an optimizer-selection target.

## Profile P1 - currently running

- seed 97;
- Sobol m=8 = 256 initial candidates;
- eight L-BFGS-B polish starts selected by training objective;
- existing parameter bounds;
- existing L-BFGS-B stopping settings;
- one BLAS/OpenMP thread;
- three OpenBLAS core kernels: Haswell, Sandybridge, Zen;
- same sub-001 data and two chronological CV directions.

### P1 PASS criteria

All conditions must hold:

1. all four model/direction optimizations report success for every kernel;
2. every kernel recovers each historical best training basin within 0.0005 normalized score;
3. for each model/direction, the range of optimized training scores across the three kernels is <= 0.001;
4. for each model, the range of the resulting two-direction CV ELPD across kernels is <= 0.005;
5. all outputs are finite and satisfy the frozen parameter bounds.

Criterion 4 is a numerical reproducibility criterion, not a preference for a favorable M3-M2 effect. M2 and M3 are assessed separately.

If P1 passes, P1 becomes the final numerical profile. No cheaper profile is selected after seeing held-out development effects. Production development and holdout fits will then force `OPENBLAS_CORETYPE=Haswell`; Sandybridge and Zen are robustness kernels only.

## Predetermined escalation P2

If P1 fails any criterion, do not open the holdout.

Run exactly one broader profile:

- seed 97;
- Sobol m=9 = 512 initial candidates;
- sixteen L-BFGS-B polish starts;
- same bounds/stopping rules;
- one thread;
- the same three OpenBLAS kernels (Haswell, Sandybridge, Zen);
- the same PASS criteria above.

If P2 passes, P2 becomes the final numerical profile. Production development and holdout fits will force the already predeclared `OPENBLAS_CORETYPE=Haswell`.

## If P2 fails

Do not continue escalating starts ad hoc and do not open the holdout.

A new explicit numerical-method protocol must then be written before further testing, potentially using a different global optimizer. The scientific model remains frozen.

## Final freeze requirement

After P1 or P2 passes:

1. record all three kernel outputs and artifact digests;
2. update the fit driver to the passing profile;
3. update validation assertions;
4. commit a new final numerical freeze;
5. rerun all 34 QC-passed development subjects from scratch;
6. aggregate and audit all 136 model/direction optimizer calls;
7. only then sign off development and open the holdout.

No profile may be chosen because its development DeltaELPD is larger, more positive, more significant, or otherwise more favorable to SMM.


## Infrastructure correction before P1 interpretation

The first attempted P1 workflow included an `OPENBLAS_CORETYPE=SkylakeX` arm. That arm terminated during environment import with CPU illegal-instruction exit code 132 before any SMM fitting or probe score was produced. The hosted runner therefore cannot safely execute the SkylakeX kernel.

Before interpreting the still-running Haswell/Zen probe outputs, the kernel set was corrected to:

- Haswell;
- Sandybridge;
- Zen.

This is an infrastructure compatibility correction only. No model, data, optimizer profile, threshold, training target, or PASS criterion changed. The corrected three-arm P1 is rerun from scratch.

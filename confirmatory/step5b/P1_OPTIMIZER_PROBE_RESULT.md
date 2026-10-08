# Step 5B P1 optimizer robustness result

**Date:** 2026-10-06  
**Decision:** P1 FAIL  
**Holdout status:** CLOSED  
**Escalation:** exactly to predeclared P2

## Provenance

- Workflow run: `37499465520`
- P1 profile: 256 Sobol candidates, 8 L-BFGS-B polish starts
- Kernels: Haswell, Sandybridge, Zen
- Decisive completed arm: Haswell
- Haswell job ID: `112393128859`
- Scientific/model code: unchanged from the pre-P1 lock
- Selection basis: training-objective recovery/reproducibility only

## Decisive Haswell result

The broad optimizer probe itself completed successfully. The subsequent
predeclared recovery assertion failed because M2 did not recover two historical
best training basins within tolerance 0.0005.

| Model | Train | P1 Haswell training score | Historical best | Recovered? |
|---|---|---:|---:|---|
| M2 | A | 535.7596930646307 | 535.8336527151650 | NO |
| M2 | B | 534.0061847445946 | 534.0933137283831 | NO |
| M3 | A | 535.8929500896384 | 535.8430963300440 | YES |
| M3 | B | 533.9434636648306 | 533.8829279760863 | YES |

For audit only, not profile selection:

- M2 held-out directions: 533.8621106072561 and 535.8040522700745
- M3 held-out directions: 533.9092498159005 and 535.7505175421107
- DeltaELPD(M3-M2): -0.0031977596596561852

The Delta value is recorded only for transparency. It played no role in the
P1 failure decision.

## Why P1 fails immediately

The frozen P1 rule requires **every kernel** to recover **every historical best
training basin** within 0.0005. Haswell failed that condition in M2 A and M2 B.

Therefore:

[
\boxed{P1=FAIL}
]

regardless of the eventual Sandybridge/Zen values.

The remaining P1 arms may finish for audit, but cannot change the decision.

## Required next action

Per `OPTIMIZER_ROBUSTNESS_DECISION_RULE.md`, run exactly P2:

- seed 97;
- Sobol m=9 = 512 candidates;
- 16 L-BFGS-B polish starts;
- same bounds and stopping rules;
- one thread;
- Haswell, Sandybridge, Zen;
- identical pass criteria.

No scientific-model change is authorized. The holdout remains closed.


## Full three-kernel audit

All three P1 kernel jobs ultimately failed the same predeclared basin-recovery requirement.

### Haswell

- M2 A train: 535.7596930646307 vs historical 535.8336527151650 -> FAIL
- M2 B train: 534.0061847445946 vs historical 534.0933137283831 -> FAIL
- M3 A train: 535.8929500896384 vs historical 535.8430963300440 -> PASS
- M3 B train: 533.9434636648306 vs historical 533.8829279760863 -> PASS
- audit-only DeltaELPD: -0.0031977596596561852

### Zen

Zen reproduced the Haswell P1 solution numerically for all four directions:

- M2 A train: 535.7596930646307 -> FAIL historical-basin recovery
- M2 B train: 534.0061847445946 -> FAIL historical-basin recovery
- M3 A train: 535.8929500896384 -> PASS
- M3 B train: 533.9434636648306 -> PASS
- audit-only DeltaELPD: -0.0031977596596561852

### Sandybridge

- M2 A train: 535.7607731393065 -> FAIL historical-basin recovery
- M2 B train: 533.9475359791719 -> FAIL historical-basin recovery
- M3 A train: 535.8929502096953 -> PASS
- M3 B train: 533.9934375840123 -> PASS
- audit-only DeltaELPD: +0.08385537300182477

The differing audit-only DeltaELPD on Sandybridge is additional evidence that P1 is not numerically robust enough for production, but the formal P1 failure remains the predeclared training-basin recovery failure, not the sign of DeltaELPD.

Final P1 run conclusion: `failure`.

All three jobs completed, so the P1 decision record is now closed.

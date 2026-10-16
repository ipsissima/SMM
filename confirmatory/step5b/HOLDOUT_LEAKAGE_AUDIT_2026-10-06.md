# Step 5B holdout-leakage audit

**Date:** 2026-10-06  
**Holdout gate:** CLOSED  
**Assigned confirmatory holdout:** sub-044..608

## Conclusion

No confirmatory holdout EEG signal has entered the Step 5B preprocessing,
development fitting, optimizer selection, or manuscript result generation.

The holdout remains unconsumed.

## Evidence

### 1. No holdout workflow has ever run

At this audit point, GitHub Actions contains **zero runs** of:

`.github/workflows/step5b-confirmatory-holdout.yml`

The prepared workflow is manual-only and additionally requires the permanent
`HOLDOUT_GATE.json` to be OPEN.

### 2. The gate is still structurally closed

`confirmatory/step5b/HOLDOUT_GATE.json` currently has:

- `status = CLOSED`;
- `development_run_id = null`;
- `development_aggregate_artifact_id = null`;
- `development_aggregate_sha256 = null`;
- `numerical_freeze_commit = null`.

`NUMERICAL_PROFILE.json` is also still `PENDING`, so the holdout gate could
not validly open even by manual dispatch.

### 3. Canonical development EEG artifacts are development-only

Canonical primary-development preprocessing run:

`37438628533`

Its successful EEG artifacts are exactly the 34 frozen-QC-passed development
subjects:

`sub-001,002,003,004,005,006,007,010,011,014,015,016,017,018,019,020,021,022,023,024,025,028,029,030,031,032,033,034,035,036,038,039,040,042`.

No artifact for sub-044 or later exists in that run.

The four frozen-QC development failures were sub-012, sub-026, sub-041 and
sub-043; the five predeclared metadata-anomalous subjects were sensitivity-only.

### 4. Optimizer robustness work is sub-001 only

The original reproducibility test, P1 robustness probe, and P2 robustness probe
all use the already exposed development subject `sub-001`.

P1 run: `37499465520`  
P2 run: `37503947541`

Optimizer profile selection therefore cannot contain holdout signal information.

### 5. The pre-EEG document's sub-044 reference is metadata-only

The pre-EEG lock states that a spot check of sub-001, sub-021, sub-043 and
sub-044 showed the same 64-channel list and identical **channel-file Git SHA**.
This was a dataset-tree/channel-schema check, not inspection, preprocessing,
feature extraction, fitting, visualization or scoring of sub-044 EEG signal.

The holdout boundary explicitly remained sub-044..608 after that metadata check.

### 6. Forward-model construction contains no EEG

The canonical forward gate was built from template head/cortex/montage geometry
and DK68 assets and was validated to contain no EEG or subject-holdout signal.

## Permitted pre-opening information

The following do not constitute holdout signal consumption:

- public dataset tree structure;
- filenames;
- acquisition-condition availability;
- channel-schema metadata;
- file/git-annex hashes and byte sizes;
- subject IDs used solely to define the frozen split.

The forbidden information before opening is holdout EEG signal or any derived
signal-dependent quantity used for preprocessing tuning, feature choice, model
selection, optimizer selection, exclusion-threshold changes or manuscript
effect interpretation.

No such signal-dependent holdout information has been used.

## Freeze implication

P1/P2 numerical work remains legitimate development because it uses sub-001
only and chooses the numerical method from training-objective robustness rather
than SMM-favorable held-out effects.

The next legitimate signal access to sub-044..608 remains the single frozen
confirmatory execution after:

1. final numerical profile freeze;
2. complete fresh 34-subject development rerun;
3. development sign-off;
4. provenance-only opening of `HOLDOUT_GATE.json`.

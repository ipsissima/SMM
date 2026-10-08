# Step 5B post-open amended analysis — execution register

**Source classification:** `POST_OPEN_AMENDED`, never `FROZEN_CONFIRMATORY_PASS`.

## Original immutable primary analysis

- Original run: https://github.com/ipsissima/SMM/actions/runs/37752719639
- Original scientific executable: `20fab272edf33d2513470c60ac9398754535cacb`
- Original frozen verdict: `NOT EVALUABLE`, four unexpected preprocessing stops.
- Original batch B (188 assignments) continues to execute original frozen N1 fits. No partial M2/M3 result was used to define this amendment.

## Post-open justification and metadata evidence

- Blind four-EDF diagnostic runs: `37772814067`, `37773401991`, both four jobs successful.
- Four-EDF forensic provenance: `diagnostics/step5b_2026_10_08/FORENSIC_FINDINGS_2026-10-08.md` on `step5b-incident-diagnostics-2026-10-08`.
- Locked decisions: `amended/step5b/AMENDMENT_LOCK_2026-10-08.md`
- `sub-075` A1 technical preprocessing run `37786081008`, success, channel reorder proven, 49 clean four-second epochs, artifact `SMM_STEP5B_AMENDED_PREPROCESS_sub-075`, artifact ID `11554432748`.
- Synthetic gate run `37786665225`, success, exactly 565 records and three separately classified structural inputs; identical untouched frozen CI/sign-flip functions.

## Execution underway

- Amended A/C fit workflow run: https://github.com/ipsissima/SMM/actions/runs/37787000597
- Source commit at dispatch: `0f1fb40addc047c3719ea3efa2b9317fbdc5e8f7`
- Batch A: 186 subjects (`sub-044..231`, omitting `sub-206`, `sub-230`); includes amended `sub-075`.
- Batch C: 188 subjects (`sub-420..608`, omitting `sub-425`).
- Reused frozen preprocessing from original run: 373 A/C + 188 B = 561 subjects. The original 561 artifacts were explicitly verified by the successful amended-gate job.
- Original frozen batch B fits: 188 (to be reused only once all have valid completed artifacts).
- Separate `.ineligible.json` records, locked prior to renewed fitting: `sub-206`, `sub-230`, `sub-425`.
- Mathematical account: `374+188+3=565`, and normal predeclared frozen-QC exclusion records may occur within the 562 A/C+B records.
- Amended aggregate step MUST fail closed if the 188 original B fit/exclusion artifacts are not yet complete, any fit fails, provenance mismatches, QC changes, duplicate or missing subject, or numerical/nesting criterion drifts.

## Results status as of initial launch

`FITTING IN PROGRESS; NO AMENDED GROUP STATISTICS AVAILABLE YET.`

Expected final artifact, when legitimate: `SMM_STEP5B_POST_OPEN_AMENDED_RESULT`, with `amended_primary_result.json`, `amended_primary_result.md`, `amended_subject_results.csv` and `amended_optimizer_diagnostics.csv`.

**Do not report a post-open amended criterion as a PASS of the original frozen primary.** The successful outcome, if any, would not replace the need for an independent prospectively untouched replication.

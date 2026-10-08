# Step 5B blinded EDF incident audit (non-scientific, read-only)

Original blocked confirmatory run: https://github.com/ipsissima/SMM/actions/runs/37752719639, SHA `20fab272edf33d2513470c60ac9398754535cacb`.

This branch adds only an independent forensic reader and a **manual** GitHub Actions workflow to investigate the four unexpected preprocessing stops in the original run. No model, data-split, forward solution, optimizer, QC criterion, threshold, prediction, original workflow, or inferential code is changed. No `fit` is run by this workflow, and no subject is excluded, corrected, imputed, rescaled, or reordered.

| Subject | Stop from original job logs | Present conclusion |
|---|---|---|
| sub-075 | Channel names/order differs from frozen list | BIDS lists same 64 names with Fp1/Fp2 swapped; actual EDF still needs checking |
| sub-206 | 1–45 Hz QC abs p999 = 10141.688 uV, above frozen 5000 uV | Potential genuine artifact or invalid EDF scale; not yet established |
| sub-230 | Recording duration less than 180s | BIDS duration=60s: **does not meet frozen minimum** |
| sub-425 | 1–45 Hz QC RMS median = 1298.395 uV; p999 = 15064.722 uV | Potential genuine artifact or invalid EDF scale; not yet established |

The source dataset README warns that EDF physical min/max metadata may be invalid. An arbitrary correction or re-scaling based on holdout is not permitted.

## Workflow

`.github/workflows/step5b-incident-edf-diagnostics.yml` is manual only. It downloads exactly four pinned EDFs plus their BIDS channel/JSON metadata, independently verifies the git-annex SHA256 and bytes, reads EDF digital samples **without modifying them**, and saves one JSON report per subject. There is no model fitting or inference and no reused holdout result.

**GitHub caveat:** GitHub may require a workflow_dispatch workflow to exist on the default branch before the manual Run workflow button is available. This PR deliberately does not merge or start anything automatically.

## Scientific stopping rule

The original primary confirmatory run remains blocked with no frozen PASS/FAIL aggregate. Its scientific verdict is **NOT EVALUABLE**, not PASS or FAIL. Do not rerun non-infrastructure failures. The diagnostic reports may establish causes, but do not themselves authorize a post-opening correction to any frozen gate, including relaxing channel order, duration, amplitude limits, or exclusion handling. Scientific assessment of any proposed correction must be independent of M2/M3 results and explicitly account for the post-opening change.

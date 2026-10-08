# Step 5B same-dataset amended completion — irrevocable decision log

**Analysis classification:** POST-OPEN AMENDED / not the untouched frozen primary confirmation
**Documented:** 2026-10-08, before inspecting any confirmatory M2/M3 contrast to decide these cases.
**Original holdout executable and lock:** \`20fab272edf33d2513470c60ac9398754535cacb\` / \`SMM_STEP5B_PRE_EEG_LOCK_2026-10-05\`.
**Frozen dataset:** OpenNeuro \`ds005385\` commit \`6a558cd5852503e66df9ac7fdac2f3a7f4ed5f12\`; 565 assignments \`sub-044..608\`.
**Original run:** \`37752719639\` — never overwritten, rerun or relabeled. Unexpected failure accounting preserved.
**Blinded forensic audit:** isolated \`step5b-incident-diagnostics-2026-10-08\` branch; full \`FORENSIC_FINDINGS_2026-10-08.md\`, 4/4 verified EDF hash and byte size, diagnostic runs \`37772814067\` and \`37773401991\`.

## Scientific settings that MUST remain frozen

Do not edit any tracked file in \`confirmatory/step5b\` or any original reusable workflow.
Use exact original EDFs, subject boundary, forward model, U20, N1 optimizer and pinned numerical environment, M2/M3 generative models, raw amplitude and QC thresholds, filter definitions, CSD, likelihood, chronological CV, seed 97, 10,000 subject-bootstrap CI and 100,000 sign flips. Success condition unchanged: CI_low > 0 AND one-sided p < 0.05.

Original immutable blob references include preprocessing \`797c7334c4431addf87c15bc6009b98ae24fd6cd\`, frozen N1 fit \`f2276f20ff6eb20666634b2b20fa21b66288780d\`, group aggregator \`164ecf6f5ce55d7a71a4b78efe12305543755eed\`, profile \`fa13dd5f898f7ef6fc5c289fc7a0c851e2cd59a1\` and channel list \`3b335f5f334a49c92ba46ea82b21d996f8a693a2\`.

## Post-opening amendments, chosen from input diagnostics only

**A1 — channel permutation for sub-075.** Real EDF and BIDS each list the exact same 64 frozen EEG electrode names, with *only first two in reverse order*. Correct the **channel object order (not just names)** from \`Fp2,Fp1,...\` to \`Fp1,Fp2,...\` on a separate amended preprocessing entrypoint, and then apply *unaltered* frozen preprocessing parameters, QC, filtering, ICA, epoching. No other channel issue may be auto-corrected. Any other failure stops the amended analysis, not an automatic new exclusion.

**A2 — explicit non-QC structural ineligibility.** The following inputs stop before the original ordinary Primary QC check. Account for them separately with verified original EDF hashes and exact observed frozen hard-gate failure reasons. Do not call them frozen \`Primary QC failed:\` cases and do not assign DeltaELPD:
- \`sub-206\`: 1–45 Hz QC amplitude p99.9 above frozen 5,000 uV; exact channel order, intact EDF; direct ADC-bound evidence.
- \`sub-230\`: EDF and BIDS exact 60 seconds, below frozen 180-second minimum.
- \`sub-425\`: 1–45 Hz QC amplitude RMS median and p99.9 above frozen limits; actual prolonged C1 digital-bound interval, intact EDF.
These classifications are **post-open structural-ineligibility accounting amendments**, not permitted under the original aggregator. They must appear as such in output and exclusion summary.

**A3 — reuse already completed identical preprocessing** from original run \`37752719639\` for all other 561 assigned subjects (except 075, 206, 230, 425). Frozen QC inclusion outcomes remain untouched. No reread or reprocessing of these inputs. Reuse exact original fit artefacts for B (232..419), but only if all 188 subjects have a valid original fit or original frozen-QC-exclusion record. No selective replacement based on model performance.

**A4 — inference implementation.** Call frozen \`aggregate_confirmatory\` functions for fit validation, 10k bootstrap, and 100k sign flip. The only new logic is recognizing the three pre-recorded \`.ineligible.json\` records in 565-assignment accounting. The output is \`amended_primary_result.json\`, not \`confirmatory_primary_result.json\`, with \`analysis_classification=POST_OPEN_AMENDED\`, \`untouched_frozen_primary_verdict=NOT_EVALUABLE\`, \`n_structural_ineligible_post_open=3\`, and the unchanged frozen numerical contrast criterion as an **amended criterion result**, not the original primary verdict.

## Mandatory failure guards

- Original 565 subject assignments; no duplicates, substitutions or deleted assignments.
- Only exactly three specified structural exclusions with exact subject identity, frozen failure class, original SHA-256 and matching provenance.
- \`sub-075\` must complete the frozen QA checks after the one-to-one channel permutation or analysis stops.
- All other records must be original frozen QC exclusions or fully validated N1 model fits.
- A/C fit results must use frozen scientific fit code with original subject epochs, unless \`sub-075\` which is explicitly corrected and logged.
- B fit results must be from original run \`37752719639\` only, with full count 188/188; missing B artifact stops computation.
- The amended inferential functions must remain identical; if they are not, stop.
- Publish comprehensive subject accounting, optimizer results, exact inputs and sha256 hashes, and both successes and anomalies.
- Never inspect original partial B ELPD or aggregate results to alter the above plan.

## Scope of the claim

This produces an amended *same-dataset* analysis, not an independent prospectively untouched primary confirmatory replication. Reporting a positive amended criterion does not justify describing the original frozen holdout as PASS. A negative result must be reported as such rather than post-hoc modified.

# Step 5B confirmatory holdout: blinded EDF incident findings

**Date:** 2026-10-08 (UTC)  
**Scientific protocol:** `SMM_STEP5B_PRE_EEG_LOCK_2026-10-05`, frozen N1 numerical profile  
**Primary holdout:** `sub-044..608` (565 assignments), `ses-1 / EyesClosed / acq-pre`  
**Original execution:** [run 37752719639](https://github.com/ipsissima/SMM/actions/runs/37752719639), original head `20fab272edf33d2513470c60ac9398754535cacb`  
**Diagnostic source:** isolated branch `step5b-incident-diagnostics-2026-10-08`; original protocol and workflow unchanged  
**Read-only diagnostic runs:** [37772814067](https://github.com/ipsissima/SMM/actions/runs/37772814067) and [37773401991](https://github.com/ipsissima/SMM/actions/runs/37773401991), both four-subject matrices 4/4 success  
**Enhanced diagnostic code commit:** `acada8e70c6912321e44bed7d61415e8199710b0`

## Scope and non-intervention

The diagnostic workflow downloaded only four exact, frozen-snapshot EDFs and their same-commit BIDS channel and recording metadata. Every EDF was verified against its git-annex SHA-256 and byte count. The independent digital/header reader never reorders, filters, rescales, replaces, interpolates, excludes, fits, or computes M2/M3 predictive contrasts. Synthetic EDF parsing, channel-swap detection, limit-run accounting, and hash-mismatch rejection were checked before inspecting actual inputs.

**No scientific parameter, cutoff, model, split, optimizer, QC criterion, likelihood, inferential rule, or subject accounting was changed.** These reports do not authorize retrospective exclusions or post-hoc preprocessing modifications.

## Four unexpected original preprocessing stops

### `sub-075` — exact channel-order anomaly

- Verified EDF SHA-256: `bc2cf91a253a490342d849f39baba530bbe4a58b5c2ea86fb78db08e037fcb02`.
- BIDS and EDF independently agree on the observed 64 EEG channel names/order.
- EDF's observed positions 1–2 are `Fp2, Fp1`; frozen canonical positions 1–2 are `Fp1, Fp2`. **All 64 channel names are present, with no other ordering mismatch.**
- EDF/BIDS duration: 207 s.
- Digital-minimum signal suffix occurs for up to 0.440 s near the end (e.g. 206.560 s for Fp1/Fp2), within the two seconds removed by the frozen edge crop.
- Original stopping point: `preprocess_ds005385.py:62`, `Channel names/order differ from frozen 64-channel ...`.
- **Interpretation:** verified representation anomaly. A permutation would restore canonical labels but is currently *not* permitted by the frozen executable's fail-closed channel-order condition. It must not be silently introduced after opening the holdout.

### `sub-230` — minimum-duration violation

- Verified EDF SHA-256: `cd11d262a93de7847828d3cb8fb75b7a5b5481b767b555aca35cb4f6a185e40b`.
- EDF and BIDS each report exactly **60 seconds**, below the frozen **180-second minimum**.
- Original stopping point: `preprocess_ds005385.py:66`, `Recording shorter than 180 s before frozen 2-s edge crop.`
- **Interpretation:** unambiguously ineligible for the original locked duration requirement. The current executable stops unexpectedly rather than producing the ordinary `Primary QC failed:` exclusion record; changing this accounting behavior after opening requires explicit methodological adjudication.

### `sub-206` — digital-extreme samples and amplitude-sanity stop

- Verified EDF SHA-256: `9a25d6079631d57c1f21c0c8a8cd3ff4f9d29e58824ef598c54a40291c4263c4`.
- EDF/BIDS duration: 211 s; exact canonical 64-channel order.
- Original 1–45 Hz sanity: median channel RMS **482.976 µV** (under 500 µV maximum), absolute p99.9 **10,141.688 µV** (over 5,000 µV maximum).
- Highest digital-limit occupancy occurs in `AF3`: **0.993365%** of its samples at an EDF digital endpoint, split **0.590% lower / 0.403% upper**, with longest lower and upper runs **0.360 s / 0.175 s**, first observed around 5.83 s.
- EDF EEG physical-digital gain shown in header: 0.499992 µV / digital count. Endpoint occupancy is a *digital observation*, independent of any questionable physical unit conversion.
- Original stopping point: `preprocess_ds005385.py:75`, unexpected `EDF amplitude sanity check failed`.
- **Interpretation:** amplitude check fails for a reproducible non-transient input. Digital endpoints indicate possible clipping or special-coded samples. The current audit cannot safely attribute cause or prescribe rescaling.

### `sub-425` — prolonged digital-minimum interval and amplitude-sanity stop

- Verified EDF SHA-256: `7d9cc41b0e7e8c4d2a7a9630da1c2af0c83a7a38eace92e8452d771d06818fb8`.
- Exact canonical 64-channel order. **EDF duration: 213 s; BIDS duration: 212 s**, a metadata disagreement that does not affect the minimum-duration requirement.
- Original 1–45 Hz sanity: median channel RMS **1,298.395 µV** (over 500 µV maximum), absolute p99.9 **15,064.722 µV** (over 5,000 µV maximum).
- Channel `C1`: **17.569014%** of all digital samples at an EDF endpoint, of which **17.517% lower** and **0.052% upper**. Its longest uninterrupted run at the exact lower digital limit is **32.129 seconds**; first such sample is at approximately 83.452 s.
- **15 EEG channels** individually exceed 1% digital-endpoint occupancy.
- EDF EEG physical-digital gain shown in header: 0.499992 µV / digital count.
- Original stopping point: `preprocess_ds005385.py:75`, unexpected `EDF amplitude sanity check failed`.
- **Interpretation:** strong direct evidence of a sustained digital-limit anomaly, not a routine network timeout or download corruption. It is not justified to discard the affected interval, replace the signal, or relax the frozen amplitude gate.

## Permanent diagnostic artifact pointers

| Subject | Enhanced audit artifact ID | Input SHA-256 (prefix) |
|---|---:|---|
| `sub-075` | `11548619057` | `bc2cf91a` |
| `sub-206` | `11547534815` | `9a25d607` |
| `sub-230` | `11548855409` | `cd11d262` |
| `sub-425` | `11549350085` | `7d9cc41b` |

The artifacts contain all 64 per-channel physical/digital ranges, observed digital extrema, raw digital-run durations and positions, and provenance metadata. They are *forensic reports*, not confirmatory fits.

## Scientific verdict and exact stop

All four original EDF downloads passed git-annex integrity checks before preprocessing. The original run therefore contains **four non-transient, input- or implementation-related preprocessing failures**. The frozen primary workflow requires *all preprocessing jobs in a batch* to succeed before launching its batch fits. Consequently, batches **A** and **C** are skipped, and the final aggregate requires all three fit batches and cannot validly run. At the most recent check the original run was still in progress completing batch B; no primary confirmatory aggregate artifact existed.

**Primary Step 5B scientific verdict: NOT EVALUABLE / NO FROZEN PASS OR FAIL.** Do not reinterpret missing inference as evidence for or against SMM.

The pre-holdout model was never compared with M2/M3 outcomes by this diagnostic investigation. No rerun of a scientifically implicated failed job is justified merely because GitHub will permit it.

## Decision boundary for recovery

There is currently **no zero-change pathway to a complete frozen primary aggregate** on this assigned dataset and executable. Restoring subject `sub-075` would require an explicit channel-order handling amendment; coding `sub-230` as an ordinary exclusion would amend unexpected-error accounting; coding `sub-206` and `sub-425` as QC exclusions or correcting their scale would require separate justification under the original lock. None is authorized by this report.

A post-opening technical amendment, if independently adjudicated, must remain traceable, model-blind and uniformly specified, with original failures preserved. Any resulting 565-assignment reanalysis must be explicitly labeled **amended analysis**, not silently substituted for the frozen primary run. Strictly new confirmatory status would require an independent held-out EEG dataset/cohort under a newly locked, audited executable defined *before* inspection of its outcomes.

## Separation and publication integrity

Keep the original run, freeze commits, failure logs, and diagnostic run IDs. Do not merge this draft PR into the scientific branch until reviewed; its base is the pre-existing `step5b-confirmatory-gate-2026-10-05` branch and it contains diagnostic-only files. Do not inspect the partial batch-B ELPD results to decide how to handle these input anomalies.

# Step 5B permanent provenance manifest

**Repository:** ipsissima/SMM  
**Branch:** step5b-confirmatory-gate-2026-10-05  
**Purpose:** permanent lightweight map from paper-facing claims to code, runs, hashes, and heavyweight artifacts.

## Scientific locks

- Pre-EEG lock ID: `SMM_STEP5B_PRE_EEG_LOCK_2026-10-05`
- Original freeze record: `confirmatory/step5b/FREEZE_COMMIT.txt`
- Development numerical freeze: `confirmatory/step5b/DEVELOPMENT_NUMERICAL_FREEZE_2026-10-06.md`
- Confirmatory holdout gate: `confirmatory/step5b/HOLDOUT_GATE.json`
- Development aggregation code: `confirmatory/step5b/aggregate_development.py`
- Confirmatory aggregation/inference code: `confirmatory/step5b/aggregate_confirmatory.py`

## Frozen public dataset snapshot

- OpenNeuro dataset: `ds005385`
- Git snapshot: `6a558cd5852503e66df9ac7fdac2f3a7f4ed5f12`
- Primary condition: `ses-1 / EyesClosed / acq-pre`
- Development exposure boundary: `sub-001..043`
- Confirmatory holdout: `sub-044..608`
- Primary metadata-clean development set: 38 subjects
- QC-passed development fit set: 34 subjects
- Prespecified metadata sensitivity-only subjects: `sub-008, sub-009, sub-013, sub-027, sub-037`
- Frozen-QC development failures: `sub-012, sub-026, sub-041, sub-043`

## Canonical forward model

Canonical forward run: `37376954431`  
Artifact: `SMM_STEP5B_FORWARD_GATE`  
Artifact ID: `11371817671`  
Artifact digest: `sha256:e6d3ff97202c3fc8276c467ee00cd4fb0f07c88ab2bba14d67834794a2ef9ef0`

Frozen material hashes:

| Asset | SHA256 |
|---|---|
| fsaverage_vertex_forward-fwd.fif | `907f7d75e31a921f5156048b45cb0736efa7320ab8b903b7cadbd5fc3d082a75` |
| fsaverage_fixed_surface_normal_average_ref_vertex_leadfield.npy | `c5172c6ed56368d52e4128704e5d6131a694bcca09546e87f1eee5e5c43826a7` |
| leadfield_DK68_64x68.csv | `86afbddbd637ef4239b5755ef3853aafb62890ac989340fd093482ad7c9853b2` |
| DK68_regional_surface_area_mm2.csv | `01ef575dc4660b236a00e49bb41b0860ae034310269b95dbdc94755ce252648e` |
| leadfield_U20.npy | `17eb1218c6cfb8b3e27ed51283e91d3e7068261a9c7595b6da2b2e358e3b8891` |
| DK68_NETWORK_ORDER.txt | `a2feee096aba9fafa822f7a51a85577c4af2ee9117dfc4ac9fb4f8df46ab7779` |
| channels_64.txt | `6445258f699531b1a7bd57401a6343a436a99493b83dc0dbed21b242889bd348` |

## Frozen network reconstruction

Upstream sources are vendored under `confirmatory/step5b/upstream_sources/`.

Expected reconstructed hashes:

| Asset | SHA256 |
|---|---|
| W_ENIGMA_DK68_spectral_normalized.csv | `9f83f8ea5ba38aa68094739395db225e2e53c1b8541fbe8d2a9a6295dc2bd210` |
| DK68_centers_proxy_mm.csv | `5857bc9473c84574e2105a464b2475676ba2fe1c349500088d2bb6308ac0384a` |
| DK68_edges_with_delay_proxy.csv | `16604f620233b3f19ff6f1917f7f62589e8b3b4ea343bf8790b38e6871b72ff1` |

## Development preprocessing

Canonical full primary-development preprocessing run: `37438628533`

Outcome:

- 34/38 pass
- `sub-012`: 13 global bad channels > frozen maximum 10
- `sub-041`: 11 global bad channels > frozen maximum 10
- `sub-043`: 11 global bad channels > frozen maximum 10
- `sub-026`: 21 clean 4-s epochs = 84 s < frozen minimum 30 epochs / 120 s

The amplitude-sanity implementation amendment preserved the original numerical thresholds but applied them to the prespecified 1-45 Hz QC stream because ds005385 documents invalid EDF physical min/max metadata and raw DC offsets. The correction is part of the auditable development history; no threshold was relaxed.

## Numerical reproducibility history

### Provisional same-image gate

Run: `37483948147`  
Commit: `5ecc084d4d193629657052c9f69448b8e0327850`  
Runner image: `ubuntu-24.04 / 20260927.320.1`

Three independent single-thread runs of `sub-001` returned exactly:

- M2 CV ELPD = `534.8237844922861`
- M3 CV ELPD = `534.7535949015517`
- Delta ELPD(M3-M2) = `-0.07018959073445785`

Identical optimizer diagnostics:

- M2 A: nit 29, nfev 504
- M2 B: nit 52, nfev 816
- M3 A: nit 34, nfev 378
- M3 B: nit 34, nfev 282

This initially motivated a single-thread implementation freeze, but that freeze was subsequently **superseded before holdout opening**.

### Cross-runner-image failure

Run `37495554793`, on runner image `ubuntu-24.04 / 20261004.327.1`, used byte-identical scientific/fit code, the same frozen sub-001 epochs, the same forward model, the same network reconstruction, the same pinned Python package versions and the same one-thread environment, but returned:

- M2 CV ELPD = `534.8734362017726`
- M3 CV ELPD = `534.7059328161354`
- Delta ELPD(M3-M2) = `-0.16750338563724654`

The provisional freeze is therefore not final. Run `37495554793` is a numerical diagnostic and is not the canonical 34-subject development result.

A development-only optimizer robustness probe now tests a broader search using training-objective recovery across deliberately different OpenBLAS kernels. No scientific-model setting has been changed, and the holdout remains closed.

### Final numerical freeze

- final optimizer profile: **PENDING**
- P1 robustness-probe run ID: `37499465520` (**currently running; verdict pending**)
- final numerical-freeze commit: **PENDING**
- canonical 34-subject rerun: **PENDING**

## Canonical development fit

The previous 34-subject attempt `37495554793` was cancelled/superseded after its first completed sub-001 result exposed cross-runner-image optimizer instability. It is not canonical.

After the optimizer robustness probe passes and a new final numerical freeze is committed, the definitive 34-subject workflow is:

`.github/workflows/step5b-development-final.yml`

Its aggregate artifact must be named:

`SMM_STEP5B_DEVELOPMENT_AGGREGATE`

When complete, record here:

- run ID: **PENDING**
- commit SHA: **PENDING**
- aggregate artifact ID: **PENDING**
- aggregate artifact digest: **PENDING**
- permanent aggregate files committed to repository: **PENDING**

## Confirmatory holdout

The primary holdout workflow is already prepared but hard-gated:

`.github/workflows/step5b-confirmatory-holdout.yml`

Reusable jobs:

- `.github/workflows/step5b-reusable-preprocess-one.yml`
- `.github/workflows/step5b-reusable-fit-one.yml`

The holdout cannot execute unless both conditions hold:

1. `confirmatory/step5b/HOLDOUT_GATE.json` has `"status": "OPEN"` and complete development provenance;
2. the manual workflow dispatch explicitly supplies `OPEN-HOLDOUT`.

At the time of this manifest revision, the gate is **CLOSED**.

The holdout workflow is required to account for every `sub-044..608` assignment. Expected frozen primary-QC failures are represented structurally rather than causing batch-wide failure; unexpected preprocessing/infrastructure failures still fail closed. Group inference is applied only to the frozen-QC-included recordings, exactly as implied by the pre-EEG primary inclusion rules.

## Frozen primary confirmatory rule

All 565 assigned holdout subjects are retained in the accounting record. The pre-EEG lock also froze primary QC/inclusion criteria, including >=30 clean 4-s epochs. Subjects failing those criteria are recorded as frozen-QC exclusions and are not replaced.

For each frozen-QC-included holdout subject:

[
Delta_i = ELPD_i(M_3)-ELPD_i(M_2).
]

Specific SMM predictive success requires both:

1. 10,000-resample subject bootstrap 95% CI lower bound > 0;
2. one-sided paired sign-flip test with 100,000 permutations gives p < 0.05.

Seed = 97.

The executable implementation is `confirmatory/step5b/aggregate_confirmatory.py`.

[
M_3>M_0
]

alone is explicitly not sufficient for SMM-specific success.

## Permanent-output rule

Large FIF/forward derivatives may remain GitHub Actions artifacts or external derivatives. The repository must permanently contain:

- frozen code;
- subject-level result CSVs;
- optimizer diagnostics;
- group summary JSON/Markdown;
- artifact/run/hash manifest;
- final numerical freeze;
- confirmatory verdict;
- replication summaries;
- exact manuscript-facing figure/table source data.

No paper number should exist only in a transient log.

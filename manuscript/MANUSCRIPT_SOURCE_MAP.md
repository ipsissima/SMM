# Definitive SMM manuscript source map

This map prevents the integrated paper from silently upgrading design assumptions into executed results. For each main manuscript claim, the canonical source and epistemic status are listed.

| Manuscript claim / section | Canonical source | Status to use in paper |
|---|---|---|
| Original SMM continuity and what is abandoned | `theory/definitive/SMM_GENEALOGY_ORIGINAL_TO_DEFINITIVE.md` | Historical/theoretical reconstruction |
| Frontiers compatibility; control-field interpretation | `theory/definitive/SMM_DEFINITIVE_FOUNDATIONS_STEP1.md` | Canonical conceptual architecture |
| Choice of QIF/MPR E-I backbone | `theory/definitive/SMM_DEFINITIVE_STEP2_NEURONAL_CORE.md` | Model-design decision |
| (K_e\to E_K\to I_{SN}\to\eta_{QIF}) bridge | `theory/definitive/SMM_DEFINITIVE_STEP2A_K_TO_QIF_BRIDGE.md` | Derivation plan / theory |
| Cellular continuation, QIF coefficients, (q_E,q_I) | `theory/definitive/SMM_DEFINITIVE_STEP2B_EXECUTED_RESULTS.md` | Executed quantitative result |
| Full cellular-calibration protocol | `theory/definitive/SMM_DEFINITIVE_STEP2B_CELLULAR_CALIBRATION.md` | Prespecified protocol |
| Two-state ionic/syncytial architecture | `theory/definitive/SMM_DEFINITIVE_STEP3_ASTROGLIAL_MESH.md` | Canonical mechanistic architecture |
| KNP reference hierarchy and reduction test | `theory/definitive/SMM_DEFINITIVE_STEP3A_KNP_REDUCTION.md` | Prespecified reduction protocol |
| Linear KNP poles and fitted reduced coefficients | `theory/definitive/SMM_DEFINITIVE_STEP3B_RESULTS.md` | Executed quantitative result |
| Nonlinear R1 and electro-chemo-mechanical R2 validation | `theory/definitive/SMM_DEFINITIVE_STEP3B_NONLINEAR_R2_RESULTS.md` | Executed robustness result |
| Validity domain of reduced mesh | same Step 3B nonlinear/R2 document | Executed bounded result |
| Closed neuron-ion-astroglial equations | `theory/definitive/SMM_DEFINITIVE_STEP4_CLOSED_LOOP_RESULTS.md` | Executed model closure |
| Slow 8.855-s pole; spatial decay hierarchy | same Step 4 document | Executed quantitative result |
| Physiological pulse magnitude and modest rate effect | same Step 4 document | Executed quantitative result |
| Hopf shift rather than inserted glial oscillator | same Step 4 document | Executed quantitative result |
| Spatial redistribution and (D_g) identifiability limits | same Step 4 document | Executed quantitative result |
| DK68 embedding, (L_g\neq W_N), delayed neuronal network | `theory/definitive/SMM_DEFINITIVE_STEP5_WHOLEBRAIN_EEG_ARCHITECTURE.md` | Executed pre-EEG architecture |
| Template-head EEG observation model and its limitations | same Step 5 document + Step 5B lock | Executed/frozen architecture |
| M0-M3 adversarial hierarchy | Step 4 + Step 5 + `SMM_DEFINITIVE_STEP5B_PRE_EEG_LOCK.md` | Frozen inferential design |
| Development/holdout split | `theory/definitive/SMM_DEFINITIVE_STEP5B_PRE_EEG_LOCK.md` + `confirmatory/step5b/subject_split.csv` | Frozen before confirmatory signal inspection |
| Frozen preprocessing/QC | Step 5B lock + `confirmatory/step5b/preprocess_ds005385.py` | Frozen implementation |
| Empirical CSD | Step 5B lock + `empirical_csd_lock.py` | Frozen implementation |
| Complex Whittle primary score | Step 5B lock + `statistical_lock.py` | Frozen implementation |
| M2/M3 frequency-domain model | `model_frequency_lock.py` | Frozen implementation |
| Numerical optimizer history | `DEVELOPMENT_NUMERICAL_FREEZE_2026-10-06.md`, `OPTIMIZER_ROBUSTNESS_DECISION_RULE.md` | Development-only implementation history |
| Final numerical profile | `NUMERICAL_PROFILE.json` | **PENDING until status=FROZEN** |
| Final 34-subject development summary | `development_summary.json` after canonical rerun | **PENDING** |
| Confirmatory primary result | `confirmatory_primary_result.json` | **PENDING; do not write before holdout run** |
| Replication hierarchy | `REPLICATION_AND_ROBUSTNESS_PLAN.md` | Frozen secondary sequence |

## Wording rules

### “Derived”
Use only when a relation follows from the mechanistic/calibration chain rather than being fitted from EEG.

### “Validated”
Use with its declared domain. In particular, the reduced astroglial mesh is validated over the tested physiological amplitude/spatial regime, not as an unrestricted pathological high-K model.

### “Identified”
Avoid for astroglia from EEG. A positive M3-vs-M2 result means unseen EEG favors the physiologically constrained SMM conditional on the upstream mechanism. It does not mean EEG directly identifies astrocytes or (D_g).

### “Confirmed”
Reserve for the actual frozen confirmatory outcome. Do not use for development data or historical v2/v3 fit correlations.

### “Falsified”
If the primary frozen criterion fails, state exactly which claim failed: the current SMM constraints did not earn the prespecified predictive advantage in the primary EEG condition. Do not imply that every possible astrocyte contribution to brain dynamics has been disproven.

## Final manuscript audit

Before submission, every numerical value in the main text, tables and figures should be traceable to one of:

1. an executed canonical source document above;
2. a permanent aggregate CSV/JSON committed after the final development/holdout run;
3. a figure source table generated from those permanent aggregates.

Transient GitHub log text is not an acceptable final manuscript source.

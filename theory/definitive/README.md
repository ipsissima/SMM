# Definitive SMM reconstruction - canonical source documents

This directory archives the full mechanistic reconstruction that precedes the Step 5B empirical test. These documents were originally produced during the reconstruction process and are now committed to the paper repository so the complete scientific chain is available from Git.

## Reading order

1. `SMM_GENEALOGY_ORIGINAL_TO_DEFINITIVE.md`  
   Genealogy from the original Brain-Mesh/SMM programme to the reconstructed theory.

2. `SMM_DEFINITIVE_FOUNDATIONS_STEP1.md`  
   Biological and mathematical root; continuity with the published *Frontiers* control-field paper and canonical abandonment of the old direct-wave interpretation.

3. `SMM_DEFINITIVE_STEP2_NEURONAL_CORE.md`  
   Choice of the next-generation QIF/MPR E-I neuronal substrate and comparator hierarchy.

4. `SMM_DEFINITIVE_STEP2A_K_TO_QIF_BRIDGE.md`  
   Derivation strategy for extracellular K -> Nernst potential -> firing threshold -> QIF excitability.

5. `SMM_DEFINITIVE_STEP2B_CELLULAR_CALIBRATION.md`  
   Prespecified cellular calibration protocol.

6. `SMM_DEFINITIVE_STEP2B_EXECUTED_RESULTS.md`  
   Executed cellular K sensitivity and reverse K-per-spike calibration.

7. `SMM_DEFINITIVE_STEP3_ASTROGLIAL_MESH.md`  
   Ionic-homeostasis/electrodiffusive derivation of the reduced syncytial controller.

8. `SMM_DEFINITIVE_STEP3A_KNP_REDUCTION.md`  
   Reference-model hierarchy and controlled reduction protocol.

9. `SMM_DEFINITIVE_STEP3B_RESULTS.md`  
   Executed linear electrodiffusive reduction.

10. `SMM_DEFINITIVE_STEP3B_NONLINEAR_R2_RESULTS.md`  
    Nonlinear and electro-chemo-mechanical robustness tests; validated physiological domain.

11. `SMM_DEFINITIVE_STEP4_CLOSED_LOOP_RESULTS.md`  
    Closed neuronal-ion-astroglial loop, timescales, spatial redistribution, bifurcation and identifiability.

12. `SMM_DEFINITIVE_STEP5_WHOLEBRAIN_EEG_ARCHITECTURE.md`  
    DK68 whole-brain embedding, delayed neuronal network and EEG observation architecture.

13. `SMM_DEFINITIVE_STEP5B_PRE_EEG_LOCK.md`  
    Final pre-EEG scientific lock: split, preprocessing, likelihood, null hierarchy and confirmatory criterion.

## Executable empirical layer

The executable Step 5B implementation lives in:

`../../confirmatory/step5b/`

Start with:

`../../confirmatory/step5b/README.md`

## Manuscript integration

The pre-holdout integrated paper draft is:

`../../manuscript/DEFINITIVE_SMM_MANUSCRIPT_PREHOLDOUT.md`

The manuscript is intentionally more compressed than these source documents. If a technical detail is omitted from the main paper, this source directory is the canonical place to recover the derivation rather than reconstructing it from memory.

## Epistemic hierarchy

The documents should not all be read as having the same status.

- Steps 1-3A include design decisions and prespecified derivation protocols.
- Step 2B Executed, Step 3B, Step 3B nonlinear/R2, Step 4 and Step 5 contain executed mechanistic/computational results.
- Step 5B is an analysis lock, not an empirical result.
- Development and holdout results are produced only by the frozen executable pipeline under `confirmatory/step5b/`.

This distinction should be preserved in the final manuscript and supplement.

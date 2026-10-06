# Definitive SMM figure and table plan

**Rule:** every plotted or tabulated manuscript number must be generated from a permanent machine-readable source file or a cryptographically identified artifact. No value should be copied from transient logs without provenance.

## Main Figure 1 - From physiology to the observable

**Purpose:** show the complete causal chain in one panel sequence.

Panels:
- **A. Cellular K physiology:** (K_e\to E_K\to I_{SN}\to\eta_{QIF}).
- **B. Microscopic astrocyte-ECS transport:** KNP/electrodiffusive reference system.
- **C. Reduced syncytial controller:** two-state (k,a) system with (D_eL_e) and (D_gL_g).
- **D. Closed E-I neuronal/ionic loop:** neuronal firing produces K source; K shifts excitability.
- **E. Whole-brain embedding:** local astroglial controller per DK68 region, long-range neuronal (W_N) only.
- **F. Observation:** pyramidal/neuronal source currents -> frozen leadfield -> 64-channel EEG -> U20 projected CSD.

Critical visual annotation:

[
L_g \neq W_N
]

and

[
(k,a) \rightarrow X_N \rightarrow EEG,
]

with **no direct glial EEG arrow**.

Source documents:
- Step 2B executed results
- Step 3B nonlinear/R2 results
- Step 4 closed-loop results
- Step 5 whole-brain/EEG architecture

## Main Figure 2 - The mesh is a dissipative homeostatic controller, not a theta oscillator

Panels:
- **A. Reference-model transfer poles** including the slow ~9.15-s component.
- **B. Reduced-model vs full-reference transfer response on calibration modes.**
- **C. Held-out spatial-mode validation.**
- **D. Nonlinear R1/R2 coarse-scale validation across perturbation amplitude.**
- **E. Spatial decay time vs wavelength/mode.**
- **F. Schematic comparison: historical oscillatory-mesh interpretation crossed out; validated spatial low-pass/control interpretation highlighted.**

Primary message:

> The reconstructed SMM preserves slow spatial mode selection but not protected 4/8/12-Hz glial harmonics.

## Main Figure 3 - Closed-loop physiological consequences and identifiability

Panels:
- **A. Physiological pulse:** (K_e(t)), (R_E(t)), neural-only vs SMM.
- **B. Hopf/stability boundary:** neural-only and SMM across anatomical prior range.
- **C. Seven-patch spatial redistribution:** (D_g>0) vs local buffering only at 0/150/300/450 micrometers.
- **D. Parameter identifiability comparison:** neuronal-rate observation vs direct (K_e) observation.
- **E. State-dependent network amplification:** same frozen glial physiology under different neuronal susceptibility.

Primary message:

> The local controller is modest but spatially structured; macroscopic effects depend on network state, and EEG cannot directly identify microscopic glial transport.

## Main Figure 4 - Adversarial model comparison and information barriers

Panels:
- **A. Model hierarchy:** M0 -> M1 -> M2 -> M3.
- **B. Explicit reason M2 is dangerous:** generic two-state slow control can approximate/realize the linearized SMM transfer class.
- **C. Information barrier timeline:** no EEG -> development 001-043 -> numerical freeze -> holdout 044-608.
- **D. Primary two-block CV:** fit A/score B and fit B/score A.
- **E. Frozen success criterion:** bootstrap CI and sign-flip gate.

This figure should make it visually impossible to misread the study as "SMM vs no slow process."

## Main Figure 5 - Primary empirical result

**Do not instantiate until holdout is opened.**

Panels:
- **A. Subject-level DeltaELPD(M3-M2), ordered by value.**
- **B. Distribution/violin or histogram with zero line.**
- **C. Mean and frozen 95% bootstrap CI.**
- **D. Fraction of subjects with Delta > 0.**
- **E. Optional M2 vs M3 held-out ELPD scatter with identity line.**

Caption must state:
- n=565 expected;
- primary condition exact;
- holdout untouched before gate;
- frozen success rule;
- PASS/FAIL without reinterpretation.

Machine-readable source:
`confirmatory_subject_results.csv`
and
`confirmatory_primary_result.json`.

## Main Figure 6 - Replication and robustness

Frozen condition order:
1. session-1 EyesOpen pre;
2. session-1 EyesClosed post;
3. session-1 EyesOpen post;
4. session-2 EyesClosed pre;
5. remaining session-2 conditions.

Preferred display:
- forest plot of mean DeltaELPD with fixed analysis per condition;
- optionally paired condition-to-condition subject effects where longitudinal overlap exists.

No replication is allowed to redefine the primary result.

---

# Main Table 1 - Multiscale parameter provenance

Columns:
- parameter/symbol;
- value/range;
- scale;
- source/calibration;
- fitted from EEG? yes/no;
- role in model.

Must clearly separate:
- microscopic/frozen biology;
- network nuisance parameters;
- observation nuisance parameters.

## Main Table 2 - Model hierarchy

Rows M0-M3. Columns:
- neuronal backbone;
- slow states;
- spatial structure;
- free slow-control parameters;
- physiological K constraints;
- purpose.

Required final row explanation:
M3 differs from M2 by physiological constraints, not merely by possessing a slow latent state.

## Main Table 3 - Development and numerical audit

Rows:
- metadata-clean development n=38;
- QC pass n=34;
- each of four QC failures and frozen reason;
- reproducibility run ID;
- three replicate M2/M3/Delta values;
- deterministic environment;
- final development run ID/artifact digest.

No development p-value.

## Main Table 4 - Primary confirmatory test

Columns:
- n;
- mean DeltaELPD;
- median;
- SD;
- bootstrap 95% CI;
- sign-flip p;
- fraction Delta>0;
- frozen criterion;
- verdict.

Source:
`confirmatory_primary_result.json`.

## Main Table 5 - Replications

Rows by frozen replication condition. Columns:
- available n;
- QC n;
- mean Delta;
- CI;
- sign-flip p or prespecified replication statistic;
- fraction positive;
- same-sign primary? yes/no.

---

# Supplementary figures

- **S1:** cellular continuation curves (I_{SN}^{E/I}(K_e)).
- **S2:** center-manifold/QIF local approximation error.
- **S3:** reference KNP poles and residues by spatial mode.
- **S4:** grid convergence N=11/21/31.
- **S5:** nonlinear validity domain and failure beyond physiological range.
- **S6:** anatomical prior sensitivity.
- **S7:** full delayed-network stability scan.
- **S8:** forward-model rank and U20 singular/eigen structure.
- **S9:** preprocessing/QC flow and subject exclusions.
- **S10:** optimizer reproducibility fingerprints.
- **S11:** development subject Delta distribution, explicitly labelled development only.
- **S12:** optimizer iterations/evaluations across all subjects.
- **S13+:** robustness backbones/metrics if executed.

# Figure-generation rule

When result data become available, figure scripts should consume only the permanent CSV/JSON tables produced by the aggregate scripts. They should not parse GitHub logs directly.

# Definitive SMM manuscript - pre-holdout integrated draft

**Status:** manuscript architecture frozen as far as possible before confirmatory data  
**Scientific state:** mechanistic reconstruction complete; deterministic development fitting in progress; confirmatory holdout unopened  
**Do not insert confirmatory language until the frozen holdout workflow has completed.**

## Working title

**From Astroglial Ionic Homeostasis to Human EEG: A Mechanistically Constrained Test of the Syncytial Mesh Model**

Alternative shorter title:

**A Mechanistically Constrained Syncytial Mesh Model of Human Brain Dynamics**

## One-sentence paper claim

Known astrocyte-ECS ionic physiology can be reduced to a slow spatially structured homeostatic controller, quantitatively coupled to an exact neuronal population model and embedded in a whole-brain EEG observation model; the decisive empirical question is whether those physiological constraints improve prediction of unseen human EEG relative to a matched generic slow-control model.

---

# Abstract - locked structure

Large-scale neuronal dynamics unfold on a structural connectome but are also regulated by slower non-neuronal physiological processes. We test whether astrocytic syncytial homeostasis can supply a quantitatively constrained control layer rather than a freely fitted latent field. Starting from published electrodiffusive astrocyte-extracellular-space models, we derive and validate a two-state mesoscale potassium-control subsystem, calibrate extracellular-potassium effects on excitatory and inhibitory neuronal excitability through conductance-based saddle-node reductions, and close the bidirectional neuron-ion-astroglial loop. The resulting controller is embedded locally in a delayed 68-region next-generation E-I neural-mass network and observed through a template-head 64-channel EEG forward model. The model introduces a multi-second control mode without inserting an intrinsic glial theta/delta oscillator and predicts spatial low-pass redistribution rather than brain-wide direct glial propagation. We compare a physiologically constrained SMM (M3) against the same neuronal network with a flexible generic two-state slow controller (M2) using two-block held-out complex multivariate Whittle prediction. The analysis plan, development/holdout boundary, preprocessing, forward model, comparator hierarchy, frequency range and success criterion were frozen before confirmatory EEG inspection.

**CONFIRMATORY SENTENCE PLACEHOLDER - insert exactly one of the approved result forms after holdout.**

The study therefore separates three questions that earlier SMM formulations conflated: whether slow non-neuronal control is physiologically derivable, whether it measurably affects neuronal dynamics, and whether its specific astroglial constraints improve prediction beyond generic slow control.

---

# 1. Introduction

## 1.1 The explanatory gap

The human connectome constrains long-range communication, but a fixed structural graph does not uniquely determine the dynamical regime expressed by the brain. Sleep, arousal, cognitive load, pathology and neuromodulatory state can reorganize coherence, metastability and spectral structure without anatomical rewiring on the same timescale. This motivates a distinction between **structural connectivity geometry** and **state-dependent control geometry**.

Astrocytes are plausible contributors to the latter because syncytial gap-junction coupling, extracellular potassium regulation, Na/K-ATPase activity, metabolic support and neuromodulator-sensitive signaling operate over slower timescales than dominant neuronal membrane and synaptic dynamics. The key mechanistic question, however, is not whether astrocytes can influence neurons. It is whether known astrocytic physiology coarse-grains into a controller with quantitative consequences that survive comparison with simpler alternatives.

## 1.2 Relation to earlier SMM formulations

The original Brain-Mesh/Syncytial Mesh Model proposed a second spatial dynamical layer in addition to neuronal structural connectivity. Its early implementation used an elastic/damped-wave analogy and treated modal resonance as a candidate route to large-scale coherence. Later SMM versions progressively weakened those literal claims and reframed the mesh as an astrocytic mesoscale control field.

The published *Frontiers in Human Neuroscience* theory paper made the decisive conceptual correction: astrocytes were not proposed as dominant direct EEG/MEG generators or as centimeter-scale calcium-wave carriers. Instead, glial syncytial fields were defined as slow effective variables modulating neuronal excitability, gain, damping, coupling and stability.

The present work is the technical completion of that conceptual architecture. It does not preserve the old metallic-mesh physics. It tests the deeper research-programme claim that neuronal structural connectivity may coexist with a distinct slow astroglial control geometry.

## 1.3 What must be shown for the claim to become mechanistic

A defensible technical SMM requires all of the following:

1. a neuronal population model with a controlled microscopic interpretation;
2. a non-arbitrary bridge from extracellular ionic state to neuronal excitability;
3. an astroglial subsystem derived from ion-conserving transport rather than imposed as a convenient field equation;
4. explicit closure of neuron-to-ion and ion-to-neuron feedback;
5. scale separation between local/mesoscale astroglial transport and long-range neuronal connectivity;
6. an observation model in which scalp EEG is generated by neuronal source currents;
7. matched alternative slow-control models;
8. held-out prediction under a frozen analysis plan.

We construct this chain sequentially and freeze biological parameters before confirmatory EEG evaluation.

---

# 2. Results

## 2.1 Extracellular potassium provides a quantitative neuron-control bridge

The glia-to-neuron coupling was derived through

[
K_e \rightarrow E_K \rightarrow I_{SN}(K_e) \rightarrow \eta_{QIF},
]

rather than through a fitted term of the form (g_A u_g).

At (K_e=3.5) mM, conductance-based excitatory and inhibitory calibration models exhibit the required local saddle-node structure. Continuation over the physiological range produced independent excitatory and inhibitory potassium-sensitivity maps. For the 8-ms MPR timescale used in the closed model, the resulting excitability shifts are

[
\delta\eta_E =
0.05210958\ell - 0.00489261\ell^2,
]

[
\delta\eta_I =
0.04673428\ell - 0.00367223\ell^2,
]

where

[
\ell=\ln(K_e/3.5\text{ mM}).
]

The reverse coupling was independently calibrated from delayed-rectifier potassium current per spike:

[
q_E=4.736\times10^{-8}
\text{ mol m}^{-2}\text{ spike}^{-1},
]

[
q_I=1.267\times10^{-8}
\text{ mol m}^{-2}\text{ spike}^{-1}.
]

Thus the central bidirectional neuronal-ionic coupling is not estimated from EEG.

## 2.2 A two-state syncytial controller is recovered from electrodiffusive physiology

A published six-ion astrocyte-ECS electrodiffusive system was reconstructed and linearized around its resting state. Its experimentally relevant input-to-extracellular-potassium response is dominated by a small number of stable dissipative poles, including a slow uniform-mode component with characteristic time approximately 9.15 s.

A prespecified two-state reduction,

[
\phi_e\dot k =
S_K-(\kappa_e+\kappa_N)k+\kappa_a a-D_eL_e k,
]

[
\phi_a\dot a =
\kappa_e k-\kappa_a a-D_gL_g a,
]

was calibrated on selected spatial modes and tested on held-out modes.

The same architecture survived nonlinear zero-flow electrodiffusive validation and a higher-fidelity electro-chemo-mechanical challenge including osmotic swelling, hydrostatic pressure, water movement and fluid-assisted ionic transport. In the latter system, fluid mechanics renormalized the effective syncytial transport coefficient rather than requiring a different state architecture.

The validated interpretation is therefore not an intrinsic glial wave oscillator. It is a dissipative spatial homeostatic controller.

## 2.3 Closing the loop produces slow control, not an inserted neuronal oscillator

The local state is

[
X=(R_E,V_E,R_I,V_I,S_{EI},S_{IE},k,a).
]

With the neuronal potassium source and potassium-to-QIF map closed, the canonical astroglial coefficients are

[
\kappa_e=2.01315923\ \mathrm{s}^{-1},
\quad
\kappa_a=0.48318016\ \mathrm{s}^{-1},
\]

[
\kappa_N=0.232\ \mathrm{s}^{-1},
]

[
D_e=9.55343\times10^{-10}\ \mathrm{m^2/s},
\quad
D_g^{eff}=3.71479\times10^{-10}\ \mathrm{m^2/s}.
]

The closed system contains a slow control pole

[
\tau_{g,slow}\approx8.855\ \mathrm{s}
]

and a faster ionic component around 0.081 s. Higher spatial frequencies decay more rapidly than coarse spatial modes.

A physiological excitatory pulse produced a peak extracellular-potassium excursion of approximately 0.464 mM, within the independently validated nonlinear domain. The direct difference between SMM and otherwise identical neural-only firing was modest, establishing a useful negative constraint: large macroscopic SMM effects cannot be assumed merely because the controller exists.

The neuronal-only E-I system undergoes its own Hopf transition. Adding the SMM shifts the stability boundary only slightly. Thus the controller modulates neuronal susceptibility; it does not manufacture an oscillatory regime by inserting a hidden theta/delta oscillator.

## 2.4 Syncytial transport has a spatial signature but is not directly identifiable from neuronal rates

In the mesoscale spatial test, gap-junction-dependent transport had little effect on the stimulated patch peak but progressively increased ionic redistribution away from the active location. This provides an operational meaning of "mesh": nonzero syncytial transport changes the spatial distribution of homeostatic load.

Sensitivity calculations also established an important limitation. The effective syncytial transport parameter is poorly identifiable from noisy neuronal firing alone but highly recoverable from direct extracellular-potassium observations. EEG therefore cannot be interpreted as direct measurement of an astroglial transport coefficient.

This motivates the paper's epistemic rule: EEG can favor a physiologically constrained SMM over matched alternatives, but cannot by itself directly identify astrocytes.

## 2.5 Whole-brain embedding preserves two distinct geometries

The local controller was embedded in a 68-region Desikan-Killiany network. Long-range structural coupling is neuronal and uses the frozen ENIGMA/HCP structural prior. Astroglial states remain local at regional scale:

[
L_g \ne W_N.
]

Long-range glial consequences therefore follow

[
\text{local homeostasis}
\rightarrow
\text{local neuronal operating point}
\rightarrow
W_N
\rightarrow
\text{remote neuronal dynamics}.
]

No centimeter-scale astrocytic edges are introduced.

The delayed neuronal network remains stable below the frozen global-coupling ceiling used for inference. The EEG observation architecture projects neuronal source currents through a frozen template-head forward model; there is no direct glial scalp generator.

## 2.6 Strong generic slow controllers are required adversarial comparators

Synthetic closed-loop analyses showed that generic slow latent dynamics can explain a substantial fraction of the SMM-induced neuronal correction. In the linear regime, an unconstrained generic two-state controller can even realize the same transfer-function class as the linearized two-state SMM.

Consequently, evidence that M3 outperforms a neural-only model would not identify the SMM mechanism. The primary hierarchy is

[
M_0=\text{neuronal only},
]

[
M_1=\text{generic one-state slow control},
]

[
M_2=\text{flexible generic two-state slow control},
]

[
M_3=\text{physiologically constrained SMM}.
]

The decisive empirical contrast is M3 versus M2.

## 2.7 Development is separated from the confirmatory cohort

The public EEG cohort contains 608 participants. Because historical SMM work had already exposed signals from subjects 001-043, those participants were assigned exclusively to development. Five prespecified metadata-anomalous participants are sensitivity-only; the primary metadata-clean development set contains 38 participants.

Frozen preprocessing yielded 34 QC-passed primary development recordings. Four failed prespecified criteria and were not rescued by threshold relaxation.

During development, repeated fits revealed sensitivity of the optimizer to numerical execution details. An initial three-run same-image reproducibility test produced bit-identical M2 and M3 scores under pinned packages and single-thread execution. However, the first subsequent development fit executed after a GitHub runner-image update returned materially different local optima despite byte-identical scientific code and inputs. The provisional numerical freeze was therefore superseded before any confirmatory holdout data were opened. A development-only robustness study was then initiated to select the final optimizer profile on the basis of training-objective recovery and cross-kernel reproducibility, not on the sign or magnitude of M3-versus-M2 held-out effects.

**Insert final 34-subject descriptive development aggregate here once complete. Do not report confirmatory significance on development.**

## 2.8 Confirmatory prediction on unseen human EEG

**LOCKED PLACEHOLDER. NO RESULT MAY BE WRITTEN UNTIL HOLDOUT EXECUTION.**

Primary condition: session 1, eyes closed, pre-cognitive-block.

For each holdout subject,

[
\Delta_i=ELPD_i(M_3)-ELPD_i(M_2).
]

Specific SMM predictive success was defined before holdout inspection as requiring both:

- lower bound of a 10,000-resample subject bootstrap 95% CI > 0;
- one-sided paired 100,000-permutation sign-flip (p<0.05).

Seed = 97.

### Approved PASS wording

> In the untouched confirmatory cohort, the physiologically constrained SMM predicted held-out EEG better than the matched generic two-state slow-control model under both prespecified inferential criteria. This result favors the SMM physiological constraints conditional on the upstream mechanistic derivation; it does not imply that EEG directly observes astrocytes.

### Approved FAIL wording: M3 approximately equals M2

> In the untouched confirmatory cohort, the physiologically constrained SMM did not outperform the matched generic two-state slow-control model under the prespecified criterion. The result does not support SMM-specific physiological identification from these EEG data, although it remains compatible with a broader role for slow control.

### Approved FAIL wording: M3 worse than M2

> In the untouched confirmatory cohort, the physiologically constrained SMM predicted held-out EEG worse than the matched generic two-state slow-control model. Under the prespecified test, this falsifies the claim that the present SMM physiological constraints improve prediction of the primary resting EEG condition.

---

# 3. Methods

## 3.1 Design principle and information barriers

The analysis was organized as a sequence of one-way information barriers:

1. cellular calibration without EEG;
2. astrocyte-ECS reduction without EEG;
3. nonlinear mechanistic validation without EEG;
4. closed-loop and whole-brain architecture without EEG;
5. pre-EEG lock of dataset split, preprocessing, observation model, likelihood, comparators and success rule;
6. development on historically exposed subjects only;
7. one permitted numerical implementation freeze;
8. untouched confirmatory holdout.

No biological mechanism, primary condition, comparator strength, success criterion, frequency interval or exclusion threshold could be changed because development performance was disappointing.

## 3.2 Next-generation neuronal mass

Describe the E-I Montbrio-Pazo-Roxin/QIF system, synaptic filters, operating point, long-range excitatory coupling and delayed structural network. Include full equations in Methods or Supplementary Note 1.

## 3.3 Cellular potassium calibration

Describe excitatory Traub-Miles and inhibitory Wang-Buzsaki calibration, continuation of the SNIC threshold, center-manifold check, QIF rescaling and potassium-current integration per spike.

## 3.4 Astrocyte-ECS reference systems and model reduction

Describe the pinned open electrodiffusive implementation, species/compartments, KNP structure, modal reduction split, held-out modes, nonlinear finite-volume reproduction and higher-fidelity electro-chemo-mechanical robustness model.

## 3.5 Closed neuron-ion-syncytium model

Present complete state equations and frozen coefficients. Clarify the source-bookkeeping convention for extracellular volume fraction.

## 3.6 Whole-brain network

Describe DK68 ENIGMA/HCP structural matrix, spectral normalization, delayed E-to-E coupling, distance proxy limitation and frozen velocity range.

## 3.7 EEG observation model

Describe the 64-channel actiCAP montage, fsaverage_1005 template geometry, fixed-orientation cortical leadfield, DK68 regional aggregation, average reference, rank limitation and U20 projection. Explicitly state that subject-specific electrode digitization/MRI coregistration is unavailable.

## 3.8 EEG cohort and primary condition

Dataset: OpenNeuro ds005385 at the frozen git snapshot. Primary recording: session 1 / EyesClosed / acq-pre. Development/holdout split follows historical signal exposure: sub-001..043 are development and sub-044..608 are the 565 assigned confirmatory holdout subjects. Primary inferential n is determined only by the already-frozen QC/inclusion rules; QC-failed recordings remain explicitly accounted for and are not replaced.

## 3.9 Frozen preprocessing and QC

Describe:
- crop 2 s from each end;
- resample to 250 Hz;
- 1-45 Hz QC stream for amplitude sanity;
- PyPREP RANSAC/correlation bad-channel detection;
- maximum 10 globally bad channels;
- extended Infomax ICA on 1-100 Hz;
- ICLabel artifact removal at frozen probability threshold;
- 0.5-45 Hz analysis stream;
- average reference;
- interpolation with bad-channel provenance retained;
- 4-s fixed epochs;
- absolute and robust epoch rejection;
- minimum 30 clean epochs / 120 s.

Document the ds005385 EDF invalid-physical-range issue and explain that the amendment changed the location of the frozen amplitude sanity check, not its thresholds.

## 3.10 Empirical cross-spectral density

Describe DPSS multitaper estimation, NW=3, Kmax=5, integer 1-40 Hz frequencies, two chronological halves and identical U20 projection for empirical and model CSDs.

## 3.11 Model fitting

Frozen deterministic environment:
- seed 97;
- 32 Sobol candidates;
- 4 L-BFGS-B polish starts;
- maxiter 120;
- ftol (10^{-9});
- gtol (10^{-6});
- maxls 30;
- relative eigenvalue floor (10^{-6});
- single-thread BLAS/OpenMP execution.

Fit A and score B; fit B and score A; average held-out score.

## 3.12 Predictive likelihood

Use the complex multivariate Whittle/complex-Wishart score

[
\ell_M
=-\sum_f\nu_f\left[
\log|S_M(f)|+
\mathrm{tr}\left(S_M(f)^{-1}\hat S(f)\right)
\right]
]

up to model-independent constants.

## 3.13 Confirmatory inference

State the frozen bootstrap and sign-flip rule exactly as implemented in
`confirmatory/step5b/aggregate_confirmatory.py`.

## 3.14 Replications

Apply the same architecture without mechanism retuning in the frozen order:
1. session-1 EyesOpen pre;
2. session-1 EyesClosed post;
3. session-1 EyesOpen post;
4. session-2 EyesClosed pre;
5. remaining session-2 conditions.

Replication failure is reported rather than used to redesign the primary model.

---

# 4. Discussion - prewritten logical structure

## 4.1 What the paper establishes regardless of holdout outcome

The mechanistic contribution is separable from the confirmatory EEG result. The paper establishes that a two-state mesoscale astroglial homeostatic subsystem can be derived from substantially richer electrodiffusive dynamics in a declared physiological regime; that its coupling to neuronal excitability can be calibrated without EEG; and that its whole-brain consequences can be formulated without treating astrocytes as direct scalp-current generators.

## 4.2 The original SMM intuition survives in a different mathematical form

The part of the original theory that survives is not elastic-wave physics or fixed harmonics. It is the distinction between neuronal structural geometry and a slower non-neuronal control geometry. In the reconstructed model, the latter is dissipative, local/mesoscale and homeostatic. Its global consequences are mediated by neuronal susceptibility and the white-matter network.

## 4.3 Why a small physiological effect is not a defect

The calibrated local SMM effect is modest under ordinary operating conditions. This prevents the model from obtaining large EEG improvements simply by inserting an unconstrained latent field. Any macroscopic predictive advantage must emerge from spatial organization, state dependence or network susceptibility.

## 4.4 Identification limits

Even a positive EEG result cannot directly measure astroglial transport. The paper should distinguish:
- predictive discrimination of physiologically constrained model classes;
- parameter identifiability;
- direct causal identification.

Direct astrocyte perturbation combined with electrophysiology remains a stronger causal test.

## 4.5 PASS-specific discussion branch

If M3 beats M2, emphasize constraint value rather than fit flexibility. The surprising result is not that adding a slow state helps; M2 was designed to capture that. The evidential content is that the externally derived physiological constraints predict unseen data better than a flexible matched controller.

## 4.6 FAIL-specific discussion branch

If M3 fails to beat M2, do not rescue the mechanism by post-hoc retuning. Distinguish among:
- failure of SMM-specific EEG discrimination in this condition;
- possible insufficiency of scalp EEG to identify the mechanism;
- possible state dependence, which may be assessed only through the frozen replication sequence;
- survival of the upstream mesoscale reduction as a mechanistic result independent of macroscopic predictive success.

## 4.7 Limitations

Required limitations:
- template-head rather than subject-specific source geometry;
- centroid-distance delay proxy unless tract lengths are replaced under a predeclared rule;
- EEG cannot directly identify (D_g);
- linearized frequency-domain primary likelihood;
- physiological validity domain of the reduced mesh;
- primary test is resting eyes-closed EEG, not direct astrocyte perturbation;
- human-specific astrocyte properties are not directly measured in ds005385.

---

# 5. Data, code and reproducibility statements

All frozen analysis code, model definitions, network reconstruction logic, preprocessing, development/holdout barriers and inferential code are version controlled in the SMM repository. Heavyweight EEG derivatives and forward-model artifacts may remain external/action artifacts, but their exact run IDs, artifact IDs and cryptographic hashes are permanently recorded in `confirmatory/step5b/RESULTS_PROVENANCE_MANIFEST.md`.

No manuscript-facing numerical result should be reported without a permanent machine-readable source table and provenance entry.

---

# 6. Supplementary architecture

- Supplementary Note 1: full QIF/MPR equations and conductance mapping
- Supplementary Note 2: cellular continuation and center-manifold derivation
- Supplementary Note 3: full KNP reference equations
- Supplementary Note 4: modal reduction and held-out-mode validation
- Supplementary Note 5: nonlinear/electro-chemo-mechanical solver validation
- Supplementary Note 6: closed-loop bifurcation and sensitivity analyses
- Supplementary Note 7: DK68 network, delays and forward model
- Supplementary Note 8: frozen preprocessing/QC
- Supplementary Note 9: M0-M3 definitions and identifiability argument
- Supplementary Note 10: optimization reproducibility experiment
- Supplementary Data 1: development subject results
- Supplementary Data 2: confirmatory subject results
- Supplementary Data 3+: frozen replication results

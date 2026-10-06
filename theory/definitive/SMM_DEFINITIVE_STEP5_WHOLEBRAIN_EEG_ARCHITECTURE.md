# SMM Definitive Reconstruction — Step 5 Executed
## Whole-brain embedding and EEG observation architecture

**Status:** executed as a pre-EEG development/falsification stage  
**EEG signals used:** none

## 1. Scale-consistent architecture

The local model from Steps 1–4 is now embedded in a 68-region neuronal
network while preserving two non-negotiable separations:

\[
L_g\neq W_N
\]

and

\[
(k,a)\rightarrow X_N\rightarrow EEG.
\]

There is no direct astroglial EEG generator.

## 2. Structural neuronal network

The development network uses the ENIGMA Toolbox HCP structural-connectivity
matrix for the 68-region Desikan–Killiany cortical parcellation.

It contains 697 undirected non-zero edges (density 0.30597). The raw largest
eigenvalue is 173.536637. All weights are divided by that value, so

\[
\rho(\widetilde W_N)=1.
\]

No structural edge was added.

Long-range coupling is excitatory neuronal coupling only:

\[
\tau_{LR}\dot s_i^{LR}
=
-s_i^{LR}
+
G_N\sum_j\widetilde W_{ij}
[R_{E,j}(t-\tau_{ij})-R_E^0].
\]

There is no corresponding inter-regional glial term.

## 3. Astroglia remains local at DK68 scale

Each region carries local \(k_i,a_i\) control states:

\[
\phi_e\dot k_i=
J_{K,i}^N-(\kappa_e+\kappa_N)k_i+\kappa_a a_i,
\]

\[
\phi_a\dot a_i=\kappa_e k_i-\kappa_a a_i.
\]

The experimentally calibrated \(D_gL_g\) remains a **subregional** syncytial
transport process established in Step 3. It is not projected onto the
centimeter-scale white-matter graph.

Long-range effects of astroglial control therefore occur by:

\[
\text{local homeostasis}
\rightarrow
\text{local neuronal operating point}
\rightarrow
W_N
\rightarrow
\text{distant neuronal dynamics}.
\]

## 4. Delay architecture

For development only, a documented DK68 centroid set supplies Euclidean
center-to-center distances. These are explicitly a **delay proxy**, not tract
lengths.

At \(v=6\) m/s the structural-edge delays are:

- minimum: **0.891 ms**
- median: **8.471 ms**
- mean: **8.912 ms**
- maximum: **21.648 ms**

Sensitivity results for 3, 6, and 12 m/s are bundled.

Before confirmatory EEG analysis, a matched tract-length matrix should replace
this proxy if available; otherwise the proxy and velocity sensitivity must be
preregistered as an approximation.

## 5. Network stability

The no-delay linear network loses stability at approximately

\[
G_N=188.4803.
\]

Direct delayed simulations locate the corresponding transition near

\[
G_N\simeq189.36.
\]

The development working point is \(G_N=150\), with a separate
near-critical susceptibility test at \(G_N=188\).

## 6. State-dependent amplification

The glial physiology is held fixed while only neuronal network susceptibility
changes. For an identical small perturbation, the relative scalp-projected
SMM-versus-neural-only difference is:

- \(G_N=100\): **0.000144**
- \(G_N=150\): **0.000570**
- \(G_N=180\): **0.007569**
- \(G_N=185\): **0.029219**
- \(G_N=188\): **0.158421**

At \(G_N=188\), peak extracellular K changes by only

\[
\Delta K_e=0.005642\ {\rm mM}.
\]

Thus a weak fixed local physiological effect can have a large relative network
consequence close to neuronal criticality without becoming the oscillator:

\[
\boxed{
\text{weak astroglial control}\times
\text{neuronal susceptibility}
\rightarrow
\text{state-dependent macroscopic modulation}.
}
\]

This is a concrete implementation of the published control-field idea.

## 7. EEG source and forward model

The observation equation is

\[
Y(t)=L\,S_N(t)+\epsilon(t),
\]

with no direct glial term.

The development regional source is the modeled net synaptic drive entering the
excitatory/pyramidal population:

\[
S_{N,i}
=
I_{E,i}^{drive}
+\tau_Es_i^{LR}
-\tau_E(s_{EI,i}-s_{EI}^0).
\]

For development testing, a 64-channel MNE-Python forward operator was built
using:

- a 64-channel subset of the standard 10-05 montage;
- 68 DK regional source locations;
- radial development source orientations;
- an MNE three-layer spherical conductor;
- explicit average reference.

The resulting \(64\times68\) lead field has rank **63** and a non-zero
condition number of approximately **4.14e+05**.

The rank 63 is expected after average reference. The severe ill-conditioning
is an expected property of the EEG forward problem.

## 8. Volume conduction is a major adversarial confound

Sixty-eight independent regional AR sources have median absolute zero-lag
correlation

\[
|r|_{\rm median}=0.0213.
\]

After projection through the 64-channel EEG forward operator, the median
absolute sensor correlation becomes

\[
|r|_{\rm median}=0.4858,
\]

with 99th percentile

\[
|r|_{99}=0.9490.
\]

Therefore ordinary scalp correlation/coherence can look strongly structured
even when the underlying regional sources are independent.

The confirmatory feature set must prioritize leakage-resistant quantities:
imaginary coherence, wPLI/debiased wPLI, lagged phase relations, and
forward-model-matched comparisons.

## 9. Synthetic recovery survives sensor projection

A synthetic development experiment was generated at

\[
G_N^{true}=150,\qquad v^{true}=6\ {\rm m/s}.
\]

A \(5\times5\) grid search over global coupling and propagation speed was
performed **after** projection through the 64-channel forward operator. Its
unique optimum was recovered at

\[
G_N=150,\qquad v=6.0\ {\rm m/s}.
\]

This does not prove practical recovery from noisy spontaneous EEG, but it shows
that the development observation operator does not make these two network
parameters structurally identical under a designed perturbation.

## 10. Generic slow-field null remains dangerous

A generic one-state slow controller fitted to full-SMM network trajectories
chose

\[
\tau_g\simeq0.241\ {\rm s}.
\]

It explains a substantial fraction of the SMM correction, but leaves
**41.9%** of the SMM-specific sensor-space glial signature unexplained.

More importantly, a sufficiently flexible generic **two-state** linear slow
field can be state-space equivalent to the linearized two-state SMM.

Therefore EEG model comparison must remain adversarial:

\[
M_0=\text{neuronal only}
\]

\[
M_1=M_0+\text{generic one-state slow control}
\]

\[
M_2=M_0+\text{generic two-state/spatial slow control}
\]

\[
M_3=\text{biophysically constrained SMM}.
\]

If \(M_3\) fails to outperform \(M_2\) on held-out data, the valid conclusion
is evidence for slow control, not astroglial identification.

## 11. Step-5 verdict

### Whole-brain embedding: PASS

A real HCP-derived DK68 structural prior can carry long-range neuronal dynamics
while the astroglial variables remain local.

### Delay architecture: PASS WITH A DEVELOPMENT PROXY

The delayed simulator is operational and synthetic \(G_N,v\) recovery works,
but centroid distance must not be mislabeled as tract length.

### EEG observation architecture: PASS FOR DEVELOPMENT

The forward operator behaves correctly as a rank-limited, strongly mixing EEG
observation operator, and it exposes why raw scalp FC is an inadequate primary
test.

### Confirmatory forward model: GATED

Before empirical EEG signals are opened, the development sphere must be
replaced by an `fsaverage` three-layer BEM using the exact BIDS
channel/electrode geometry of ds005385. The bundle includes
`build_fsaverage_BEM_gate.py`.

## 12. What is now frozen

Before EEG signal analysis, freeze:

1. the local MPR equations;
2. \(K_e\to\eta_E,\eta_I\);
3. microscopic \(q_E,q_I\);
4. astroglial/homeostatic coefficients and priors;
5. the prohibition on inter-regional astroglial edges;
6. the ENIGMA DK68 neuronal structural prior and normalization rule;
7. delayed E-to-E long-range architecture;
8. null-model hierarchy \(M_0\)–\(M_3\);
9. neuronal-only EEG-generation principle;
10. leakage-resistant primary EEG metrics.

Only a small network/observation set remains eligible for empirical inference:
global neuronal coupling, delay/conduction scale, neuronal stochastic drive,
observation scale/noise, and tightly controlled regional variation.

Microscopic glial coefficients are not free subject-level EEG parameters.

## 13. Next operation

The next step is **Step 5B / empirical-analysis lock**:

1. read BIDS metadata only;
2. build the actual 64-channel `fsaverage` BEM forward solution;
3. freeze preprocessing and QC;
4. freeze feature definitions and likelihood/model-comparison criteria;
5. create an immutable analysis snapshot;
6. then open the historical 43-subject development EEG;
7. freeze again before opening the remaining ds005385 holdout subjects.

The old 4.65% threshold result, 4/8/12-Hz mesh harmonics, and historical
critical-scale fitting are excluded from this pipeline.
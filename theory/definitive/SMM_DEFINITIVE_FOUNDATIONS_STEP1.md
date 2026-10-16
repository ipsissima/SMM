# SMM Definitive Reconstruction — Canonical Foundations
## Step 1: Biological and Mathematical Root

**Project:** The Syncytial Mesh Model (definitive technical version)  
**Target standard:** Nature Communications-level technical paper  
**Status:** Canonical design decision after reconstruction of the original SMM, published Frontiers control-field paper, legacy Colab notebooks, raw EEG data provenance, and current GitHub implementation.

---

## 1. Core Continuity Claim

The definitive SMM is **not a different theory with the same name**. It preserves the causal and conceptual core of the original Syncytial Mesh Model while replacing a physically incorrect implementation of that idea with a biologically grounded multiscale control architecture.

The original deep intuition was:

> Astrocytic syncytial organization contributes a spatially distributed dynamical structure that cannot be reduced to conventional neuronal connectivity alone and that can alter large-scale neuronal coherence, stability, and mode selection.

That claim is preserved.

### Old implementation
\[
\text{syncytium}
\rightarrow
\text{glial wave}
\rightarrow
\text{brain-scale coherence}
\]

### Definitive implementation
\[
\boxed{
\text{syncytium}
\rightarrow
\text{slow homeostatic control state}
\rightarrow
\text{neuronal parameter geometry}
\rightarrow
\text{brain-scale neuronal dynamics}
}
\]

---

## 2. Exact Compatibility with the Published Frontiers Paper

The published paper **Glial Syncytial Control Fields and the Low-Frequency Resonome of the Human Brain** already establishes the conceptual architecture required by the definitive SMM.

Its published commitments are:

1. Glial syncytial control fields are **effective mesoscale variables**.
2. Their role is to modify neuronal:
   - gain,
   - damping,
   - coupling,
   - precision weighting,
   - excitability,
   - dynamical stability.
3. They are **not** to be identified with:
   - a microscopic calcium wave moving across centimeters;
   - a primary astrocytic electromagnetic field generating EEG/MEG.
4. Measurable EEG/MEG remains primarily neuronal in electrical generation.
5. The strong thesis is multiscale:
   \[
   \text{slow astrocytic state}
   \rightarrow
   \text{modification of neuronal dynamics}
   \rightarrow
   \text{observable low-frequency organization}.
   \]
6. The Frontiers paper explicitly defers to a later technical SMM paper:
   - full graph-to-field derivation,
   - numerical simulations,
   - parameter inference,
   - null-model comparison,
   - empirical validation.

Therefore:

\[
\boxed{\text{Frontiers paper} = \text{conceptual/physiological architecture}}
\]

\[
\boxed{\text{Definitive SMM} = \text{technical derivation + computation + empirical test}}
\]

There is no conceptual contradiction. The definitive SMM is the technical completion that the published paper explicitly leaves open.

---

## 3. What Is Canonically Abandoned

The following claims or mechanisms from the older SMM do **not** survive into the definitive version unless independently re-derived from the new model:

### 3.1 Direct glial generation of delta/theta
Abandoned:
\[
\text{astrocytic Ca}^{2+}\text{ wave}
\rightarrow
4-8\text{ Hz EEG carrier}.
\]

Slow astrocytic calcium/IP3 dynamics are not the direct centimeter-scale electrical carrier of delta/theta EEG.

### 3.2 Literal 4/8/12 Hz glial eigenmodes
The old interpretation of 4, 8 and 12 Hz as first geometric eigenmodes of a brain-scale astrocytic wave field is abandoned.

### 3.3 Telegraph equation as a derived astrocyte equation
The present repository claims the microscopic linear system

\[
\dot c=-\alpha c+\beta p
\]

\[
\dot p=\gamma c-\delta p+D\nabla^2p
\]

reduces to a telegraph equation with

\[
c_{\rm eff}^2=\frac{D(\delta-\alpha)}2.
\]

This is mathematically incorrect as an exact reduction.

Exact elimination gives:

\[
\boxed{
\ddot c+
(\alpha+\delta)\dot c+
(\alpha\delta-\beta\gamma)c
-D\nabla^2\dot c
-\alpha D\nabla^2c=0
}
\]

For a Fourier mode \(e^{\lambda t+i\mathbf{k}\cdot\mathbf{x}}\),

\[
\lambda^2+
(\alpha+\delta+Dk^2)\lambda+
(\alpha\delta-\beta\gamma+\alpha Dk^2)=0.
\]

Its discriminant is

\[
\boxed{
\Delta=(Dk^2+\delta-\alpha)^2+4\beta\gamma>0
}
\]

for positive \(\beta,\gamma\).

Thus the declared linear micro-model has two real relaxation modes; it does not justify a propagating oscillatory telegraph field.

### 3.4 Independent Kuramoto layer
Kuramoto oscillators will not constitute a third ontological model layer.

Phase coherence, order parameters, phase gradients and traveling-wave metrics may be computed **from the neuronal dynamics**, but synchronization will not be inserted as an independent oscillator system by construction.

### 3.5 Historical 4.65% / critical-scale result as confirmation
The historical \(2/43=4.65\%\) result was constructed using a percentile threshold on the same sample and later used as a fitting target. It is not an independent prediction and will not be used as confirmatory evidence.

### 3.6 Historical model outputs are not protected
Old PSD fits, 4/8/12 Hz modes, critical-scale curves, cusp/bifurcation outputs, legacy figures, and fitted lambda corrections remain only as historical development artifacts. They may reappear only if independently generated by the new model.

---

## 4. What Is Canonically Preserved

The following constitute the stable conceptual core of the SMM:

1. **Astroglia is dynamically central**, not merely supportive.
2. Astrocytes form a **spatially distributed syncytial organization**.
3. The geometry of astroglial coupling differs from the neuronal long-range connectome.
4. Glial dynamics operate on slower timescales than dominant neuronal oscillations.
5. Glial state can change excitability, gain, effective damping, local coupling, stability, metastability, and mode selection of neuronal systems.
6. Large-scale neuronal coherence is potentially neuronal in electrical expression but glio-neural in control architecture.
7. The theory must generate falsifiable predictions that differ from neuron-only systems, generic slow-control systems, and spatially unstructured latent modulators.

---

## 5. Canonical Biological Root

### 5.1 Core mechanism: astroglial homeostatic control

The strongest physiological root is extracellular ionic homeostasis, especially potassium regulation, together with astroglial gap-junction coupling.

Minimal causal loop:

\[
\boxed{
\text{neuronal activity}
\rightarrow
[K^+]_e
\rightarrow
\text{astroglial uptake/redistribution}
\rightarrow
\text{neuronal excitability}
}
\]

Principal candidate mechanisms include astrocytic Kir4.1 conductance, Na/K ATPase, connexin-mediated astroglial coupling (especially Cx30/Cx43), extracellular K+ redistribution, and local ionic/metabolic homeostasis.

The core SMM must **not depend** on controversial universal gliotransmission assumptions.

### 5.2 Calcium/IP3 status

Ca2+/IP3 remains biologically important, but as a slow astroglial signaling channel rather than a direct EEG carrier.

If included mechanistically, it should use a defensible established class such as Li-Rinzel-type calcium dynamics, G-ChI-type calcium/IP3 extensions, or explicit graph/reaction-diffusion coupling. It must not be represented by the current false telegraph reduction.

### 5.3 Gliotransmission status

Gliotransmission is optional, not foundational.

Model hierarchy:

\[
\text{SMM}_{core}
\]

= ionic/homeostatic + syncytial coupling + neuronal feedback.

Possible extensions:

\[
\text{SMM}_{Ca/IP3}
\]

\[
\text{SMM}_{gliotransmission}
\]

These extensions must earn their inclusion through model comparison.

---

## 6. Minimal Microscopic Astroglial Architecture

For astroglial domain \(j\), define at minimum extracellular potassium \(K^e_j\), astroglial intracellular ionic state \(K^A_j\), and astroglial membrane potential \(V^A_j\).

Extracellular potassium balance:

\[
\frac{dK^e_j}{dt}
=
J^N_{K,j}(X_j)
-
J^A_{K,j}
+
D_e\sum_k L^e_{jk}K^e_k .
\]

Astroglial potassium uptake:

\[
J^A_{K,j}
=
J_{\rm Kir,j}
+
2J_{\rm NKA,j}
+\cdots.
\]

Astroglial membrane dynamics:

\[
C_A\frac{dV^A_j}{dt}
=
-\left(
I_{\rm Kir,j}
+
I_{\rm NKA,j}
+
I_{\rm leak,j}
+
I_{\rm gap,j}
\right).
\]

Syncytial electrical coupling:

\[
I_{\rm gap,j}
=
\sum_k g^A_{jk}(V^A_j-V^A_k).
\]

The direct bridge back to neuronal excitability is physically constrained because extracellular K+ changes the neuronal potassium reversal potential:

\[
\boxed{
E^N_{K,j}
=
\frac{RT}{F}
\ln\frac{K^e_j}{K^N_j}
}
\]

and therefore changes neuronal membrane excitability.

---

## 7. Multiscale Reduction

The microscopic model is the **biophysical justification layer**, not necessarily the object fitted to EEG.

After linearization/coarse-graining around a physiological operating point, define a slow regional astroglial state \(g\):

\[
\tau_g \dot g
=
-A_g g
-D_gL_g g
+B_g r_N(X)
+\xi_g.
\]

The neuronal state \(X\) evolves under parameters controlled by \(g\):

\[
\boxed{
\dot X
=
F\left(
X;
\Theta_0+Mg,
W_N
\right)
+
\Sigma_N\eta.
}
\]

Here \(W_N\) is long-range neuronal connectivity, \(L_g\) is local astroglial/syncytial coupling geometry, and \(M\) maps glial state to neuronal control parameters.

The two geometries remain distinct:

\[
W_N \neq L_g.
\]

---

## 8. Core Slow-Fast Formulation

The definitive SMM is fundamentally a slow-fast control system:

\[
\boxed{
\dot X=F(X;\Theta[g],W_N)+\Sigma_N\eta
}
\]

\[
\boxed{
\dot g=\varepsilon G(g,X)+\Sigma_g\xi,
\qquad \varepsilon\ll1.
}
\]

For quasi-static \(g\), local neuronal dynamics may be linearized:

\[
\dot{\delta X}=A(g)\delta X+\eta.
\]

The frequency-domain transfer function is:

\[
H(\omega;g)=\left(i\omega I-A(g)\right)^{-1}.
\]

The neuronal cross-spectrum is:

\[
\boxed{
S_X(\omega|g)
=
H(\omega;g)QH(\omega;g)^\ast.
}
\]

This gives the formal statement of the published control-field idea: a slow glial variable need not oscillate at a neuronal frequency in order to change the gain, damping, stability, coherence, or spectral expression of that neuronal frequency.

Equivalently:

\[
A_0\rightarrow A(g),
\qquad
\lambda_k(A_0)\rightarrow\lambda_k(A(g)).
\]

This is the mathematical form of glial **control geometry**.

---

## 9. Spatial Principle

The definitive SMM does **not** posit a centimeter-scale astrocytic connectome parallel to white-matter connectivity.

Instead:

\[
\text{local astroglial coupling}
\rightarrow
\text{local neuronal parameter control}
\rightarrow
\text{long-range neuronal network}
\rightarrow
\text{macroscopic dynamics}.
\]

Thus long-distance effects can arise without requiring centimeter-scale propagation through astrocytic gap junctions.

---

## 10. Observation Principle

The model must distinguish latent neuronal dynamics from measured EEG.

\[
g(t)\rightarrow X(t)\rightarrow Y(t),
\]

where \(g(t)\) is latent slow astroglial state, \(X(t)\) is latent neuronal state, and \(Y(t)\) is measured EEG after an observation/volume-conduction operator.

The SMM must never equate a latent field directly with measured EEG.

---

## 11. Empirical Standard for the Definitive Paper

Required model hierarchy:

1. neural-only model;
2. neural + generic non-spatial slow control;
3. neural + spatial generic slow field;
4. full biologically constrained SMM;
5. optional SMM + Ca/IP3 extension;
6. optional SMM + gliotransmission extension.

The central scientific question is not whether the SMM can reproduce an EEG spectrum. It is:

\[
\boxed{
\text{Does the biologically constrained SMM predict held-out neural dynamics better than equally flexible alternatives?}
}
\]

The strongest target result is:

\[
\text{SMM}>\text{neural-only}
\]

and, more importantly,

\[
\boxed{
\text{SMM}>\text{generic slow-field control}
}
\]

on held-out human EEG.

---

## 12. Data Principles

The historical local data correspond to OpenNeuro **ds005385** (Dortmund Vital Study), not ds003633.

The definitive pipeline must return to raw EDF EEG, document all exclusions, avoid inherited \(N=43\) assumptions, treat the historical 43-subject analysis as development history, use genuinely unseen participants/conditions for confirmation, exploit eyes-open / eyes-closed, pre / post cognitive block, and longitudinal/session structure where appropriate, and compute more than PSD: robust functional connectivity, phase-gradient/traveling-wave statistics, metastability, temporal persistence, aperiodic structure, and condition-dependent state changes.

---

## 13. Canonical Definition of the Definitive SMM

> **The Syncytial Mesh Model is a multiscale dynamical model in which local gap-junction-coupled astrocytic homeostatic networks generate slow spatial control variables that modify neuronal excitability and coupling, while long-range neuronal interactions generate the measurable electromagnetic dynamics.**

This definition is canonical unless later evidence forces revision.

---

## 14. Design Rule Going Forward

No equation enters the definitive paper merely because it is convenient or visually plausible.

Each equation must satisfy one of two standards:

1. **Derived/justified mechanistically** from an explicit lower-scale system; or
2. **Declared phenomenological**, with dimensional consistency, parameter-identifiability analysis, sensitivity analysis, and explicit null comparison.

No hidden switching between those two statuses is allowed.

---

## 15. Step 2

The next task is to choose and derive the **minimal neuronal substrate** compatible with physiologically interpretable glial control, low-frequency EEG, whole-brain/mesoscale network dynamics, tractable inference, explicit observation model, metastability and phase/coherence extraction, parameter recovery, and adversarial model comparison.

Candidates to audit include Wilson-Cowan neural masses, Jansen-Rit / Wendling-type neural masses, reduced dynamic mean-field models, Hopf/Stuart-Landau models, and conductance-based population reductions.

The choice will be made on scientific and inferential grounds, not historical continuity with the old repository.
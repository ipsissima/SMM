# SMM Definitive Reconstruction — Step 3
## Deriving the astroglial syncytial mesh from ionic homeostasis and electrodiffusion

**Status:** Canonical model architecture for implementation and falsification  
**Depends on:** Step 1 foundations, Step 2 neuronal core, Step 2A K->QIF bridge, Step 2B cellular calibration  
**Primary principle:** the mesh must be derived from physiological transport, not imposed as a wave equation.

---

# 1. Scientific objective

Define a minimal astroglial subsystem that simultaneously:

1. respects ionic mass balance;
2. separates extracellular from astroglial ionic compartments;
3. includes astrocytic membrane transport;
4. includes gap-junction-mediated syncytial redistribution;
5. includes extracellular diffusion;
6. couples bidirectionally to neuronal population activity;
7. admits a controlled reduction to regional/mesoscale dynamics;
8. has no intrinsic oscillatory mechanism inserted by hand.

The target is not a detailed simulation of every astrocyte.

The target is a **biophysically derived effective mesh** suitable for whole-brain inference.

---

# 2. Reference microscopic model: electrodiffusive truth layer

The reference layer follows the Kirchhoff-Nernst-Planck / electrodiffusive tradition of astrocyte-ECS modelling.

At fine-scale astroglial domain \(j\), distinguish:

- extracellular compartment \(e\);
- astroglial compartment \(a\).

For ion species

\[
s\in\{K^+,Na^+,Cl^-\}
\]

define concentrations

\[
c^{e}_{s,j},\qquad
c^{a}_{s,j},
\]

and electrical potentials

\[
\phi^{e}_{j},\qquad
\phi^{a}_{j}.
\]

Ca2+ is not required in the core potassium-buffering model.

It may be added later as a signalling extension.

---

# 3. Electrodiffusive transport

For an ion species \(s\) with valence \(z_s\), transport along either the extracellular domain or the astroglial syncytium follows a Nernst-Planck-type flux.

For an astroglial edge \(j\leftrightarrow k\):

\[
\boxed{
J^{a}_{s,jk}
=
-\mathcal G^{a}_{jk}
\frac{D^{a}_s}{\lambda_a^2}
\left[
\frac{c^a_{s,k}-c^a_{s,j}}{\ell_{jk}}
+
\frac{z_sF}{RT}
\bar c^a_{s,jk}
\frac{\phi^a_k-\phi^a_j}{\ell_{jk}}
\right]
}
\]

where:

- \(D_s^a\) is the free diffusion coefficient;
- \(\lambda_a\) is an effective tortuosity / hindrance term;
- \(\mathcal G^a_{jk}\) is the effective gap-junction coupling area/permeability;
- \(\ell_{jk}\) is edge length;
- \(\bar c\) is an edge concentration average.

For extracellular transport:

\[
\boxed{
J^{e}_{s,jk}
=
-\mathcal G^{e}_{jk}
\frac{D^{e}_s}{\lambda_e^2}
\left[
\frac{c^e_{s,k}-c^e_{s,j}}{\ell_{jk}}
+
\frac{z_sF}{RT}
\bar c^e_{s,jk}
\frac{\phi^e_k-\phi^e_j}{\ell_{jk}}
\right].
}
\]

Thus the syncytium does not transmit a fictional displacement variable.

It transmits ions and electrical influence by diffusion plus field-driven migration.

---

# 4. Membrane potassium flux

Define positive astrocytic membrane potassium flux as movement from extracellular space into the astrocyte:

\[
J^A_{K,j}>0
\quad\Longleftrightarrow\quad
K^+:e\rightarrow a.
\]

The total flux is

\[
\boxed{
J^A_{K,j}
=
J_{K,\mathrm{Kir},j}
+
J_{K,\mathrm{NKA},j}
+
J_{K,\mathrm{other},j}.
}
\]

The model must not assume that Kir4.1 is always an inward uptake channel.

Its direction follows the electrochemical driving force.

Let

\[
E^A_{K,j}
=
\frac{RT}{F}
\ln\frac{K^e_j}{K^a_j}.
\]

If outward ionic current is defined as positive,

\[
I_{\mathrm{Kir},j}
=
g_{\mathrm{Kir},j}
\,\rho_{\mathrm{Kir}}(V^A_j,E^A_{K,j})
\,
(V^A_j-E^A_{K,j}),
\]

and therefore

\[
J_{K,\mathrm{Kir},j}
=
-\frac{I_{\mathrm{Kir},j}}{F}.
\]

The rectification factor \(\rho_{\mathrm{Kir}}\) is calibrated from published astrocyte electrophysiology.

---

# 5. Na/K-ATPase

The astrocytic Na/K pump transports:

\[
3Na^+:\ a\rightarrow e,
\]

\[
2K^+:\ e\rightarrow a.
\]

Let the pump cycle rate be

\[
P_j
=
P_{\max}
f_{\mathrm{Na}}(Na^a_j)
f_K(K^e_j),
\]

with saturating kinetics taken from a validated astrocyte model.

Then

\[
\boxed{
J_{K,\mathrm{NKA},j}=2P_j
}
\]

and

\[
J_{Na,\mathrm{NKA},j}=-3P_j
\]

under the positive-inward convention.

No claim is made that a single transport mechanism dominates every physiological regime.

The total derivative of \(J_K^A\), not a verbal assignment of "uptake to Kir", determines the reduced model.

---

# 6. Astrocytic membrane voltage

At the detailed reference level:

\[
C_A\dot V^A_j
=
-
\left(
I_{\mathrm{Kir},j}
+
I_{\mathrm{Na},j}
+
I_{\mathrm{Cl},j}
+
I_{\mathrm{NKA},j}
+
I_{\mathrm{gap},j}
\right).
\]

Alternatively, in a KNP implementation, potentials are obtained consistently from electroneutrality / charge-capacitor constraints.

The second route is preferred for the reference truth model.

Important distinction:

\[
\boxed{
\text{rapid electrical isopotentiality}
\neq
\text{instantaneous ionic redistribution}.
}
\]

Gap junctions can rapidly equalize astroglial voltage while actual ionic mass transport remains slower.

The definitive SMM must never conflate these two processes.

---

# 7. Fine-scale mass balance

Let \(\phi_e\) and \(\phi_a\) be extracellular and astroglial volume fractions.

For extracellular potassium:

\[
\boxed{
\phi_e\frac{dK^e_j}{dt}
=
S^N_{K,j}
-
A_mJ^A_{K,j}
-
J^N_{\mathrm{reuptake},j}
-
\sum_k J^e_{K,jk}.
}
\]

For astroglial potassium:

\[
\boxed{
\phi_a\frac{dK^a_j}{dt}
=
A_mJ^A_{K,j}
-
\sum_kJ^a_{K,jk}.
}
\]

Here:

- \(S^N_K\) is neuronal K release;
- \(J^N_{\rm reuptake}\) is neuronal return to the intracellular neuronal reservoir;
- \(A_m\) is astrocytic membrane area per tissue volume.

The astrocytic membrane term appears with opposite signs and therefore conserves potassium between ECS and astroglia.

---

# 8. Neuronal source and feedback

From Step 2B:

\[
\boxed{
S^N_{K,j}
=
\Gamma_E R_{E,j}
+
\Gamma_I R_{I,j}
+
S_{K,\mathrm{sub}}(V_E,V_I).
}
\]

The feedback onto the neuronal mass is:

\[
\boxed{
\eta_{q,j}
=
\eta_{q0}
+
\chi_q
\ln\frac{K^e_j}{K_{e0}},
\qquad q\in\{E,I\}.
}
\]

Both directions are therefore calibrated independently of EEG.

---

# 9. Why the core model must include both local uptake and syncytial transport

Current evidence does not justify a simplistic statement:

\[
\text{"Kir4.1 + gap junctions are the sole K clearance mechanism"}.
\]

Instead, physiological K homeostasis is redundant and state dependent.

Important empirical constraints:

- astrocytic Na/K-ATPase contributes substantially to K uptake;
- Kir4.1 strongly shapes astrocyte membrane K conductance and local excitability control;
- Kir flux can reverse direction depending on electrochemical driving force;
- gap-junction coupling redistributes ionic/electrical load across the syncytium;
- gap-junction-dependent effects can be more visible during sustained / stronger activity;
- individual astrocytes can dissipate moderate K loads locally.

Therefore the core model separates:

\[
\boxed{
\text{local buffering}
}
\]

from

\[
\boxed{
\text{syncytial redistribution}.
}
\]

This separation creates a direct experimental/null-model test of the "mesh" component.

---

# 10. Reduced mesoscale variables

The full electrodiffusive model is too large for whole-brain inference.

Linearize around a physiological equilibrium:

\[
K^e=K_{e0}+k,
\qquad
K^a=K_{a0}+a.
\]

Here:

- \(k_j\) = excess extracellular K;
- \(a_j\) = excess astroglial K load.

Linearize total astrocytic membrane transport:

\[
J^A_K
\simeq
\kappa_e k-\kappa_a a.
\]

The coefficients are derivatives of the detailed membrane model:

\[
\boxed{
\kappa_e
=
\left.
\frac{\partial J^A_K}{\partial K^e}
\right|_0
}
\]

and

\[
\boxed{
\kappa_a
=
-
\left.
\frac{\partial J^A_K}{\partial K^a}
\right|_0.
}
\]

They are not EEG fitting coefficients.

Let neuronal reuptake linearize as

\[
J^N_{\rm reuptake}\simeq\kappa_N k.
\]

---

# 11. The definitive reduced syncytial mesh

Let:

- \(L_e\) = physical extracellular-space diffusion Laplacian;
- \(L_g\) = astroglial gap-junction / syncytial Laplacian;
- \(D_e\) = effective ECS K diffusivity;
- \(D_g\) = effective syncytial K transport coefficient.

Then:

\[
\boxed{
\phi_e\dot k
=
S_K^N
-
(\kappa_e+\kappa_N)k
+
\kappa_a a
-
D_eL_e k
}
\tag{G1}
\]

\[
\boxed{
\phi_a\dot a
=
\kappa_e k
-
\kappa_a a
-
D_gL_g a.
}
\tag{G2}
\]

These two equations are the **minimal canonical SMM mesh**.

They replace the old damped wave PDE.

---

# 12. What the variable \(a\) means

\(a\) is not a calcium amplitude and not an arbitrary field.

It is a coarse-grained **astroglial ionic buffering load**.

At the microscopic level it summarizes the perturbation in intracellular astroglial ionic content associated with K handling and redistribution.

At regional level it functions as a slow hidden state encoding how much ionic load has been absorbed and spatially redistributed by the local syncytium.

This is a genuine physiological state variable.

---

# 13. Mathematical character: the mesh is dissipative, not oscillatory

Suppose, initially, that the ECS and astroglial operators share spatial eigenfunctions:

\[
L\varphi_m=\lambda_m\varphi_m,
\qquad
\lambda_m\ge0.
\]

For each spatial mode \(m\),

\[
\frac{d}{dt}
\begin{pmatrix}
k_m\\
a_m
\end{pmatrix}
=
M(\lambda_m)
\begin{pmatrix}
k_m\\
a_m
\end{pmatrix}
+
\begin{pmatrix}
S_m/\phi_e\\
0
\end{pmatrix},
\]

where

\[
M(\lambda)
=
\begin{pmatrix}
-\dfrac{\kappa_e+\kappa_N+D_e\lambda}{\phi_e}
&
\dfrac{\kappa_a}{\phi_e}
\\[1.2ex]
\dfrac{\kappa_e}{\phi_a}
&
-\dfrac{\kappa_a+D_g\lambda}{\phi_a}
\end{pmatrix}.
\]

Its eigenvalues are

\[
\boxed{
\mu_{\pm}(\lambda)
=
-\frac12(A_\lambda+B_\lambda)
\pm
\frac12
\sqrt{
(A_\lambda-B_\lambda)^2
+
\frac{4\kappa_a\kappa_e}{\phi_e\phi_a}
}
}
\]

with

\[
A_\lambda
=
\frac{\kappa_e+\kappa_N+D_e\lambda}{\phi_e},
\]

\[
B_\lambda
=
\frac{\kappa_a+D_g\lambda}{\phi_a}.
\]

The discriminant is strictly non-negative:

\[
\boxed{
(A_\lambda-B_\lambda)^2
+
\frac{4\kappa_a\kappa_e}{\phi_e\phi_a}
>0.
}
\]

Therefore the core ionic mesh has **real relaxation modes**, not intrinsic oscillatory wave modes.

With positive physiological rates and \(\kappa_N>0\),

\[
\mu_\pm(\lambda)<0.
\]

Thus:

\[
\boxed{
\text{the astroglial mesh is a stable dissipative control system}.
}
\]

Observable delta/theta oscillation must arise in the neuronal subsystem.

---

# 14. How spatial modes survive without resonance

Although the glial subsystem does not oscillate, it still has spatial modes.

Increasing \(\lambda_m\) strengthens the negative diffusion terms.

Thus high-spatial-frequency modes decay more rapidly.

The slowest surviving perturbations are therefore spatially smooth low modes.

This yields a biologically grounded version of one of the useful intuitions in the old SMM:

\[
\boxed{
\text{low spatial modes are persistent}
}
\]

but not because they are standing-wave resonances.

They are persistent because diffusion and syncytial redistribution are spatial low-pass processes.

This is a major conceptual repair.

---

# 15. Astroglial memory kernel

Equation (G2) can be solved formally for \(a\):

\[
a(t)
=
e^{-B_gt}a(0)
+
\frac{\kappa_e}{\phi_a}
\int_0^t
e^{-B_g(t-s)}k(s)\,ds,
\]

where

\[
B_g
=
\frac{\kappa_a I+D_gL_g}{\phi_a}.
\]

Substituting into (G1) gives a neuronal-interface variable \(k(t)\) with a spatially structured memory term.

Therefore the syncytium acts as a **mode-dependent physiological memory kernel**.

This is a rigorous implementation of the published idea that glial control variables can persist longer than the neuronal oscillations they regulate.

---

# 16. Relation to the old telegraph equation

Eliminating \(a\) from the two-state system produces a second-order-in-time relaxation equation for \(k\).

This superficial resemblance to a telegraph equation does **not** imply propagating glial waves.

The exact modal roots remain real in the physiological linear regime.

Thus the correct statement is:

> A second-order effective equation can emerge after eliminating a hidden astroglial buffer state, but its origin is coupled ionic relaxation, not elastic inertia or a calcium-wave carrier.

This provides a mathematically honest bridge to the historical SMM without preserving the invalid wave interpretation.

---

# 17. What mathematically constitutes the "mesh"

The definitive mesh is not one scalar PDE.

It is the operator

\[
\boxed{
\mathcal M_g
=
(\kappa_e,\kappa_a,D_g,L_g,\phi_a)
}
\]

acting on the hidden astroglial load \(a\), coupled through the extracellular interface \(k\).

Its distinctive property is that

\[
L_g
\]

is a spatial operator physically implemented by astrocytic syncytial coupling.

Without \(L_g\), there may still be astrocytic local buffering, but there is no syncytial mesh.

This yields a clean definition:

\[
\boxed{
\text{astrocyte contribution}
=
\text{local buffering}
+
\text{syncytial redistribution}.
}
\]

The specifically **mesh** contribution is the second term.

---

# 18. Spatial geometry of \(L_g\)

The core model does not assume a whole-brain small-world astrocyte graph.

At the fine scale, edges represent local gap-junction coupling.

At larger scales, homogenization yields either:

\[
\nabla\cdot(D_g(\mathbf x)\nabla a)
\]

on cortical tissue, or a graph operator

\[
L_g
\]

on a spatial cortical mesh.

Long-range neuronal connectivity remains separately represented by

\[
W_N.
\]

Canonical constraint:

\[
\boxed{
L_g\neq W_N.
}
\]

No direct centimeter-scale gap-junction edge is added merely to improve EEG fit.

---

# 19. Anisotropy

If tissue architecture supports directional differences, replace scalar \(D_g\) by a tensor:

\[
\mathbf D_g(\mathbf x).
\]

Then:

\[
\partial_t a
=
\cdots
+
\nabla\cdot(\mathbf D_g\nabla a).
\]

Anisotropy must be:

- measured;
- literature constrained;
- or tested as a prespecified extension.

It must not be inferred as an unconstrained high-dimensional field from the EEG dataset.

---

# 20. Core model variants / adversarial ablations

The mesh hypothesis becomes experimentally separable.

## \(G_0\): no astroglia

\[
J^A_K=0,\qquad D_g=0.
\]

## \(G_1\): local astrocytic buffering only

\[
\kappa_e,\kappa_a>0,\qquad D_g=0.
\]

## \(G_2\): full syncytial mesh

\[
\kappa_e,\kappa_a,D_g>0.
\]

## \(G_3\): generic spatial slow field

Same number of effective degrees of freedom and matched time constants, but no ionic/Nernst constraints.

The crucial comparison is:

\[
\boxed{
G_2>G_1
}
\]

for evidence that **syncytial redistribution** matters, and

\[
\boxed{
G_2>G_3
}
\]

for evidence that the **astroglial biophysical constraints** add predictive value beyond a generic slow spatial field.

---

# 21. Perturbation mapping

The reduced coefficients have direct biological perturbations.

### Kir4.1 perturbation

Changes the derivatives embedded in

\[
\kappa_e,\kappa_a
\]

and astrocytic membrane response.

### Na/K-ATPase perturbation

Changes local uptake kinetics and therefore

\[
\kappa_e,\kappa_a.
\]

### Cx30/Cx43 perturbation

Primarily changes

\[
D_g
\]

and potentially the effective geometry \(L_g\).

### Extracellular-space/tortuosity change

Changes

\[
D_e.
\]

This gives the model unusually direct experimental falsification routes.

---

# 22. Important state-dependence prediction

Experimental literature indicates that the contribution of gap-junction-dependent redistribution need not be equally large at every activity level.

Therefore the definitive SMM predicts, rather than assumes, that the mesh contribution may be activity dependent.

Possible ordering:

\[
\text{rest / low load}
:
G_2-G_1\ \text{small}
\]

while

\[
\text{sustained activity / recovery}
:
G_2-G_1\ \text{larger}.
\]

This is especially relevant to datasets containing pre/post cognitive activity.

The effect size is an empirical question.

---

# 23. Whole-system core after Step 3

The minimal definitive SMM is now:

## Neuronal subsystem

\[
\dot X
=
F_{\rm MPR}(X;W_N,\eta_E(K_e),\eta_I(K_e)).
\]

## Neuronal K source

\[
S_K^N
=
S_K(R_E,V_E,R_I,V_I).
\]

## Extracellular interface

\[
\phi_e\dot k
=
S_K^N
-
(\kappa_e+\kappa_N)k
+
\kappa_a a
-
D_eL_e k.
\]

## Astroglial mesh

\[
\phi_a\dot a
=
\kappa_e k
-
\kappa_a a
-
D_gL_g a.
\]

## Observation model

\[
Y
=
\mathcal H[X]+\epsilon.
\]

This is a closed causal model:

\[
\boxed{
X
\rightarrow
K_e
\rightarrow
\text{astroglial mesh}
\rightarrow
K_e
\rightarrow
X
\rightarrow
EEG.
}
\]

---

# 24. What Step 3 has achieved

The term **Syncytial Mesh** now has a precise physical meaning.

It no longer means:

> a global wave-supporting lattice inspired by metal mesh.

It means:

> a spatial operator generated by locally coupled astrocytic ionic homeostasis, whose hidden buffer state redistributes activity-dependent extracellular ionic load and thereby changes neuronal operating points over slow timescales.

That is a mechanistic, measurable and falsifiable definition.

---

# 25. Failure criteria

The core mesh hypothesis fails or must be weakened if:

1. calibrated \(D_g\) is too small to produce any measurable mesoscale effect in the relevant physiological regime;2. \(G_2\) does not outperform local buffering \(G_1\);
3. a generic matched slow field \(G_3\) explains the data equally well or better;
4. the required K excursions exceed physiological bounds;
5. the model requires unrealistically strong gap-junction coupling;
6. estimated effects exist only because of EEG volume conduction;
7. the result disappears in held-out subjects/conditions.

---

# 26. Immediate next step: Step 3A implementation/calibration

Before whole-brain EEG fitting:

1. reproduce a published electrodiffusive astrocyte-ECS K-buffering model;
2. reproduce the effect of weak vs strong glial coupling;
3. linearize numerically around physiological baseline;
4. estimate \(\kappa_e,\kappa_a,D_e,D_g\);
5. validate the reduced two-state model against the full electrodiffusive model;
6. quantify the range of activity for which reduction error is acceptably small;
7. freeze coefficient ranges and uncertainties;
8. only then couple the reduced mesh to MPR.

The implementation should treat the full electrodiffusive model as the **reference truth layer** and the two-state SMM mesh as its **validated coarse-grained reduction**.
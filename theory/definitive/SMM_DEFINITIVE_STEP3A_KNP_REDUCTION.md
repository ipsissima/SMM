# SMM Definitive Reconstruction — Step 3A
## Reproduce the electrodiffusive astrocyte–ECS reference system and prove the reduced Syncytial Mesh

**Status:** Canonical calibration and model-reduction protocol  
**Language:** English  
**Target:** Nature Communications-level mechanistic validation  
**Dependencies:** Steps 1, 2, 2A, 2B, and 3  
**Hard rule:** no EEG data may be used anywhere in this step.

---

# 1. Scientific purpose

Step 3 proposed a minimal two-state astroglial mesh:

\[
\phi_e\dot k
=
S_K^N
-
(\kappa_e+\kappa_N)k
+
\kappa_a a
-
D_eL_e k,
\tag{G1}
\]

\[
\phi_a\dot a
=
\kappa_e k
-
\kappa_a a
-
D_gL_g a.
\tag{G2}
\]

Here

- \(k\) is excess extracellular potassium;
- \(a\) is excess astroglial ionic buffering load;
- \(L_e\) is extracellular spatial transport geometry;
- \(L_g\) is the astroglial syncytial transport geometry.

These equations are **not yet entitled to be the definitive SMM mesh**.

Step 3A must establish whether they are a quantitatively valid coarse-grained reduction of a published, ion-conserving, electrodiffusive astrocyte–ECS model over the physiological operating range relevant to the SMM.

The central question is:

\[
\boxed{
\text{Can the full electrodiffusive astrocyte/ECS dynamics be reduced to a small,
stable, spatially coupled K-control system without losing the dynamics that matter
for neuronal control?}
}
\]

If the answer is no, the reduced SMM must acquire the additional state(s) required by the reference model.

---

# 2. Reference-model hierarchy

We use a three-level reference hierarchy.

## Reference R1 — primary reduction target

**Halnes et al. (2013), _Electrodiffusive Model for Astrocytic and Neuronal Ion Concentration Dynamics_, PLOS Computational Biology 9:e1003386.**

This is the primary theoretical anchor because it:

- explicitly models astrocytic intracellular space and extracellular space;
- enforces particle and charge conservation;
- uses generalized Nernst–Planck transport;
- derives electrical potentials consistently from ionic dynamics;
- contains Na\(^+\), K\(^+\), and Cl\(^-\);
- includes Kir-like K conductance and Na/K-ATPase;
- directly compares spatial buffering, local astrocytic storage, and ECS diffusion.

The numerical implementation used for reproduction will be the authors' later open-source **`ffian.zero_flow_model`**, which states explicitly that it implements the Halnes et al. 2013 model with minor adjustments.

This avoids reimplementing a complex electrodiffusive model from prose and accidentally creating a different system.

## Reference R2 — higher-fidelity robustness target

**Sætra, Ellingsrud & Rognes (2023), _Neural activity induces strongly coupled electro-chemo-mechanical interactions and fluid flow in astrocyte networks and extracellular space_, PLOS Computational Biology 19:e1010996.**

Its open `ffian.flow_model` extends the zero-flow electrodiffusive system with:

- variable compartment volumes;
- osmotic forces;
- hydrostatic pressures;
- transmembrane water movement;
- intracellular and extracellular fluid flow;
- advective ionic transport.

R2 is **not** the model that the SMM must carry into whole-brain inference.

It is the adversarial high-fidelity test of whether omission of fluid/volume degrees of freedom invalidates the proposed reduced mesh.

## Reference R3 — neuron–ECS–glia consistency target

**Sætra, Einevoll & Halnes (2021), _An electrodiffusive neuron-extracellular-glia model for exploring the genesis of slow potentials in the brain_, PLOS Computational Biology 17:e1008143.**

The open `edNEGmodel` couples:

- neuron;
- extracellular space;
- glia;

inside the same KNP/electrodiffusive framework.

R3 is used to verify that the way we later connect MPR-derived neuronal K release to the astroglial mesh is compatible with an explicit ion-conserving neuron–glia model.

---

# 3. Experimental constraints external to the computational references

The reference-model reduction must also respect several experimentally established directional constraints.

## Kir4.1

Glial-conditional Kir4.1 loss slows recovery from moderate activity-induced extracellular K elevations in vivo.

Therefore the reduced model must predict:

\[
g_{\mathrm{Kir}}\downarrow
\quad\Longrightarrow\quad
\tau_{K_e,\mathrm{recovery}}\uparrow
\]

in the physiological regime.

## Gap-junctional coupling

Experimental work in hippocampus shows that astrocytic gap-junction coupling contributes to potassium buffering.

Therefore:

\[
D_g\downarrow
\]

must not improve syncytial redistribution.

However, the model must **not** assume that gap-junction transport is the only important K-clearance mechanism.

Local uptake and syncytial redistribution remain separate mechanisms.

---

# 4. Exact primary numerical system

The pinned zero-flow reference implementation uses:

\[
s\in\{\mathrm{Na},\mathrm K,\mathrm{Cl}\},
\]

with concentrations

\[
c^s_i(x,t),\qquad c^s_e(x,t),
\]

and potentials

\[
\phi_i(x,t),\qquad \phi_e(x,t).
\]

The spatial ion flux in compartment \(r\in\{i,e\}\) is of generalized Nernst–Planck form:

\[
\boxed{
J^s_r
=
-\frac{D_s}{\lambda_r^2}
\left[
\nabla c^s_r
+
\frac{z_sF}{RT}c^s_r\nabla\phi_r
\right].
}
\tag{R1}
\]

Thus transport contains both:

- chemical diffusion;
- electrical migration.

The conservation equations have the generic structure

\[
\alpha_i\partial_t c_i^s
=
-\gamma_m j_m^s
-\nabla\cdot(\alpha_iJ_i^s),
\tag{R2}
\]

\[
\alpha_e\partial_t c_e^s
=
+\gamma_m j_m^s
-\nabla\cdot(\alpha_eJ_e^s)
+S_e^s.
\tag{R3}
\]

The reference model enforces the electrodiffusive charge constraints required to obtain \(\phi_i,\phi_e\) consistently from the concentration state.

This is crucial: voltage is not independently prescribed.

---

# 5. Membrane mechanisms in the pinned reference implementation

At the astrocytic membrane:

\[
E_s=\frac{RT}{z_sF}\ln\frac{c_e^s}{c_i^s}.
\]

The implementation contains:

- passive Na conductance;
- passive Cl conductance;
- inward-rectifying K conductance;
- Na/K-ATPase.

The pump cycle rate has the form

\[
P
=
\rho_{\rm pump}
\left[
\frac{Na_i^{3/2}}
{Na_i^{3/2}+P_{Na}^{3/2}}
\right]
\left[
\frac{K_e}
{K_e+P_K}
\right].
\tag{R4}
\]

With outward membrane flux defined as positive, total K membrane flux is

\[
\boxed{
j_m^K=j_{\mathrm{Kir}}-2P.
}
\tag{R5}
\]

The sign of the Kir contribution is determined by the electrochemical driving force.

We therefore do not encode "Kir = uptake" as a rule.

---

# 6. Pinned reference baseline

The open `ffian.zero_flow_model` baseline will be version-pinned before analysis.

The published implementation currently specifies:

| Quantity | Baseline |
|---|---:|
| Temperature | \(310.15\) K |
| \(D_{Na}\) | \(1.33\times10^{-9}\ {\rm m^2/s}\) |
| \(D_K\) | \(1.96\times10^{-9}\ {\rm m^2/s}\) |
| \(D_{Cl}\) | \(2.03\times10^{-9}\ {\rm m^2/s}\) |
| Astroglial/ICS volume fraction | \(0.4\) |
| ECS volume fraction | \(0.2\) |
| ICS tortuosity | \(3.2\) |
| ECS tortuosity | \(1.6\) |
| membrane area/volume | \(8.0\times10^6\ {\rm m^{-1}}\) |
| \(g_{Na}\) | \(1.0\ {\rm S/m^2}\) |
| \(g_{Cl}\) | \(0.5\ {\rm S/m^2}\) |
| \(g_K\) | \(16.96\ {\rm S/m^2}\) |
| max pump rate | \(1.12\times10^{-6}\ {\rm mol\,m^{-2}s^{-1}}\) |
| pump \(Na_i\) half-scale | \(10.0\ {\rm mol/m^3}\) |
| pump \(K_e\) half-scale | \(1.5\ {\rm mol/m^3}\) |

The baseline equilibrium in the same implementation is approximately:

\[
Na_i=15.4746\ {\rm mM},
\]

\[
K_i=99.8921\ {\rm mM},
\]

\[
Cl_i=5.36369\ {\rm mM},
\]

\[
Na_e=144.0908\ {\rm mM},
\]

\[
K_e=3.21580\ {\rm mM},
\]

\[
Cl_e=133.2726\ {\rm mM},
\]

\[
\phi_i=-85.861\ {\rm mV},
\qquad
\phi_e=0.
\]

These values are reference-model values, **not universal physiological constants**.

Their role is to make the reproduction exact.

---

# 7. Input must preserve charge consistency

We will not inject K alone into the KNP reference model.

The published Halnes framework explicitly requires external input to preserve local charge consistency.

Therefore the canonical neuronal-activity proxy in R1 is:

\[
S_e^K=+u(x,t),
\]

\[
S_e^{Na}=-u(x,t),
\]

\[
S_e^{Cl}=0.
\]

This reproduces the reference-model stimulation convention.

Only after validating the reduction may Step 2B's explicit neuronal membrane K source replace this artificial reference input.

---

# 8. Reproduction tests before reduction

Before fitting any reduced model, the implementation must reproduce the qualitative and quantitative reference behavior.

Required R1 checks:

1. stable resting equilibrium;
2. local extracellular K rise under the published charge-neutral input;
3. astrocytic depolarization;
4. local K uptake in the stimulated region;
5. intracellular/syncytial redistribution away from the stimulated region;
6. distal K release where the extracellular concentration is lower;
7. recovery after input termination;
8. comparison of:
   - full astrocyte spatial buffering;
   - local storage without spatial redistribution;
   - ECS diffusion without astrocyte.

No model reduction is attempted until these are reproduced.

---

# 9. Small-signal regime for mathematical reduction

Let the full reference state be

\[
x(x,t)=x_0+\delta x(x,t),
\]

where \(x\) contains all six ion concentrations plus constrained electrical variables.

Define the physiological perturbation set

\[
\mathcal P_{\rm lin}
\]

as activity-induced K deviations small enough that the reference-model response is approximately linear around \(x_0\).

The exact amplitude boundary is determined numerically, not assumed.

For candidate amplitude \(A\), test superposition:

\[
y[u_1+u_2]
\approx
y[u_1]+y[u_2]
\]

and scaling:

\[
y[c\,u]
\approx
c\,y[u].
\]

Define normalized nonlinearity error

\[
\epsilon_{\rm NL}(A)
=
\frac{
\|y_{2A}-2y_A\|_2
}{
\|2y_A\|_2
}.
\]

The linear reduction range is the largest range for which \(\epsilon_{\rm NL}\) remains acceptably small under prespecified tolerance analysis.

---

# 10. Linearized full system

After spatial discretization and linearization about equilibrium, write the reference system as a descriptor system

\[
\boxed{
E\dot{\delta x}
=
A\delta x
+
Bu,
}
\tag{L1}
\]

\[
y=C\delta x.
\tag{L2}
\]

The descriptor form is important because the KNP electrical constraints need not produce a naïve unconstrained ODE.

The primary output is

\[
y_K=\delta K_e.
\]

Additional diagnostic outputs include:

\[
\delta K_i,\quad
\delta Na_i,\quad
\delta Na_e,\quad
\delta Cl_i,\quad
\delta Cl_e,\quad
\delta\phi_m.
\]

---

# 11. Spatial modal decomposition

For the spatial operator of the reference domain, use eigenfunctions

\[
L\varphi_m=\lambda_m\varphi_m.
\]

Project both input and outputs into these spatial modes.

For each \(m\):

\[
E_m\dot x_m
=
A_m x_m+B_m u_m.
\]

This separates two questions:

1. what are the temporal poles of a given spatial mode?
2. how do those poles change with spatial wavenumber \(\lambda_m\)?

The definitive SMM predicts a spatial low-pass structure:

\[
\lambda_m\uparrow
\quad\Longrightarrow\quad
\text{faster decay of high-spatial-frequency perturbations}.
\]

That prediction must emerge from R1 rather than be imposed.

---

# 12. Transfer-function identification

For each spatial mode define the reference transfer function

\[
\boxed{
H_m(s)
=
C_m(sE_m-A_m)^{-1}B_m.
}
\tag{L3}
\]

The central reduction question is whether \(H_m(s)\), over the frequency range relevant to astroglial control, is dominated by two real stable poles.

We explicitly test:

\[
H_m(s)
\approx
\frac{r_{m1}}{s-p_{m1}}
+
\frac{r_{m2}}{s-p_{m2}},
\]

with

\[
p_{m1},p_{m2}<0.
\]

If a third pole carries non-negligible slow-system weight, the two-state reduction is rejected.

---

# 13. What counts as a dynamically relevant pole

A pole is not retained merely because it exists in the full system.

For each mode, quantify:

- timescale;
- residue magnitude in \(K_e\);
- contribution to impulse-response energy;
- contribution across the physiological control-frequency band;
- sensitivity to perturbation amplitude.

The control-frequency band is defined from the actual cellular/astroglial timescales and later checked against the EEG experiment, but **not selected using EEG fit**.

Fast electrical constraint modes that contribute negligibly to slow \(K_e\) behavior may be adiabatically eliminated.

Slow Na/pump or volume modes may not.

---

# 14. Physical reduced coordinates

If a two-dimensional slow subspace exists, we do not keep arbitrary balanced coordinates.

We require a physically interpretable coordinate transformation.

Coordinate 1 is fixed:

\[
\boxed{k=\delta K_e.}
\]

Candidate coordinate 2 is:

\[
\boxed{
a=
\text{astroglial excess K load per tissue volume}.
}
\]

At the reference level this can be constructed from the astroglial K concentration perturbation and volume fraction:

\[
a
\propto
\alpha_i\,\delta K_i
\]

with the exact normalization chosen so that total K mass balance is explicit.

The reduced coordinate must correlate strongly with the second dominant slow mode.

If it does not, the second physical state must be reidentified from the dominant eigenvector composition.

---

# 15. Candidate reduced model to be tested

The constrained two-state family is

\[
\boxed{
\phi_e\dot k
=
u
-
(\kappa_e+\kappa_N)k
+
\kappa_a a
-
D_eL_e k,
}
\tag{M1}
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
\tag{M2}
\]

The coefficients are shared across spatial modes.

Mode dependence enters **only** through the physical spatial eigenvalues.

This prevents the reduction from cheating by assigning a separate time constant to every spatial pattern.

---

# 16. Parameter extraction

Parameter estimation is performed against R1, not EEG.

The order is:

## 16.1 Direct quantities

Use directly:

- \(\phi_e\);
- \(\phi_a\);
- reference spatial geometry.

## 16.2 Effective extracellular transport

Estimate \(D_e\) from the full reference model with astrocytic membrane transport disabled, using K/Na charge-consistent perturbations.

## 16.3 Effective syncytial transport

Estimate \(D_g\) from the intracellular redistribution component of the reference model.

In the homogenized reference, this is an **effective astroglial transport coefficient** that subsumes:

- ionic diffusivity;
- tortuosity;
- syncytial pathway geometry.

It must not be mislabeled as a single gap-junction permeability.

## 16.4 Local membrane exchange

Estimate

\[
\kappa_e,\kappa_a
\]

from the linearized membrane-flux Jacobian plus system-level refinement.

Initial derivatives come from:

\[
\frac{\partial J_K^A}{\partial K_e},
\qquad
\frac{\partial J_K^A}{\partial K_i},
\]

at equilibrium.

Then all four dynamic coefficients are jointly refined against the full modal impulse responses under positivity and mass-conservation constraints.

---

# 17. Global fitting across modes

Let \(\hat H_m\) be the reduced-model transfer function.

Estimate the shared parameter vector

\[
\theta_g=
(\kappa_e,\kappa_a,D_e,D_g)
\]

by minimizing

\[
\boxed{
\mathcal L(\theta_g)
=
\sum_{m\in\mathcal M_{\rm train}}
w_m
\int_{\Omega_c}
\left|
H_m(i\omega)
-
\hat H_m(i\omega;\theta_g)
\right|^2d\omega.
}
\tag{F1}
\]

Constraints:

\[
\kappa_e,\kappa_a,D_e,D_g\ge0.
\]

The modal training set excludes some spatial modes for genuine reduction validation.

---

# 18. Modal holdout

Do not fit all modes.

Example:

- train on low modes \(m=0,1,3,5\);
- hold out \(m=2,4,6\) and higher modes within numerical reliability.

The precise partition is preregistered in the analysis configuration.

The reduction succeeds only if shared coefficients predict held-out modes.

This is a stringent test of whether the Laplacian structure is genuine.

---

# 19. Nonlinear perturbation validation

After fitting exclusively in the small-signal regime, freeze all reduced parameters.

Then challenge the reduction with nonlinear but physiological R1 simulations.

Required perturbations:

### P1 — brief local pulse
Tests impulse recovery.

### P2 — sustained local activity
Tests load accumulation and recovery.

### P3 — two separated active zones
Tests superposition breakdown and spatial interaction.

### P4 — spatially broad low-amplitude drive
Tests low spatial modes.

### P5 — narrow high-amplitude but still physiological drive
Tests the boundary of the reduction.

### P6 — post-stimulation recovery
Tests the predicted astroglial memory state.

No reduced-model refitting is allowed on P1–P6.

---

# 20. Comparison metrics

For \(\delta K_e(x,t)\), compare full and reduced systems on:

1. peak K elevation;
2. time to peak;
3. recovery time;
4. area under the K-excess curve;
5. spatial center of mass;
6. spatial variance / spread;
7. modal amplitudes;
8. modal decay times;
9. distal K redistribution;
10. total K mass error.

Primary normalized trajectory error:

\[
\mathrm{NRMSE}
=
\frac{
\|K^{\rm full}_e-K^{\rm red}_e\|_2
}{
\|K^{\rm full}_e-K_{e0}\|_2
}.
\]

All metrics are reported, not only a composite score.

---

# 21. Predefined acceptance standard

The two-state mesh is accepted as the core reduction only if:

### Linear/modal criteria

- all retained poles are stable;
- the two dominant slow poles explain at least 95% of slow-response energy for the trained low spatial modes;
- held-out spatial modes are predicted without mode-specific refitting;
- high-\(\lambda\) modes decay faster than low-\(\lambda\) modes.

### Nonlinear criteria

Across P1–P6 within the declared physiological validity region:

- median \(K_e\) NRMSE \(\le 0.10\);
- peak error \(\le 10\%\);
- recovery-time error \(\le 10\%\);
- spatial-spread error \(\le 10\%\);
- no violation of K mass balance attributable to the reduction.

These tolerances are engineering/model-reduction criteria, not biological laws, and the paper will show sensitivity to stricter and looser thresholds.

---

# 22. If two states fail

Failure of the two-state reduction does **not** falsify the SMM.

It falsifies the proposed coarse-graining.

The next state is selected from the slow eigenvector/residue structure.

Priority interpretations:

## Candidate third state A — astroglial Na / pump activation

If the missing slow mode is dominated by

\[
Na_i
\]

and Na/K-ATPase dynamics, introduce

\[
n=\delta Na_i.
\]

Then pump-mediated K handling becomes explicitly state dependent.

## Candidate third state B — volume fraction / osmotic state

If R2 shows that fluid/volume dynamics materially alter K control, introduce

\[
v=\delta\alpha_i
\]

or another minimal volume/osmotic state.

## Candidate third state C — membrane/potential state

Only retain an electrical state if it remains slow and carries substantial \(K_e\) response weight after the KNP constraints are enforced.

The number of reduced states is determined by the reference dynamics, not by elegance.

---

# 23. High-fidelity challenge with R2

After the R1 reduction is frozen, run the corresponding perturbation family in `ffian.flow_model`.

Compare:

\[
K_e^{R2}
\]

against predictions from the R1-calibrated reduced SMM.

Questions:

1. Does swelling change peak \(K_e\)?
2. Does advection materially change spatial spread?
3. Do effective recovery time constants change?
4. Does the rank/order of slow modes change?
5. Does an extra slow pole appear?

The two-state core survives R2 only if its predictive error remains within the declared reduction tolerance or if R2 corrections can be absorbed by uncertainty ranges already justified by R1.

If not, a volume/osmotic state becomes mandatory.

Sætra et al. 2023 found that advection can accelerate ionic transport inside astrocytic networks by approximately \(1{-}5\times\) relative to diffusion alone, so this test is substantive rather than ceremonial.

---

# 24. R3 neuron–glia consistency check

The final Step 3A test uses `edNEGmodel`.

Purpose:

verify that the Step 2B mapping

\[
(R_E,V_E,R_I,V_I)
\rightarrow
S_K^N
\]

and the Step 3 reduced mesh are consistent with an explicit neuron–ECS–glia electrodiffusive system.

We do **not** require the MPR neural mass to reproduce every edNEG spike waveform.

We require consistency of slow ionic bookkeeping:

- net neuronal K release;
- extracellular K excursion;
- glial uptake/redistribution;
- recovery timescale;
- sign and magnitude range of neuronal feedback.

---

# 25. Syncytial-coupling ablation

The reference homogenized model does not give us a single literal connexin conductance parameter.

Therefore we define an explicit effective coupling multiplier

\[
\rho_g
\]

on intracellular/syncytial ionic mobility:

\[
D^{\rm eff}_{i,s}
\rightarrow
\rho_gD^{\rm eff}_{i,s}.
\]

Canonical levels:

\[
\rho_g=1
\]

reference coupling,

\[
0<\rho_g<1
\]

partial uncoupling,

\[
\rho_g=0
\]

local-buffering-only limit.

This is a mechanistic ablation parameter, not an EEG fit parameter.

The reduced model must map this monotonically onto its syncytial transport coefficient:

\[
\boxed{
\rho_g\downarrow
\Rightarrow
D_g\downarrow.
}
\]

Its qualitative effect is checked against gap-junction perturbation literature.

---

# 26. Crucial distinction: local buffering versus mesh buffering

Step 3A must produce two independently validated reduced models.

## Local astrocyte model

\[
D_g=0.
\]

This retains membrane K exchange but eliminates syncytial redistribution.
## Full mesh model

\[
D_g>0.
\]

The difference

\[
\boxed{
\Delta_{\rm mesh}
=
\text{full syncytium}
-
\text{local buffering only}
}
\]

is the actual mechanistic quantity associated with the word **mesh**.

This distinction must survive into every later whole-brain and EEG comparison.

---

# 27. Expected qualitative result — but not assumed

Based on the reference literature, the working expectation is:

### Short/local times

local astrocytic uptake/storage can remove a substantial fraction of the extracellular K load.

### Longer/spatial times

syncytial redistribution becomes increasingly important for moving K away from a persistently active region.

Therefore the mesh-specific effect may appear more strongly in:

- sustained activity;
- recovery after sustained activity;
- spatially structured repeated drive;

than during weak stationary rest.

This is a prediction to be measured, not a premise built into later EEG interpretation.

---

# 28. Numerical verification requirements

The reference simulation itself must pass:

- spatial convergence;
- temporal convergence;
- positivity of concentrations;
- conservation checks;
- equilibrium stability;
- reproducibility under fixed configuration.

R2's published study already demonstrates convergence testing of its finite-element scheme; our reproduction must independently verify the subset of observables used for reduction.

---

# 29. Provenance and reproducibility

Every calibration run stores:

- exact source repository;
- exact commit hash/tag;
- environment/container specification;
- all parameter files;
- spatial mesh;
- time step;
- perturbation definition;
- random seed where applicable;
- raw output fields;
- reduction configuration;
- fit/holdout mode split;
- software version.

No manually copied numerical result is considered canonical if it can instead be generated from the pipeline.

---

# 30. Step 3A output objects

When Step 3A is computationally executed, it must generate:

```text
calibration/astroglia/
    provenance.yaml
    reference_R1/
        baseline_state.*
        pulse_responses.*
        modal_transfer_functions.*
        coupling_ablation.*
    reference_R2/
        flow_robustness.*
    reference_R3/
        neuron_glia_consistency.*
    reduction/
        coefficients.yaml
        uncertainty.yaml
        poles_and_residues.csv
        modal_holdout.csv
        nonlinear_validation.csv
        validity_domain.yaml
    figures/
        reference_reproduction.*
        modal_poles.*
        full_vs_reduced_impulse.*
        full_vs_reduced_spatial.*
        coupling_ablation.*
        R2_flow_robustness.*
```

---

# 31. What coefficients may be frozen after Step 3A

Only after passing all tests may we freeze physiological priors/ranges for:

\[
\boxed{
\kappa_e,\quad
\kappa_a,\quad
D_e,\quad
D_g
}
\]

and, if required,

\[
\kappa_{\rm Na/pump}
\]

or volume-state coefficients.

These become cellular/tissue-informed quantities in the whole-brain SMM.

They are not tuned freely against the EEG cohort.

---

# 32. What Step 3A would prove

A successful Step 3A does **not** prove that astrocytes explain human EEG.

It establishes something more basic and necessary:

\[
\boxed{
\text{known astrocyte/ECS electrodiffusion}
\Longrightarrow
\text{a small, stable, spatially coupled slow control system}
}
\]

with quantitative parameters and uncertainty.

This is the physical derivation of the **Syncytial Mesh**.

Only after this has been established are we entitled to ask whether that mesh materially improves a model of human brain dynamics.

---

# 33. Decision rule after Step 3A

## If two-state reduction passes R1 and R2

Canonical mesh:

\[
(k,a).
\]

Proceed to Step 4: couple the calibrated mesh to the calibrated E–I MPR network.

## If R1 requires three slow states

Adopt the smallest physically interpretable three-state model.

Do not force \((k,a)\).

## If R1 reduces well but R2 fails

Add the minimal fluid/volume correction required by R2 before whole-brain modeling.

## If syncytial redistribution is negligible in all physiological R1/R2 regimes

Downgrade the SMM from a syncytial-mesh theory to a local astroglial-homeostasis model.

That outcome is allowed.

The name **Syncytial Mesh Model** must earn its "mesh" component quantitatively.

---

# 34. Canonical conclusion of Step 3A

The definitive SMM will therefore not inherit the mesh operator from the original theory.

It will **derive the mesh operator by model reduction from a published electrodiffusive astrocyte/ECS system**.

The chain is:

\[
\boxed{
\begin{array}{c}
\text{Nernst--Planck ion transport}\\
+\text{Kir/NaK membrane transport}\\
+\text{charge conservation}\\
+\text{astroglial spatial coupling}
\end{array}
}
\]

\[
\Downarrow
\]

\[
\boxed{
\text{full KNP astrocyte/ECS dynamics}
}
\]

\[
\Downarrow\quad\text{validated model reduction}\quad
\]

\[
\boxed{
\text{small dissipative syncytial control system}
}
\]

\[
\Downarrow
\]

\[
\boxed{
\text{calibrated neuronal operating-point modulation}.
}
\]

That is the hard-science replacement for the original metallic-mesh wave equation.
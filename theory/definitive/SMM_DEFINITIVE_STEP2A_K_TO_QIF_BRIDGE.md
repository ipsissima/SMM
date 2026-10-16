# SMM Definitive Reconstruction — Step 2A
## The biophysical bridge: extracellular K+ -> neuronal excitability -> QIF/MPR

**Status:** Canonical derivation plan  
**Role in paper:** mechanistic bridge between the astroglial syncytial subsystem and the next-generation neural mass  
**Target standard:** no free glia->neuron coupling fitted from EEG

---

# 1. Design objective

The definitive SMM must not contain an arbitrary term of the form

\[
I_{\rm glia}=g_A\,u_g .
\]

Instead, the core glia-to-neuron coupling must be derived through an experimentally grounded physiological variable.

The core bridge is:

\[
\boxed{
[K^+]_e
\longrightarrow
E_K
\longrightarrow
\text{distance to spike threshold}
\longrightarrow
\eta_{\rm QIF}
}
\]

with the reverse pathway

\[
\boxed{
R(t)
\longrightarrow
K^+\text{ efflux}
\longrightarrow
[K^+]_e
}
\]

closing the neuron-astrocyte feedback loop.

---

# 2. Physiological starting point

For potassium,

\[
E_K(K_e,K_i)=\frac{RT}{F}\ln\frac{K_e}{K_i}.
\]

Around a reference extracellular concentration \(K_{e0}\), if intracellular potassium is approximately constant on the time scale of the resting-state analysis,

\[
\Delta E_K
=
\frac{RT}{F}\ln\frac{K_e}{K_{e0}}.
\]

At 37 C,

\[
\frac{RT}{F}\approx 26.7\ {\rm mV}.
\]

Thus even physiologically modest extracellular changes are not negligible:

\[
K_e:3.5\to4.0\ {\rm mM}
\quad\Rightarrow\quad
\Delta E_K\approx +3.6\ {\rm mV},
\]

\[
K_e:3.5\to4.5\ {\rm mM}
\quad\Rightarrow\quad
\Delta E_K\approx +6.7\ {\rm mV}.
\]

An increase in \(E_K\) makes the potassium reversal potential less negative and reduces outward hyperpolarizing K+ drive, thereby tending to increase excitability in the physiological regime.

---

# 3. Conductance-based neuronal root

Let the single-cell fast subsystem be

\[
\dot{\mathbf z}
=
\mathbf f(\mathbf z;I,K_e),
\]

where

\[
\mathbf z=(V,\mathbf w)
\]

contains membrane voltage and gating variables.

The voltage equation has the generic form

\[
C_m\dot V
=
I_{\rm app}
-I_L
-I_{\rm Na}
-\sum_m I_{K,m}
-I_{\rm pump}
-I_{\rm syn}
+\cdots
\]

with potassium currents

\[
I_{K,m}
=
\bar g_{K,m}\,
q_m(V,\mathbf w)\,
\bigl(V-E_K(K_e)\bigr).
\]

The extracellular K+ concentration therefore changes neuronal dynamics through a physically explicit reversal-potential dependence.

---

# 4. Exact local reduction near the type-I firing threshold

The QIF model is the normal form of a saddle-node-on-invariant-circle (SNIC) onset.

Assume that at the physiological reference state

\[
(\mathbf z_\ast,I_\ast,K_{e0})
\]

the fast neuronal subsystem is near a saddle-node firing threshold.

Let \(J=D_{\mathbf z}\mathbf f\) be its Jacobian at the saddle-node.

Let

\[
Jq=0,\qquad
p^\top J=0,\qquad
p^\top q=1,
\]

where \(q\) and \(p\) are the right and left zero-eigenvectors.

Center-manifold reduction gives, to leading order,

\[
\dot x
=
a x^2
+
b_I\,\delta I
+
b_K\,\delta \ell_K
+
O(x^3,x\delta\theta,\delta\theta^2),
\]

where

\[
\ell_K=\ln(K_e/K_{e0}),
\]

\[
a=
\frac12
p^\top
D_{\mathbf z}^2\mathbf f[q,q],
\]

\[
b_I=p^\top\partial_I\mathbf f,
\]

and

\[
b_K=
p^\top\partial_{\ell_K}\mathbf f.
\]

This is a saddle-node normal form. After the standard rescaling it becomes a QIF equation

\[
\tau_q\dot u=u^2+\eta_{\rm eff}.
\]

Consequently,

\[
\boxed{
\eta_{\rm eff}
=
\eta_0
+
\chi_K
\ln\frac{K_e}{K_{e0}}
+
O\!\left(
\ln^2\frac{K_e}{K_{e0}}
\right)
}
\]

where the coefficient

\[
\chi_K
\]

is determined by the normal-form projection and rescaling.

It is **not** an EEG fitting parameter.

---

# 5. Interpretation as an equivalent current

Because \(K_e\) enters the voltage equation through potassium reversal potentials, the normal-form shift can equivalently be expressed as a current displacement.

For a potassium current

\[
I_K=g_{K,\rm eff}(V-E_K),
\]

\[
\frac{\partial(-I_K)}{\partial E_K}
=
g_{K,\rm eff}.
\]

Since

\[
\frac{\partial E_K}{\partial\ln K_e}
=
\frac{RT}{F},
\]

the leading equivalent depolarizing-current contribution is

\[
\boxed{
\Delta I_{\rm eq}^{K}
\approx
g_{K,\rm eff}(\mathbf z_\ast)
\frac{RT}{F}
\ln\frac{K_e}{K_{e0}}
}
\]

plus contributions caused by other K-sensitive currents, leak terms and Na/K-pump dependence.

The full coefficient will therefore be obtained from continuation of a conductance-based neuron model rather than guessed from one channel.

---

# 6. Numerical calibration route

The preferred implementation is to determine the mapping numerically from a validated conductance-based model.

For each neuronal class \(a\in\{E,I\}\):

1. Choose a published conductance-based model that is type-I/SNIC in the physiological reference regime.
2. Set a reference \(K_{e0}\).
3. Sweep \(K_e\) through the physiological range.
4. At every \(K_e\), use numerical continuation to find the firing threshold / saddle-node current
   \[
   I_{\rm SN}^{(a)}(K_e).
   \]
5. Define the potassium-induced equivalent input:
   \[
   \boxed{
   \Delta I_{\rm eq}^{(a)}(K_e)
   =
   I_{\rm SN}^{(a)}(K_{e0})
   -
   I_{\rm SN}^{(a)}(K_e)
   }.
   \]
6. Reduce the single-cell dynamics locally to QIF coordinates.
7. Convert the equivalent-current shift into the QIF excitability shift:
   \[
   \eta_a(K_e)
   =
   \eta_{a0}
   +
   \mathcal S_a
   \Delta I_{\rm eq}^{(a)}(K_e),
   \]
   where \(\mathcal S_a\) is fixed by the normal-form scaling.
8. Test whether a linear log-Nernst approximation is sufficient:
   \[
   \eta_a(K_e)
   \simeq
   \eta_{a0}
   +
   \chi_a\ln(K_e/K_{e0}).
   \]
9. If curvature is material, retain the next term:
   \[
   \eta_a(K_e)
   =
   \eta_{a0}
   +
   \chi_{a1}\ell_K
   +
   \chi_{a2}\ell_K^2.
   \]

The polynomial order is selected from the single-cell calibration data, not from EEG fit quality.

---

# 7. Primary calibration models

## Excitatory population

Primary calibration candidate:

**Traub-Miles cortical pyramidal neuron model with dynamic ionic concentrations**, following the formulation used by Contreras et al. (2021).

Reasons:

- mammalian cortical neuron model;
- type-I/SNIC at physiological extracellular K+;
- explicitly includes dynamic extracellular K+;
- experimentally tested against mouse cortical pyramidal neurons;
- published bifurcation analysis shows how changing extracellular K+ alters neuronal firing dynamics.

## Inhibitory population

Primary calibration candidate:

a published type-I fast-spiking inhibitory conductance model, with the Wang-Buzsaki family as a leading candidate.

The inhibitory mapping is calibrated independently:

\[
\chi_E\neq\chi_I
\]

in general.

No assumption of identical K+ sensitivity across E and I populations is allowed.

---

# 8. Domain of validity

This reduction is intentionally local.

QIF is the correct normal form while firing onset remains in the type-I/SNIC regime.

At sufficiently elevated extracellular K+, conductance-based cortical models can move through a saddle-node-loop transition into HOM/bistable firing.

Therefore the definitive core model has an explicit physiological validity region:

\[
K_e\in\mathcal D_{\rm SNIC}.
\]

The exact numerical boundary is determined by continuation of the selected calibration models.

If simulated \(K_e\) leaves this domain, the run is flagged as outside the core SMM's validity.

The model will **not** silently extrapolate the QIF reduction into pathological high-K+ regimes.

A future pathological extension may explicitly incorporate the SNIC-to-HOM transition.

---

# 9. Population-level MPR equations

For each cortical region \(r\) and population \(a\in\{E,I\}\),

\[
\tau_a\dot R_{a,r}
=
\frac{\Delta_a}{\pi\tau_a}
+
2R_{a,r}V_{a,r},
\]

\[
\tau_a\dot V_{a,r}
=
V_{a,r}^2
+
\bar\eta_{a,r}(K_{e,r})
-
(\pi\tau_a R_{a,r})^2
+
I^{\rm syn}_{a,r}
+
I^{\rm ext}_{a,r}.
\]

The potassium dependence is

\[
\boxed{
\bar\eta_{a,r}(K_{e,r})
=
\eta_{a0}
+
\chi_a
\ln\frac{K_{e,r}}{K_{e0}}
}
\]

to first order.

For the core model, extracellular K+ shifts the **mean excitability**.

The heterogeneity parameter \(\Delta_a\) remains fixed.

A K-dependent \(\Delta_a\) is an optional later extension only if cell-to-cell variation in potassium sensitivity proves empirically necessary.

---

# 10. Neuron -> extracellular K+ coupling without a fitted free parameter

The reverse coupling can also be calibrated from the same single-cell models.

For one spike, define the outward potassium charge

\[
Q_K^{\rm sp}
=
\int_{\rm spike}
I_K^{\rm out}(t)\,dt.
\]

Convert charge to amount of K+ using Faraday's constant:

\[
n_K^{\rm sp}
=
\frac{Q_K^{\rm sp}}{F}.
\]

For neuronal membrane-area density \(\rho_A\) and extracellular volume fraction \(\phi_e\),

\[
\boxed{
\left.\frac{dK_e}{dt}\right|_{\rm spikes}
=
\Gamma_K R,
\qquad
\Gamma_K
=
\frac{\rho_A Q_K^{\rm sp}}{F\phi_e}.
}
\]

Separate coefficients can be derived for excitatory and inhibitory populations:

\[
\left.\dot K_{e,r}\right|_N
=
\Gamma_E R_{E,r}
+
\Gamma_I R_{I,r}
+
J_{K,\rm subthreshold}.
\]

Thus the forward neuronal drive into the ionic subsystem is also calibrated from biophysics rather than EEG.

---

# 11. Closed slow-fast loop

The minimal mechanistic loop becomes

\[
R_E,R_I
\longrightarrow
K_e
\longrightarrow
\text{astrocytic uptake and spatial buffering}
\longrightarrow
K_e
\longrightarrow
\eta_E,\eta_I
\longrightarrow
R_E,R_I.
\]

At regional level,

\[
\boxed{
\dot K_{e,r}
=
\Gamma_E R_{E,r}
+
\Gamma_I R_{I,r}
-
J_{\rm uptake,A}
-
J_{\rm pump,N}
+
D_e(\mathcal L_e K_e)_r
}
\]

with the astrocytic subsystem determining \(J_{\rm uptake,A}\) through Kir4.1, Na/K-ATPase and syncytial redistribution.

The neuronal dynamics then uses

\[
\boxed{
\eta_{a,r}(t)
=
\eta_{a0}
+
\chi_a\ln\frac{K_{e,r}(t)}{K_{e0}}.
}
\]

This provides a closed causal loop with physically interpretable parameters.

---

# 12. What is and is not fitted to EEG

## Not fitted to EEG

The following are fixed or tightly constrained using cellular physiology / calibration:

- Nernst relation;
- \(K_{e0}\) physiological range;
- \(\chi_E,\chi_I\) mapping from K+ to QIF excitability;
- potassium charge per spike / \(\Gamma_E,\Gamma_I\);
- Kir4.1 current law;
- pump kinetics where retained;
- physiological validity bounds;
- signs of all couplings.

## Potentially inferred from EEG, within priors

Only parameters that genuinely operate at the population / subject scale may be inferred, for example:

- long-range neuronal coupling scale;
- background drive;
- observation-model scale/noise;
- limited regional heterogeneity;
- possibly a small number of astroglial buffering-state parameters if not identifiable independently.

The number of such parameters must be controlled and matched in null models.

---

# 13. Why this bridge matters scientifically

The old SMM used an arbitrary glia-to-neuron coupling.

The definitive SMM instead has:

\[
\boxed{
\text{astroglial physiology}
\to
K_e
\to
E_K
\to
\text{single-neuron bifurcation}
\to
\eta_{\rm QIF}
\to
\text{exact neural-mass dynamics}
}
\]

This is the multiscale bridge required to make the published "control geometry" hypothesis computationally testable.

It also produces an important falsifiable restriction:

If the empirical effect attributed to glial control requires potassium excursions or \(\eta\)-shifts outside the range permitted by the cellular calibration, the SMM fails.

---

# 14. Immediate next step: 2B

Implement and validate the cellular calibration layer **before** modifying the whole-brain model:

1. reproduce the published Traub-Miles / dynamic-K+ bifurcation structure;
2. continue the SNIC threshold as a function of \(K_e\);
3. derive \(\chi_E\);
4. select and calibrate an inhibitory type-I model to derive \(\chi_I\);
5. integrate potassium current per spike to derive \(\Gamma_E,\Gamma_I\);
6. verify the QIF normal-form approximation against the parent conductance models across the physiological K+ range;
7. freeze these coefficients and uncertainties before any EEG fitting.

Only after those checks pass should the new MPR network be implemented.
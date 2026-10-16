# SMM Definitive Reconstruction — Step 4 Executed
## Closing the neuronal–ionic–syncytial feedback loop before EEG

**Status:** executed.  
**No EEG data were used.**

## 1. Closed local architecture

The definitive local SMM is now

\[
X=(R_E,V_E,R_I,V_I,S_{EI},S_{IE},k,a).
\]

The neuronal core uses a published exact-QIF/MPR E–I operating set:

\[
\tau_E=\tau_I=8\ {\rm ms},\qquad
\Delta_E=\Delta_I=1,\qquad
\eta_E=\eta_I=-5,
\]

\[
\tau_{s,E}=1\ {\rm ms},\qquad
\tau_{s,I}=5\ {\rm ms},\qquad
J_{EI}=J_{IE}=13.
\]

At the 8-ms neuronal timescale, the Step-2B potassium-to-excitability
calibration becomes

\[
\delta\eta_E
=
0.05210958\,\ell
-
0.00489261\,\ell^2,
\]

\[
\delta\eta_I
=
0.04673428\,\ell
-
0.00367223\,\ell^2,
\]

where

\[
\ell=\ln\frac{K_e}{3.5\ {\rm mM}}.
\]

The neuronal potassium source is

\[
\boxed{
J_K^N
=
\rho_{A,E}^{eff}q_E(R_E-R_E^0)
+
\rho_{A,I}^{eff}q_I(R_I-R_I^0)
}
\]

with

\[
q_E=4.736\times10^{-8}
\ {\rm mol\,m^{-2}\,spike^{-1}},
\]

\[
q_I=1.267\times10^{-8}
\ {\rm mol\,m^{-2}\,spike^{-1}}.
\]

The canonical extracellular and astroglial balances are

\[
\boxed{
\phi_e\dot k
=
J_K^N
-(\kappa_e+\kappa_N)k
+\kappa_a a
-D_eL_e k
}
\]

and

\[
\boxed{
\phi_a\dot a
=
\kappa_e k
-\kappa_a a
-D_gL_g a.
}
\]

The R2/M3-calibrated coefficients are

\[
\kappa_e=2.01315923\ {\rm s^{-1}},
\qquad
\kappa_a=0.48318016\ {\rm s^{-1}},
\]

\[
\kappa_N=0.232\ {\rm s^{-1}},
\]

\[
D_e=9.55343\times10^{-10}\ {\rm m^2/s},
\qquad
D_g^{eff}=3.71479\times10^{-10}\ {\rm m^2/s}.
\]

## 2. Source-bookkeeping correction

The factor \(1/\phi_e\) appears exactly once. If \(J_K^N\) is defined per
tissue volume, then the balance is \(\phi_e\dot k=J_K^N-\cdots\).

This supersedes an ambiguous Step-2B wording that could be read as dividing
by extracellular volume fraction twice. The microscopic \(q_E,q_I\)
coefficients themselves are unchanged.

## 3. Anatomical scaling

A cortical morphometric anchor was constructed from published axonal and
dendritic length densities using the cylindrical surface approximation
\(S_A=\pi dL_V\).

The resulting anchor is approximately

\[
S_{\rm axon}=3.70\ \mu{\rm m}^2/\mu{\rm m}^3,
\]

\[
S_{\rm dendrite}=1.23\ \mu{\rm m}^2/\mu{\rm m}^3,
\]

and therefore

\[
\boxed{
S_{\rm neuronal}\simeq4.929\
\mu{\rm m}^2/\mu{\rm m}^3
=
4.929\times10^6\ {\rm m^2/m^3}.
}
\]

This is an anatomical anchor, not a claim that every neuritic membrane patch
contributes the single-compartment Step-2B spike charge identically.

The conservative effective active-membrane prior is therefore

\[
\boxed{
\rho_A^{eff}\in[2.5,8.0]\times10^6\ {\rm m^2/m^3}
}
\]

with central value \(4.929\times10^6\).

The excitatory membrane share is tested over

\[
f_E\in[0.65,0.85],
\]

with central value \(f_E=0.75\).

## 4. Baseline equilibrium

The closed model has the homeostatic equilibrium

\[
R_E^0=8.089047\ {\rm Hz},
\qquad
R_I^0=9.686712\ {\rm Hz},
\]

\[
V_E^0=-2.459420,
\qquad
V_I^0=-2.053779,
\]

with

\[
k^0=a^0=0.
\]

The ionic source is written in deviations from this operating point because
the reduced mesh describes perturbations around an equilibrium in which tonic
neuronal potassium release is already balanced by pumps and transport.

## 5. Timescale separation

The two dominant slow closed-loop poles are

\[
\lambda_{g,slow}=-0.112933\ {\rm s^{-1}},
\]

\[
\boxed{\tau_{g,slow}=8.855\ {\rm s}},
\]

and

\[
\lambda_{g,fast}=-12.31164\ {\rm s^{-1}},
\]

\[
\tau_{g,fast}\simeq0.0812\ {\rm s}.
\]

The remaining neural/synaptic modes are much faster.

Thus the closed model has the intended hierarchy:

\[
\boxed{
{\rm millisecond\ neuronal}
\ll
{\rm tens\!-\!of\!-\!ms\ ionic}
\ll
{\rm multi\!-\!second\ syncytial\ memory}.
}
\]

The glial subsystem adds slow control poles. It does **not** insert a
theta/delta oscillator.

## 6. Spatial modes after closing the loop

In the validated 300-\(\mu{\rm m}\) reference geometry, the slow decay times are

- uniform mode: \(8.855\) s;
- 300-\(\mu{\rm m}\) wavelength: \(1.527\) s;
- 150-\(\mu{\rm m}\): \(0.467\) s;
- 100-\(\mu{\rm m}\): \(0.227\) s.

Therefore

\[
\boxed{\text{higher spatial frequencies decay faster}.}
\]

The original mesh intuition survives as spatial modal filtering, not
oscillatory resonance.

## 7. Physiological pulse test

At the central anatomical prior, a transient excitatory perturbation yields

\[
\boxed{\max\Delta K_e\simeq0.464\ {\rm mM}},
\]

comfortably inside the nonlinear Step-3B validation range.

Peak excitatory firing is

\[
R_E^{peak}\simeq13.445\ {\rm Hz}.
\]

The maximum difference between the full SMM and the otherwise identical
neural-only model is only

\[
0.01442\ {\rm Hz},
\]

and after the fast neural transient it is approximately

\[
0.00162\ {\rm Hz}.
\]

This is an important negative constraint: a calibrated SMM does not generate
large neuronal effects merely because an astroglial state was added.

Large macroscopic effects, if observed later, must arise through network
susceptibility, brain state, spatial organization, or near-critical
amplification.

## 8. Anatomical robustness

Across

\[
\rho_A^{eff}\in[2.5,8.0]\times10^6\ {\rm m^2/m^3}
\]

and \(f_E\in[0.65,0.85]\), the equilibrium remains stable and the multi-second
control pole persists.

Thus the existence of the slow mesh state is not an artifact of one exact
surface-density choice.

## 9. Bifurcation result

The neural-only E–I system undergoes a Hopf bifurcation at

\[
\boxed{I_H^{neural}=8.132639}
\]

with frequency

\[
32.982735\ {\rm Hz}.
\]

When the ionic system is recentered at each tonic operating point, the closed
SMM moves the threshold only slightly:

- \(\rho_A=2.5\times10^6\): \(I_H=8.133056\), 32.9763 Hz;
- \(\rho_A=4.929\times10^6\): \(I_H=8.133461\), 32.9700 Hz;
- \(\rho_A=8.0\times10^6\): \(I_H=8.133972\), 32.9621 Hz.

Hence

\[
\boxed{\text{SMM shifts neuronal stability; it does not create the oscillator.}}
\]

That is precisely the control-field interpretation required by the published
Frontiers theory.

## 10. The specifically syncytial contribution

A seven-patch mesoscale chain with 150-\(\mu{\rm m}\) spacing was compared in
two conditions:

1. full mesh: \(D_g=3.71479\times10^{-10}\ {\rm m^2/s}\);
2. local buffering only: \(D_g=0\).

At the stimulated patch, peak \(K_e\) is nearly unchanged:

\[
0.49917\ {\rm mM}
\quad{\rm vs}\quad
0.50143\ {\rm mM}.
\]

But away from the stimulus the mesh redistributes substantially more ionic
load.

At 150 \(\mu{\rm m}\):

\[
0.02953\ {\rm mM}
\quad{\rm vs}\quad
0.01611\ {\rm mM}
\]

(\(\sim1.83\times\)).

At 300 \(\mu{\rm m}\):

\[
0.005807\ {\rm mM}
\quad{\rm vs}\quad
0.001617\ {\rm mM}
\]

(\(\sim3.59\times\)).

At 450 \(\mu{\rm m}\):

\[
0.001553\ {\rm mM}
\quad{\rm vs}\quad
0.000203\ {\rm mM}
\]

(\(\sim7.65\times\)).

This gives the term **mesh** an operational meaning:

\[
\boxed{
D_g>0
\Rightarrow
\text{enhanced redistribution away from the active patch}.
}
\]

## 11. Identifiability of the mesh

The same simulation gives an important epistemic result.

For neuronal-rate observations with 0.01-Hz independent noise,

\[
\boxed{\sigma_{\log D_g}\approx6.31,}
\]

so \(D_g\) is effectively non-identifiable.

With direct extracellular-potassium observations at 0.005-mM noise,

\[
\boxed{\sigma_{\log D_g}\approx0.027,}
\]

so it is highly recoverable.

Therefore

\[
\boxed{
\text{neuronal electrophysiology alone cannot identify }D_g
\text{ as an astroglial parameter}.
}
\]

This is not merely a philosophical limitation; it appears directly in the
sensitivity calculation.

## 12. Other microscopic parameters

In a designed three-perturbation local experiment, the sensitivity matrix for

\[
(\rho_A^{eff},\kappa_e,\kappa_a)
\]

has condition number approximately 6.1, so the parameters are structurally
distinguishable under ideal observations.

At 0.01-Hz rate noise, approximate relative 1-sigma uncertainties are:

- \(\rho_A^{eff}\): 12.3%;
- \(\kappa_e\): 31.1%;
- \(\kappa_a\): 37.7%.

At 0.05-Hz noise they degrade to roughly:

- 61%;
- 156%;
- 188%.

Therefore subject-level EEG inference must **not** freely estimate these
microscopic glial parameters. They must remain fixed or tightly constrained by
the upstream physiology.

## 13. Adversarial generic-slow-field null

For synthetic trajectories generated by the full SMM:

- neural-only RMSE: \(0.002621\) Hz;
- fitted generic one-state slow field: \(0.001193\) Hz;
- generating two-state SMM: 0.

More importantly, in the strict linear regime an unconstrained generic
two-state slow field can realize the same state-space transfer function as the
linearized two-state SMM.

Thus

\[
\boxed{
\text{a good EEG fit cannot, by itself, identify the latent field as astroglial}.
}
\]

The later empirical hierarchy must be adversarial:

\[
\text{neural-only}
\rightarrow
\text{generic slow field}
\rightarrow
\text{generic spatial slow field}
\rightarrow
\text{biophysically constrained SMM}.
\]

If EEG distinguishes SMM from neural-only but not from a matched generic slow
field, the valid conclusion is evidence for slow spatial control, not evidence
that EEG has identified astroglia.

## 14. Step-4 decision

**Step 4 passes.**

The model is now mechanistically closed from neuronal firing through K release,
astroglial buffering/redistribution, extracellular K, and back to neuronal
excitability.

The executed findings are:

1. no free glia-neuron gain is required;
2. the mesh adds a multi-second control mode and a faster ionic mode;
3. spatial low-pass filtering survives closure of the loop;
4. ordinary stable operation produces modest neuronal modulation, not an
   artificially dramatic effect;
5. the SMM shifts neuronal stability boundaries but does not manufacture the
   neuronal oscillator;
6. \(D_g\) produces a clear spatial ionic signature;
7. \(D_g\) is essentially not identifiable from neuronal rates alone;
8. generic slow fields remain necessary adversarial nulls.

## 15. Next step

The next justified stage is **Step 5: whole-brain embedding and EEG observation
model**.

It must preserve

\[
L_g\neq W_N
\]

and

\[
g\rightarrow X\rightarrow EEG.
\]

The intended architecture is:

- one E–I next-generation neural mass per cortical region;
- long-range white-matter coupling only through neuronal populations;
- conduction delays from tract lengths;
- local/mesoscale syncytial control variables;
- no centimeter-scale astrocytic edges;
- an explicit neuronal-current/source-to-scalp observation model;
- frozen biological priors before held-out EEG analysis.

Only after Step 5 and network-level parameter recovery should the empirical
EEG cohort be opened for definitive inference.

---

## External anchors

- Reyner-Parra & Huguet et al. (2022), *Phase-locking patterns underlying
  effective communication in exact firing rate models of neural networks*,
  PLOS Computational Biology, DOI 10.1371/journal.pcbi.1009342.
- Karbowski (2015), *Cortical Composition Hierarchy Driven by Spine
  Proportion Economical Maximization or Wire Volume Minimization*,
  PLOS Computational Biology, DOI 10.1371/journal.pcbi.1004532.
- Senk et al. (2024), *Multi-scale spiking network model of human cerebral
  cortex*, Cerebral Cortex, DOI 10.1093/cercor/bhae409.
- Forrester et al. (2024), *Whole brain functional connectivity: Insights
  from next generation neural mass modelling incorporating electrical
  synapses*, PLOS Computational Biology, DOI 10.1371/journal.pcbi.1012647.
---
title: 'The Syncytial Mesh Model: Identifying Independent Astroglial Transport Geometry in Neuronal–Ionic Dynamics'
author: Andreu Ballús Santacana
date: Department of Philosophy, Universitat Autònoma de Barcelona
lang: en-US
fontsize: 11pt
geometry: margin=0.88in
colorlinks: true
linkcolor: blue
urlcolor: blue
mainfont: STIX
mathfont: STIX Math
header-includes:
  - \usepackage{float}
  - \floatplacement{figure}{H}
  - \setlength{\emergencystretch}{3em}
  - \usepackage{microtype}
---

# Abstract {.unnumbered}

The spatial organization of neuronal excitability may depend on an astroglial transport geometry distinct from neuronal connectivity and extracellular diffusion. The Syncytial Mesh Model (SMM) formalizes this hypothesis as an operator-identification problem. Complete extracellular potassium transfers determine the weighted glial operator in a calibrated passive system, but the inverse is severely ill-conditioned in some regimes. Both passive chemical transport and a quasi-static electrical buffering operator lose graph-shape sensitivity at the two extremes of glial mobility. In full electrodiffusive simulations, axial electrical migration shifts the sampled topology-contrast maximum from 5 to 60 µm. The calculated two-conductor electrical length tracks the migration-dependent contrast window. An effective-medium estimate based on published astrocyte anatomy and paired-cell resistance gives an electrical length of 41 µm. Targeted full-model refinement with chemical and electrical transport jointly scaled confirms a sampled intrinsic-scale maximum of 30 µm for the short transmembrane pulse (25–40 µm at the conductivity sensitivity endpoints). In simulations near the empirically informed conductivity, neutral KCl loading at 37 µm yields differences of up to 0.81 mM extracellular potassium and 0.60 mV neuronal membrane potential relative to a total-edge-weight-matched path. Relative dose amplification is modest and regime-dependent. Fixed-tissue refinement confirms the importance of distributed source–sensor coverage. Aggregate CA1 screening underestimates experimental uncoupling effects. The results establish mechanisms and conditions for testing independent astroglial transport support, not its direct identification in tissue.

**Keywords:** astroglial networks; Syncytial Mesh Model; potassium buffering; graph Laplacians; electrodiffusion; neuronal excitability; structural and practical identifiability; experimental design.

# Introduction

Neuronal connectivity specifies a principal organization of interactions between neurons, but it does not exhaust the spatial organization of the conditions under which those interactions take place. Extracellular ionic concentrations, membrane transport and astrocytic networks affect excitability, and the spatial routes through which these quantities change need not coincide with synaptic connections. The Syncytial Mesh Model (SMM) advances the hypothesis that a partially independent astroglial geometry contributes to the organization of neuronal responses. In this sense a glial network acts as a spatial controller of the conditions of neuronal activity, not necessarily as a separate generator of electromagnetic activity.

SMM develops the spatial architecture introduced in the Brain-Mesh and Syncytial Mesh preprints [1,33] and the subsequent conceptual account of glial syncytial control [2]. Here we restrict attention to a potassium-mediated realization (SMM-K), leaving the original calcium-wave and low-frequency resonome hypotheses to separate investigations.

Several research lineages motivate this architecture. The classic observations of Orkand and colleagues [3], the analysis of current-driven potassium transport by Gardner-Medwin [35], and its analytical and numerical extension by Chen and Nicholson [36] distinguish intracellular electrical spatial buffering from extracellular diffusion and local uptake. Experiments and reviews establish the role and limitations of astroglial coupling in ionic regulation, synaptic activity and plasticity [4–6,20,21]. Houades et al. [37] demonstrated spatially structured astrocytic coupling in barrel cortex; Lallouette et al. [38] showed computationally that changing gap-junction network topology can alter intercellular calcium-wave propagation even when cell positions and biophysical parameters are held fixed. Ma et al. [39] showed that electrical coupling promotes syncytial isopotentiality and potassium clearance. Thus neither the importance of topology in astroglial signaling nor the distinction between electrical and chemical transport is a new claim of SMM.

A complementary functional lineage establishes that astrocytes can organize neuronal activity at the network level. Cahill et al. [40] described network-level astrocytic encoding of local neurotransmitter signals; Visser et al. [41] showed that disrupting astroglial gap-junction connectivity alters spatial visual representations and associated behavior in the mouse superior colliculus. Cooper et al. [13,34] demonstrate selective, plastic, long-range astrocytic networks whose architecture can diverge substantially from neuronal projections. Their tracer measurements demonstrate anatomical and molecular connectivity, not a potassium conductance matrix. Oliveira et al. [42] synthesize evidence for astrocytic functional units across scales, and Zhang et al. [43] explicitly discuss the emerging astrocytic connectome. A cortical astrocytic territory may contain numerous synapses [14]; regional anatomy, subcellular morphology and micrometric ionic transport graphs are related but distinct levels of description.

The functional effect of coupling is dependent on preparation, intensity and timescale. Breithausen et al. [15] found a limited acute gap-junction contribution to ordinary CA1 stratum radiatum potassium transients but a clearer contribution when highly localized extracellular loads produced elevations on the order of 10 mM or greater. Bojovic et al. [16] found that interference with Kir channels and astrocytic coupling alters neuronal excitability during sustained cortical activity; their results also indicate compensation by other mechanisms. Neither finding requires coupling to dominate all physiological potassium clearance. Together they motivate experiments in which the delivered ionic load, spatial profile, and duration are independently measured rather than treating the contribution of a glial network as a fixed fraction of any neuronal response.

The modeling ingredients are also established. Coupled neuronal–ionic models include potassium-mediated feedback and regime transitions [22,23]. Halnes and colleagues and Sætra and colleagues have developed increasingly complete electrochemical models of neuron–extracellular–glia interactions [7–10]. The two-module microcircuit model of Cafiso et al., first circulated in June 2026 [45] and published in November 2026 [19], shows how local and syncytial astrocytic buffering can regulate synchronization and pathological excitability through extracellular potassium. Nakatani and De Schutter [44] investigate the dependence of astrocytic membrane responses on stimulus scale and intracellular electrical compartmentalization, showing why somatic recordings can miss local peripheral depolarization. These studies address coupling strength and synchronization, or within-cell electrical response geometry, complementing the present question of identifying intercellular transport support from extracellular and neuronal observations. General methods for compartmental systems, dynamical-network identification, profile likelihood and optimal discrimination design are likewise established [18,26–32].

What distinguishes SMM-K is the problem addressed by combining these ingredients: given extracellular and neuronal alternatives, can the *geometry* of the glial transport operator be recovered, and can a preparation be shown to require connections outside the support of extracellular transport? This is stronger than detecting a difference after gap-junction blockade. A change in local uptake, global diffusivity, membrane excitability, or a hidden feedback process may account for such a difference without requiring a distinct spatial organization. A meaningful test must compare independently constrained model families under matched inputs, observations and uncertainty.

This article therefore makes three connected contributions. It first derives exact implications of a distinct glial operator in a calibrated passive model, including conditions for recovery and sensitivity. It then calculates both the conditioning of that inverse problem and finite separation from shared-support alternatives under an explicit experimental design. Finally, it tests the graph representation against an independent conservative electrodiffusive model, identifies the dependence of the topology-sensitive spatial window on axial electrical migration, examines dose-dependent nonlinear contrast and distal current mechanisms, and derives an independently informed electrical parameter constraint. The original neuronal-population, metabolic, stability and EEG audits are retained in the supplementary material because they constrain the physiological interpretation, but the theory is organized here around its distinctive prospective test.

# Model and hypothesis

## Three spatial organizations

Let $V=\{0,\ldots,N-1\}$ index spatial compartments. We distinguish a neuronal interaction operator $W_n$, which may be directed; an extracellular transport Laplacian $L_e$; and an astroglial transport Laplacian $L_g$. For a symmetric ionic graph with nonnegative weights $w_{ij}$,

$$L=B\operatorname{diag}(w)B^\top=\sum_{\{i,j\}\in E}w_{ij}(e_i-e_j)(e_i-e_j)^\top,$$

where $B$ is an oriented incidence matrix and $e_i$ is the $i$th coordinate vector. Edge orientation is bookkeeping; flux is determined by gradients. The Laplacian is positive semidefinite and annihilates the constant vector. A nonnegative symmetric ionic edge is not the same physical quantity as a directed synaptic weight. Distinct organizations may differ by edge support, relative weights, or both.

![Illustrative three-layer SMM architecture. Solid horizontal links in the ionic layers denote the shared path; dashed curves denote additional astroglial edges. The directed neuronal links are schematic, not the synaptic graph used in the full electrodiffusive simulations, where synapses were disabled. This is an abstract transport network, not an anatomical reconstruction.](architecture.png){width=95%}

A graph on spatial compartments does not have to correspond to a literal graph of individually identified astrocytes. It can describe a coarse-grained transport operator at a specified scale. At that scale, a meaningful comparison requires the same sites and a common spatial embedding or independently defined operational support. The analysis below uses a symmetric nonnegative graph as a restricted first model; changing support or coarse-graining changes the hypothesis to be tested.

## Potassium-mediated passive realization

Let $k,a\in\mathbb R^N$ denote extracellular and astroglial excess potassium concentrations in mM, relative to a reference state. Let $\phi_e,\phi_a>0$ be the respective volume fractions. With homogeneous coefficients, the local and spatial deviation balances are

$$\begin{aligned}
\phi_e\dot k &= J_K^n-(\kappa_e+\kappa_N)k+\kappa_a a-D_eL_e k+u_k,\\
\phi_a\dot a &= \kappa_e k-\kappa_a a-D_gL_g a+u_a .
\end{aligned}$$

Here $\kappa_e$ and $\kappa_a$ describe effective uptake and return, $\kappa_N$ a local clearance slope, $J_K^n$ excess neuronal release, and $u_k,u_a$ externally specified inputs in mM of tissue-volume-equivalent potassium per second. The $\kappa$ coefficients have units s$^{-1}$ in this convention. If $D_e,D_g$ are in m$^2$/s, $L_e,L_g$ carry m$^{-2}$. Specifically, for nominal spacing $d$, write $L_r=d^{-2}\mathcal L_r$, where $\mathcal L_r$ is a dimensionless weighted graph Laplacian. We denote the weighted glial operator in s$^{-1}$ by $G=D_gL_g$ and the extracellular operator by $E=D_eL_e$.

The passive system sets $J_K^n=0$ to isolate transport, with $\kappa_e,\kappa_a>0$ and $\kappa_N\geq0$. It is a model of deviations, not an absolute ionic inventory law for a full tissue. When $\kappa_N=0$ and the external inputs vanish, the summed potassium inventory $\mathbf1^\top(\phi_e k+\phi_a a)$ is conserved; otherwise its change is governed by the specified sources and clearance. The two-compartment model is useful precisely because it provides a tractable conditional identification problem. It does not represent membrane capacitance, electrical migration, variation in volumes, all ionic species or the feedback of neuronal spikes. Its connection to complete physiology is tested below, rather than asserted by matching the names of parameters.

For homogeneous passive coefficients the weighted quadratic energy

$$\mathcal E=\tfrac12\big(\kappa_e\phi_e\|k\|^2+\kappa_a\phi_a\|a\|^2\big)$$

satisfies, in the unforced passive system,

$$\begin{aligned}
\dot{\mathcal E}={}&-\|\kappa_e k-\kappa_a a\|^2-\kappa_e\kappa_N\|k\|^2\\
&-\kappa_e D_e k^\top L_e k-\kappa_a D_g a^\top L_g a\le0 .
\end{aligned}$$

For $\kappa_N>0$ the inequality is strict away from the origin and the finite-dimensional linear system is asymptotically stable. For zero clearance, neutral inventory modes may remain. This dissipation result does not require commuting graphs and does not imply stability of a closed neuronal feedback loop.

## Distinctive hypothesis and nested alternatives

The broad SMM hypothesis is that the control geometry of a non-neuronal network can be partially independent of neuronal and extracellular geometries. A stronger, preparation-specific prediction for SMM-K is that the weighted glial operator contains functionally non-negligible edges outside the support $E_e$ of an independently established extracellular transport operator. Define the restricted family

$$\mathcal H_0=\left\{\sum_{\{i,j\}\in E_e}w_{ij}(e_i-e_j)(e_i-e_j)^\top:\ w_{ij}\geq0\right\}.$$

The alternative contains a prespecified set of possible additional glial edges, with independently justified constraints on their weights. The prediction is $G\notin\mathcal H_0$, to a practically and biologically specified tolerance. This excludes not merely a scaled extracellular path but *every* weighted Laplacian with the specified shared support. Unequal weights on the same support can still constitute a distinct geometry under the broader architecture; the extra-edge prediction is a deliberately sharper test. The biological attribution also requires evidence that the inferred coupling is astroglial rather than an arbitrary unobserved controller.

# Mathematical results

## Exact recovery without commuting graphs

Consider initially unperturbed glia and extracellular potassium with calibrated extracellular drive $u_k=u$, $u_a=0$, no neuronal feedback, and homogeneous local parameters. For real $s>0$ define

$$\begin{aligned}
P(s)&=(\phi_e s+\kappa_e+\kappa_N)I+E,\\
Q(s)&=(\phi_a s+\kappa_a)I+G,\\
c&=\kappa_a\kappa_e .
\end{aligned}$$

Laplace transformation gives the block system

$$\begin{pmatrix}P&-\kappa_a I\\-\kappa_e I&Q\end{pmatrix}\begin{pmatrix}\widehat k\\\widehat a\end{pmatrix}=\begin{pmatrix}\widehat u\\0\end{pmatrix}.$$

**Proposition 1 (conditional graph recoverability).** With the stated assumptions, the full extracellular transfer matrix $H$ satisfies

$$\widehat k=H(s)\widehat u,\qquad H(s)=(P-cQ^{-1})^{-1}.$$

If $P,c,\phi_a,\kappa_a$ are known and $H(s)$ is measured exactly with full input and output coverage, then

$$\boxed{G=c(P-H^{-1})^{-1}-(\phi_a s+\kappa_a)I.}$$

*Proof.* Since $Q\succeq(\phi_a s+\kappa_a)I\succ0$, elimination of $\widehat a$ is valid. Moreover,

$$P-cQ^{-1}\succeq\left(\phi_e s+\kappa_N+\frac{\kappa_e\phi_a s}{\phi_a s+\kappa_a}\right)I\succ0.$$

Thus $H$ exists and $P-H^{-1}=cQ^{-1}$ is positive definite. Rearranging gives the recovery expression. Neither elimination nor the inequalities exchange the order of the two graph operators. $\square$

The identified quantity is the product $D_gL_g$: multiplying $D_g$ by a positive scalar and dividing all glial weights by the same scalar leaves the transfer unchanged. A conductivity measurement or graph weight convention is necessary to distinguish these factors. For partial source and observation matrices $B_k,C_k$, it is generally not legitimate to invert $C_kHB_k$ as if it were $H$. Unknown neuronal feedback or incorrectly specified local exchange invalidates the same reconstruction.

A further consequence follows from the high-frequency expansion

$$\begin{aligned}
H^{-1}(s)={}&\phi_e s I+(\kappa_e+\kappa_N)I+E-\frac{c}{\phi_a s}I\\
&+\frac{c}{\phi_a^2s^2}(\kappa_a I+G)+O(s^{-3}).
\end{aligned}$$

The off-diagonal coefficients of orders $s^0$ and $s^{-2}$ separate extracellular and glial contributions under the stated homogeneous calibration. In a joint identification problem the required scalar factors must also be identified. This expansion is an algebraic result, not a recommended estimator of small high-order coefficients from noisy data.

For $G=D_gL_g$ with fixed $L_g$, differentiating the transfer gives

$$\frac{\partial H}{\partial D_g}=-cH Q^{-1}L_gQ^{-1}H\preceq0.$$

The quadratic-form sign follows from symmetry and positive semidefiniteness, regardless of whether $L_e$ and $L_g$ commute. This is a Loewner-order statement for real positive $s$, not an assertion that every sitewise time-domain transient decreases with coupling.

## Conditioning of the inverse map

Exact recovery does not establish that the data needed for it can be measured with sufficient precision. With the other parameters held fixed, perturb $H$ by a symmetric matrix $Z$. The first differential of the recovered operator is

$$\delta G=-c^{-1}QH^{-1}ZH^{-1}Q.$$

Indeed, $\delta H^{-1}=-H^{-1}ZH^{-1}$ and differentiating $H^{-1}=P-cQ^{-1}$ yields $\delta H^{-1}=cQ^{-1}(\delta G)Q^{-1}$. With $A=QH^{-1}$, the differential is $-AZA^\top/c$. Its exact induced Frobenius norm on unrestricted symmetric perturbations is $\|A\|_2^2/c$: the upper bound is submultiplicativity, and a unit-Frobenius rank-one perturbation $Z=vv^\top$ along a leading right singular vector $v$ attains equality. For $G\ne0$ the relative local condition number is therefore

$$\mathcal K(s)=\frac{\|QH^{-1}\|_2^2\,\|H\|_F}{c\,\|G\|_F}.$$

This is a worst-case local gain for the exact unconstrained map, not a variance bound for a structured estimator. Constraints on Laplacian support, positivity, reciprocity and multiple independent frequencies can reduce effective amplification. The expression provides a direct mathematical bridge between the recovery theorem and the need for calibrated forward fitting.

## Source-route dependence and uniform blindness

Let $F$ be the $2N$-dimensional passive generator, and $b=(u/\phi_e,0)^\top$ the input vector for a constant extracellular step switched on at $t=0$. For zero initial deviations,

$$x(t)=\sum_{j\geq0}\frac{t^{j+1}}{(j+1)!}F^j b,\qquad x=(k,a)^\top.$$

**Proposition 2 (pathway order).** In the stated passive model the first possible nonzero extracellular response sensitivities to the two scalar diffusion coefficients are

$$\begin{aligned}
\partial_{D_e}k(t)&=-\frac{L_eu}{2\phi_e^2}t^2+O(t^3),\\
\partial_{D_g}k(t)&=-\frac{\kappa_a\kappa_e L_gu}{24\phi_e^2\phi_a^2}t^4+O(t^5).
\end{aligned}$$

For a strictly downstream neuronal state $n$, with $\dot n=A_nn+B_nk$, $n(0)=0$, and no direct glial drive,

$$\partial_{D_g}n(t)=-\frac{\kappa_a\kappa_e B_nL_gu}{120\phi_e^2\phi_a^2}t^5+O(t^6).$$

For a direct constant glial input $v$ instead of extracellular drive, the corresponding first possible extracellular term is

$$\partial_{D_g}k(t)=-\frac{\kappa_a L_gv}{6\phi_e\phi_a^2}t^3+O(t^4),$$

and the downstream neuronal sensitivity acquires one additional order of time.

*Proof.* The extracellular operator acts immediately on the populated upper block of $Fb$. The glial operator must first receive a local exchange contribution in the lower block and then return its effect to the upper block. Its first upper-block term is in $\partial_{D_g}(F^3b)$, with coefficient $-\kappa_a\kappa_e L_gu/(\phi_e^2\phi_a^2)$. With direct glial forcing the lower block is already populated at input, so the first upper-block term is in $\partial_{D_g}(F^2b)$. Integrating these sensitivities through a strictly downstream first-order neuronal system adds a time power and the corresponding factorial. This is a path-length result, and coefficients can vanish for specific sources or projections. $\square$

**Proposition 3 (uniform blindness).** If the initial conditions and inputs are uniform and local coefficients homogeneous, then the passive solution stays in $\operatorname{span}(\mathbf1)\oplus\operatorname{span}(\mathbf1)$ and is independent of both graph mobilities. The proof is invariance of this subspace, because $L_e\mathbf1=L_g\mathbf1=0$. No commutativity is required. Heterogeneous coefficients or boundaries can break this invariance.

The onset hierarchy and uniform control constrain an experiment, but do not distinguish SMM from another two-compartment diffusion-with-exchange model with the same causal path length. They should be used as tests of the model scaffold, rather than the main biological falsifier. In particular, the onset result presumes a purely chemical forcing in the passive model; electrical buffering can open a faster pathway even under charge-neutral KCl forcing, because a local change in potassium concentration changes the local reversal potential. Charge-separating membrane stimulation can additionally generate a direct electrical footprint. The passive expansion is not a general onset theorem for electrodiffusion.

## Intermediate topology-sensitive mobility

Consider two distinct connected symmetric glial Laplacians on the same sites, identical extracellular and local coefficients, zero initial glial deviation, and a bounded extracellular source. Multiply each glial transport operator by $\mu>0$.

**Proposition 4 (loss of graph-shape contrast at both mobility extremes).** For a fixed bounded observation interval, the difference $\Delta k_{12}(t;\mu)$ between the two extracellular responses tends uniformly to zero as $\mu\downarrow0$ and as $\mu\to\infty$. If some intermediate mobility gives a nonzero finite-time contrast, its continuous norm achieves a positive maximum at an intermediate mobility.

*Proof.* Continuity of the linear generator and its exponential gives the zero-mobility limit. For the infinite-mobility limit, decompose $a=\Pi a+a_\perp$, where $\Pi=\mathbf1\mathbf1^\top/N$. Since each connected symmetric graph has a positive spectral gap on $\mathbf1^\perp$, the nonuniform component obeys

$$\phi_a\dot a_\perp=\kappa_e(I-\Pi)k-(\kappa_a I+\mu G)a_\perp.$$

On any fixed time interval the forcing is bounded uniformly in $\mu$ by the passive dissipative estimate with the bounded input. With zero initial nonuniform load, variation of constants therefore gives $\|a_\perp\|=O(\mu^{-1})$ uniformly on that interval. The uniform glial state solves the same averaged exchange equation in both graphs. The remaining extracellular and averaged-glial equations form a fixed linear system perturbed by the vanishing $a_\perp$ term, hence converge to a common limit. Continuity in $\mu$ and the two vanishing endpoints imply the maximum statement. $\square$

The positive maximum is conditional on a discriminating input, graph pair and observation; it need not be unique. Disconnected graphs can have different nullspaces and different limiting operators. Neither this theorem nor the maximum of a relative norm implies that glia become the dominant potassium transport route. It is a statement about the information carried by internal graph shape. In the source-reconciled parameterization below $D_g/D_e\simeq0.218$ at nominal mobility, independently of spacing.

## Electrical buffering on a graph

Current-driven spatial buffering [3,35,36] provides a complementary, quasi-static electrical limit. Consider a frozen-concentration two-conductor network with connected symmetric glial and extracellular conductance Laplacians $A_g=mG_c$ and $A_e=E_c$, and a uniform positive membrane slope conductance density $g_m$. The axial operators have the same conductance-per-volume units as $g_m$. Let $e$ denote a prescribed vector of perturbations of the effective membrane reversal voltage, $v=\varphi_g-\varphi_e$, and let $j=g_m(v-e)$ denote outward membrane current density. In the absence of imposed net electrical current, current balance gives

$$A_g\varphi_g+j=0,\qquad A_e\varphi_e-j=0.$$

On the mean-zero subspace $H=\mathbf1^\perp$ the restricted Laplacians are positive definite and invertible, and $\mathbf1^\top j=0$. Define the parallel sum on $H$ by

$$B_m=\left[(mG_c)^{-1}+E_c^{-1}\right]^{-1}.$$

Eliminating the potentials gives $j=-B_m v$, and therefore

$$v=g_m(B_m+g_m I)^{-1}e,\qquad j=-g_m B_m(B_m+g_m I)^{-1}e.$$

These expressions apply to the projected mean-zero perturbation; the spatially uniform reversal-voltage component induces the same uniform membrane-voltage shift and zero axial or membrane current. No commutativity of the two graph Laplacians is assumed. The equations represent a linearized frozen-concentration electrical response, not the time evolution of the full ionic model.

**Proposition 5 (electrical graph-shape window).** For two connected glial conductance Laplacians with the same extracellular Laplacian, local slope, and prescribed reversal-voltage perturbation, the difference of their quasi-static membrane-current operators converges to zero as $m\downarrow0$ and $m\to\infty$. If a nonzero contrast occurs, its continuous norm attains a positive maximum at an intermediate $m$.

*Proof.* On $H$, the identity $B_m=m[G_c^{-1}+mE_c^{-1}]^{-1}$ gives $B_m\to0$ as $m\downarrow0$, while the defining inverse expression gives $B_m\to E_c$ as $m\to\infty$. The resolvent is continuous because $g_m>0$, so both limiting membrane-current operators are independent of glial graph shape. Continuity and vanishing contrast at both endpoints imply the intermediate maximum. $\square$

In a uniform one-dimensional continuum with axial tissue conductivities $m\sigma_g$ and $\sigma_e$, the corresponding voltage difference satisfies $v''=g_m[(m\sigma_g)^{-1}+\sigma_e^{-1}](v-e)$ away from imposed current sources. Its homogeneous decay length is

$$\ell(m)=\left[g_m\left(\frac1{m\sigma_g}+\frac1{\sigma_e}\right)\right]^{-1/2}.$$

The electrical length saturates at $\sqrt{\sigma_e/g_m}$ as $m\to\infty$; neither this length nor Proposition 5 determines the quantitative maximum of a full ionic topology contrast. The electrical current map can be nonlocal in the reversal-voltage perturbation. Under a neutral localized KCl step, $e(t)=O(t)$ near onset, so this quasi-static current map can produce an $O(t)$ remote current and an $O(t^2)$ integrated ionic contribution when the relevant coefficient is nonzero. This is a possible faster pathway than Proposition 2, not a universal onset exponent for EDNM, which retains capacitive dynamics, multiple ions and concentration-dependent fluxes.

# Numerical methods: a matched experiment

## Graphs, kinetics and input convention

The seven-site experiments use the extracellular path $0-1-\cdots-6$, with unit nearest-neighbor edge weights. Write its dimensionless Laplacian as $\mathcal L$, and define the additional-edge Laplacian

$$\mathcal C=0.8b_{04}b_{04}^\top+0.5b_{15}b_{15}^\top+0.6b_{26}b_{26}^\top,$$

where $b_{ij}=e_i-e_j$. The passive family is

$$E=D_e\mathcal L/d^2,\qquad G=\mu D_g(\mathcal L+g\mathcal C)/d^2.$$

The target $g$ controls the three chords relative to the common path; $g=0$ is the shared path and $g=1$ the nominal alternative. The separate mobility multiplier $\mu$ allows overall strength to change. Chord weights are effective weights at the chosen nominal discretization length, not conductances derived from the physical lengths of the long links. The full EDNM graph adapter uses the same weighted adjacency convention. No measured astrocyte geometry or topology is assumed. Changing $d$ rescales the intrinsic tissue graph, including the physical lengths of the chords; it is not a change of electrode spacing on one fixed tissue. The three four-step chords of weights 0.8, 0.5 and 0.6 correspond to own-length effective diffusivities 12.8, 8 and 9.6 times the nominal glial diffusivity, respectively, when represented as ordinary diffusion across their entire physical lengths. They are deliberately synthetic high-conductance conduits, not anatomical measurements. Useful dimensionless groups include $D_g/(d^2\kappa_a)$, $D_e/(d^2\kappa_a)$, $\kappa_e/\kappa_a$, $\kappa_N/\kappa_a$, the volume fractions, mobility, and $\kappa_a T$ for protocol duration $T$.

Table 1. Fixed passive parameters for the source-reconciled experiment. The exchange and transport coefficients are inherited from the earlier explicit source-reconciliation exercise; the clearance slope is inherited separately and the volume fractions are those of the EDNM resting state. The combination is a specified synthetic parameter set, not an independently fitted passive model of the seven-site EDNM trajectories.

| Parameter | Value | Units |
|:--|--:|:--|
| $\phi_e$ | 0.2 | dimensionless |
| $\phi_a$ | 0.4 | dimensionless |
| $\kappa_e$ | 1.29629 | s$^{-1}$ |
| $\kappa_a$ | 0.52863 | s$^{-1}$ |
| $\kappa_N$ | 0.232 | s$^{-1}$ |
| $D_e$ | $1.05270\times10^{-9}$ | m$^2$/s |
| $D_g$ | $2.29581\times10^{-10}$ | m$^2$/s |

The passive extracellular input is a rectangular pulse at one site with amplitude $q=0.1714683$ mM tissue/s, on from 0.08 to 0.40 s. This is obtained from the original EDNM membrane flux magnitude $10^{-4}$ mM cm/s multiplied by the source membrane-area density $\gamma_m=1714.683$ cm$^{-1}$. This is a conversion of the potassium mass delivered per tissue volume, not equality of the electrical action of the inputs: the full model transfers potassium across a neuronal membrane and has a direct capacitive and electrical footprint. The pulse has tissue-volume-integrated dose $q(0.32)\simeq0.05487$ mM. The initial full graph experiment uses 5 µm spacing and sealed boundaries with a 0.8 s horizon; the extended sweep uses the same normalized forcing and graph weights across intrinsic scales 5–150 µm.

For passive simulations, if $F$ is the stable generator and $B=(I/\phi_e,0)^\top$, the exact step response matrix is $S(t)=F^{-1}(e^{Ft}-I)B$ for $t>0$, zero otherwise. The rectangular pulse is $q[S(t-t_{\mathrm{on}})-S(t-t_{\mathrm{off}})]$. In the matched comparison, the original EDNM time labels and strict on/off inequalities are used in a zero-order-hold propagation at the actual 1 ms step. For scale sweeps a diagonally weighted symmetric eigendecomposition evaluates the exact passive solution. These two evaluations agree to the intended temporal-resolution accuracy; there is no stochastic fitting in the scale sweep.

We use the same dimensionless contrast for both models,

$$\mathcal R(d,\mu)=\frac{\|K_1-K_0\|_F}{\|K_0\|_F},$$

where the matrices stack the stimulus-minus-control extracellular potassium responses at all seven somatic sites and the specified time points, subscripts label the two graph variants, and the denominator is nonzero. The statistic is not an absolute signal-to-noise ratio. A short-pulse grid uses 100 evenly spaced samples from 0.081 to 0.8 s; a longer step grid uses a sustained focal extracellular input starting at zero with observations at 0.5, 1, 3 and 10 s. Both grids span spacing 2–1000 µm (75 logarithmic values) and mobility multipliers $10^{-4}$–$10^5$ (82 logarithmic values). The nominal-mobility one-dimensional maximum was located numerically in that spacing interval; it is not an analytically proved unique global optimum or a maximization over biological uncertainty. The illustrative grid is deliberately much broader than a measured physiological range.

## Fisher information under ideal source–sensor coverage

For the new ideal passive design, each site is stimulated separately by the same calibrated pulse; all seven extracellular potassium channels are recorded in every experiment at 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5 and 3 s. This yields $7\times7\times8=392$ scalar observations per complete protocol. Measurement errors are modeled as independent Gaussian deviations of standard deviation $\sigma=0.001$ mM (1 µM). The covariance and perfect source specificity are design assumptions, not instrument specifications. The sensitivity is to the dimensionless chord multiplier $g$ at $g=1$. Five nuisance multipliers, all at unity, scale glial mobility, extracellular transport, uptake, return and clearance independently. The volumes and pulse are fixed. Central finite differences with relative multiplier increments $10^{-4}$ and $5\times10^{-5}$ cross-check the sensitivities; relative changes in the complete whitened Jacobian are below $8\times10^{-9}$ in the reported calculations.

For a mean data vector $m(g,\eta)$ and parameter-independent positive-definite covariance $\Sigma$, let $s_g=\partial_g m$ and $S_\eta=\partial_\eta m$. With $\widetilde s_g=\Sigma^{-1/2}s_g$ and $\widetilde S_\eta=\Sigma^{-1/2}S_\eta$, the efficient local information after nuisance adjustment is

$$\begin{aligned}
I_{g\mid\eta}&=\min_b\|\widetilde s_g-\widetilde S_\eta b\|^2\\
&=\|(I-\widetilde S_\eta\widetilde S_\eta^+)\widetilde s_g\|^2.
\end{aligned}$$

Known parameters give $I_g=\|\widetilde s_g\|^2$. The formula states exactly whether the target score can be reproduced by linear combinations of nuisance scores in the specified data. It is a local measure and scales as the square of dose in this passive linear model. For an unbiased regular scalar estimator, the local Cramér–Rao lower bound is $\operatorname{SD}(\widehat g)\geq I_{g\mid\eta}^{-1/2}$. A nominal three-standard-deviation local separation for change $\Delta g$ and $M$ independent complete protocols requires $M I_{g\mid\eta}(\Delta g)^2\geq9$; this is a lower-bound planning calculation rather than a calibrated test's detection probability. Values below one complete protocol are not fractional experimental repetitions.

## Finite discrimination against shared support

Because a local derivative does not settle the finite comparison of graph families, we also generate a noiseless mean data vector from the independent graph with $g=1$ under the same 392-observation pulse design. Three restricted models fit those means: a single scalar glial path multiplier; six positive independently adjustable glial path-edge weights; and six path weights together with four multipliers for extracellular diffusivity, uptake, return and clearance. The latter is substantially more flexible than a globally rescaled extracellular path. The scalar and edge multipliers and local nuisances vary from 0.01 to 100 in logarithmic coordinates. These bounds define a broad, illustrative subset of the full nonnegative shared-support family; exact zero or unbounded weights are not included. The target source normalization and measurement covariance remain fixed.

The squared separation is

$$\mathcal D^2=\min_{\theta\in\Theta_0}\|\Sigma^{-1/2}[m_1-m_0(\theta)]\|^2.$$

Numerically we use bounded nonlinear least squares with ten deterministic or seeded starts, recording the best obtained distance, not claiming a certified global lower bound over the full mathematical cone. The same data vector is used to fit and compare synthetic means, so this is a *finite design-discrimination calculation*, not held-out predictive validation or observed statistical significance. A prospective biological test must fit on training sources and predict held-out responses.

# Results: spatial scale and discriminability

## Conditional spatial sensitivity window

At the source-reconciled nominal coefficients, $\sqrt{D_g/\kappa_a}=20.84$ µm, a characteristic length associated with glial exchange and diffusion in the passive equations. The spacing maximizing a graph contrast need not be that length, because the extracellular operator, input duration and observation schedule also set scales. The matched short-pulse schedule yields a numerical maximum at approximately 14.76 µm, where $\mathcal R=0.04645$; the longer step schedule gives approximately 36.98 µm, where $\mathcal R=0.19233$. At 5 µm and nominal mobility, the corresponding contrasts are 0.00392 and 0.00153. The two protocols differ in both the input waveform and the observation schedule; they show that changing the combined source and observation protocol can shift the optimum and need not increase relative graph-shape contrast at fixed intrinsic scale.

![Mobility–spacing windows for the two passive protocols. The circle marks the source-reconciled reduced point at 5 µm and unit mobility; the triangle marks the numerical spacing maximum along unit mobility. The circle is not a fitted effective location of the full multi-ion EDNM. Color scales differ by panel. The spacing and mobility axes span synthetic values, not measured tissue uncertainty.](matched_windows.png){width=100%}

These results make a risky but precisely delimited prediction: in this parameterized family of geometrically rescaled graphs, intrinsic scales of tens of micrometres rather than 5 µm yield greater topology sensitivity for these protocols. Changing the time protocol shifts the predicted optimum. The model does not predict a universal tissue length scale or a universal optimal astrocyte spacing; what is testable is the change in discriminability when the independently estimated coefficients and physical source–sensor schedules are specified. Varying $\mu$ can move the window substantially. Since the sweep changes the strength as well as the weights of the alternative relative to the path, its norm alone does not isolate shape from rescaling. The shared-support comparison below performs that isolation more directly.

## Exact recovery can be severely ill-conditioned

For the synthetic independent graph at $s=1$ s$^{-1}$, the relative worst-case recovery gain $\mathcal K$ is $8.51\times10^5$ at 5 µm and $2.32\times10^3$ at 14.76 µm. For $s=0.25$ and $4$ s$^{-1}$, the corresponding 5 µm gains are approximately $1.69\times10^6$ and $4.57\times10^5$, while those at 14.76 µm are $3.82\times10^3$ and $2.11\times10^3$. Thus the exact inverse can amplify small response perturbations by orders of magnitude. Changing spacing improves its conditioning markedly even when the underlying relative graph weights are unchanged. This resolves the apparent tension between an exact transfer inverse and a practically weak observed effect: the hidden operator is not absent, but its imprint can be overwhelmed by the size of the extracellular transfer.

The normalized operator-space residual after optimal scalar projection of $G$ onto the shared path is 0.27437 at these spacings. An appreciable difference between operators therefore coexists with a very small response difference. The former is not a proxy for the latter. A reconstruction algorithm that simply inverts a noisy transfer without checking uncertainty can generate spurious off-support edges even if the true graph is shared.

## Local information and finite shared-support fit

Table 2 gives the ideal passive Fisher results. At 5 µm, all-site observation still yields weak information even when all nuisance coefficients are known. With five nuisance multipliers the local information is only $2.40\times10^{-5}$. With just one stimulated site at 5 µm and seven sensors it falls to $4.38\times10^{-6}$ (known-parameter information $1.50\times10^{-3}$). Complete source coverage is useful, but it cannot solve the intrinsic geometric suppression at this spacing. At 36.98 µm the same 392 observations yield a known-parameter information of 152.6 and nuisance-adjusted information of 17.19. The latter is about $7.17\times10^5$ times its 5 µm value. This is a design prediction at the chosen nominal coefficients and assumed independent 1 µM noise, not a measurement of instrument feasibility.

Table 2. Local information in the new passive all-site design. The last column is the formal lower bound $9/I_{g\mid\eta}$ for nominal local three-standard-deviation separation at unit change; at least one complete protocol is required. The same short pulse and eight recording times are used at all spacings, including the spacing selected using the longer step contrast.

| Spacing (µm) | Known parameters | Unknown glial scale only | Five nuisances | $9/I_{g\mid\eta}$ |
|--:|--:|--:|--:|--:|
| 5.00 | 0.005692 | $2.400\times10^{-5}$ | $2.398\times10^{-5}$ | $3.75\times10^5$ |
| 14.76 | 19.72 | 0.3589 | 0.3502 | 25.7 |
| 36.98 | 152.58 | 19.90 | 17.19 | 0.524 |

The comparison between known and nuisance-adjusted parameters at 5 µm is informative: the scalar glial scale accounts for most of the information loss, but the raw information is already small. The bottleneck is therefore not merely insufficient nuisance calibration. At the larger spacing the target score is appreciably independent of nuisance directions even though overall glial mobility remains uncertain. This shows why a well-conditioned design can make calibration and sensor improvements much more productive than attempting to overcome an unfavorable operating regime with repetition alone.

Table 3. Best obtained squared noise-normalized distance between independent-topology synthetic means and fitted shared-support rivals. The last column allows all six path weights and four local/transport nuisance multipliers to adapt. Bounds and noise are those specified in Methods.

| Spacing (µm) | Scaled path | Six free path weights | Free path weights + nuisances |
|--:|--:|--:|--:|
| 5.00 | $4.747\times10^{-5}$ | $1.539\times10^{-5}$ | $7.885\times10^{-6}$ |
| 14.76 | 1.5176 | 0.7483 | 0.6508 |
| 36.98 | 133.66 | 98.26 | 90.74 |

At 5 µm the most flexible fitted rival leaves a maximum absolute sample discrepancy of only 0.000707 µM. At 36.98 µm that maximum increases to 2.42 µM; the squared whitened distance over the full response vector is 90.74. All ten optimization starts converge to effectively the same best cost for the reported cases. This numerical consistency is evidence against a simple local-minimum explanation, not a proof of global optimality. The larger spacing is the stronger discriminatory regime in this particular design, and the result persists when edge-specific transport weights rather than just one common strength are allowed to vary.

![Local Fisher information (A) and finite forward-model separation (B) under the same passive pulse design. Both panels use logarithmic ordinates. The finite comparison concerns noiseless simulated means under an illustrative noise model, not experimental p-values.](information.png){width=100%}

The result clarifies the difference between recovering all entries of a graph without constraints and testing a specific biologically motivated additional-edge hypothesis. The latter may require fewer identifiable parameters and can remain useful even when the full inverse has a large worst-case condition number. Conversely, rejecting one scalar rescaling of the extracellular graph is too weak a criterion: a flexible model with the same support can be much closer to the independent-topology model. The strongest numerical result is therefore the large change in separation from this *more flexible* null as the spatial regime changes.

## Fixed-tissue observation design and discretization

The preceding intrinsic scale sweep changes the physical graph. To isolate observation design from this transformation, we independently construct a sealed, one-dimensional finite-volume domain of length 259 µm. Each of seven 37-µm reference cells is subdivided into $r=1,3,7,15,21$ equal cells of width $h=37/r$ µm. The original three conduits connect the centers of the reference cells, with fixed integrated conductance $D_g w/d$ per unit cross-sectional area. Dividing the resulting flux by control-volume length yields the chord term $D_gw/(dh)$; local edges have the usual $D_g/h^2$ coefficient. Each source delivers the same spatially integrated mass per unit area, and the block-average designs preserve the physical support of both sources and measurements at every refinement level. The passive equations, time samples, nominal parameters and illustrative 1 µM independent measurement noise are those of the preceding design. This construction is a synthetic fixed-tissue discretization, not a reconstruction of actual CA1 tissue.

Table 4. Nuisance-adjusted Fisher information for fixed block-average inputs and measurements as the same physical domain is refined.

| Compartments | Nominal passive mobility | Reduced-mobility sensitivity case |
|--:|--:|--:|
| 7 | 17.2117 | 2.31976 |
| 21 | 9.25605 | 1.35952 |
| 49 | 8.55547 | 1.22013 |
| 105 | 8.43150 | 1.19015 |
| 147 | 8.41456 | 1.18584 |

The reduced-mobility case uses 0.17988 times nominal glial mobility as an illustrative cross-model sensitivity value motivated by the electrical estimate below; it is not a fitted passive reduction of the full electrodiffusive model. The coarse mesh overestimates nominal information by approximately a factor of two, but the block-average information converges toward a nonzero value. At 147 cells the relative whole-field K contrast is 0.08073 at nominal mobility and 0.02215 at reduced mobility, with maximum sample differences of 2.47 and 0.788 µM, respectively.

Holding the 49-cell tissue fixed, we next compare equal-count distributed and clustered focal source/sensor arrays. Seven sources and seven sensors distributed over the reference-cell centers yield information 12.10 (nominal) and 2.085 (reduced mobility). Seven sources and sensors clustered in the first seven cells yield 2.451 and 0.2363. Retaining the distributed sources and observing all 49 cells yields 79.57 and 12.20. These values account for the same five nuisance directions. Across a 32-corner sensitivity box combining mobility 0.1190–0.3108, extracellular diffusivity multipliers 0.7–1.3 and independent uptake, return and clearance multipliers 0.5–2, plus the central parameter point, distributed seven-source/seven-sensor information ranges from 0.124 to 18.43 (median 1.33). This is a finite illustrative sensitivity analysis, not a bound over the full continuous parameter box or a measured probability distribution. The point-source comparison holds mesh and source support fixed between designs; it is not itself a mesh-convergence claim.

Thus observation geometry remains consequential when the tissue itself is held fixed. Distributed coverage of the candidate conduits is more informative than dense sampling restricted to one end of this example. The original intrinsic-scale optimum must not be interpreted as an optimal electrode pitch.

# Full electrodiffusive implementation and physiological constraints

## Implementation and conservation

The passive analysis deliberately omits physiological pathways that can change both amplitude and sign of observed responses. To evaluate those omissions we use the independently published multicompartmental electrodiffusive network model (EDNM) of Sætra and Mori [10], pinned to commit `826c211c70d9027a4acb02202ca91a60de79a15d`. Each unit contains soma and dendrite layers with neuronal, glial and extracellular domains; it evolves Na, K, Cl and Ca concentrations, membrane and extracellular potentials, gating variables and water-dependent volume fractions. It enforces a capacitive charge relationship rather than computing extracellular potential from potassium concentration alone.

For an ionic species $i$ in domain $r$, the interunit flux on an oriented edge $u\to v$ uses the centered Nernst–Planck discretization

$$j_{r,uv}^i=-\frac{D_i\upsilon_{ir}w_{uv}}{\lambda_r^2d}\left[c_v^i-c_u^i+\frac{z_i(c_u^i+c_v^i)}{2\psi}(\varphi_v-\varphi_u)\right],$$

where $\lambda_r$ is tortuosity, $\upsilon_{ir}$ the mobile fraction, $\psi=RT/F_{\!A}$ and $F_{\!A}$ Faraday's constant. The source finite-volume assembly includes the previous-step domain volume fraction and a second factor of inverse unit spacing. Consequently both chemical and electrical axial transport have the same explicit inverse-square scaling in the volume-normalized inventory equations. Membrane area per tissue volume, interlayer distance, local kinetics and normalized forcing remain fixed; absolute delivered ionic mass need not remain fixed across rescaled tissues. Each edge correction is inserted with opposite signs at its two endpoints. Glial and extracellular weights can be changed separately without changing local membrane fluxes, pumps or gating equations. The adapter replaces the original closed-path contribution by adding the difference of the new and old edge residuals and their analytic sparse Jacobian. The compiled parent local equations are retained. Earlier finite-difference Jacobian checks and the fine-step validation are archived with the source.

This structure preserves total inventory of each ionic species under internal transfers and sealed boundaries. A separate capacitive balance must hold for charge; mass conservation alone cannot establish electrical consistency. For the neuronal soma,

$$\gamma_m C_m V_m=F_{\!A}\left(\alpha_n\sum_i z_i c_{n,i}+z_0a_n^{\mathrm{imm}}\right),$$

with constant immobile-ion inventory $a_n^{\mathrm{imm}}$ and $V_m=\varphi_n-\varphi_e$. We use the discrete current balance implied by this relation and the actual semi-implicit time update to diagnose the neighboring response.

The matched 5-µm full-model comparison gives a relative K contrast of 0.000438 versus 0.00393 in the source-reconciled passive approximation. This quantitative difference motivates the axial-migration intervention below; the detailed matched comparison and its flux accounting appear in Supplement F.

## Electrical migration shifts the topology-sensitive window

We extend the matched short-pulse seven-site experiment to intrinsic scales 5, 15, 37, 60, 100 and 150 µm. The native-glial sweep retains the original electrodiffusive constitutive law. A diagnostic counterfactual sets only the axial glial electrical-migration coefficient to zero in both the graph replacement residual and its Jacobian, retaining chemical diffusion, the extracellular and local membrane terms and the capacitive constraints. This changes a transport law and is not equivalent to a selective physiological blocker. In both conditions internal edge contributions remain conservative. Each statistic is calculated from paired baseline-subtracted trajectories at the same scale. Both timesteps, 1 and 0.5 ms, were tested. The interpretation record commits to a predicted shift toward smaller scales before these follow-up outputs were inspected, but after execution had started and completed; it is not a preregistration before execution ([outcome-blind interpretation record](https://github.com/ipsissima/SMM/blob/main/docs/MIGRATION_HYPOTHESIS.md)).

Table 5. Whole-field relative extracellular potassium contrast in the full EDNM (percent, 0.5 ms timestep). The maximum is on the sampled grid, not a continuous or universal tissue optimum.

| Intrinsic scale (µm) | Native migration | Glial axial migration off |
|--:|--:|--:|
| 5 | 0.043825 | 0.397940 |
| 15 | 0.728079 | 0.226676 |
| 37 | 2.019322 | 0.051258 |
| 60 | 2.364808 | 0.020855 |
| 100 | 2.012149 | 0.007735 |
| 150 | 1.479953 | 0.003472 |

The native response confirms suppression at 5 µm and a pronounced intermediate window, but its sampled maximum is 60 µm, beyond the passive short-pulse maximum near 15 µm. At 60 µm the relative contrast is about 113 times that of the migration-off counterfactual. At 5 µm the direction of the effect reverses: migration suppresses contrast. The counterfactual shifts the sampled maximum to 5 µm, with monotonically decreasing values across the rest of the sampled grid. The refined timesteps agree closely and the extended runs satisfy the positivity, convergence and ionic inventory checks. This result identifies the axial migration term as a cause of the shift within the solver, rather than identifying its magnitude with an effective scalar diffusivity. Because the transmembrane stimulus has an electrical footprint, the primary endpoint here is extracellular K, not voltage. The passive theorem concerns mobility limits at fixed geometry and does not itself prove the existence of a full nonlinear maximum as intrinsic scale changes.

To test the length-scale interpretation independently of the zero-migration contrast, we additionally multiply only the axial glial electrical term by $m=0.25$ or $4$. A frozen-concentration, two-conductor approximation gives a voltage-difference length

$$\ell(m)=\left[g_m\left(\frac{1}{m\sigma_g}+\frac{1}{\sigma_e}\right)\right]^{-1/2},$$

where $\sigma_g,\sigma_e$ are axial tissue conductivities and $g_m$ is glial membrane slope conductance per tissue volume. In this reduction the oppositely directed axial conductor currents conserve total current and membrane exchange is proportional to their voltage difference. Eliminating the currents gives the differential relation $v''=g_m[(m\sigma_g)^{-1}+\sigma_e^{-1}]v$ away from sources. The frozen slope and conductivities are calculated from the native EDNM resting state, independently of the graph contrast maxima. This is not the exact electrical mode of the full multilayer, concentration- and volume-dependent system.

At $m=0.25,1,4$, respectively, the calculated lengths are 47.84, 88.34 and 140.07 µm and the sampled contrast maxima occur at 37, 60 and 100 µm. These coarse maxima motivated the scaling hypothesis $d_*\approx0.7\ell$ under this forcing and graph family. A targeted refinement at 0.5 ms, with additional intrinsic scales bracketing each maximum, yields maxima 35, 60 and 85 µm for $m=0.25,1,4$, with respective ratios 0.732, 0.679 and 0.607. The larger migration multiplier therefore exhibits a materially lower ratio than the initial coarse-grid calculation suggested. The electrical length predicts the ordering and approximate scale, not a universal proportionality. Each maximum is internal to the newly sampled interval; no claim of a continuous or unique global maximum is made. Current-driven spatial buffering is established physiology [3,35,36,39]. In particular, Chen and Nicholson [36] derived an optimum at equality of geometric and electrotonic length for the endfoot current of an equivalent leaky cable. The present result concerns a different observable: the difference between responses generated by distinct intercellular transport graphs under matched local physiology and forcing. Thus the matching of electrical and geometrical scales in buffering is not new; the result here concerns its consequences for topology discrimination, including the noncommuting two-conductor graph operator and the protocol-specific optimum in the full electrodiffusive model. Recent within-cell electrical compartmentalization results [44] provide complementary motivation, not a prior derivation of this topology-identification result.

The independent two-unit pump-clamp analysis and its exact current balance are reported in Supplement G. It establishes a model-specific mechanism for voltage inversion but is not itself a topology test.

## Matched-dose neutral KCl topology challenge

The seven-site topology comparison is repeated using a state-independent extracellular source that delivers equal K and Cl inventories to one somatic ECS control volume during 0.08–0.28 s. This is externally charge-neutral but introduces osmoles; it is not a physical reconstruction of iontophoresis. The two rates, 3 and 30 mM tissue/s, deliver 0.6 and 6 mM of each ion per source-cell tissue volume. We compare the independent graph to the ordinary path, a path whose six edge weights are all $7.9/6$ (equal total glial edge weight and Laplacian trace), and an axially uncoupled graph. All other quantities are held fixed within a matched comparison. The intrinsic scale is 37 µm, native glial mobility is used, sources are nodes 0 and 2, and the simulation horizon is 0.8 s. Independent baseline trajectories and timestep checks at 0.5 and 0.25 ms control drift and numerical error.

Table 6. Topology effects relative to the equal-total-weight path at the finer timestep. The relative K statistic divides by the norm of the rival's baseline-subtracted K response; absolute maxima use all observed nodes and times.

| Source | Rate (mM tissue/s) | Relative K contrast (%) | Maximum K difference (mM) | Maximum membrane voltage difference (mV) |
|--:|--:|--:|--:|--:|
| 0 | 3 | 2.15090 | 0.032603 | 0.045324 |
| 0 | 30 | 3.01463 | 0.574726 | 0.339096 |
| 2 | 3 | 1.13519 | 0.015345 | 0.025019 |
| 2 | 30 | 1.28343 | 0.163860 | 0.256193 |

Relative K contrast increases by 40.16% and 13.06% for the two sources, respectively; this dose dependence cannot occur for a linear system under identical waveform scaling. These are nonlinear contrasts relative to a particular rival, not a lower bound on topology separation after arbitrary edge fitting. Absolute differences of hundreds of µM in K and hundreds of µV in neuronal voltage are plausible targets for appropriately characterized sensors, not measured signals. The high-load source elevations are 23–26 mM, versus 2.3–2.5 mM at low load. Removing axial glial coupling from the path increases peak source K by 3.80% and 4.38% at low load and 4.79% and 6.12% at high load for sources 0 and 2. No matched-dose run reached the model's spike threshold or crossed −40 mV; there is no categorical firing or depolarization-block effect in these protocols. The separate published-source coupling-strength benchmark is reported in Supplement E, not counted as a topology test.

At distal node 5 for source 0, the high-dose shortcut-minus-weight-matched-path difference at 0.8 s is +0.207535 mM extracellular K, +1.370738 mV K reversal potential, and −0.339096 mV neuronal voltage. Four-way subtraction of baseline and stimulated trajectories from both geometries gives integrated pump, K leak, delayed-rectifier K, Cl leak, Na leak and gated Na contributions of −12.443, +5.495, +1.897, +4.014, +1.033 and −0.567 mV, respectively, with net remaining axial terms +0.232 mV. The current sum reconstructs the voltage difference with maximum paired closure error $1.46\times10^{-8}$ mV. The maximum somatic extracellular volume-fraction difference at this distal node is $5.01\times10^{-6}$ in absolute fraction. The small differential swelling and current accounting support the ionic-current interpretation but do not independently exclude indirect effects of water dynamics. The earlier pump clamp (Supplement G) establishes a separate two-site causal result; this four-way accounting is not itself an intervention. Because the residual voltage is a small difference of much larger opposing currents, we interpret its pathway and magnitude at the specified kinetics rather than predict a robust sign across uncertain pump affinity and leak conductance.



## Empirically informed electrical parameter constraint

Ma et al. [39] report a paired-astrocyte coupling resistance of 4.2 ± 1.5 MΩ under special recording conditions designed to reduce membrane shunting. Independent CA1 measurements by Kiyoshi et al. [46, Fig. 5B–D] give cell density 11,238 ± 331 mm$^{-3}$, nearest-neighbor separation 39.45 ± 1.14 µm and 8.4 ± 0.16 nearest neighbors. A simple affine effective-medium calculation combines these independent summaries. For an isotropic resistor network with density $n$, mean degree $z$, identical edge conductance $1/R$ and representative edge length $d_a$, the dissipation per tissue volume under an affine potential gradient is

$$\frac{P}{V}=\frac{nz}{2R}\mathbb E[(\nabla\varphi\cdot\mathbf d_a)^2]
=\frac{nzd_a^2}{6R}|\nabla\varphi|^2,\qquad
\sigma_g^{\mathrm{aff}}=\frac{nzd_a^2}{6R}.$$

The factor $1/2$ avoids double counting edges, and isotropy gives the factor $1/3$. This estimate assigns the isolated-pair resistance to all representative neighbors and neglects edge heterogeneity, anisotropy, nonaffine electrical relaxation and correlations between anatomical variables. For a specified resistor network, affine boundary-compatible voltages give a variational dissipation upper bound relative to the relaxed voltages; the estimate here is not a rigorous upper bound for real CA1. CA1 coupling can itself be anisotropic [47].

The conditional estimate is $5.83\times10^{-5}$ S/cm, or 0.180 of the native EDNM axial glial value ($3.24\times10^{-4}$ S/cm). Changing the four inputs independently to their favorable or unfavorable reported one-SEM endpoints yields $3.86\times10^{-5}$ to $1.01\times10^{-4}$ S/cm, or 0.119–0.311 of native conductivity. This is an endpoint sensitivity box, not a confidence interval. With the native EDNM frozen glial membrane slope $g_m=3.335$ S/cm$^3$ and extracellular conductivity $0.0013213$ S/cm, the two-conductor estimate yields an electrical length of 40.9 µm (33.5–53.0 µm in the endpoint box). Extrapolating the coarse migration-only relationship gave a prospective estimate near 29 µm (23–37 µm at the conductivity sensitivity endpoints). We tested this separately by scaling both chemical diffusion and electrical migration in the Nernst–Planck graph law, keeping the short transmembrane pulse, local kinetics and graph family unchanged. Across 22 additional constitutively scaled cases at 0.5 ms, the lower, central and upper conductivity estimates give sampled maxima 25, 30 and 40 µm, respectively; the corresponding ratios to the electrical length are 0.746, 0.733 and 0.755. The sampled maximum at the central estimate therefore corroborates the prospective approximately 30-µm prediction in the full model. The 25–40 µm range is a finite endpoint sensitivity result, not a confidence interval or a bound over continuous physiological uncertainty. This is an intrinsic scale prediction for a synthetic graph and specified input, not an electrode-spacing rule or a directly measured optimum in CA1.

![Refined electrical topology-sensitivity windows. Left: whole-field short-pulse K contrast for migration-only interventions and jointly scaled chemical and electrical transport at the empirical conductivity estimate and sensitivity endpoints. Right: normalized contrast against intrinsic scale divided by the independently calculated electrical length; symbols mark sampled maxima. The full model does not exhibit exact universal collapse, but the three empirically informed cases cluster near 0.74. All curves use 0.5 ms steps.](electrical_peak_refinement.png){width=100%}

Breithausen et al. [15] also provide independent experimental targets: focal K transients with pre-blockade mean peaks 2.30 and 8.28 mM under distinct weak and strong iontophoresis protocols, and a measured spatial peak-decay length 13.08 ± 0.79 µm. The strong group showed a significant increase relative to time controls after pharmacological uncoupling; the weak group did not. The published group mean changes and time constants motivate a coupled-only source calibration followed by prediction of the uncoupled responses without fitting to them. A one-dimensional neutral-KCl screening model is not a faithful reconstruction of their three-dimensional charged iontophoresis and blocker experiment; the individual traces and recording geometry are required for quantitative trace-level validation. We do not count aggregate target extraction or a fitted source amplitude as an independent validation of SMM's extra-edge prediction.

## Topology contrast in the empirically informed conductivity regime

We test the independent graph with both chemical and electrical axial glial fluxes scaled together, rather than scaling only migration. Thirty-two configurations compare ordinary, independent and total-glial-edge-weight-matched paths, each with its own baseline. The design varies mobility over the effective-medium sensitivity values 0.1190, 0.1799 and 0.3108, intrinsic scale 15, 37 and 60 µm, neutral KCl dose 3 and 30 mM tissue/s, selected source sites 0 and 2, and selected extracellular axial multipliers 0.7 and 1.3. The primary timestep is 1 ms with central 37-µm cases repeated at 0.5 ms. All configurations complete the positivity, mass and Newton-convergence checks. The following results are for the total-edge-weight-matched path.

Table 7. Central empirically informed mobility 0.17988 at 37 µm.

| Source | Dose (mM tissue/s) | Timestep (ms) | Relative K contrast (%) | Max K difference (mM) | Max voltage difference (mV) |
|--:|--:|--:|--:|--:|--:|
| 0 | 3 | 1 | 3.66765 | 0.062014 | 0.085459 |
| 0 | 30 | 1 | 3.86802 | 0.811412 | 0.598052 |
| 0 | 30 | 0.5 | 3.86830 | 0.811510 | 0.598588 |
| 2 | 30 | 0.5 | 2.09264 | 0.320331 | 0.542941 |

At the central high-dose source-0 point, the absolute differences exceed the native-conductivity results, with close agreement under timestep refinement. At the central low-dose point the relative K contrast is already 3.67%, so high load increases it by about 5.46%, less than the native-mobility proportional increase. This is evidence that the absolute topology contrast need not vanish near the estimated biological conductivity. It does not identify the true graph in tissue or establish the residual against freely weighted alternatives at this mobility. The upper conductivity endpoint at 37 µm and high dose gives 0.826 mM maximum K difference and 0.541 mV maximum voltage difference. The lower endpoint at 60 µm gives 0.473 mM and 0.522 mV. Across all 16 low-dose configurations (including refinement runs), maximum absolute K differences range from 0.00905 to 0.06203 mM; across the 16 high-dose configurations they range from 0.1052 to 0.8255 mM. Corresponding high-dose voltage differences range from 0.1462 to 0.6084 mV. All configurations retain nonzero separation from the restricted total-weight-matched rival; none produces spikes. The relative K contrast ratio for high versus low dose ranges from 0.785 to 1.468, increasing in nine of sixteen paired comparisons (eight of fourteen distinct parameter and source combinations, excluding the two timestep repeats). In particular, the 60-µm cases show a relative contrast decrease with dose. Nonlinearity can amplify or attenuate relative discrimination according to regime; the native-mobility amplification is not a universal monotonic law. At the central mobility and 37 µm, changing extracellular axial transport by ±30% retains high-dose absolute K differences of 0.796–0.816 mM. These are finite sensitivity calculations rather than robustness bounds over a continuous parameter domain. No physiological transfer of the derived empirical mobility to the particular synthetic long edges is implied.

## Freely weighted nonlinear rivals

The native-conductivity joint-source glial-only fit (Supplement H) provides an initial stress test: six independently variable path weights are fitted jointly to sources 0 and 2 and frozen for prediction of source 4. The held-out K and neuronal voltage RMS discrepancies are 0.0662 mM and 0.0539 mV. The squared whitened discrepancy at 15 times, seven sites and two channels is $5.18648\times10^5$ under illustrative independent noise scales 1 µM and 10 µV, corresponding to a whitened norm of approximately 720. These numbers are synthetic design distances, not an experimental significance test. Large fitted weights and the absence of independent edge-level conductance bounds motivate comparison in the empirically informed mobility regime. A bulk effective-conductivity estimate does not mathematically determine an upper bound on every individual edge, so candidate edge-wise bounds must be specified as explicit sensitivity assumptions. Independent measurement of conductance heterogeneity and extracellular transport is needed to turn such constraints into preparation-specific bounds.

## Aggregate CA1 screening: agreement and discrepancies

In a separate screening calculation, published pre-CBX means are used only to determine two effective neutral KCl source rates in a sealed 15-site, 5-µm one-dimensional path, with a recording proxy 10 µm from the source. Local EDNM physiology is unchanged, and the uncoupled responses use the fitted source rates without fitting to the blocker effects. The comparison is with the aggregate difference of the experimental CBX and no-drug group means in Fig. 5 of [15], not a reanalysis of individual paired data.

Table 8. Uncoupling-induced peak K change. Experimental differences are in percentage points; simulated changes compare the zero-glial-axial-coupling condition with its coupled control.

| Condition | Weak pulse | Strong pulse |
|:--|--:|--:|
| Experiment, CBX minus control | +8.18 | +16.88 |
| EDNM native, 1 ms | +3.140 | +4.199 |
| EDNM reduced mobility, 1 ms | +2.860 | +3.756 |

The native weak-pulse fitted single-exponential decay time is 2.141 s, close to the reported pre-CBX mean 2.20 s. The native strong-pulse fitted fast/slow time constants are 0.321 and 4.629 s, compared with 1.20 and 6.79 s experimentally. The simulated spatial peak-decay lengths are 15.94 µm (weak) and 23.67 µm (strong), whereas the experimental overall decay estimate is 13.08 ± 0.79 µm. These spatial estimates are from different observation designs and are not a direct paired comparison. Both mobility conditions give closely agreeing results at 2 and 1 ms. The simulations reproduce the direction but underestimate the experimental difference in group means; neither a sharp experimental threshold nor quantitative physiological validation is established. In the separate current- and duration-ramp calculations, the largest coupled recorder peak is only about 9.31 mM, below the reported approximately 10–15 mM recorder-amplitude band at which significant uncoupling effects emerge. Thus these ramps do not test that threshold directly. The independent Fig. 6b near-source subgroup is especially instructive: at mean recorder distances of 9.20 and 9.86 µm, the reported CBX peak changes are 1.15 ± 9.49% (weak, five observations) and 34.7 ± 10.9% (strong, seven observations). Although pooled Fig. 5 amplitudes were used for the source fits rather than the near-source subgroup amplitudes, this subgroup emphasizes the magnitude of the unresolved high-load discrepancy. We do not interpret it as a matched quantitative prediction or test of statistical equivalence. The pooled distances, electrode geometry and delivered ion mass are not known; three-dimensional dilution and bath exchange are absent, native active neuronal physiology is retained rather than implementing the experimental activity blockade, neutral salt injection differs from iontophoresis, and complete axial glial uncoupling is not identical to CBX, which also has non-junctional neuronal effects [48]. The discrepancy therefore cannot be assigned uniquely to underestimated glial transport. The individual traces are needed for a faithful calibration. These limitations concern this screening model and motivate more faithful preparation-specific modeling; they do not by themselves determine whether a distinct glial support is needed.

# Discussion: what would identify an independent control geometry?

## From possibility to a model-specific biological test

The mathematics explains a potentially important dissociation: a glial graph can be structurally different and dynamically coupled yet leave almost no distinctive trace in a particular observation. At fixed homogeneous kinetics and low mobility, the graph does little redistribution. At very high mobility, connected graphs collapse toward the same uniform glial state. In between, geometry matters. Under the specified source-reconciled model, changing spatial scale can make both the local score and finite shared-support discrepancy orders of magnitude larger. This is a substantive prediction about the relationship between geometry, timescale and observability, not merely a statement that adding an edge changes an equation.

It is equally important to keep the levels separate. The passive model identifies a weighted glial operator conditional on its assumptions. The full EDNM demonstrates conservation, a migration-dependent shift of the sampled sensitivity window and dose-dependent nonlinear topology contrast. Independent aggregate physiological measurements constrain a plausible electrical parameter regime, rather than validating the particular synthetic graph. The broader architectural claim is motivated by independent anatomical and functional observations [5,6,13,15,16,39,46]; whether the distinct organization is required to predict neuronal responses in a preparation needs a discriminating experiment. No new animal or human recordings are presented here.

A core experimental prediction should accordingly concern support. In a preparation with an independently characterized extracellular graph, the fitted glial operator should require reproducible, functionally relevant off-support coupling. The primary analysis should compare a shared-support model with free nonnegative edge weights against a prespecified independent-support model, while applying identical local kinetic, source, sensor and covariance constraints. Parameters should be trained on one subset of stimulus sites and conditions, then frozen for prediction of withheld spatial and temporal responses. A measured residual beyond uncertainty from the restricted family and a cell-specific intervention affecting the inferred route with downstream neuronal consequences would support the independent-geometry realization. A powered equivalence test placing off-support coupling and its downstream effect below prespecified biologically meaningful bounds would reject that realization in the preparation. A null result in a demonstrably insensitive experiment is not the same as such a test.

This approach is stronger than simply testing whether recovered weights are statistically nonzero: one must establish source specificity, rule out changes in effective extracellular support, and avoid confounding local membrane currents or generic latent feedback with spatial glial transport. Similarity-equivalent hidden realizations can share all recorded neuronal and extracellular transfer functions. Identifiability of the mathematical operator within a restricted model is not unrestricted identification of a cell type. Direct astroglial observations or validated interventions are essential for that final attribution.

## Experimental design and biological feasibility

The proposed sequence is first to characterize the preparation: local extracellular spread and volume fraction; astroglial coupling and relevant local uptake/return kinetics; source spatial profile and ionic/electrical dose; and neuronal synaptic/excitability controls. Potassium reporter kinetics, pH sensitivity, range and correlated noise should be measured in the exact regime, rather than assumed from published large-event detection [24,25]. An independently measured coupling marker can constrain the candidate graph family, but a tracer network cannot be converted directly into an absolute ionic conductivity matrix. Particularly valuable preparations would offer an astroglial route that differs from a measured extracellular transport route at the recording scale.

The rescaled-graph calculations identify candidate intrinsic transport regimes, not an electrode-spacing rule. On a fixed tissue, characterize the physical routes first and select source–sensor configurations that cover both ends of candidate conduits, with separations spanning the experimentally measured extracellular spread during the actual forcing protocol. Independently estimate effective parameters, propagate uncertainty and compare source duration and observation schedules without conflating them with changes in tissue geometry. Focal source locations and distributed inputs provide multiple spatial modes. Uniform stimulation in a sufficiently homogeneous preparation is a useful negative control; direct glial input, when selective and physically quantified, changes the pathway order and may provide a more informative design. Electrical artifacts must be monitored at onset and offset, particularly for membrane-current-driven sources. Directly applied extracellular K loads may require explicit counterion transfer and volume accounting that are different from a transmembrane K exchange.

A prospective robust design objective is

$$\mathcal D^*=\arg\max_{D\in\mathcal A}\inf_{\theta\in\Theta}I_{g\mid\eta}(D;\theta),$$

where $\mathcal A$ is a set of experimentally feasible calibrated protocols and $\Theta$ an independently justified parameter uncertainty set. A finite predictive-distance criterion against the flexible shared-support family is a complementary objective [27,32]. The present computations do not solve this optimization over measured biological uncertainty; they show why spatial and temporal design can dominate nominal repetition counts.

For a nonlinear validation, a first step is to reproduce the contrasting low- and large-load responses of the CA1 experiments [15] with their measured focal geometry, source dose, recording positions and glial-coupling manipulation. The experimental threshold is protocol-specific; the artificial strong-glia EDNM reproduction cannot supply it. Then a full-model graph sweep should match local calibration and compare freely weighted shared support with additional edges. Strong-coupling or high-K regimes may increase absolute effects but may also introduce nonlinear saturation, volume changes and excitability transitions, and thus cannot simply inherit the passive information numbers.

Cell-type-specific and reversible coupling interventions with independent coupling assays and rescue are particularly useful. Connexin interventions can modify metabolite exchange, glutamate handling, volume and neuronal excitability; Kir4.1 manipulations alter more than spatial transport [20,21]. Measuring those routes prevents a glial-coupling manipulation from being interpreted as a pure change in $D_g$ when it is not. The prediction requires coupling-dependent *spatial organization* beyond a generic change in total clearance.

## Implications for neuronal coordination and electrophysiology

The role of glial networks in determining neuronal response geometry is compatible with the fact that neuronal membrane currents principally generate EEG and related extracellular fields. A physiologically identified glial transport graph could change the pattern of ionic states, neuronal membrane currents and thus the fields those currents generate. The required causal chain is glial transport $\to$ extracellular and local ionic state $\to$ neuronal current generation $\to$ a calibrated volume-conductor or lead-field map. An EEG spectrum alone does not identify the hidden spatial controller. The original local spectral comparison omitted the glial transport operator, so it neither tests this defining prediction nor supplies evidence for it (Supplement D). The low-frequency resonome remains a motivating hypothesis for a separately validated extension [2].

## Limitations

The passive parameter set is reconstructed from an earlier source calibration and uses an inherited local clearance coefficient; it is not a fit to a preparation or an exact reduction of EDNM. The comparison to full EDNM shows that its quantitative extrapolation is not yet warranted. The graph edges are synthetic and the source-reconciled coefficients, volumes, spatial grid, nuisance bounds and noise assumptions define the reported numerical results. The conditional spacing maxima and information values are not biological optima or actual experimental power estimates. The finite forward fit is numerically reproducible but not a certified global optimization over all nonnegative shared-support weights. The original independent graph experiment does not normalize graph strength before adding chords; the flexible shared-support fit is the more stringent strength-adjusted comparison. The electrodiffusive contrasts and fitted rivals are conditional on the synthetic graphs, fixed local kinetics and specified parameter bounds. The fixed-tissue example is a passive sealed one-dimensional model and its sensitivity box is illustrative. The independent aggregate-data screening reproduces direction but underestimates the experimentally observed uncoupling effect; the particular source and geometry cannot be quantitatively validated without individual traces and more faithful preparation modeling. The aggregate electrical constraint uses an affine homogenization assumption and separately measured input quantities; it does not calibrate all transport and membrane coefficients. A prospective biological support test, faithful CA1 source geometry and trace-level calibration, and broader robustness to uncertain parameters remain necessary. The high-load reproduction demonstrates the source model's nonlinear behavior at an intentionally artificial mobility multiplier. Older source-reconstruction quantities whose exact historical forcing or fixed-inventory procedure cannot be verified are reported as historical diagnostics in the supplement, not prospective validation. No new biological, human or animal data were acquired.

# Conclusion

SMM proposes that the spatial conditions shaping neuronal currents can have a partially independent astroglial organization. The present potassium-mediated theory specifies that idea as an operator-level and experimental hypothesis. Exact conditional glial operator recovery coexists with severe practical ill-conditioning, and graph-shape contrast is suppressed at both mobility extremes. Full electrodiffusive simulations identify a shift of the topology-sensitive window controlled by axial migration. In the empirically informed mobility regime, the sampled maximum occurs at 30 µm for the specified short pulse, consistent with the independently calculated electrical length and its prospective extrapolation. Matched neutral KCl dosing shows regime-dependent changes in relative topology contrast and substantial absolute differences against a total-weight-matched rival; current accounting relates distal effects to competing neuronal pump and channel currents. Published anatomical and electrophysiological measurements constrain a plausible conductivity regime. The decisive biological prediction is not that glia influence neurons or that their effects are always large, but that a correctly characterized preparation requires independent glial transport support to predict spatial ionic and neuronal responses. This is a precise, falsifiable realization of the original SMM architecture.

# Code and data availability

The independent EDNM implementation is available from <https://github.com/martejulie/electrodiffusive-network-model>, commit `826c211c70d9027a4acb02202ca91a60de79a15d`. The two-site validation and graph correction are in the SMM research repository (<https://github.com/ipsissima/SMM>), branches [full-solver validation](https://github.com/ipsissima/SMM/tree/research/ednm-full-solver-validation-20261126) and [graph topologies](https://github.com/ipsissima/SMM/tree/research/graph-ednm-topologies-20261126). Original trajectory artifacts are associated with runs [45839123584](https://github.com/ipsissima/SMM/actions/runs/45839123584), [45840811721](https://github.com/ipsissima/SMM/actions/runs/45840811721), and [45847600616](https://github.com/ipsissima/SMM/actions/runs/45847600616). The pump clamp and high-load reproduction are in [mechanism/high-load branch](https://github.com/ipsissima/SMM/tree/research/ednm-mechanism-high-load-20261126), commit `3798c749cdbed18e4222dc805e94f28a71cde5cb`, workflow run [45859252351](https://github.com/ipsissima/SMM/actions/runs/45859252351). Relevant artifact IDs are 15137282506 and 15137766579 (pump clamp, 1 and 2 ms), 15137517741 and 15137242820 (weak and strong coupling, original 50 µs), and 15138690510 (weak coupling, 100 µs). Workflow artifacts have finite retention; the accompanying diagnostic code and machine-readable summaries identify the exact inputs and checks. The completed core trajectory arrays and checksum manifests are preserved in a staged repository release. A permanent DOI deposit of the final numerical and source archive remains a submission prerequisite.

The accompanying reproducibility package contains the matched passive/full-model diagnostic, inverse conditioning, Fisher calculations, shared-support fits, extended migration and electrical-length sweeps, matched KCl dose and distal current/volume summaries, and empirical electrical constraint. Exact driver commits, workflow links, data conventions and known limitations are indexed in the repository validation documents. Full-state trajectories are currently stored as time-limited workflow artifacts; permanent raw-data archiving is a submission prerequisite. Historical neuronal-population and EEG analyses are in the separate Supplementary material and are not used as empirical support for spatial transport.

# References {.unnumbered}

1. Ballús, A. (2024). *The Brain-Mesh Model: A Unified Framework for Neural Synchrony, Plasticity, and Coherence*. bioRxiv, version 1, posted 25 November 2024. <https://doi.org/10.1101/2024.11.22.624908>; arXiv:2412.12106v1. <https://arxiv.org/abs/2412.12106v1>.

2. Ballús Santacana, A. (2026). Glial syncytial control fields and the low-frequency resonome of the human brain. *Frontiers in Human Neuroscience*, 20, 1896291. <https://doi.org/10.3389/fnhum.2026.1896291>.

3. Orkand, R. K., Nicholls, J. G., & Kuffler, S. W. (1966). Effect of nerve impulses on the membrane potential of glial cells in the central nervous system of amphibia. *Journal of Neurophysiology*, 29, 788–806. <https://doi.org/10.1152/jn.1966.29.4.788>.

4. Kofuji, P., & Newman, E. A. (2004). Potassium buffering in the central nervous system. *Neuroscience*, 129, 1043–1054. <https://doi.org/10.1016/j.neuroscience.2004.06.008>.

5. Wallraff, A., Köhling, R., Heinemann, U., Theis, M., Willecke, K., & Steinhäuser, C. (2006). The impact of astrocytic gap junctional coupling on potassium buffering in the hippocampus. *Journal of Neuroscience*, 26, 5438–5447. <https://doi.org/10.1523/JNEUROSCI.0037-06.2006>.

6. Pannasch, U., et al. (2011). Astroglial networks scale synaptic activity and plasticity. *PNAS*, 108, 8467–8472. <https://doi.org/10.1073/pnas.1016650108>.

7. Halnes, G., et al. (2013). Electrodiffusive model for astrocytic and neuronal ion concentration dynamics. *PLOS Computational Biology*, 9, e1003386. <https://doi.org/10.1371/journal.pcbi.1003386>.

8. Sætra, M. J., Einevoll, G. T., & Halnes, G. (2021). An electrodiffusive neuron-extracellular-glia model for exploring the genesis of slow potentials in the brain. *PLOS Computational Biology*, 17, e1008143. <https://doi.org/10.1371/journal.pcbi.1008143>.

9. Sætra, M. J., Ellingsrud, A. J., & Rognes, M. E. (2023). Neural activity induces strongly coupled electro-chemo-mechanical interactions and fluid flow in astrocyte networks and extracellular space—A computational study. *PLOS Computational Biology*, 19, e1010996. <https://doi.org/10.1371/journal.pcbi.1010996>.

10. Sætra, M. J., & Mori, Y. (2024). An electrodiffusive network model with multicompartmental neurons and synaptic connections. *PLOS Computational Biology*, 20, e1012114. <https://doi.org/10.1371/journal.pcbi.1012114>.

11. Montbrió, E., Pazó, D., & Roxin, A. (2015). Macroscopic description for networks of spiking neurons. *Physical Review X*, 5, 021028. <https://doi.org/10.1103/PhysRevX.5.021028>.

12. Howarth, C., Gleeson, P., & Attwell, D. (2012). Updated energy budgets for neural computation in the neocortex and cerebellum. *Journal of Cerebral Blood Flow and Metabolism*, 32, 1222–1232. <https://doi.org/10.1038/jcbfm.2012.35>.

13. Cooper, M. L., Selles, M. C., Cammer, M., et al. (2026). Astrocytes connect specific brain regions through plastic networks. *Nature*, 655, 183–191. <https://doi.org/10.1038/s41586-026-10426-6>.

14. Halassa, M. M., Fellin, T., Takano, H., Dong, J.-H., & Haydon, P. G. (2007). Synaptic islands defined by the territory of a single astrocyte. *Journal of Neuroscience*, 27, 6473–6477. <https://doi.org/10.1523/JNEUROSCI.1419-07.2007>.

15. Breithausen, B., Kautzmann, S., Boehlen, A., Steinhäuser, C., & Henneberger, C. (2020). Limited contribution of astroglial gap junction coupling to buffering of extracellular K+ in CA1 stratum radiatum. *Glia*, 68, 918–931. <https://doi.org/10.1002/glia.23751>.

16. Bojovic, D., Dagostin, A., Sullivan, S. J., Emery, B., von Gersdorff, H., & Mishra, A. (2025). Astrocyte gap junctions and Kir channels contribute to K+ buffering and regulate neuronal excitability. *Frontiers in Cellular Neuroscience*, 19, 1571218. <https://doi.org/10.3389/fncel.2025.1571218>.

17. Carter, B. C., & Bean, B. P. (2009). Sodium entry during action potentials of mammalian neurons: incomplete inactivation and reduced metabolic efficiency in fast-spiking neurons. *Neuron*, 64, 898–909. <https://doi.org/10.1016/j.neuron.2009.12.011>.

18. Jacquez, J. A., & Simon, C. P. (1993). Qualitative theory of compartmental systems. *SIAM Review*, 35, 43–79. <https://doi.org/10.1137/1035003>.

19. Cafiso, M., Casagrande, G., Angiolelli, M., Paradisi, P., Sorrentino, P., & Depannemaecker, D. (2026). A microcircuit model of astrocytic potassium buffering and neural synchronization. *Brain Research*, article 150630 (published online 19 November 2026). <https://doi.org/10.1016/j.brainres.2026.150630>.

20. Giaume, C., Koulakoff, A., Roux, L., Holcman, D., & Rouach, N. (2010). Astroglial networks: a step further in neuroglial and gliovascular interactions. *Nature Reviews Neuroscience*, 11, 87–99. <https://doi.org/10.1038/nrn2757>.

21. Djukic, B., Casper, K. B., Philpot, B. D., Chin, L.-S., & McCarthy, K. D. (2007). Conditional knock-out of Kir4.1 leads to glial membrane depolarization, inhibition of potassium and glutamate uptake, and enhanced short-term synaptic potentiation. *Journal of Neuroscience*, 27, 11354–11365. <https://doi.org/10.1523/JNEUROSCI.0723-07.2007>.

22. Cressman, J. R., Jr., Ullah, G., Ziburkus, J., Schiff, S. J., & Barreto, E. (2009). The influence of sodium and potassium dynamics on excitability, seizures, and the stability of persistent states: I. Single neuron dynamics. *Journal of Computational Neuroscience*, 26, 159–170. <https://doi.org/10.1007/s10827-008-0132-4>.

23. Ullah, G., Cressman, J. R., Jr., Barreto, E., & Schiff, S. J. (2009). The influence of sodium and potassium dynamics on excitability, seizures, and the stability of persistent states: II. Network and glial dynamics. *Journal of Computational Neuroscience*, 26, 171–183. <https://doi.org/10.1007/s10827-008-0130-6>.

24. Yang, L., Pathiranage, V., Zhou, S., et al. (2025). Sensitive red fluorescent indicators for real-time visualization of potassium ion dynamics in vivo. *PLOS Biology*, 23, e3002993. <https://doi.org/10.1371/journal.pbio.3002993>.

25. Groschup, B., Calandra, G. M., Raitmayr, C., et al. (2024). Probing intracellular potassium dynamics in neurons with the genetically encoded sensor lc-LysM GEPII 1.0 in vitro and in vivo. *Scientific Reports*, 14, 13753. <https://doi.org/10.1038/s41598-024-62993-1>.

26. Raue, A., Kreutz, C., Maiwald, T., Bachmann, J., Schilling, M., Klingmüller, U., & Timmer, J. (2009). Structural and practical identifiability analysis of partially observed dynamical models by exploiting the profile likelihood. *Bioinformatics*, 25, 1923–1929. <https://doi.org/10.1093/bioinformatics/btp358>.

27. Atkinson, A. C., & Fedorov, V. V. (1975). Optimal design: experiments for discriminating between several models. *Biometrika*, 62, 289–303. <https://doi.org/10.1093/biomet/62.2.289>.

28. Gevers, M., Bazanella, A. S., & Parraga, A. (2017). On the identifiability of dynamical networks. *IFAC-PapersOnLine*, 50, 10580–10585. <https://doi.org/10.1016/j.ifacol.2017.08.1310>.

29. Hendrickx, J. M., Gevers, M., & Bazanella, A. S. (2019). Identifiability of dynamical networks with partial node measurements. *IEEE Transactions on Automatic Control*, 64, 2240–2253. <https://doi.org/10.1109/TAC.2018.2867336>.

30. Bortner, C., & Meshkat, N. (2022). Identifiable paths and cycles in linear compartmental models. *Bulletin of Mathematical Biology*, 84, 53. <https://doi.org/10.1007/s11538-022-01007-5>.

31. Pant, S. (2018). Information sensitivity functions to assess parameter information gain and identifiability in dynamical systems. *Journal of the Royal Society Interface*, 15, 20170871. <https://doi.org/10.1098/rsif.2017.0871>.

32. Liu, Y., Maini, P. K., & Baker, R. E. (2026). Optimal experiment design for practical parameter identifiability and model discrimination. *Mathematical Biosciences*, 399, 109710. <https://doi.org/10.1016/j.mbs.2026.109710>.

33. Ballús Santacana, A. (2025). *The Syncytial Mesh Model: A Biophysical Framework for Scale-Dependent Coherence in the Brain*. arXiv:2412.12106v2, dated 5 June 2025. <https://arxiv.org/abs/2412.12106v2>.

34. Cooper, M. L., Selles, M. C., Cammer, M., et al. (2025). Astrocytes connect specific brain regions through plastic gap junctional networks. *bioRxiv*, July 2025. <https://doi.org/10.1101/2025.07.18.665573>.

35. Gardner-Medwin, A. R. (1983). Analysis of potassium dynamics in mammalian brain tissue. *Journal of Physiology*, 335, 393–426. <https://doi.org/10.1113/jphysiol.1983.sp014541>.

36. Chen, K. C., & Nicholson, C. (2000). Spatial buffering of potassium ions in brain extracellular space. *Biophysical Journal*, 78, 2776–2797. <https://doi.org/10.1016/S0006-3495(00)76822-6>.

37. Houades, V., Koulakoff, A., Ezan, P., Seif, I., & Giaume, C. (2008). Gap junction-mediated astrocytic networks in the mouse barrel cortex. *Journal of Neuroscience*, 28, 5207–5217. <https://doi.org/10.1523/JNEUROSCI.5100-07.2008>.

38. Lallouette, J., De Pittà, M., Ben-Jacob, E., & Berry, H. (2014). Sparse short-distance connections enhance calcium wave propagation in a 3D model of astrocyte networks. *Frontiers in Computational Neuroscience*, 8, 45. <https://doi.org/10.3389/fncom.2014.00045>.

39. Ma, B., Buckalew, R., Du, Y., et al. (2016). Gap junction coupling confers isopotentiality on astrocyte syncytium. *Glia*, 64, 214–226. <https://doi.org/10.1002/glia.22924>.

40. Cahill, M. K., Collard, M., Tse, V., et al. (2024). Network-level encoding of local neurotransmitters in cortical astrocytes. *Nature*, 629, 146–153. <https://doi.org/10.1038/s41586-024-07311-5>.

41. Visser, J., Milior, G., Breton, R., et al. (2024). Astroglial networks control visual responses of superior collicular neurons and sensory-motor behavior. *Cell Reports*, 43, 114504. <https://doi.org/10.1016/j.celrep.2024.114504>.

42. Oliveira, J. F., Agarwal, A., Beckervordersandforth, R., et al. (2026). The multiple scales of astrocytic functional units. *Nature Neuroscience*, 29, 1279–1292. <https://doi.org/10.1038/s41593-026-02308-x>.

43. Zhang, J.-Q., Sun, W., & Jin, W.-L. (2026). The astrocytic connectome: Gap-Junction topology beyond diffuse syncytium. *Trends in Cell Biology*, 36, 629–631. <https://doi.org/10.1016/j.tcb.2026.05.005>.

44. Nakatani, R. J., & De Schutter, E. (2026). Global Nernstian astrocytic depolarization breaks down during local synaptic input. *eLife*, reviewed preprint, version 1. <https://doi.org/10.7554/eLife.112921.1>.

45. Cafiso, M., Casagrande, G., Angiolelli, M., Paradisi, P., Sorrentino, P., & Depannemaecker, D. (2026). A microcircuit model of astrocytic potassium buffering and neural synchronization. *bioRxiv*, June 2026. <https://doi.org/10.64898/2026.06.15.732376>.

46. Kiyoshi, C. M., Du, Y., Zhong, S., Wang, W., Taylor, A. T., Xiong, B., Ma, B., Terman, D., & Zhou, M. (2018). Syncytial isopotentiality: A system-wide electrical feature of astrocytic networks in the brain. *Glia*, 66, 2756–2769. <https://doi.org/10.1002/glia.23525>.

47. Anders, S., Minge, D., Griemsmann, S., Herde, M. K., Steinhäuser, C., & Henneberger, C. (2014). Spatial properties of astrocyte gap junction coupling in the rat hippocampus. *Philosophical Transactions of the Royal Society B*, 369, 20130600. <https://doi.org/10.1098/rstb.2013.0600>.

48. Rouach, N., Segal, M., Koulakoff, A., Giaume, C., & Avignone, E. (2003). Carbenoxolone blockade of neuronal network activity in culture is not mediated by an action on gap junctions. *Journal of Physiology*, 553, 729–745. <https://doi.org/10.1113/jphysiol.2003.053439>.

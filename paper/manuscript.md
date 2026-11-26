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

The spatial organization of neuronal excitability may differ from both neuronal connectivity and extracellular diffusion if astroglial transport contributes a partially independent control geometry. The Syncytial Mesh Model (SMM) formalizes this hypothesis and develops a potassium-mediated realization with distinct extracellular and astroglial transport operators. In a calibrated passive two-compartment system, complete extracellular potassium transfer measurements determine the weighted glial operator exactly, without assuming commutativity of the spatial operators. We derive the inverse map's local conditioning, the source-route dependence of transport onset, and the disappearance of graph-shape contrast at both extremes of glial mobility. In a source-reconciled seven-site model, the predicted spacing maximizing relative graph contrast at nominal mobility is 14.8 µm for a short pulse and 37.0 µm for a longer step protocol. In an explicitly specified all-site observation design, increasing spacing from 5 to 37 µm raises nuisance-adjusted topology information from $2.40\times10^{-5}$ to 17.2, and the squared noise-normalized discrepancy of the best obtained flexible shared-support rival rises from $7.89\times10^{-6}$ to 90.7. Full electrodiffusive simulations establish conservative graph implementation and expose a quantitative gap with the passive approximation at small ionic loads. Current accounting and selective pump intervention explain the neighboring neuronal hyperpolarization; a separate reproduction of a published high-load benchmark confirms that strong glial coupling can prevent modeled depolarization block. The distinctive empirical prediction is that, in an independently characterized preparation, a glial operator with transport edges outside extracellular support is required to predict ionic and neuronal responses. These results identify conditions under which independent astroglial geometry becomes observable, rather than establishing its physiological magnitude or its contribution to human EEG.

**Keywords:** astroglial networks; Syncytial Mesh Model; potassium buffering; graph Laplacians; electrodiffusion; neuronal excitability; structural and practical identifiability; experimental design.

# Introduction

Neuronal connectivity specifies a principal organization of interactions between neurons, but it does not exhaust the spatial organization of the conditions under which those interactions take place. Extracellular ionic concentrations, membrane transport and astrocytic networks affect excitability, and the spatial routes through which these quantities change need not coincide with synaptic connections. The Syncytial Mesh Model (SMM) advances the hypothesis that a partially independent astroglial geometry contributes to the organization of neuronal responses. In this sense a glial network acts as a spatial controller of the conditions of neuronal activity, not necessarily as a separate generator of electromagnetic activity.

The present theory develops the 2024 Brain-Mesh proposal [1] and its subsequent published biological interpretation [2]. The earlier work conjectured a non-neuronal mesh contributing to slow coordination. Its architectural insight is retained here, while the putative literal metallic analogy, an insufficiently derived calcium-wave mechanism, and the proposed low-frequency resonome are separated from the experimentally testable potassium-mediated realization (SMM-K). The question is not whether astrocytes influence neurons in general; it is whether a distinct astroglial transport organization makes a measurable, specifically topological contribution to neuronal response geometry.

The biological premises are well established in restricted senses. Classical work linked glial membrane responses to extracellular potassium [3]; potassium uptake and spatial redistribution through astrocytic networks have been studied extensively [4–6,20,21]. Astroglial coupling can influence synaptic activity and plasticity [6]. Cooper et al. [13] demonstrated spatially selective, plastic, gap-junction-connected astrocytic networks spanning mouse brain regions, with patterns that sometimes differ substantially from neuronal projections. Their molecular tracing establishes anatomical connectivity and transport of tracer cargo; it does not measure a regional potassium conductance matrix. A cortical astrocytic territory may contain numerous synapses [14]; a micrometric ionic graph and a regional anatomical network are consequently different levels of description, even if they instantiate the same architectural distinction.

The functional effect of coupling is dependent on preparation, intensity and timescale. Breithausen et al. [15] found a limited acute gap-junction contribution to ordinary CA1 stratum radiatum potassium transients but a clearer contribution when highly localized extracellular loads produced elevations on the order of 10 mM or greater. Bojovic et al. [16] found that interference with Kir channels and astrocytic coupling alters neuronal excitability during sustained cortical activity; their results also indicate compensation by other mechanisms. Neither finding requires coupling to dominate all physiological potassium clearance. Together they motivate experiments in which the delivered ionic load, spatial profile, and duration are independently measured rather than treating the contribution of a glial network as a fixed fraction of any neuronal response.

The modeling ingredients are also established. Coupled neuronal–ionic models include potassium-mediated feedback and regime transitions [22,23]. Halnes and colleagues and Sætra and colleagues have developed increasingly complete electrochemical models of neuron–extracellular–glia interactions [7–10]. The recent microcircuit model of Cafiso et al. [19] demonstrates in simulation that local and syncytial astrocytic buffering can regulate synchronization and pathological excitability. These models are not rivals simply because they incorporate potassium buffering; they supply physiological mechanisms on which the present work builds. General methods for compartmental systems, dynamical-network identification, profile likelihood and optimal discrimination design are likewise established [18,26–32].

What distinguishes SMM-K is the problem addressed by combining these ingredients: given extracellular and neuronal alternatives, can the *geometry* of the glial transport operator be recovered, and can a preparation be shown to require connections outside the support of extracellular transport? This is stronger than detecting a difference after gap-junction blockade. A change in local uptake, global diffusivity, membrane excitability, or a hidden feedback process may account for such a difference without requiring a distinct spatial organization. A meaningful test must compare independently constrained model families under matched inputs, observations and uncertainty.

This article therefore makes three connected contributions. It first derives exact implications of a distinct glial operator in a calibrated passive model, including conditions for recovery and sensitivity. It then calculates both the conditioning of that inverse problem and finite separation from shared-support alternatives under an explicit experimental design. Finally, it tests the graph representation and its small-load limitations against an independent conservative electrodiffusive model, including the electrical mechanism of a seemingly reversed neuronal response and a published nonlinear positive control. The original neuronal-population, metabolic, stability and EEG audits are retained in the supplementary material because they constrain the physiological interpretation, but the theory is organized here around its distinctive prospective test.

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

The onset hierarchy and uniform control constrain an experiment, but do not distinguish SMM from another two-compartment diffusion-with-exchange model with the same causal path length. They should be used as tests of the model scaffold, rather than the main biological falsifier. In particular, the onset result presumes a purely chemical forcing in the passive model; a real intervention that separates charge across a membrane can add a much faster electrical pathway, as shown in the full model below.

## Intermediate topology-sensitive mobility

Consider two distinct connected symmetric glial Laplacians on the same sites, identical extracellular and local coefficients, zero initial glial deviation, and a bounded extracellular source. Multiply each glial transport operator by $\mu>0$.

**Proposition 4 (loss of graph-shape contrast at both mobility extremes).** For a fixed bounded observation interval, the difference $\Delta k_{12}(t;\mu)$ between the two extracellular responses tends uniformly to zero as $\mu\downarrow0$ and as $\mu\to\infty$. If some intermediate mobility gives a nonzero finite-time contrast, its continuous norm achieves a positive maximum at an intermediate mobility.

*Proof.* Continuity of the linear generator and its exponential gives the zero-mobility limit. For the infinite-mobility limit, decompose $a=\Pi a+a_\perp$, where $\Pi=\mathbf1\mathbf1^\top/N$. Since each connected symmetric graph has a positive spectral gap on $\mathbf1^\perp$, the nonuniform component obeys

$$\phi_a\dot a_\perp=\kappa_e(I-\Pi)k-(\kappa_a I+\mu G)a_\perp.$$

On any fixed time interval the forcing is bounded uniformly in $\mu$ by the passive dissipative estimate with the bounded input. With zero initial nonuniform load, variation of constants therefore gives $\|a_\perp\|=O(\mu^{-1})$ uniformly on that interval. The uniform glial state solves the same averaged exchange equation in both graphs. The remaining extracellular and averaged-glial equations form a fixed linear system perturbed by the vanishing $a_\perp$ term, hence converge to a common limit. Continuity in $\mu$ and the two vanishing endpoints imply the maximum statement. $\square$

The positive maximum is conditional on a discriminating input, graph pair and observation; it need not be unique. Disconnected graphs can have different nullspaces and different limiting operators. Neither this theorem nor the maximum of a relative norm implies that glia become the dominant potassium transport route. It is a statement about the information carried by internal graph shape. In the source-reconciled parameterization below $D_g/D_e\simeq0.218$ at nominal mobility, independently of spacing.

# Numerical methods: a matched experiment

## Graphs, kinetics and input convention

The seven-site experiments use the extracellular path $0-1-\cdots-6$, with unit nearest-neighbor edge weights. Write its dimensionless Laplacian as $\mathcal L$, and define the additional-edge Laplacian

$$\mathcal C=0.8b_{04}b_{04}^\top+0.5b_{15}b_{15}^\top+0.6b_{26}b_{26}^\top,$$

where $b_{ij}=e_i-e_j$. The passive family is

$$E=D_e\mathcal L/d^2,\qquad G=\mu D_g(\mathcal L+g\mathcal C)/d^2.$$

The target $g$ controls the three chords relative to the common path; $g=0$ is the shared path and $g=1$ the nominal alternative. The separate mobility multiplier $\mu$ allows overall strength to change. Chord weights are effective weights at the chosen nominal discretization length, not conductances derived from the physical lengths of the long links. The full EDNM graph adapter uses the same weighted adjacency convention. No measured astrocyte geometry or topology is assumed.

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

The passive extracellular input is a rectangular pulse at one site with amplitude $q=0.1714683$ mM tissue/s, on from 0.08 to 0.40 s. This is obtained from the original EDNM membrane flux magnitude $10^{-4}$ mM cm/s multiplied by the source membrane-area density $\gamma_m=1714.683$ cm$^{-1}$. This is a conversion of the potassium mass delivered per tissue volume, not equality of the electrical action of the inputs: the full model transfers potassium across a neuronal membrane and has a direct capacitive and electrical footprint. The pulse has tissue-volume-integrated dose $q(0.32)\simeq0.05487$ mM. The full graph experiment uses 5 µm spacing and sealed boundaries with a 0.8 s horizon.

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

At the source-reconciled nominal coefficients, $\sqrt{D_g/\kappa_a}=20.84$ µm, a characteristic length associated with glial exchange and diffusion in the passive equations. The spacing maximizing a graph contrast need not be that length, because the extracellular operator, input duration and observation schedule also set scales. The matched short-pulse schedule yields a numerical maximum at approximately 14.76 µm, where $\mathcal R=0.04645$; the longer step schedule gives approximately 36.98 µm, where $\mathcal R=0.19233$. At 5 µm and nominal mobility, the corresponding contrasts are 0.00392 and 0.00153. The latter two results show that a longer observation does not monotonically increase a *relative graph-shape* contrast at fixed spacing.

![Mobility–spacing windows for the two passive protocols. The circle marks the source-reconciled reduced point at 5 µm and unit mobility; the triangle marks the numerical spacing maximum along unit mobility. The circle is not a fitted effective location of the full multi-ion EDNM. Color scales differ by panel. The spacing and mobility axes span synthetic values, not measured tissue uncertainty.](matched_windows.png){width=100%}

These results make a risky but precisely delimited prediction: in this parameterized graph family, tens of micrometres rather than 5 µm are the more informative nominal site spacings for these protocols. Changing the time protocol shifts the predicted optimum. The model does not predict a universal tissue length scale or a universal optimal astrocyte spacing; what is testable is the change in discriminability when the independently estimated coefficients and physical source–sensor schedules are specified. Varying $\mu$ can move the window substantially. Since the sweep changes the strength as well as the weights of the alternative relative to the path, its norm alone does not isolate shape from rescaling. The shared-support comparison below performs that isolation more directly.

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

# Full electrodiffusive implementation and physiological constraints

## Implementation and conservation

The passive analysis deliberately omits physiological pathways that can change both amplitude and sign of observed responses. To evaluate those omissions we use the independently published multicompartmental electrodiffusive network model (EDNM) of Sætra and Mori [10], pinned to commit `826c211c70d9027a4acb02202ca91a60de79a15d`. Each unit contains soma and dendrite layers with neuronal, glial and extracellular domains; it evolves Na, K, Cl and Ca concentrations, membrane and extracellular potentials, gating variables and water-dependent volume fractions. It enforces a capacitive charge relationship rather than computing extracellular potential from potassium concentration alone.

For an ionic species $i$ in domain $r$, the interunit flux on an oriented edge $u\to v$ uses the centered Nernst–Planck discretization

$$j_{r,uv}^i=-\frac{D_i\upsilon_{ir}w_{uv}}{\lambda_r^2d}\left[c_v^i-c_u^i+\frac{z_i(c_u^i+c_v^i)}{2\psi}(\varphi_v-\varphi_u)\right],$$

where $\lambda_r$ is tortuosity, $\upsilon_{ir}$ the mobile fraction, $\psi=RT/F_{\!A}$ and $F_{\!A}$ Faraday's constant. The source finite-volume assembly includes the previous-step domain volume fraction and unit spacing. Each edge correction is inserted with opposite signs at its two endpoints. Glial and extracellular weights can be changed separately without changing local membrane fluxes, pumps or gating equations. The adapter replaces the original closed-path contribution by adding the difference of the new and old edge residuals and their analytic sparse Jacobian. The compiled parent local equations are retained. Earlier finite-difference Jacobian checks and the fine-step validation are archived with the source.

This structure preserves total inventory of each ionic species under internal transfers and sealed boundaries. A separate capacitive balance must hold for charge; mass conservation alone cannot establish electrical consistency. For the neuronal soma,

$$\gamma_m C_m V_m=F_{\!A}\left(\alpha_n\sum_i z_i c_{n,i}+z_0a_n^{\mathrm{imm}}\right),$$

with constant immobile-ion inventory $a_n^{\mathrm{imm}}$ and $V_m=\varphi_n-\varphi_e$. We use the discrete current balance implied by this relation and the actual semi-implicit time update to diagnose the neighboring response.

## Small-load seven-site comparison under matched conditions

At 5 µm spacing, the seven-site full source simulation transfers potassium from the neuronal soma of site 0 into extracellular space from 0.08 to 0.40 s using $j_{\mathrm{stim}}=-10^{-4}$ mM cm/s in the original source convention; synaptic connections are disabled. The 1 ms path run yields a stimulated-site peak extracellular elevation of 64.49 µM and a nearest-neighbor peak of 45.68 µM. Adding the three specified glial chords changes the path-distal site 6 extracellular signal by at most 0.02053 µM; the corresponding neuronal membrane contrast is about 0.021 µV. This is a very small graph-specific effect. Inventory drifts are below $3\times10^{-13}$ relative, and the graph contrast was previously checked at 2 and 1 ms, with relative RMS discrepancies of approximately 0.51–0.52% for K and 0.27% for voltage. The focal membrane voltage excursion is about 32 mV despite small extracellular loads, so the full protocol is not a globally small-signal neuronal experiment.

The original passive and full model results appeared to differ by roughly two orders of magnitude when contrasts with different observation windows and normalizations were compared. Using the same 1 ms time labels, weighted graph, pulse duration, mass source and whole-field statistic removes much of that disparity. The contrast ratio between the source-reconciled passive and the complete model is now approximately 8.96. The difference remains scientifically significant. A passive model using the EDNM concentration-diffusion coefficients alone, $D_e=\phi_e D_K/\lambda_e^2=1.53125\times10^{-10}$ m$^2$/s and $D_g=\phi_a D_K/\lambda_g^2=7.65625\times10^{-11}$ m$^2$/s (after unit conversion), predicts an even larger contrast of 0.05219. Those coefficients omit electrical migration and are not effective fitted EDNM coefficients.

Table 4. Matched relative extracellular graph contrast at 5 µm. All models use the same source-site pulse convention and full 0.8 s sampling labels; the passive model matches delivered potassium mass, not the electrical action of transmembrane injection.

| Model | Relative contrast $\mathcal R$ | Peak path K (µM) | Peak graph difference, site 6 (µM) |
|:--|--:|--:|--:|
| Full EDNM | 0.0004380 | 64.49 | 0.02053 |
| Source-reconciled passive | 0.0039262 | 23.17 | 0.08865 |
| Passive, EDNM diffusion only | 0.0521922 | 45.80 | 1.31430 |

![Comparison of source-reconciled passive and full EDNM trajectories under matched small-load conditions. Different total response magnitudes and distal graph contrasts demonstrate that matching mass, geometry and protocol is insufficient to make the passive model a quantitatively equivalent physiological surrogate.](matched_waveforms.png){width=100%}

Even after fitting a single amplitude multiplier (2.057) to the passive path response, the relative whole-field RMS mismatch is 0.420. Because the graph statistic is invariant to common amplitude scaling, this mismatch is not simply a different source amplitude. The underlying glial flux also cannot be identified with a single chemical diffusion coefficient: on the three added somatic glial edges, the L2 norms of stimulus-induced electrical migration terms in the actual trajectories are about 1.97, 2.13 and 2.19 times those of their chemical-diffusion terms. The combined changes are 2.77–3.00 times the diffusion-only norms. Their correlations are positive but far from perfect. This is descriptive flux accounting, not a causal decomposition of the full passive/full-model discrepancy. In particular, migration is not a uniformly suppressive mechanism; membrane transport, electric fields, ionic feedback, and local dynamics all matter. An effective scalar mobility that would locate full EDNM on the passive sensitivity surface has not been established. The circle in the scale figure therefore marks the specified passive reference, not a pretend full-model fit.

This benchmark demonstrates graph-specific numerical feasibility and provides a negative quantitative constraint on strong-effect interpretations at the tested small-load protocol. It is not the evidential basis for the larger-spacing prediction, which is conditional on the passive parameterization. It also cannot establish the contribution of glial coupling at the large localized loads observed experimentally in CA1 [15].

## Mechanism of the neighboring voltage inversion

The independent two-unit EDNM challenge uses 5 µm spacing, sealed boundaries, no synapses, a 0.5 s neuronal-to-extracellular K transfer of magnitude $10^{-4}$ mM cm/s (0.05–0.55 s), and a 2 s horizon. The original 2 ms run gives a neighboring extracellular K peak of 0.13611 mM and a peak absolute neuronal voltage response of 0.14986 mV. Turning off glial interunit transport changes those neighboring signals by at most 0.04113 µM and 0.06834 µV, respectively; relative RMS convergence of the intervention contrast from 2 to 1 ms is approximately 0.93% and 0.69%. Thus the overall ionic response should not be conflated with the specific contribution of interunit glial transport.

At $t=0.4$ s in the native 2 ms run, neighboring extracellular K increases by 0.101726 mM and the K Nernst potential by 0.753641 mV, but the neighboring membrane potential decreases by 0.111094 mV. The changes in neuronal intracellular and extracellular potential are respectively $-0.08030$ and $+0.03080$ mV. This decomposition is a kinematic identity, not an explanation of the sign. There are two mechanistically distinguishable contributions in the original stimulus and response.

First, the imposed flux transfers positively charged ions between compartments: it is globally potassium- and charge-conserving, not locally electrically neutral. Its outward transmembrane current has magnitude $F_{\!A}|j_{\mathrm{stim}}|=9.648$ µA/cm$^2$ in the source units. In the 1 ms trajectories, the first stimulated timestep at the neighboring soma produces an extracellular potential change of $+34.86$ µV while extracellular K changes by only $+0.0220$ µM. At the onset of stimulus removal the extracellular potential falls by approximately 34.86 µV within one step, whereas K remains continuous to that temporal resolution. The rapid response is consistent with the quasistatic electrical constraints responding to imposed membrane current. It should not be interpreted as exceptionally fast chemical transport or as a topology-specific glial signal.

Second, the longer voltage response reflects a balance of electrogenic membrane processes. With outward current positive, multiplying the ionic balance by $F_{\!A}z_i$ and summing gives at the unstimulated neighboring soma the discrete identity

$$C_m\frac{V_m^j-V_m^{j-1}}{\Delta t}=-I_{\mathrm{mem}}^{j}-I_{\mathrm{ax}}^{j}.$$

The axial term includes soma–dendrite ionic flux at the source volume and area conventions. The neuronal Na/K pump exchanges three Na for two K and generates net outward current $F_{\!A}j_{\mathrm{pump}}$. KCC2, NKCC1 and the source two-Na/Ca exchanger are directly electroneutral by stoichiometry, although they affect the concentrations that determine other currents. Currents are reconstructed with the exact source semi-implicit convention (previous-step pump and gating factors, current-step voltage and reversal potentials). Integrating paired stimulated-minus-baseline currents over the time steps reconstructs the voltage change without fitting.

Table 5. Current decomposition of the neighboring soma at 0.4 s (native 2 ms trajectory). Outward-positive current differences are in µA/cm$^2$; integrated voltage contributions use the actual discrete capacitive balance. Totals may differ in the last displayed digit owing to rounding.

| Current component | Instantaneous difference | Integrated contribution (mV) |
|:--|--:|--:|
| Sodium leak | −0.002640 | +0.169354 |
| Gated sodium | +0.001805 | −0.116674 |
| Potassium leak | −0.021186 | +1.301747 |
| Delayed-rectifier potassium | −0.006943 | +0.436137 |
| Chloride leak | −0.009736 | +0.633226 |
| Na/K ATPase | +0.049874 | −3.019603 |
| Axial ionic current, summed | −0.010354 | +0.484720 |
| **Net** | **+0.000819** | **−0.111094** |

The maximum difference between reconstructed and actual paired voltage across the complete trajectory is $5.25\times10^{-9}$ mV at 2 ms and $8.75\times10^{-9}$ mV at 1 ms. At 1 ms the integrated pump contribution is −3.01203 mV, with the potassium channels contributing +1.72835 mV and the net voltage −0.111302 mV. The pump provides the dominant negative contribution; the residual is the difference of larger opposing processes. This is exact current accounting along the trajectory, not a counterfactual on its own.

To test causality within the model, we independently modify only the extracellular K argument of the neuronal Na/K pump at the unstimulated soma, clamping it at the initial reference value 3.543992 mM while retaining sodium dependence and the pump's 3:2 stoichiometry. All other equations are unchanged; each intervention uses its own matched unstimulated baseline. The clamped neighboring response at 0.4 s becomes +0.045166 mV at 1 ms and +0.045317 mV at 2 ms, while extracellular K elevations are 0.102358 and 0.102571 mM, close to the native values of 0.101515 and 0.101726 mM. The sign reversal persists under timestep refinement. Both intervention runs finish with no flagged unconverged steps and relative ionic inventory drifts of order $10^{-12}$ or below.

![Neighboring membrane-voltage response with native versus clamped pump K sensitivity (A), and integrated native current components at 0.4 s (B). Shading marks the stimulus interval. The clamped-pump intervention tests the causal role of K-sensitive pump activation in this model while preserving ionic stoichiometry.](pump.png){width=100%}

The inversion is thus explained within the parent physiology by the competition between K-dependent reversal-potential shifts and electrogenic pump activation, together with other membrane and axial currents. It persists in the native glial-off control, so it is not itself evidence of distinct astroglial topology. Nor does the pump intervention prove that the identical current competition governs every biological preparation. It demonstrates why a K-only neuronal response map cannot be assumed to have the sign of the Nernst shift under all conditions.

## Independent nonlinear positive control

The small-load graph and clamp experiments do not test whether the solver is capable of reproducing a large coupling-dependent neuronal transition. We therefore reproduced the published EDNM two-unit high-load protocol [10, Table 4 and Figs. 9–11]: 1 mm interunit spacing, sealed boundaries, synapses off, a depolarizing constant stimulus $j_{\mathrm{stim}}=+2\times10^{-4}$ mM cm/s from 0.1 to 10 s, and the original 50 µs timestep. The source's strong-glia variant multiplies interunit glial Nernst–Planck fluxes by 2000, leaving the other domains and the local membrane equations unchanged. The multiplier is an artificial computational positive control, not a measurement of gap-junction conductivity. Unlike the earlier small-load negative K-transfer pulse, the positive stimulus drives neuronal depolarization and activity-associated potassium accumulation.

At 10 s, our native (weak-glia) reproduction yields an extracellular K increase of 13.644 mM at the stimulated soma, and the strong-glia reproduction yields 2.617 mM. These match the independently reported values of about 13.6 and 2.6 mM. In the native case, the voltage remains near −29.7 mV without spikes over the final second; in the strong case, voltage repeatedly traverses a spiking range (−65.9 to +9.1 mV over the final second). The strong-coupling trajectory peaks at a K elevation of 7.82 mM at 6.63 s, close to the reported peak of approximately 7.7 mM at 6.7 s. The weak-glia 100 µs run also terminates in depolarization block, with K elevated 13.677 mM; precise spike counts and the time of transition differ under timestep refinement. The original-resolution runs are used for the positive-control comparison, not for a claim of pointwise convergence of spike timing.

Table 6. Original-resolution high-load reproduction against the independently published computational benchmark. Both variants use the source-published stimulus and original 50 µs timestep.

| Outcome | Weak glia | Strong glia |
|:--|--:|--:|
| Published final K elevation (mM) [10] | 13.6 | 2.6 |
| Reproduced final K elevation (mM) | 13.644 | 2.617 |
| Reproduced peak K elevation (mM) | 13.644 | 7.822 |
| Voltage range in final 1 s (mV) | −29.68 to −29.62 | −65.88 to +9.13 |
| Threshold-detected spikes in final 1 s | 0 | 3 |
| Max relative ionic inventory drift | $6.17\times10^{-9}$ | $1.20\times10^{-8}$ |

![Reproduction of the independent high-load EDNM weak/strong coupling experiment at 50 µs. Large K accumulation accompanies the native-coupling depolarization block, whereas the artificially enhanced glial mobility permits continuing spiking. Voltage traces are displayed with light decimation for legibility.](high_load.png){width=86%}

The full trajectories are finite and satisfy positive volume fractions and nonnegative concentrations (glial calcium is zero in the source). Inventory checks were repeated directly on the archived arrays. The automated post-processing encountered a NumPy integer JSON serialization error *after* saving the full state and observables; the figures and values above are recovered from those raw numerical arrays, not from a successful workflow exit status. These are computational source-model reproduction results. They show that the source physiology can yield strong dependence of neuronal state on glial transport under a deliberately different regime. They do not demonstrate that the synthetic shortcut edges determine that transition, that 2000-fold mobility is physiological, or that an externally applied local KCl load and connexin manipulation reproduce it. The CA1 large-load experiment [15] requires its own calibrated source, counterion/charge-balancing representation, geometry and off-target controls; it remains an experimental validation target.

# Discussion: what would identify an independent control geometry?

## From possibility to a model-specific biological test

The mathematics explains a potentially important dissociation: a glial graph can be structurally different and dynamically coupled yet leave almost no distinctive trace in a particular observation. At fixed homogeneous kinetics and low mobility, the graph does little redistribution. At very high mobility, connected graphs collapse toward the same uniform glial state. In between, geometry matters. Under the specified source-reconciled model, changing spatial scale can make both the local score and finite shared-support discrepancy orders of magnitude larger. This is a substantive prediction about the relationship between geometry, timescale and observability, not merely a statement that adding an edge changes an equation.

It is equally important to keep the levels separate. The passive model identifies a weighted glial operator conditional on its assumptions. The full EDNM establishes a conservative implementation and produces a much smaller small-load contrast than the nominal passive model. The high-load reproduction establishes a nonlinear computational positive control for glial *strength*, not a topology-specific physiological amplification. The broader architectural claim is supported by independent anatomical and functional observations [5,6,13,15,16]; the mechanistic conjunction—whether the distinct organization is required to predict neuronal responses—needs a discriminating experiment. No new animal or human recordings are presented here.

A core experimental prediction should accordingly concern support. In a preparation with an independently characterized extracellular graph, the fitted glial operator should require reproducible, functionally relevant off-support coupling. The primary analysis should compare a shared-support model with free nonnegative edge weights against a prespecified independent-support model, while applying identical local kinetic, source, sensor and covariance constraints. Parameters should be trained on one subset of stimulus sites and conditions, then frozen for prediction of withheld spatial and temporal responses. A measured residual beyond uncertainty from the restricted family and a cell-specific intervention affecting the inferred route with downstream neuronal consequences would support the independent-geometry realization. A powered equivalence test placing off-support coupling and its downstream effect below prespecified biologically meaningful bounds would reject that realization in the preparation. A null result in a demonstrably insensitive experiment is not the same as such a test.

This approach is stronger than simply testing whether recovered weights are statistically nonzero: one must establish source specificity, rule out changes in effective extracellular support, and avoid confounding local membrane currents or generic latent feedback with spatial glial transport. Similarity-equivalent hidden realizations can share all recorded neuronal and extracellular transfer functions. Identifiability of the mathematical operator within a restricted model is not unrestricted identification of a cell type. Direct astroglial observations or validated interventions are essential for that final attribution.

## Experimental design and biological feasibility

The proposed sequence is first to characterize the preparation: local extracellular spread and volume fraction; astroglial coupling and relevant local uptake/return kinetics; source spatial profile and ionic/electrical dose; and neuronal synaptic/excitability controls. Potassium reporter kinetics, pH sensitivity, range and correlated noise should be measured in the exact regime, rather than assumed from published large-event detection [24,25]. An independently measured coupling marker can constrain the candidate graph family, but a tracer network cannot be converted directly into an absolute ionic conductivity matrix. Particularly valuable preparations would offer an astroglial route that differs from a measured extracellular transport route at the recording scale.

The passive design calculations suggest candidate spacings on the order of tens of micrometres at nominal rates. The next experiment should not assume that the modeled maxima apply to tissue. It should estimate effective parameters independently, propagate their uncertainty, and test a spacing and duration series spanning the predicted band. Focal source locations and distributed inputs provide multiple spatial modes. Uniform stimulation in a sufficiently homogeneous preparation is a useful negative control; direct glial input, when selective and physically quantified, changes the pathway order and may provide a more informative design. Electrical artifacts must be monitored at onset and offset, particularly for membrane-current-driven sources. Directly applied extracellular K loads may require explicit counterion transfer and volume accounting that are different from a transmembrane K exchange.

A prospective robust design objective is

$$\mathcal D^*=\arg\max_{D\in\mathcal A}\inf_{\theta\in\Theta}I_{g\mid\eta}(D;\theta),$$

where $\mathcal A$ is a set of experimentally feasible calibrated protocols and $\Theta$ an independently justified parameter uncertainty set. A finite predictive-distance criterion against the flexible shared-support family is a complementary objective [27,32]. The present computations do not solve this optimization over measured biological uncertainty; they show why spatial and temporal design can dominate nominal repetition counts.

For a nonlinear validation, a first step is to reproduce the contrasting low- and large-load responses of the CA1 experiments [15] with their measured focal geometry, source dose, recording positions and glial-coupling manipulation. The experimental threshold is protocol-specific; the artificial strong-glia EDNM reproduction cannot supply it. Then a full-model graph sweep should match local calibration and compare freely weighted shared support with additional edges. Strong-coupling or high-K regimes may increase absolute effects but may also introduce nonlinear saturation, volume changes and excitability transitions, and thus cannot simply inherit the passive information numbers.

Cell-type-specific and reversible coupling interventions with independent coupling assays and rescue are particularly useful. Connexin interventions can modify metabolite exchange, glutamate handling, volume and neuronal excitability; Kir4.1 manipulations alter more than spatial transport [20,21]. Measuring those routes prevents a glial-coupling manipulation from being interpreted as a pure change in $D_g$ when it is not. The prediction requires coupling-dependent *spatial organization* beyond a generic change in total clearance.

## Implications for neuronal coordination and electrophysiology

The role of glial networks in determining neuronal response geometry is compatible with the fact that neuronal membrane currents principally generate EEG and related extracellular fields. A physiologically identified glial transport graph could change the pattern of ionic states, neuronal membrane currents and thus the fields those currents generate. The required causal chain is glial transport $\to$ extracellular and local ionic state $\to$ neuronal current generation $\to$ a calibrated volume-conductor or lead-field map. An EEG spectrum alone does not identify the hidden spatial controller. The original local spectral comparison omitted the glial transport operator, so it neither tests this defining prediction nor supplies evidence for it (Supplement D). The low-frequency resonome remains a motivating hypothesis for a separately validated extension [2].

## Limitations

The passive parameter set is reconstructed from an earlier source calibration and uses an inherited local clearance coefficient; it is not a fit to a preparation or an exact reduction of EDNM. The comparison to full EDNM shows that its quantitative extrapolation is not yet warranted. The graph edges are synthetic and the source-reconciled coefficients, volumes, spatial grid, nuisance bounds and noise assumptions define the reported numerical results. The conditional spacing maxima and information values are not biological optima or actual experimental power estimates. The finite forward fit is numerically reproducible but not a certified global optimization over all nonnegative shared-support weights. The original independent graph experiment does not normalize graph strength before adding chords; the flexible shared-support fit is the more stringent strength-adjusted comparison. Full electrodiffusive discrimination across calibrated spacing and large-load topology, independently measured ionic conductances, and a prospective biological support test are still required. The high-load reproduction demonstrates the source model's nonlinear behavior at an intentionally artificial mobility multiplier. Older source-reconstruction quantities whose exact historical forcing or fixed-inventory procedure cannot be verified are reported as historical diagnostics in the supplement, not prospective validation. No new biological, human or animal data were acquired.

# Conclusion

SMM proposes that the spatial conditions shaping neuronal currents can have a partially independent astroglial organization. The present potassium-mediated theory specifies that idea as an operator-level and experimental hypothesis. Distinct glial and extracellular operators admit exact conditional graph recovery, yet the inverse can be severely ill-conditioned. Their topological contrast vanishes at both glial mobility extremes, and the nominal passive example predicts protocol-dependent windows where separation from shared-support alternatives increases dramatically. The source-resolved electrodiffusive work confirms the feasibility of graph-specific conservative transport and constrains its quantitative interpretation. A counterfactual pump clamp explains a neighboring voltage inversion, and a separate published high-load reproduction establishes a nonlinear computational positive control. The distinctive prediction is not that all astroglial effects are large or that their existence is new; it is that a correctly designed and calibrated preparation requires a transport geometry different from extracellular support to explain spatial ionic and neuronal responses. That is a precise, falsifiable realization of the original architectural hypothesis.

# Code and data availability

The independent EDNM implementation is available from <https://github.com/martejulie/electrodiffusive-network-model>, commit `826c211c70d9027a4acb02202ca91a60de79a15d`. The two-site validation and graph correction are in the SMM research repository (<https://github.com/ipsissima/SMM>), branches [full-solver validation](https://github.com/ipsissima/SMM/tree/research/ednm-full-solver-validation-20261126) and [graph topologies](https://github.com/ipsissima/SMM/tree/research/graph-ednm-topologies-20261126). Original trajectory artifacts are associated with runs [45839123584](https://github.com/ipsissima/SMM/actions/runs/45839123584), [45840811721](https://github.com/ipsissima/SMM/actions/runs/45840811721), and [45847600616](https://github.com/ipsissima/SMM/actions/runs/45847600616). The pump clamp and high-load reproduction are in [mechanism/high-load branch](https://github.com/ipsissima/SMM/tree/research/ednm-mechanism-high-load-20261126), commit `3798c749cdbed18e4222dc805e94f28a71cde5cb`, workflow run [45859252351](https://github.com/ipsissima/SMM/actions/runs/45859252351). Relevant artifact IDs are 15137282506 and 15137766579 (pump clamp, 1 and 2 ms), 15137517741 and 15137242820 (weak and strong coupling, original 50 µs), and 15138690510 (weak coupling, 100 µs). Workflow artifacts have finite retention; the accompanying diagnostic code and machine-readable summaries identify the exact inputs and checks. The original full-state arrays must be retained in a durable research archive before publication.

The accompanying reproducibility package contains the matched passive/full-model diagnostic, inverse conditioning, Fisher calculations, shared-support fit, current audit, figures and result files. The full-state high-load trajectories are referenced by artifact ID because of their size. Historical neuronal-population and EEG analyses are separately archived and are not used as empirical support for spatial transport.

# Supplementary material: historical model constraints and extended derivations

## Supplement A. Neuronal realization and exact hidden-state reduction

The original seven-site SMM-K population model contained eight local coordinates per site,

$$X_j=(R_{E,j},V_{E,j},R_{I,j},V_{I,j},S_{EI,j},S_{IE,j},k_j,a_j)^\top,$$

for 56 states. The rate–voltage pairs are based on the Montbrió–Pazó–Roxin mean-field construction [11]. In a convenient notation the local excitatory and inhibitory populations obey

$$\begin{aligned}
\tau_b\dot R_b&=\frac{\Delta_b}{\pi\tau_b}+2R_bV_b+F_{R,b}(R,V,S),\\
\tau_b\dot V_b&=V_b^2+\eta_b(k)+F_{V,b}(R,V,S)-(\pi\tau_bR_b)^2,
\end{aligned}$$

where $b\in\{E,I\}$, the additional functions summarize the documented synaptic and population interactions, and the source-reconciled potassium response is $\eta_b(k)=\eta_{b0}+\chi_b\log(1+k/K_{e0})$. This response map is not independently calibrated by the mean-field equations alone. It must be related to the conductance-based parent neuron and to the operative source convention. The directed neuronal graph enters via the population interactions; the two ionic graphs enter via the balance equations.

Partition a fixed equilibrium linearization into 21 retained states $z$ and the omitted states $w$:

$$\begin{pmatrix}\dot z\\\dot w\end{pmatrix}=\begin{pmatrix}A_{zz}&A_{zw}\\A_{wz}&A_{ww}\end{pmatrix}\begin{pmatrix}z\\w\end{pmatrix}+\begin{pmatrix}B_z\\B_w\end{pmatrix}u.$$

Exact elimination of $w$ gives

$$\begin{aligned}
\dot z(t)={}&A_{zz}z(t)+B_zu(t)+A_{zw}e^{A_{ww}t}w(0)\\
&+\int_0^t A_{zw}e^{A_{ww}(t-r)}[A_{wz}z(r)+B_wu(r)]\,dr .
\end{aligned}$$

Under the quasistatic approximation, if $A_{ww}$ is invertible,

$$A_{\mathrm{DC}}=A_{zz}-A_{zw}A_{ww}^{-1}A_{wz},\qquad B_{\mathrm{DC}}=B_z-A_{zw}A_{ww}^{-1}B_w.$$

The two Schur expressions preserve the zero-frequency retained-state relation under the stated assumptions. Keeping $B_z$ alone requires $B_w=0$ or vanishing correction for the selected input. In general the dynamic memory kernel prevents the reduced system from preserving the entire transfer map. The earlier source reconstruction reported slow-response relative differences of approximately 0.0472% at 1 s, 0.0122% at 3 s and 0.000451% at 20 s, with much larger 40 Hz errors (about 33% for glial input and 40% for excitatory-voltage input). Three archived values helped reconstruct the model and a fourth served as an internal holdout, not an independent experiment. The exact historical implementation's forcing matrix cannot be established from the available archive; the above formula is the corrected theoretical statement and the historical numerical results are not used to establish a frequency-valid reduction.

## Supplement B. Physiological and energetic audits

The initial source-reconciliation exercise revealed a substantial discrepancy between the archived excitatory QIF mapping coefficient (about 0.05211 in its normalization) and the recalibrated parent-neuron estimate (about 5.04214). The corrected excitatory spike-associated excess potassium transfer was approximately $7.0\times10^{-8}$ mol m$^{-2}$ spike$^{-1}$ rather than $4.736\times10^{-8}$. These are source-model calculations, not independently measured population parameters. Inhibitory response and release remained provisional. The exchange and effective diffusivities used in the present passive calculations derive from that exercise; the original $\kappa_N=0.232$ s$^{-1}$ remained a separately inherited clearance slope.

The linearized neuronal feedback need not preserve the stability of the passive sector. Let $g_E,g_I$ denote DC rate derivatives with respect to extracellular K, $\rho_A$ the effective active membrane-area density, $f_E$ the excitatory population fraction and $q_E,q_I$ spike-associated release. A conditional zero-eigenvalue crossing in the homogeneous spatial mode obeys

$$\kappa_N=\rho_A[f_Eq_Eg_E+(1-f_E)q_Ig_I],$$

if the remaining block is nonsingular. This follows by setting the spatially uniform DC response to equilibrium, eliminating the local glial exchange ($\kappa_e k=\kappa_a a$), and equating neuronal release and clearance slopes. It is not a general Hurwitz criterion. At the historical central coefficients, the reported critical density was $4.62424\times10^6$ m$^2$/m$^3$; at $4.929\times10^6$ the reported dominant growth rate was +0.01305 s$^{-1}$ and at $3.5\times10^6$ it was −0.04634 s$^{-1}$. In later conservative extensions the reported small positive rates became +0.003232, +0.000677 and +0.000447 s$^{-1}$ as more source pump/cotransporter terms were included, under their respective provisional closures. Introducing an uncalibrated Na-entry feedback yielded a stable illustrative point. These are historical conditional spectra, not validated tissue instability measurements. For a closed inventory-conserving model the spectrum must be checked on the fixed-inventory tangent space: if $M x$ denotes the conserved linear inventories, restrict the linearization to $\ker M$. The exact historical procedure is unverified; the very small reported growth rates must not be construed as verified restricted eigenvalues. This is one reason those values are supplementary and no new stability claim for the full feedback realization rests on them.

The tonic potassium release implied by the historical active membrane area and firing baseline was 1.594 mmol per L tissue/s. If all potassium is restored by neuronal Na/K ATPase with two K per ATP, the associated cost is 0.797 mmol ATP per L tissue/s. With a Na-based accounting and Na/K release ratios 1 and 1.2, the corresponding pump estimates are 0.531 and 0.638 mmol ATP per L tissue/s. Howarth et al.'s illustrative rat-cortex budget [12] is 27.2 µmol/g/min, about 0.453 mmol/L tissue/s at 1 g/mL. These differing restoration assumptions require a consistent absolute flux balance before interpretation as total cost. They indicate that the original quantitative calibration cannot be accepted without resolving active surface area, spike charge and the restoration pathways. They are not an unconditional thermodynamic bound on potassium buffering.

A membrane capacitance of 1 µF/cm$^2$ and a 100 mV voltage swing imply a monovalent capacitive charge of $1.04\times10^{-8}$ mol/m$^2$. The reconstructed spike potassium transfer is about 6.75 times this charge. The comparison alone does not establish sodium entry or ATP use; it motivates reconciliation of the spike-charge convention with experimentally measured low Na overlap in some mammalian pyramidal neurons [17]. A prior illustrative 16% action-potential allocation reduced the original imposed 0.400 mM central K response to about 0.0201 mM and the glial-transport-specific adjacent rate contrast from 0.01322 to 0.000620 Hz. These numbers are conditional diagnostics, not a physiological calibration. A conservative absolute 91-state Na/K extension and later pump and cotransporter variants improved stoichiometry but did not independently establish the equilibrium or complete capacitive closure. This motivated use of the independent EDNM rather than treating the simplified potassium gain as sufficient physiological evidence.

## Supplement C. General observability and earlier graph competition

For a general linearized realization

$$\dot x=A(g,\eta)x+Bu,\qquad y=Cx+\epsilon,\qquad T(s)=C(sI-A)^{-1}B,$$

assume $A\in\mathbb R^{m\times m}$ differentiable at the nominal point and $B,C$ independent of $g$. Put $A_g=\partial_gA$. The following exact finite characterization is a useful specialization of established linear-systems realization theory [28–30]. The target derivative is identically zero as a rational transfer matrix if and only if

$$M_j(A_g)=C\sum_{\ell=0}^{j-1}A^\ell A_g A^{j-1-\ell}B=0,\qquad j=1,\ldots,2m-1.$$

To prove the equivalence, define the augmented matrix $\mathcal A=\begin{pmatrix}A&A_g\\0&A\end{pmatrix}$, augmented input $(0,B)^\top$ and output $(C,0)$. Its resolvent transfer equals $C(sI-A)^{-1}A_g(sI-A)^{-1}B=\partial_gT$. Expansion at infinity gives the displayed Markov coefficients and a zero coefficient at power zero. Cayley–Hamilton in dimension $2m$ gives all higher coefficients from the first $2m$; vanishing is therefore necessary and sufficient. A nuisance direction is locally equivalent under the full transfer map if its difference from the target direction satisfies these same tests. In stiff models direct evaluation of high matrix powers is ill-conditioned; finite-window Fréchet exponential derivatives or a stable numerical alternative are preferred.

The earlier conservative seven-site GraphED realization has 70 states (nine Na/K/Cl inventories per site plus one neuronal synaptic-like state), with three distinct graphs and separate local mechanisms. Previous held-out comparisons fitted nuisance parameters on training sources and predicted withheld source protocols. The no-spatial-glia rival gave 0.00394 µM K and 0.0361 µV neuronal-voltage RMS errors relative to the synthetic reference; the shared-path rival gave 0.000711 µM and 0.00662 µV. Under the illustrative independent noise levels of 1 µM and 10 µV these are negligible. This is compatible with the unfavorable small-scale regime in the new passive experiment; the two model levels, protocols and source conventions are not numerically identical.

The earlier 70-state information design used four focal protocols and eight observation times spanning 0.25–3 s, five nuisance mechanisms and assumed 1 µM K, 10 µV voltage and 5 µM added glial K noise. Its archived nuisance-adjusted information values were $2.9119\times10^{-10}$ (K and voltage), $5.5835\times10^{-6}$ (added glial potassium), and $0.0564344$ (additional idealized direct glial forcing). Nominal unit-change three-standard-deviation repetition lower bounds were $3.09\times10^{10}$, $1.61\times10^6$ and 159. The exact historical stimulus amplitudes are not verified in the available source, so these numbers are retained only as historical, non-dose-comparable design diagnostics. The main article uses the new fully specified passive dose and time schedule instead, and does not substitute that dose for the missing historical values.

Similarity-equivalent unobserved realizations illustrate a further limitation: for an invertible $T_0$, the realization $(T_0AT_0^{-1},T_0B,CT_0^{-1})$ has identical transfer. The same observations cannot distinguish arbitrary interpretations of the hidden coordinates. Cellular specificity comes from separately established constraints, readouts and interventions, not names assigned to latent states.

## Supplement D. Earlier resonome and EEG claims

The original research programme's two-state cooperative calcium/IP3 kinetics do not yield a free telegraph wave simply by eliminating a variable. For $\dot c=-\alpha c+\beta p+D\Delta c$ and $\dot p=\gamma c-\delta p$, direct differentiation and substitution, with constant coefficients, gives

$$c_{tt}+(\alpha+\delta)c_t+(\alpha\delta-\beta\gamma)c-\delta D\Delta c-D\Delta c_t=0.$$

The spatial Laplacian term carries $\delta D$, and the mixed space–time term is essential to the exact elimination. Discarding it is a further approximation, not an exact wave equation. Cooperative coupling by itself does not imply oscillatory poles; a passive potassium model is not a self-sustained ionic oscillator without an additional active feedback mechanism.

The earlier EEG predictor used a local two-pole ionic filter and omitted the glial interunit spatial operator. Its spectral prediction was therefore exactly independent of the defining spatial parameter. Its originally frozen primary comparison was not evaluable. The amended comparison of 535 participants gave a mean held-out score difference of approximately −0.00918 for the constrained local predictor relative to a more flexible generic two-state competitor (participant-bootstrap 95% percentile interval approximately [−0.01016, −0.00823] in the archived score normalization). This is a negative result for that particular predictor; neither the frozen outcome nor a direct test of spatial SMM was established. Neuron-only M0 or single-feedback M1 comparative performance was not established by that analysis. The published low-frequency resonome hypothesis [2] therefore remains separate from the validated mathematical and computational statements in this work. The appropriate future electrophysiological model must carry the glial geometry through ionic dynamics and neuronal transmembrane currents to an independently calibrated lead field.

# References {.unnumbered}

1. Ballús, A. (2024; revised through 2026). *The Brain-Mesh Model: A Unified Framework for Neural Synchrony, Plasticity, and Coherence*; later *The Syncytial Mesh Model*. arXiv:2412.12106; bioRxiv. <https://doi.org/10.1101/2024.11.22.624908>.

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

19. Cafiso, M., Casagrande, G., Angiolleli, M., Paradisi, P., Sorrentino, P., & Depannemaecker, D. (2026). A microcircuit model of astrocytic potassium buffering and neural synchronization. *Brain Research*, article 150630 (published online 19 November 2026). <https://doi.org/10.1016/j.brainres.2026.150630>.

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

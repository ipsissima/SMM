---
title: 'Supplementary material: The Syncytial Mesh Model'
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

## Supplement E. Independent nonlinear source-model positive control

The small-load graph and clamp experiments do not test whether the solver is capable of reproducing a large coupling-dependent neuronal transition. We therefore reproduced the published EDNM two-unit high-load protocol [10, Table 4 and Figs. 9–11]: 1 mm interunit spacing, sealed boundaries, synapses off, a depolarizing constant stimulus $j_{\mathrm{stim}}=+2\times10^{-4}$ mM cm/s from 0.1 to 10 s, and the original 50 µs timestep. The source's strong-glia variant multiplies interunit glial Nernst–Planck fluxes by 2000, leaving the other domains and the local membrane equations unchanged. The multiplier is an artificial computational positive control, not a measurement of gap-junction conductivity. Unlike the earlier small-load negative K-transfer pulse, the positive stimulus drives neuronal depolarization and activity-associated potassium accumulation.

At 10 s, our native (weak-glia) reproduction yields an extracellular K increase of 13.644 mM at the stimulated soma, and the strong-glia reproduction yields 2.617 mM. These match the independently reported values of about 13.6 and 2.6 mM. In the native case, the voltage remains near −29.7 mV without spikes over the final second; in the strong case, voltage repeatedly traverses a spiking range (−65.9 to +9.1 mV over the final second). The strong-coupling trajectory peaks at a K elevation of 7.82 mM at 6.63 s, close to the reported peak of approximately 7.7 mM at 6.7 s. The weak-glia 100 µs run also terminates in depolarization block, with K elevated 13.677 mM; precise spike counts and the time of transition differ under timestep refinement. The original-resolution runs are used for the positive-control comparison, not for a claim of pointwise convergence of spike timing.

Table S1. Original-resolution high-load reproduction against the independently published computational benchmark. Both variants use the source-published stimulus and original 50 µs timestep.

| Outcome | Weak glia | Strong glia |
|:--|--:|--:|
| Published final K elevation (mM) [10] | 13.6 | 2.6 |
| Reproduced final K elevation (mM) | 13.644 | 2.617 |
| Reproduced peak K elevation (mM) | 13.644 | 7.822 |
| Voltage range in final 1 s (mV) | −29.68 to −29.62 | −65.88 to +9.13 |
| Threshold-detected spikes in final 1 s | 0 | 3 |
| Max relative ionic inventory drift | $6.17\times10^{-9}$ | $1.20\times10^{-8}$ |

![Reproduction of the independent high-load EDNM weak/strong coupling experiment at 50 µs. Large K accumulation accompanies the native-coupling depolarization block, whereas the artificially enhanced glial mobility permits continuing spiking. Voltage traces are displayed with light decimation for legibility.](high_load.png){width=86%}

The full trajectories are finite and satisfy positive volume fractions and nonnegative concentrations (glial calcium is zero in the source). Inventory checks were repeated directly on the archived arrays. The trajectories and reconstructed values are documented in the repository validation and provenance records. These are computational source-model reproduction results. They show that the source physiology can yield strong dependence of neuronal state on glial transport under a deliberately different regime. They do not demonstrate that the synthetic shortcut edges determine that transition, that 2000-fold mobility is physiological, or that an externally applied local KCl load and connexin manipulation reproduce it. The CA1 large-load experiment [15] requires its own calibrated source, counterion/charge-balancing representation, geometry and off-target controls; it remains an experimental validation target.

## Supplement F. Matched small-load numerical benchmark

At 5 µm spacing, the seven-site full source simulation transfers potassium from the neuronal soma of site 0 into extracellular space from 0.08 to 0.40 s using $j_{\mathrm{stim}}=-10^{-4}$ mM cm/s in the original source convention; synaptic connections are disabled. The 1 ms path run yields a stimulated-site peak extracellular elevation of 64.49 µM and a nearest-neighbor peak of 45.68 µM. Adding the three specified glial chords changes the path-distal site 6 extracellular signal by at most 0.02053 µM; the corresponding neuronal membrane contrast is about 0.021 µV. This is a very small graph-specific effect. Inventory drifts are below $3\times10^{-13}$ relative, and the graph contrast was previously checked at 2 and 1 ms, with relative RMS discrepancies of approximately 0.51–0.52% for K and 0.27% for voltage. The focal membrane voltage excursion is about 32 mV despite small extracellular loads, so the full protocol is not a globally small-signal neuronal experiment.

The original passive and full model results appeared to differ by roughly two orders of magnitude when contrasts with different observation windows and normalizations were compared. Using the same 1 ms time labels, weighted graph, pulse duration, mass source and whole-field statistic removes much of that disparity. The contrast ratio between the source-reconciled passive and the complete model is now approximately 8.96. The difference remains scientifically significant. A passive model using the EDNM concentration-diffusion coefficients alone, $D_e=\phi_e D_K/\lambda_e^2=1.53125\times10^{-10}$ m$^2$/s and $D_g=\phi_a D_K/\lambda_g^2=7.65625\times10^{-11}$ m$^2$/s (after unit conversion), predicts an even larger contrast of 0.05219. Those coefficients omit electrical migration and are not effective fitted EDNM coefficients.

Table S2. Matched relative extracellular graph contrast at 5 µm. All models use the same source-site pulse convention and full 0.8 s sampling labels; the passive model matches delivered potassium mass, not the electrical action of transmembrane injection.

| Model | Relative contrast $\mathcal R$ | Peak path K (µM) | Peak graph difference, site 6 (µM) |
|:--|--:|--:|--:|
| Full EDNM | 0.0004380 | 64.49 | 0.02053 |
| Source-reconciled passive | 0.0039262 | 23.17 | 0.08865 |
| Passive, EDNM diffusion only | 0.0521922 | 45.80 | 1.31430 |

![Comparison of source-reconciled passive and full EDNM trajectories under matched small-load conditions. Different total response magnitudes and distal graph contrasts demonstrate that matching mass, geometry and protocol is insufficient to make the passive model a quantitatively equivalent physiological surrogate.](matched_waveforms.png){width=100%}

Even after fitting a single amplitude multiplier (2.057) to the passive path response, the relative whole-field RMS mismatch is 0.420. Because the graph statistic is invariant to common amplitude scaling, this mismatch is not simply a different source amplitude. The underlying glial flux also cannot be identified with a single chemical diffusion coefficient: on the three added somatic glial edges, the L2 norms of stimulus-induced electrical migration terms in the actual trajectories are about 1.97, 2.13 and 2.19 times those of their chemical-diffusion terms. The combined changes are 2.77–3.00 times the diffusion-only norms. Their correlations are positive but far from perfect. This is descriptive flux accounting, not a causal decomposition of the full passive/full-model discrepancy. In particular, migration is not a uniformly suppressive mechanism; membrane transport, electric fields, ionic feedback, and local dynamics all matter. An effective scalar mobility that would locate full EDNM on the passive sensitivity surface has not been established. The circle in the scale figure therefore marks the specified passive reference, not a pretend full-model fit.

This benchmark provides a negative quantitative constraint on strong-effect interpretations at the tested small-load protocol. The following extended sweep tests the larger-scale prediction directly in the full physiology.

## Supplement G. Mechanism of the neighboring voltage inversion

The independent two-unit EDNM challenge uses 5 µm spacing, sealed boundaries, no synapses, a 0.5 s neuronal-to-extracellular K transfer of magnitude $10^{-4}$ mM cm/s (0.05–0.55 s), and a 2 s horizon. The original 2 ms run gives a neighboring extracellular K peak of 0.13611 mM and a peak absolute neuronal voltage response of 0.14986 mV. Turning off glial interunit transport changes those neighboring signals by at most 0.04113 µM and 0.06834 µV, respectively; relative RMS convergence of the intervention contrast from 2 to 1 ms is approximately 0.93% and 0.69%. Thus the overall ionic response should not be conflated with the specific contribution of interunit glial transport.

At $t=0.4$ s in the native 2 ms run, neighboring extracellular K increases by 0.101726 mM and the K Nernst potential by 0.753641 mV, but the neighboring membrane potential decreases by 0.111094 mV. The changes in neuronal intracellular and extracellular potential are respectively $-0.08030$ and $+0.03080$ mV. This decomposition is a kinematic identity, not an explanation of the sign. There are two mechanistically distinguishable contributions in the original stimulus and response.

First, the imposed flux transfers positively charged ions between compartments: it is globally potassium- and charge-conserving, not locally electrically neutral. Its outward transmembrane current has magnitude $F_{\!A}|j_{\mathrm{stim}}|=9.648$ µA/cm$^2$ in the source units. In the 1 ms trajectories, the first stimulated timestep at the neighboring soma produces an extracellular potential change of $+34.86$ µV while extracellular K changes by only $+0.0220$ µM. At the onset of stimulus removal the extracellular potential falls by approximately 34.86 µV within one step, whereas K remains continuous to that temporal resolution. The rapid response is consistent with the quasistatic electrical constraints responding to imposed membrane current. It should not be interpreted as exceptionally fast chemical transport or as a topology-specific glial signal.

Second, the longer voltage response reflects a balance of electrogenic membrane processes. With outward current positive, multiplying the ionic balance by $F_{\!A}z_i$ and summing gives at the unstimulated neighboring soma the discrete identity

$$C_m\frac{V_m^j-V_m^{j-1}}{\Delta t}=-I_{\mathrm{mem}}^{j}-I_{\mathrm{ax}}^{j}.$$

The axial term includes soma–dendrite ionic flux at the source volume and area conventions. The neuronal Na/K pump exchanges three Na for two K and generates net outward current $F_{\!A}j_{\mathrm{pump}}$. KCC2, NKCC1 and the source two-Na/Ca exchanger are directly electroneutral by stoichiometry, although they affect the concentrations that determine other currents. Currents are reconstructed with the exact source semi-implicit convention (previous-step pump and gating factors, current-step voltage and reversal potentials). Integrating paired stimulated-minus-baseline currents over the time steps reconstructs the voltage change without fitting.

Table S3. Current decomposition of the neighboring soma at 0.4 s (native 2 ms trajectory). Outward-positive current differences are in µA/cm$^2$; integrated voltage contributions use the actual discrete capacitive balance. Totals may differ in the last displayed digit owing to rounding.

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

## Supplement H. Native-conductivity nonlinear shared-support stress tests

A single-source nonlinear fit using six independently adjustable nonnegative glial path weights, fixed extracellular and local physiology and bounds 0–50 is a stronger comparison than equal-total-weight normalization. At 37 µm, high load and 1 ms timestep, we fit source 0 using K and voltage at seven sites and 15 time samples, with illustrative independent scale factors 1 µM and 10 µV. Two initializations converge to the same fitted weights (approximately 1.918, 4.708, 50, 50, 50, 1.262). Three weights reach the upper bound. The training K RMS discrepancy falls from 0.177 mM for the total-weight-matched path to 0.0109 mM, while the withheld source 2 K RMS discrepancy is 0.1021 mM (0.0667 mM for the total-weight-matched path). The fitted rival therefore reproduces the training source far better but generalizes worse in K; its held-out voltage RMS is 0.0492 mV versus 0.0796 mV for the total-weight-matched path. The comparison uses the independent graph as reference. These are synthetic model-discrimination and generalization results, not independent biological validation or a certified lower bound over all shared-support models. Broader fits allowing ECS weights, multiple training sources, wider conductance bounds and a third withheld source are required to assess sensitivity to the restriction.

Allowing all six extracellular path weights to vary in addition to the six glial weights further reduces single-source training K RMS discrepancy to 0.00286 mM and voltage RMS discrepancy to 0.00512 mV. The better of two obtained fits has held-out source-2 K RMS discrepancy 0.2865 mM and voltage RMS discrepancy 0.1902 mV. Both twelve-parameter optimizations exhausted their prescribed iteration budget; the values describe obtained candidates, not converged minima. The additional flexibility illustrates a severe risk of fitting only one stimulation site.

A broader joint-source glial-only optimization fits sources 0 and 2 together and withholds source 4, with bounds 0–500. It yields training potassium RMS discrepancies 0.0422 and 0.0482 mM, and held-out potassium RMS discrepancy 0.0662 mM (maximum 0.187 mM). The held-out neuronal voltage RMS discrepancy is 0.0539 mV (maximum 0.156 mV). The fitted path weights span 2.59–217.46; the large weights are not empirically constrained. These comparisons show finite predictive separation for the fitted alternatives, not a certificate of separation from the entire shared-support cone. The joint fit allowing independent ECS weights is a further stress test.

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

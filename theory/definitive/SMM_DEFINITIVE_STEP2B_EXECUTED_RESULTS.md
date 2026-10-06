# SMM Definitive Reconstruction — Step 2B Executed Results

**Status:** executed and provisionally frozen at the cellular level.  
**No EEG data were used.**

## Result

The neuronal side of the SMM can be connected to extracellular potassium
without an arbitrary glia-to-neuron coupling constant.

The forward bridge is

\[
K_e\rightarrow E_K\rightarrow I_{\rm SN}(K_e)\rightarrow\eta_{\rm QIF}.
\]

At \(K_e=3.5\) mM, the excitatory reduced Traub–Miles calibration has

\[
V_{\rm SN}=-64.0118053\ {\rm mV},\qquad
I_{\rm SN}=0.119345708\ \mu{\rm A/cm^2},
\]

and the Wang–Buzsáki inhibitory calibration has

\[
V_{\rm SN}=-59.9658163\ {\rm mV},\qquad
I_{\rm SN}=0.160086327\ \mu{\rm A/cm^2}.
\]

Each Jacobian has one numerically zero eigenvalue and two stable eigenvalues,
as required locally for the saddle-node/QIF normal form.

With \(\ell=\ln(K_e/3.5)\), continuation over \(K_e=2.5\)–\(6\) mM gives

\[
I_{\rm SN}^{E}\simeq
0.11935071-0.02826069\ell+0.00265342\ell^2,
\]

\[
I_{\rm SN}^{I}\simeq
0.16009272-0.05060487\ell+0.00397637\ell^2.
\]

Independent center-manifold projection reproduces the local continuation
slopes to numerical precision:

\[
\left.\frac{dI_{\rm SN}^{E}}{d\ln K_e}\right|_{3.5}
=-0.02822513
=-\frac{b_K}{b_I},
\]

\[
\left.\frac{dI_{\rm SN}^{I}}{d\ln K_e}\right|_{3.5}
=-0.05055987
=-\frac{b_K}{b_I}.
\]

For an MPR reference time constant \(\tau=10\) ms, the frozen K-induced
excitability shifts are

\[
\boxed{
\delta\eta_E=
0.08142122\ell-0.00764471\ell^2
}
\]

and

\[
\boxed{
\delta\eta_I=
0.07302231\ell-0.00573786\ell^2.
}
\]

For another MPR timescale, multiply both coefficients by
\((\tau/10{\rm ms})^2\).

The local QIF prediction reproduces the parent conductance-model firing rates
to better than about 10% for \(\Delta I\le0.02\ \mu{\rm A/cm^2}\). Its error
then increases with distance from the SNIC, as expected. The QIF/MPR mapping is
therefore a controlled local reduction, not a global replacement for the
conductance models.

## Reverse K source

Integration of the delayed-rectifier K current per spike gives approximately

\[
\boxed{
q_E=4.736\times10^{-8}\ {\rm mol/m^2/spike}
}
\]

for the excitatory calibration and

\[
\boxed{
q_I=1.267\times10^{-8}\ {\rm mol/m^2/spike}
}
\]

for the inhibitory calibration at \(K_e=3.5\) mM.

These values vary only modestly across the tested physiological K range and
across moderate firing rates.

The tissue source entering the mesh is therefore

\[
\boxed{
S_K^N=
\frac{\rho_{A,E}q_E R_E+\rho_{A,I}q_I R_I}{\phi_e}.
}
\]

The microscopic \(q_E,q_I\) values are frozen. The membrane-area densities
\(\rho_{A,E},\rho_{A,I}\) remain independent morphometric scale factors and
must not be hidden inside an EEG-fitted coupling coefficient.

## Closed loop

Combined with the already validated astroglial mesh,

\[
(R_E,V_E,R_I,V_I)
\rightarrow S_K^N
\rightarrow(k,a)
\rightarrow K_e
\rightarrow(\eta_E,\eta_I)
\rightarrow(R_E,V_E,R_I,V_I).
\]

Thus the cellular and mesoscale causal loop is now closed up to the separate
anatomical conversion from membrane area to tissue volume.

## Limitations

1. Contreras et al. provide their dynamic-ion code through an external
   Humboldt GitLab repository. It was not reachable from this execution
   environment. The excitatory result here is therefore a canonical reduced
   Traub–Miles SNIC calibration checked against the published Contreras
   bifurcation result, not a byte-identical reproduction of their full code.

2. The MPR/QIF reduction is local to the SNIC regime.

3. The per-spike K values quantify spike-associated delayed-rectifier K
   efflux; slower pump/leak bookkeeping must not be double counted.

4. The remaining tissue conversion requires independent morphometric priors
   for neuronal membrane area density.

## Decision

**Step 2B passes.**

The next step is Step 4: constrain the anatomical membrane-area scaling,
assemble the full E–I MPR + ionic interface + syncytial mesh, establish its
equilibria and bifurcations, perform parameter-recovery tests, and compare it
against local-buffering and generic-slow-field controls before any EEG fit.
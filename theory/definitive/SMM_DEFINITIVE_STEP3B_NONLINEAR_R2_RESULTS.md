# SMM Definitive Reconstruction — Step 3B
## Executed nonlinear R1 validation and R2 electro-chemo-mechanical robustness test

**Status:** Executed and provisionally passed for the mesoscale physiological regime  
**Date:** 2026-10-05  
**No EEG data used.**  
**Reference code:** `martejulie/fluid-flow-in-astrocyte-networks`, pinned at commit `99bfe1671d9c5735333fa24c2406aa32bde010fc`.

---

## 1. Question tested

Step 3 proposed the reduced Syncytial Mesh

\[
\phi_e\dot k
= S_K-(\kappa_e+\kappa_N)k+\kappa_a a-D_eL_e k,
\]

\[
\phi_a\dot a
= \kappa_e k-\kappa_a a-D_gL_g a.
\]

Step 3B asks whether this two-state architecture survives:

1. nonlinear zero-flow KNP dynamics (R1 / M0), and
2. the higher-fidelity Sætra–Ellingsrud–Rognes electro-chemo-mechanical model (R2 / M1–M3), including osmotic swelling and fluid advection.

The reduced model is evaluated only after projecting the reference solutions into the validated mesoscale spatial band (cosine modes 0–5). Raw microscopic peaks are reported separately and are not treated as quantities the mesoscale model must reproduce.

---

## 2. Numerical reference implementation

The nonlinear reference solver was independently ported from the published/open `ffian` equations into a finite-volume 1D implementation with:

- Na+, K+, Cl− in astroglial ICS and ECS;
- generalized Nernst–Planck electrodiffusion;
- quasi-static charge/current constraints for intra/extracellular potentials;
- Kir K flux;
- Na/K-ATPase;
- charge-neutral K-in / Na-out stimulation;
- published K-decay term;
- no-flux boundaries.

For R2/M3 it additionally includes:

- intracellular volume fraction dynamics;
- transmembrane osmotic water flux;
- membrane elastic pressure;
- hydrostatic flow;
- astroglial osmotic flow;
- ECS electro-osmosis;
- ionic advection.

The 30 μm stimulation zone is represented by exact cell-overlap fractions, so total injected flux is grid-independent.

---

## 3. R1 nonlinear result

The R1 coefficients remained frozen from the linear modal reduction:

| Parameter | R1 value |
|---|---:|
| \(\kappa_e\) | 2.03390469 |
| \(\kappa_a\) | 0.46424873 |
| \(D_e\) | \(9.84620\times10^{-10}\,\mathrm{m^2/s}\) |
| \(D_g\) | \(1.32905\times10^{-10}\,\mathrm{m^2/s}\) |

Nonlinear validation, N=31, dt=0.01 s:

| Input flux | raw peak ΔKe | coarse-field NRMSE | coarse peak error | recovery fraction full / reduced |
|---:|---:|---:|---:|---:|
| 2.5e-8 | 0.137 mM | 3.37% | 0.25% | 0.1532 / 0.1546 |
| 5e-8 | 0.277 mM | 3.82% | 1.51% | 0.1491 / 0.1546 |
| 1e-7 | 0.569 mM | 5.15% | 3.87% | 0.1414 / 0.1546 |
| 2e-7 | 1.193 mM | 8.28% | 8.07% | 0.1281 / 0.1546 |
| 5e-7 | 3.365 mM | **16.80%** | **17.56%** | 0.0998 / 0.1546 |

### R1 decision

The two-state reduction passes the prespecified 10% mesoscale error criterion through approximately

\[
\boxed{\Delta K_e\lesssim 1.2\ \mathrm{mM}}
\]

in this stimulation family.

It fails clearly in the multi-mM regime.

Therefore the nonlinear amplitude validity boundary is real rather than rhetorical.

---

## 4. R1 numerical convergence

For the 5e-8 input, the N=31 solution differs from N=41 by only ~0.35% in the stored trajectory convergence metric. The N=21 solution differs from N=41 by ~2.36%.

Temporal refinement at N=31 shows that dt=0.02 s differs from the dt=0.005 s reference by ~3.12%; dt=0.01 s is therefore retained as the standard validation timestep.

All concentrations remained positive.

---

## 5. R2: applying the R1 coefficients without modification

The higher-fidelity R2 model was decomposed into the published flow scenarios:

- **M1:** hydrostatic flow only;
- **M2:** hydrostatic + astroglial osmotic flow;
- **M3:** M2 + ECS electro-osmosis.

With the R1 coefficients frozen, transfer-function mismatch is:

| Mode | M1 NRMSE | M2 NRMSE | M3 NRMSE |
|---:|---:|---:|---:|
| 0 | 1.46% | 1.46% | 1.46% |
| 1 | 3.17% | 24.91% | 21.73% |
| 2 | 4.21% | 33.70% | 29.90% |
| 3 | 4.37% | 26.88% | 23.83% |
| 4 | 5.11% | 16.65% | 14.47% |
| 5 | 8.21% | 6.30% | 4.85% |

### Interpretation

Hydrostatic flow alone scarcely changes the reduced mesh.

The dominant correction comes from **astroglial osmotic flow**.

ECS electro-osmosis modifies that correction but does not create it.

Thus R2 genuinely challenges the zero-flow calibration.

---

## 6. Does R2 require a new state?

A direct eigenanalysis of the full M3 linearized system shows additional ionic/osmotic poles, including composite Na/Cl/K dynamics. A naïve interpretation would be to add a volume state.

However, the volume-fraction component itself is not dominant in the K-output slow eigenvectors. The relevant new modes are composite ionic/osmotic modes.

The decisive test is therefore structural model reduction: can the *same two-state architecture* reproduce R2 if its effective transport coefficients are recalibrated from R2 rather than inherited from R1?

The answer is **yes**.

---

## 7. R2 two-state refit

Fitting the same two-state equations to M1, M2, and M3 gives:

| Scenario | κe | κa | De (m²/s) | Dg (m²/s) |
|---|---:|---:|---:|---:|
| R1/M0 | 2.0339 | 0.46425 | 9.846e-10 | 1.329e-10 |
| M1 | 2.0294 | 0.46818 | 9.794e-10 | 1.259e-10 |
| M2 | 2.0530 | 0.48141 | 9.753e-10 | 3.857e-10 |
| M3 | 2.0132 | 0.48318 | 9.553e-10 | 3.715e-10 |

Relative to R1, M3 changes approximately:

\[
\kappa_e:\ -1.0\%,
\]

\[
\kappa_a:\ +4.1\%,
\]

\[
D_e:\ -3.0\%,
\]

but

\[
\boxed{D_g:\times 2.80.}
\]

M2 gives \(D_g\times 2.90\).

### Central mechanistic result

The electro-chemo-mechanical corrections do **not** require changing the state dimension of the mesoscale SMM in the physiological coarse regime.

They are captured primarily by renormalizing the **effective syncytial transport coefficient** \(D_g\).

Therefore:

\[
\boxed{
D_g\text{ must be interpreted as effective syncytial ionic transport,}
}
\]

including diffusion, electrical drift, and osmotically driven advective enhancement—not as a literal bare K diffusion coefficient or a single connexin permeability.

---

## 8. Held-out modal validation after R2 recalibration

R2/M3 was fitted on modes 0,1,3,5.

Held-out modes:

| Mode | NRMSE |
|---:|---:|
| 2 | 2.39% |
| 4 | 5.04% |
| 6 | 10.47% |

As in R1, error increases at sufficiently high spatial frequency.

Thus R2 preserves the earlier conclusion that the SMM is a **mesoscale low-spatial-mode theory**, not a microscopic continuum valid down to arbitrary wavelengths.

---

## 9. Direct nonlinear M3 test

The R2-renormalized coefficients were frozen and compared to an independently executed nonlinear M3 finite-volume solution.

N=31, dt=0.01 s, exact 30 μm input zone:

| Input | raw peak ΔKe | coarse NRMSE | coarse peak error | recovery full / reduced | αi range | max intracellular speed |
|---:|---:|---:|---:|---:|---:|---:|
| 2.5e-8 | 0.125 mM | **1.97%** | **0.17%** | 0.1609 / 0.1606 | 0.399972–0.400381 | 1.13 μm/min |
| 5e-8 | 0.253 mM | **2.48%** | **1.24%** | 0.1573 / 0.1606 | 0.399944–0.400767 | 2.28 μm/min |

These comparisons are performed after projecting the full nonlinear M3 solution onto modes 0–5, i.e. at the scale at which the reduced SMM has been validated.

The raw microscopic center peak is deliberately not used as the primary reduction metric.

---

## 10. Nonlinear M3 grid convergence

For the 5e-8 input:

| N | dx | coarse NRMSE | coarse peak error | recovery full / reduced |
|---:|---:|---:|---:|---:|
| 11 | 27.27 μm | 3.01% | 2.95% | 0.1518 / 0.1606 |
| 21 | 14.29 μm | 2.45% | 0.85% | 0.1579 / 0.1606 |
| 31 | 9.68 μm | 2.48% | 1.24% | 0.1573 / 0.1606 |

The N=21 and N=31 mesoscale results are effectively converged for the quantities used in the SMM reduction.

---

## 11. Final Step 3B decision

### PASSED — with a clarified interpretation

The core two-state Syncytial Mesh survives:

- linear R1 modal reduction;
- held-out spatial modes;
- nonlinear R1 finite-amplitude perturbations;
- R2 hydrostatic/osmotic/electro-osmotic physics;
- nonlinear R2/M3 perturbations;
- spatial grid refinement.

It survives **not** because fluid mechanics is negligible.

It survives because fluid mechanics renormalizes the effective syncytial transport operator while leaving the coarse state architecture intact.

The canonical mesoscale mesh remains:

\[
\boxed{
\phi_e\dot k
= S_K-(\kappa_e+\kappa_N)k+\kappa_a a-D_eL_e k
}
\]

\[
\boxed{
\phi_a\dot a
= \kappa_e k-\kappa_a a-D_g^{\rm eff}L_g a.
}
\]

But now:

\[
\boxed{
D_g^{\rm eff}
=
\text{coarse syncytial transport produced jointly by electrodiffusion and fluid-assisted transport.}
}
\]

---

## 12. Canonical validity domain after Step 3B

The present evidence supports the reduced mesh only within a declared domain.

### Spatial

Reliable primarily for coarse modes corresponding, in the 300 μm reference domain, to wavelengths of roughly \(\gtrsim100\) μm; accuracy degrades below that scale.

### Amplitude

R1 passes the 10% nonlinear criterion through approximately \(\Delta K_e\sim1.2\) mM in the tested stimulation family and fails by multi-mM excursions (~3.4 mM).

R2/M3 has been directly nonlinear-validated in the small physiological range up to ~0.25 mM peak ΔKe with excellent mesoscale agreement.

### Interpretation

The SMM must not be used as a pathological high-K model without a separate extension.

---

## 13. What has now been earned scientifically

Before this reconstruction, the 'mesh' was an imposed field equation.

After Step 3B we have an evidence chain:

\[
\text{published KNP astrocyte/ECS physics}
\]

\[
\Downarrow
\]

\[
\text{full electrodiffusive nonlinear dynamics}
\]

\[
\Downarrow\ \text{model reduction + held-out modes}
\]

\[
\text{two-state mesoscale mesh}
\]

\[
\Downarrow\ \text{R2 osmotic/fluid challenge}
\]

\[
\boxed{\text{same two-state architecture with }D_g^{\rm eff}\text{ renormalization}}
\]

\[
\Downarrow\ \text{nonlinear and grid-converged validation}
\]

\[
\boxed{\text{validated physiological mesoscale control subsystem}.}
\]

This is the first part of the definitive SMM that can now be treated as an actual quantitative result rather than a design hypothesis.

---

## 14. Next required step

Do **not** couple this immediately to EEG.

The next unresolved mechanistic link is the Step 2B neuronal calibration that was specified but not yet numerically executed:

\[
K_e
\longrightarrow
E_K
\longrightarrow
I_{\rm SN}(K_e)
\longrightarrow
\eta_{\rm QIF}(K_e),
\]

and

\[
R,V
\longrightarrow
S_K^N.
\]

Only after those coefficients and their uncertainty are numerically frozen should the validated astroglial mesh be coupled to the E–I MPR network.
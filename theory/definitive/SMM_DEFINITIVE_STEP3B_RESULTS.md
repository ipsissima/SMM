# SMM Definitive Reconstruction — Step 3B Results
## Linear electrodiffusive reduction of the astrocyte–ECS reference model

**Date:** 2026-10-05  
**Status:** Linear KNP reconstruction and model reduction executed. Nonlinear FEniCS R1/R2 reproduction remains to be executed in a compatible FEniCS runtime.  
**Reference implementation pinned:** `martejulie/fluid-flow-in-astrocyte-networks`, commit `99bfe1671d9c5735333fa24c2406aa32bde010fc`.

---

## 1. What was executed

The open `ffian.zero_flow_model` equations and baseline parameters were reconstructed directly from the authors' source code and linearized around the published resting state.

The reference model contains six ionic concentration variables per spatial mode:

\[
( Na_i, Na_e, K_i, K_e, Cl_i, Cl_e),
\]

with intracellular and extracellular potentials determined by the electrodiffusive charge constraints.

The membrane model includes:

- passive Na flux;
- passive Cl flux;
- inward-rectifying K flux;
- Na/K-ATPase;
- the reference charge-neutral K/Na activity input;
- the reference extracellular K decay term used by the open example implementation.

For each Neumann spatial mode

\[
q_n=\frac{n\pi}{L},\qquad L=300\ \mu\mathrm m,
\]

the full linearized concentration operator was constructed and its transfer function from charge-neutral activity input to extracellular K was calculated.

No EEG data were used.

---

## 2. Baseline numerical checks

Pinned reference baseline:

\[
K_{e0}=3.2157956\ \mathrm{mM},
\qquad
K_{i0}=99.8921022\ \mathrm{mM},
\]

\[
\phi_{i0}=-85.8612\ \mathrm{mV},
\qquad
\phi_{e0}=0.
\]

The linearization reproduces the expected stable dissipative structure. Near-zero eigenvalues correspond to conserved/unforced concentration combinations and have negligible residue in the experimentally relevant input-to-\(K_e\) transfer function.

The observable K response is controlled by a small number of stable decaying poles.

For the spatially uniform mode, the dominant full-model poles include approximately

\[
-0.1093\ \mathrm{s}^{-1},\qquad
-0.6794\ \mathrm{s}^{-1},\qquad
-12.11\ \mathrm{s}^{-1}.
\]

The slowest dominant component therefore has a characteristic time of about

\[
\tau\approx9.15\ \mathrm s.
\]

This is already qualitatively incompatible with the historical interpretation of the mesh as a theta-range oscillator and consistent with a slow control system.

---

## 3. Prespecified two-state reduction fit

The candidate reduced mesh was

\[
\phi_e\dot k
=
 u-(\kappa_e+\kappa_N)k+\kappa_a a-D_e q^2 k,
\]

\[
\phi_a\dot a
=
\kappa_e k-\kappa_a a-D_gq^2a.
\]

The reference volume fractions were fixed:

\[
\phi_e=0.2,\qquad \phi_a=0.4.
\]

The reference K-decay sink fixes

\[
\kappa_N=0.232.
\]

Following the prespecified modal split, modes

\[
n=0,1,3,5
\]

were used for calibration and modes

\[
n=2,4,6,7,8,9,10,12
\]

were held out.

A single shared parameter set was fitted to the complex transfer functions over

\[
10^{-3}\le\omega\le5\ \mathrm{rad/s}.
\]

The resulting coefficients were:

\[
\boxed{\kappa_e=2.03390}
\]

\[
\boxed{\kappa_a=0.464249}
\]

\[
\boxed{D_e=9.8462\times10^{-10}\ \mathrm{m^2/s}}
\]

\[
\boxed{D_g=1.3291\times10^{-10}\ \mathrm{m^2/s}}.
\]

The exact dimensional interpretation of \(\kappa_e,\kappa_a\) is tied to the reduced mass-balance convention and will be retained with the reference normalization rather than relabeled prematurely as bare channel rates.

---

## 4. Strong physical sanity check

The fitted spatial coefficients were not unconstrained to equal the microscopic effective diffusivities, yet they landed close to them.

Reference effective ECS K diffusivity:

\[
\frac{D_K}{\lambda_e^2}
=
7.6563\times10^{-10}\ \mathrm{m^2/s}.
\]

Fitted reduced value:

\[
D_e=9.8462\times10^{-10}\ \mathrm{m^2/s},
\]

or about **1.29×** the microscopic reference value.

Reference effective intracellular/syncytial K diffusivity:

\[
\frac{D_K}{\lambda_i^2}
=
1.9141\times10^{-10}\ \mathrm{m^2/s}.
\]

Fitted reduced value:

\[
D_g=1.3291\times10^{-10}\ \mathrm{m^2/s},
\]

or about **0.69×** the microscopic reference value.

Thus the transfer-function reduction recovers spatial coefficients of the correct physical order without being forced to reproduce them.

---

## 5. Uniform-mode reduced poles

At \(q=0\), the fitted two-state mesh has poles

\[
\boxed{-12.3814\ \mathrm{s}^{-1}}
\]

and

\[
\boxed{-0.108737\ \mathrm{s}^{-1}}.
\]

Their characteristic times are approximately

\[
0.0808\ \mathrm s
\]

and

\[
9.20\ \mathrm s.
\]

The reduction therefore separates a rapid local equilibration component from a slow homeostatic memory component.

This is a much more precise realization of the published "slow control field" concept than the old damped-wave equation.

---

## 6. Transfer-function validation

### Calibration modes

| Mode | Spatial wavelength | NRMSE |
|---:|---:|---:|
| 0 | uniform | 1.24% |
| 1 | 600 µm | 3.60% |
| 3 | 200 µm | 4.91% |
| 5 | 120 µm | 6.84% |

### Held-out modes

| Mode | Spatial wavelength | NRMSE |
|---:|---:|---:|
| 2 | 300 µm | **5.23%** |
| 4 | 150 µm | **4.34%** |
| 6 | 100 µm | 12.15% |
| 7 | 85.7 µm | 18.41% |
| 8 | 75 µm | 24.75% |
| 9 | 66.7 µm | 30.78% |
| 10 | 60 µm | 36.30% |
| 12 | 50 µm | 45.66% |

The two-state reduction therefore passes stringent held-out tests for the low spatial modes but progressively fails at fine spatial scales.

---

## 7. Emergent coarse-graining scale

For Neumann modes on the 300-µm reference domain,

\[
\Lambda_n=\frac{2L}{n}.
\]

The first clear failure of the 10% transfer-function criterion occurs around

\[
\Lambda\sim100\ \mu\mathrm m.
\]

The practical conclusion is therefore:

\[
\boxed{
\text{The two-state SMM mesh is a mesoscale reduction, not a single-astrocyte-scale model.}
}
\]

A conservative current validity statement is:

\[
\boxed{
\Lambda\gtrsim120\ \mu\mathrm m
}
\]

for the present 1-D reference and linear operating point, with a transition region around 100–120 µm.

This scale is an output of the reduction analysis, not an input chosen to fit EEG.

---

## 8. Time-domain perturbation test

A centered 10%-of-domain input profile, matching the reference geometry, was projected onto the validated coarse modes \(n=0\ldots5\).

The full linear KNP system and the reduced two-state model were then driven from 10 to 20 s and followed through recovery.

Across the full coarse spatiotemporal field:

\[
\boxed{\mathrm{NRMSE}=3.27\%}.
\]

At the stimulation center:

\[
\boxed{\mathrm{NRMSE}=1.18\%}.
\]

At 75 µm from the center:

\[
\boxed{\mathrm{NRMSE}=2.96\%}.
\]

At the extreme edge of the coarse reconstruction the error rises to approximately 9.7%, still below the provisional 10% criterion.

Because this test uses the linearized reference model, the absolute amplitude is not interpreted as a nonlinear physiological prediction; the relevant result is the agreement in spatial and temporal response shape.

---

## 9. Scientific conclusion from the executed linear Step 3B

The candidate two-state mesh survives its first serious mechanistic test.

It is **not** a good reduction of the full electrodiffusive system at arbitrarily fine spatial scales.

It **is** a quantitatively strong reduction of the low spatial modes that define a mesoscale astroglial control field.

This yields a much sharper canonical interpretation:

\[
\boxed{
\text{Syncytial Mesh}
=
\text{coarse low-spatial-mode dynamics of ion-conserving astrocyte/ECS electrodiffusion}.
}
\]

The original metallic-mesh intuition is therefore retained at the operator/topological level:

\[
L\phi_m=\lambda_m\phi_m,
\]

but the eigenvalues control **relaxation and filtering**, not intrinsic EEG-frequency resonance.

---

## 10. What has *not* yet been claimed

This result does **not** yet establish that:

- the reduction survives large nonlinear K excursions;
- swelling/fluid flow can be ignored;
- the same coefficients apply unchanged in 2-D/3-D cortical tissue;
- the mesh improves human EEG prediction;
- the mesh-specific component \(D_gL_g\) is empirically necessary.

Those are subsequent tests.

---

## 11. Runtime limitation encountered

The authors' exact reference code depends on classic FEniCS/DOLFIN.

The present execution environment has Python 3.13 but no DOLFIN/FEniCS runtime, and external package/network installation is unavailable here.

Therefore the nonlinear finite-element source implementation could not be executed directly in this runtime.

Instead, the linearized KNP operator was independently reconstructed from the authors' source equations and pinned numerical parameters, and all modal/reduction calculations reported above were actually executed locally with NumPy/SciPy.

This limitation is material only to the **nonlinear R1/R2 challenge**, not to the linear modal result reported here.

---

## 12. Decision

### Two-state mesh status

**PROVISIONALLY ACCEPTED for coarse spatial modes.**

### Current validity domain

- small-signal / linearized physiological operating point;
- low spatial modes corresponding approximately to \(\gtrsim120\ \mu\mathrm m\) in the 300-µm 1-D reference geometry;
- reference zero-flow KNP physiology.

### Next mandatory test

Run the frozen reduction against the **nonlinear** `ffian.zero_flow_model`, then against `ffian.flow_model` without refitting.

If nonlinear or flow-enabled errors exceed the preregistered tolerance, add the minimal extra slow state identified by the residual dynamics.

Only after that may the mesh be coupled to the MPR neuronal network.
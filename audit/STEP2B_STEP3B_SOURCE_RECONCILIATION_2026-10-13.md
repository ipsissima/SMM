# SMM — post-open source-model reconciliation, 13 October 2026

**Classification:** NEW independent computational audit, not historical pre-EEG Step 2B/3B/4 and not a change to frozen Step 5B. Original confirmatory primary remains **NOT EVALUABLE**; original completed same-dataset comparison remains **POST_OPEN_AMENDED**.

## Immutable upstream sources
- Contreras 2021 model source: https://itbgit.biologie.hu-berlin.de/contreras/activity-mediated_accumulation_potassium_induces_switch_firing at `4b44a276712dce4870c3c34f686d9f81d37f1927`.
- Sætra et al. 2023 original FEniCS source: `martejulie/fluid-flow-in-astrocyte-networks` at `99bfe1671d9c5735333fa24c2406aa32bde010fc`.
- Executed FEniCS reference run: https://github.com/ipsissima/SMM/actions/runs/38313546854 .
- Frozen FEniCS parent output `parent_spatiotemporal.csv`: SHA256 `fee6c5f6cd4027b18a147492c274735214a17a6cb97620014fd0b553803d9532`, 6355 rows, supplied in conversation package `SMM_STEP2B_STEP3B_RESOLUTION_2026-10-13.zip`. Store externally before any artifact expiry.

## Step 2B and conditional Step 4
The directly source-coded Contreras fast neuron with clamped `Na_i=10 mM`, `K_i=150 mM`, `K_e0=3.5 mM` has excitatory SN current `2.36356927 microA/cm²`, voltage `-63.8107084 mV`, and derivative `dI_SN/dlog(K_e)=-2.59357850 microA/cm²`. The archived reduced-SMM values are `0.119345708`, `-64.0118053 mV`, and `-0.02822513`, respectively. No documented affine re-normalization identifies the complete curves; the older reduction map is unverified.

The 10-ms parent QIF coefficient is `7.87834287` (8-ms coefficient ~`5.04214`), versus historical 10-ms `0.08142122` (8-ms ~`0.05210958`). Direct parent firing-rate QIF errors rise from 7.71% at +0.02 microA/cm² above SN to 15.83% at +0.40 microA/cm².

Gross delayed-rectifier spike K charge ~7.2–7.4e-8 mol/m² and a baseline-corrected total K excess ~7.0–7.2e-8, not archived `q_E=4.736e-8`.

For the 8-state uniform local circuit, the zero-frequency boundary is
`rho_A,crit = kappa_N/[f_E q_E (dR_E/dk)+(1-f_E) q_I (dR_I/dk)]`. This follows from the steady glial cancellation `kappa_a*a=kappa_e*k`; `D_g, D_e, kappa_a, kappa_e` do NOT affect the static boundary. With `f_E=.75` and source E mapping, rho critical is ~`4.430e6 m²/m³` using full qE and ~`4.616e6` using corrected K excess, below the archived central anatomical `4.929e6`. This is a CONDITIONAL model inconsistency, not an inference of biological pathology.

## Step 3B re-estimation preserving physical source units
The FEniCS weak-form source directly implies `gamma_m*j_in/alpha_e` (gamma_m = `8e6 m^-1`, baseline alpha_e = 0.2). The old fitted factor `1.322` is an effective transfer discrepancy, not a physical unit conversion, and rises to `1.363` with higher pulse amplitude.

Refitting only `kappa_e,kappa_a,D_e,D_g` against the lowest-amplitude original FEniCS M3 response, **fixing source amplitude and kappa_N=.232 s^-1**, yields new spatial `kappa_e=1.29629`, `kappa_a=.52863`, `D_e=1.05270e-9 m²/s`, `D_g=2.29581e-10 m²/s`. Low-six-mode NRMSE train is 2.79%; larger-amplitude holdouts 2.88%, 3.66%, 4.11% (last at reference peak 0.481 mM). A new separately fitted Dg=0 two-pool model gives 5.01%, 5.63%, 5.99% on holdouts. These are **newly estimated reduction coefficients**, not recovered historical R2 coefficients.

The central symmetric parent stimulus carries an odd DCT-II modal energy fraction only `3.652e-27`; a training split `[0,1,3,5]` would be noninformative for this specific input. New even-mode split results at 0.481 mM:
- `train [0,2,4] -> test [6,8,10]`: spatial SMM-K 15.72%, local 24.24%, 4-parameter scale-dependent extracellular diffusion 13.08%.
- `train [0,2,6] -> test [4,8,10]`: spatial SMM-K 7.81%, local 15.66%, extracellular 11.12%.

Consequently SMM-K fits the source physics more accurately than a basic local null but does not robustly demonstrate unique glial spatial identifiability against all equal-complexity extracellular spatial alternatives.

## Preservation

Full scripts, input FEniCS trajectories with SHA256, saved outputs, a clean-run validation, an integrated working manuscript and supplementary S11 are in `SMM_STEP2B_STEP3B_RESOLUTION_2026-10-13.zip` supplied in the user conversation. The **original Contreras code is intentionally not redistributed**; fetch the pinned source.

**Scientific clearance:** No positive empirical validation of the whole SMM class. These results may support a narrower methodological study or an honestly negative predictive model comparison, but source–population scaling and mechanistic identity remain open.

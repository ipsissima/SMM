# Full-physics follow-up: geometry and neutral ionic loading

Status: numerical research addendum, not experimental validation. The original version-3 manuscript is preserved. Results below are computed by the pinned workflows from paired full-state trajectories, extracted from their logged machine-readable JSON, and archived in [scale and high-dose results](../reproducibility/full_physics_followup.json).

## 1. Axial scaling

The original EDNM model (commit [826c211](https://github.com/martejulie/electrodiffusive-network-model/blob/826c211c70d9027a4acb02202ca91a60de79a15d/src/ednm/model.py), interunits_flux and residual) uses a chemical gradient and migration coefficient proportional to 1/d and multiplies the interunit flux by dt*alpha_previous/(d*gamma_m) in the inventory residual. Thus both contributions have an explicit 1/d^2 coefficient in the volume-normalized inventory equations, including the graph adapter. The code holds membrane area per tissue volume, interlayer distance, local kinetics and normalized source flux fixed; it does not represent an independently varied physical cell volume. Within each run all units have the same reference tissue volume. For geometrically scaled tissues the volume-normalized stimulus is held constant, not absolute administered mass. Neither comparison is an electrode-spacing sweep.

The passive maximum at approximately 14.76 micrometres for this specific pulse is not reproduced by the three sampled full-model scales. They establish suppression at small scale and greater contrast at 15 and 37 micrometres; the sampled maximum is 37 micrometres. The location of a continuous full-model maximum is not determined. A mobility-extreme theorem for the passive model alone does not prove a full nonlinear spatial optimum. Nor does this establish a migration mechanism.

## 2. Matched small-pulse scale results

Original [six-job run](https://github.com/ipsissima/SMM/actions/runs/45863596046), driver [541b642](https://github.com/ipsissima/SMM/tree/541b6429ea1cadbe9c5a83916d854f1e4c361318). Relative whole-field K contrast is the Euclidean norm of the difference of paired baseline-subtracted shortcut/path responses divided by the norm of the path response. All 24 baseline/stimulated trajectories satisfy the recorded positivity and inventory tolerances; no unconverged timesteps. At the finer 0.5 ms timestep:

| d (µm) | Relative K contrast (%) | Max absolute K difference (µM) | Max absolute Vm difference (µV) |
|---:|---:|---:|---:|
| 5 | 0.0438 | 0.0219 | 0.0251 |
| 15 | 0.7281 | 0.5357 | 0.7840 |
| 37 | 2.0193 | 2.4990 | 4.1795 |

The applied transmembrane K transfer has a direct electrical footprint; therefore the K contrast, not voltage, is the principal endpoint here. The extended scale/migration diagnostic has been submitted at [2ca3ae2](https://github.com/ipsissima/SMM/commit/2ca3ae20a2efe4eec80dc80d5a90c1b9a98c2e30), [run 45869133888](https://github.com/ipsissima/SMM/actions/runs/45869133888). Native extension at 60, 100 and 150 µm; zero axial glial migration at 5, 15, 37, 60, 100 and 150 µm, each at two timesteps. The latter intervention leaves local charge equations and pairwise ionic inventory accounting intact but modifies the constitutive electrochemical transport law: it is a diagnostic, not a physiological model of a specific blocker.

## 3. High-dose neutral KCl topology challenge

[Driver 3bb4a51](https://github.com/ipsissima/SMM/commit/3bb4a51a33ab5ff640d2a72521599ef754c6eb71), [completed workflow](https://github.com/ipsissima/SMM/actions/runs/45867493269). Seven sites, d = 37 µm, native mobility, sources 0 and 2, pulse 0.08–0.28 s, external K and Cl each 30 mM of source-cell tissue per second (6 mM delivered each per source-cell tissue volume), zero externally delivered net charge. The weight-matched path has all six adjacent weights equal to 7.9/6, matching the shortcut graph's total undirected edge weight (equivalently its Laplacian trace). It does not match effective resistance or independently fit six weights. At the finer 0.25 ms timestep:

| Source | Rival | Relative K contrast (%) | Relative Vm contrast (%) | Max absolute K difference (mM) | Max absolute Vm difference (mV) |
|---:|---|---:|---:|---:|---:|
| 0 | path | 4.022 | 20.722 | 0.7969 | 0.4265 |
| 0 | weight_matched_path | 3.015 | 16.102 | 0.5747 | 0.3391 |
| 2 | path | 1.863 | 8.965 | 0.2395 | 0.3292 |
| 2 | weight_matched_path | 1.283 | 6.987 | 0.1639 | 0.2562 |

All source peak increases exceed 23 mM, so this is a deliberately substantial local load. These are not equivalent to the published CA1 iontophoresis experiments; KCl injection adds osmoles and the seven-site graphs are synthetic. In the six finer-timestep cases no neuronal site crossed the -20 mV spike threshold (logged summary JSON). Do not infer a topology-induced depolarization block. Voltage contrast is a relative norm of baseline-subtracted responses, not a percentage change of resting voltage. The matched-total-weight comparison is a constrained rival, not a minimum over the complete shared-support family.

The complementary [matched-dose and glial-off experiment](https://github.com/ipsissima/SMM/actions/runs/45869212368) at [d80c3ad](https://github.com/ipsissima/SMM/commit/d80c3ad783733deefb215cafa5b2fd696be832ee) uses identical timing, geometry and native transport, lower KCl dose 3 mM tissue/s, and uncoupled controls at both doses. It prespecifies peak and relative contrast, threshold crossings, spike times and sustained-depolarization metrics, with direct cross-run comparisons. Its results are not yet included here.

## 4. Pending validation and archiving

An unrestricted shared-support fit under high load, independently documented fixed-tissue design/discretization checks, calibration against CA1 data, complete paper/supplement update and a persistent raw-state archive are outstanding. Workflow artifacts are temporary; the JSON records in Git preserve exact reported numerical summaries, not full raw state. No DOI deposit is claimed.

## Update

The subsequent extended migration, matched-dose and distal-current results are documented in [the new addendum](MIGRATION_AND_DOSE_RESULTS.md). These supersede the earlier pending-status descriptions in this historical record.

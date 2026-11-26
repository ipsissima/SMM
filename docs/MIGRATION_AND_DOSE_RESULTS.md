# Extended EDNM results: migration, matched dose and distal currents

This addendum records results computed from pinned source, following the [outcome-blind interpretation statement](MIGRATION_HYPOTHESIS.md). That statement was committed after the jobs were dispatched and completed, before their outputs were inspected in the current analysis; it is not a pre-execution preregistration. The v3 manuscript is preserved pending integration.

## Scale and axial glial migration

At 0.5 ms the relative whole-field K contrasts (%) are:

| Intrinsic graph scale (µm) | Native | Glial axial migration off |
|---:|---:|---:|
| 5 | 0.0438248 | 0.3979401 |
| 15 | 0.7280791 | 0.2266759 |
| 37 | 2.0193215 | 0.0512582 |
| 60 | 2.3648082 | 0.0208545 |
| 100 | 2.0121487 | 0.0077353 |
| 150 | 1.4799527 | 0.0034716 |

The maximum on the tested grid shifts from 60 µm to 5 µm. This confirms the specified directional migration hypothesis within the solver. At 60 µm migration enables approximately 113-fold higher contrast; at 5 µm it suppresses contrast. All eighteen extension/counterfactual jobs completed, with close agreement at 1 and 0.5 ms. The native maximum is not a continuous optimum estimate, and the diagnostic retains chemical diffusion while altering the glial axial constitutive migration law. It is not an experimental intervention or evidence for a new spatial-buffering mechanism. The novel question is the scale of *discrimination* of distinct glial geometry, not the existence of current-driven buffering. An exact full-system electrical length-scale explanation is not yet established.

Data: [fine-step summaries](../reproducibility/scale_migration_fine.json), [coarse summaries](../reproducibility/matched_dose_and_scale_coarse.json), [original native scale summaries](../reproducibility/full_physics_followup.json); [extension workflow](https://github.com/ipsissima/SMM/actions/runs/45869133888), pinned [2ca3ae2](https://github.com/ipsissima/SMM/commit/2ca3ae20a2efe4eec80dc80d5a90c1b9a98c2e30).

## Matched external neutral KCl dose

37 µm, native mobility, 0.08–0.28 s, doses 3 and 30 mM tissue/s. The restricted rival matches total undirected glial edge weight (equivalently Laplacian trace), not effective resistance or independently fitted weights. Fine timestep 0.25 ms:

| Source | Dose | Max absolute K difference (mM) | Max absolute voltage difference (mV) | Relative K contrast (%) |
|---:|---|---:|---:|---:|
| 0 | low | 0.032603 | 0.045324 | 2.15090 |
| 0 | high | 0.574726 | 0.339096 | 3.01463 |
| 2 | low | 0.015345 | 0.025019 | 1.13519 |
| 2 | high | 0.163860 | 0.256193 | 1.28343 |

Relative K contrast increases 40.16% and 13.06%, respectively. The absolute changes are not alone evidence of nonlinear amplification. Native path uncoupling raises peak source K relative to the path by 3.80% and 4.38% at low load, and 4.79% and 6.12% at high load (sources 0 and 2). This is a model-specific modest dose dependence, not a quantitative reproduction of the CA1 threshold. The low and high peak source elevations are about 2.3–2.5 and 23–26 mM, respectively. No configuration in the matched-dose comparison produced a spike or a crossing of -40 mV. Thus no categorical firing/block effect is established.

All twenty new simulations and the contrast job succeeded. [Exact matched-dose summaries](../reproducibility/matched_dose_and_scale_coarse.json); [run](https://github.com/ipsissima/SMM/actions/runs/45869212368), pinned [d80c3ad](https://github.com/ipsissima/SMM/commit/d80c3ad783733deefb215cafa5b2fd696be832ee).

## Distal current and volume reconstruction

At 0.8 s, with source 0 and distal node 5, independent shortcuts relative to the matched path yield +0.207535 mM extracellular K, +1.370738 mV potassium reversal shift and -0.339096 mV neuronal membrane potential. The four-way paired discrete capacitive current accounting gives integrated Na/K pump contribution -12.442951 mV, K leak +5.495478 mV, delayed rectifier +1.897020 mV, Cl leak +4.013892 mV, Na leak +1.032785 mV, gated Na -0.566994 mV and net remaining axial terms approximately +0.232 mV. These contributions largely cancel. The maximum four-way voltage closure error is 1.46e-8 mV. The maximum somatic ECS volume-fraction difference is 5.0073e-6 in absolute fraction (0.000501 percentage points). At node 6 with source 2 the matched-rival response has +0.163860 mM K, -0.256193 mV voltage and a similarly dominant opposing pump contribution. This is an exact along-trajectory accounting conditional on the implemented model, not a pump or water-flow intervention. Small *differences* in volume fractions do not exclude an indirect effect of volume dynamics on the full trajectories.

[Current audit summaries](../reproducibility/distal_current_volume.json), [workflow](https://github.com/ipsissima/SMM/actions/runs/45870243172), pinned [9109c58](https://github.com/ipsissima/SMM/commit/9109c5890a14610433a41005817c1588e9710a66).

## Remaining numerical and archival checks

* [Freely weighted path optimization](https://github.com/ipsissima/SMM/actions/runs/45870291419) at [2506116](https://github.com/ipsissima/SMM/commit/2506116d32b3da551f72a5956c2abebddc572133): six independently variable glial edges, with a stronger variant also allowing six ECS edge weights. Source 0 training, source 2 held out; two initializations, broad finite bounds, fixed membrane kinetics, exact full-model solves at 1 ms. This is a local numerical fit, not global certification; refine best candidates at finer timestep.
* [Graded electrical transport test](https://github.com/ipsissima/SMM/actions/runs/45870371151) at [5b0cdc4](https://github.com/ipsissima/SMM/commit/5b0cdc45e20ae013d0cf8b5d290a4b4e6803f61a): 0.25 and 4 times native axial glial migration, 5–150 µm. Calculate a frozen-concentration two-conductor length independently from rest conductivity and membrane slope; test directional correspondence with the observed contrast maximum. This is a constitutive diagnostic, not a physiological change of conductivity, nor an exact full-system mode reduction.
* Fixed-tissue spatial sampling/discretization check: independent referee calculations supplied in correspondence; original code not available here. Separate manuscript and supplement, raw state DOI deposit, and verified references still outstanding.

CA1 quantitative calibration is a distinct future empirical/experiment-matching task; avoid portraying the synthetic KCl protocol as a direct reproduction.

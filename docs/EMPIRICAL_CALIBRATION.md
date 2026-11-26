# Empirical constraints and CA1 aggregate-data screening

## Electrical transport: independent measurements and explicit conversion

Sources: [Ma et al., 2016, Fig. 4F](https://doi.org/10.1002/glia.22924); [Kiyoshi et al., 2018, Fig. 2F–H](https://doi.org/10.1002/glia.23525). Measured means and SEM: paired astrocyte resistance 4.2 ± 1.5 MΩ (n=3, Na-based pipettes minimizing membrane shunting, isolated doublets separated by 30.5 ± 3.1 µm); CA1 cell density 11238 ± 331 mm^-3, nearest-neighbor distance 39.45 ± 1.14 µm and 8.4 ± 0.16 neighbors (mouse, three animals; different sampling units). The resistance and anatomical samples are distinct.

For an affine imposed voltage gradient E on a uniform isotropically oriented resistor network, dissipation density is n z g d² |E|² /6: n is density, z mean degree (edges counted once), g=1/R, d representative edge length, and the isotropic orientation tensor is I/3. Thus the affine effective conductivity estimate is sigma=n z d²/(6R). It assumes homogeneous edge resistance, representative squared distance, and ignores nonaffine voltage relaxation and anisotropy. For a heterogeneous resistor network the affine energy is a variational upper bound on the homogenized dissipation at prescribed macroscopic voltage; this particular estimate is conditional on assigning the measured doublet resistance to all edges. It is not an experimentally measured conductivity or a rigorous bound for actual CA1.

Result: 5.829909165e-5 S/cm, 0.179881685 of native EDNM glial axial conductivity (3.240968736e-4 S/cm). Varying all four input means to the favorable/unfavorable ±1 SEM endpoints gives 3.856832008e-5 to 1.007146560e-4 S/cm (ratio 0.1190–0.3108); this is an input sensitivity box, not a joint confidence interval.

At native EDNM frozen membrane slope 3.335314393 S/cm³ and ECS conductivity 0.001321300159 S/cm, the two-conductor approximation gives lambda=[g_m(1/sigma_g+1/sigma_e)]^(-1)=40.915 µm (sensitivity 33.520–52.970 µm), versus native 88.335 µm. The slope and ECS conductivity are model quantities, not independently fitted empirical values.

The previously completed [graded migration-only experiment](https://github.com/ipsissima/SMM/actions/runs/45870371151) yields sampled maxima at 37, 60, 100 µm for migration multipliers 0.25, 1, 4 and frozen length estimates 47.84, 88.34, 140.07 µm. It supports the directional length prediction, not an exact equality or universal optimum. An empirically constrained mobility of approximately 0.18 is near the 0.25 regime; however a graph weight changes both migration and diffusion, whereas the previous graded diagnostic varied migration only. Do not interpolate these curves as though the interventions were identical.

[Machine-readable inputs and calculations](../reproducibility/empirical_electrical_constraint.json); [graded sweep](../reproducibility/electrical_length_results.json).

## Potassium experiments: published aggregate values

Source: Breithausen et al., Glia 68:918–931 (2020), DOI [10.1002/glia.23751](https://doi.org/10.1002/glia.23751), sections 2.3–2.4, 3.2–3.3, Figures 4–6. Data are mean ± SEM unless specified. Rat acute CA1 slices, neuronal activity pharmacologically suppressed for iontophoresis. Two different pipette concentrations and pulse durations were used; physical delivered ion mass cannot be inferred solely from applied current without transference and geometry.

| Quantity | Weak | Strong |
|---|---:|---:|
| Iontophoretic current, nA | 800 | 900 |
| Duration, s | 0.2 | 0.5 |
| Pipette KCl, mM | 150 | 3000 |
| Pre-CBX baseline peak, mM | 2.30 ± 0.71 (n=16) | 8.28 ± 1.62 (n=20) |
| Before treatment control-group peak, mM | 2.18 ± 0.56 (n=14) | 9.25 ± 1.65 (n=15) |
| CBX group peak change, percent | +8.60 ± 4.97 (n=16) | +10.41 ± 5.78 (n=20) |
| No-drug control peak change, percent | +0.42 ± 2.99 (n=14) | −6.47 ± 3.13 (n=15) |
| Difference of reported group means, percentage points | 8.18 | 16.88 |

Additional independent targets: spatial peak-decay length 13.08 ± 0.79 µm (n=22); pre-CBX weak single decay time 2.20 ± 0.301 s (n=15); strong fast 1.20 ± 0.115 s and slow 6.79 ± 0.648 s (n=19). Ramp comparisons indicate significant blockade effects beginning around baseline peak elevation 10–15 mM, not a mathematically sharp threshold; duration series begins showing significant effect at 300 ms, current series at 570 nA. These are not exact population response thresholds. Raw matched traces and positions are not publicly linked in the article; the data availability statement directs reasonable requests to the corresponding author. PDF figure screenshot retrieval failed, so no invented digitized individual points are included.

## Prespecified numerical screening (not yet quantitative validation)

Workflow: [corrected driver e131bc9](https://github.com/ipsissima/SMM/commit/e131bc9cb9bf7e5032b925305e19946b31fae19e), [run 45872886590](https://github.com/ipsissima/SMM/actions/runs/45872886590). Initial driver 622bebd was superseded because a local summary variable shadowed the baseline array. Four jobs at two timesteps and two glial mobility conditions (native and the empirical effective-conductivity estimate, applied to both chemical and electrical axial flux under a Nernst–Einstein assumption).

Fifteen 5-µm units on a closed 1D path, source center, recorder at 10 µm as a proxy for the near-source observations. Fit two separate effective KCl rates using only the published pre-CBX pooled amplitude means and respective pulse durations, then hold rates fixed in uncoupled simulations and current/duration ramps. Report spatial profile, fitted decay times, peak response, spike events, mass and convergence diagnostics. Strong-pulse source rate scales with current in the same pipette condition; weak and strong rates are independently fitted, not inferred from nominal current. No post-CBX value participates in fitting.

Limitations: pooled Figure 5 recordings have heterogeneous positions; 10 µm is a proxy, not their known mean distance. The 1D model cannot reconstruct 3D point-source dilution. Neutral KCl mass insertion is not the charged iontophoresis protocol, bath sinks and actual electrode geometry are not represented, and the solver retains its native active neuronal conductances rather than explicitly applying the TTX cocktail. Spike events are reported. Zero coupling is not identical to pharmacological block. The model's native resting potassium also differs from 3 mM in the experiment. Hence any agreement is aggregate-data-informed screening, not a trace-level calibration or a test of SMM's off-support topology hypothesis. No model parameters fitted to blocker effects. Raw data and a more faithful slice model would be needed for a quantitative held-out validation claim.

## Scope

The electrical analysis constrains a plausible parameter regime and links it to a successfully tested mechanism. The ionic simulation is a separate test of the physiological substrate, not an independent test of the distinctive topology hypothesis. Historical EEG analyses are not used as validation.

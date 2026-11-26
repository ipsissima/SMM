# Prospective interpretation of the glial axial-migration counterfactual

Recorded on 2026-11-26 **before inspecting the outputs of the already-dispatched extended scale/migration and matched-dose workflows in this analysis**. The jobs may have completed before this commit; this is an outcome-blind interpretation statement, not a claim of timestamped preregistration before execution or result availability. The preceding independent referee supplied the hypothesis after inspecting the original three native-scale results.

## Hypothesis and competing result

The primary endpoint is the whole-field relative extracellular potassium contrast between the synthetic shortcut and path graphs under the original 0.8 s transmembrane-transfer pulse. At native mobility the full electrodiffusive model shows a strong increase from 5 to 37 micrometres, unlike the maximum near 15 micrometres in the source-reconciled passive model. The scale variable changes the physical graph and its conduits, not the sensor array.

Working mechanism: coupling of the axial glial migration flux to the electrical state changes the length and timescales on which glial topology affects observable extracellular potassium. A scalar change of passive glial diffusivity is not in general an equivalent reduction. Classical current-driven potassium spatial buffering is not novel; the proposed new result concerns the scale of *topology discrimination*.

Prediction: with glial axial electromigration set to zero (chemical diffusion, ECS transport, membrane kinetics, charge constraints and other processes retained), the sampled maximum of the short-pulse relative K contrast should move to the 5–15 micrometre range. In particular, the contrast at 15 micrometres should exceed the contrast at 37 micrometres, or the maximum should lie at 5 micrometres. The native sampled maximum is expected above 15 micrometres, with eventual decline in the extended 60–150 micrometre range. Treat the latter expectation as empirical within this solver; the passive mobility theorem does not prove the shape of the full nonlinear sweep.

If the migration-off sampled maximum remains at 37 micrometres or beyond, the specified migration-based prediction fails. A reduction in magnitude alone does not establish the proposed scale-setting mechanism. A partial shift supports a contribution, not exclusive determination. Compare both timesteps, absolute differences and path response norms, and preserve any negative outcomes. This counterfactual changes the constitutive transport law and is not a selective physiological blocker.

If the prediction is supported, an independent parameter intervention is needed before claiming an electrotonic length explanation: vary a specified glial conductance parameter while keeping other transport coefficients fixed as far as possible; identify the electrical mode and its spatial decay or cable length from the coupled linearized system; test whether the topology contrast maximum tracks it. No assertion that a scalar cable length suffices is made beforehand.

## Matched-dose endpoints

For the neutral KCl pulse at 37 micrometres, use the relative whole-field K contrast, and separately absolute K and voltage contrasts. Nonlinear amplification requires an increase in relative contrast between the identical 3 and 30 mM tissue/s KCl protocols. The matched total edge weight (equivalent to Laplacian trace) is a restricted rival, so its residual is an upper bound on the minimum attainable shared-support residual, not a lower bound. Glial-off controls test a coupling-strength effect independently of graph support. Report spike events and sustained depolarization without inferring depolarization block from absence of spikes. Distal membrane-current accounting and volume fractions are needed for mechanistic attribution.

Workflows under interpretation: [scale/migration](https://github.com/ipsissima/SMM/actions/runs/45869133888), [matched dose](https://github.com/ipsissima/SMM/actions/runs/45869212368). Earlier independent referee calculations are motivating evidence supplied in the report, not a reproduced archive of code.

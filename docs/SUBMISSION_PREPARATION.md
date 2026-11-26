# eLife submission preparation (working, not submitted)

## Editorial positioning

Primary contribution: an identifiable and falsifiable mathematical hypothesis about independent astroglial ionic transport geometry, together with a full electrodiffusive demonstration of the electrical mechanism determining its observable scale. This is a computational/theoretical neuroscience article, not a new experimental demonstration of astroglial anatomy or validation of a brain-wide EEG mechanism. The central predicted 30-µm intrinsic-scale maximum is conditional on the synthetic graph and specified short pulse, and the 0.81 mM/0.60 mV contrast is against the total-weight-matched path under a different neutral KCl challenge. Do not conflate these protocols, their endpoints, or the effective-medium estimate with a direct measurement.

## Draft editorial letter (update after the decisive fits)

Dear Editors,

I would like to submit “The Syncytial Mesh Model: Identifying Independent Astroglial Transport Geometry in Neuronal–Ionic Dynamics” for consideration as a Research Article in eLife.

Astroglial networks contribute to potassium buffering and regulate neuronal excitability, but detecting such effects does not establish that their spatial organization is functionally distinct from extracellular transport. This manuscript addresses that distinction as an inverse problem: under which conditions can spatial ionic responses identify a separate astroglial transport operator, and how can experiments discriminate independent transport geometry from alternative models with the same extracellular support?

The work combines exact conditional recovery and sensitivity results for coupled transport operators with conservative simulations using an independently developed electrodiffusive neuron–extracellular–glia model. The simulations reveal an intermediate regime in which graph geometry is observable, and an intervention on electrical migration identifies a mechanism controlling its spatial scale. An effective-medium estimate informed by published astrocytic anatomy and paired-cell electrical resistance motivates a short-pulse prediction near 30 µm intrinsic scale, corroborated in the full model. A separate neutral-KCl challenge yields substantial differences in extracellular potassium and neuronal membrane potential relative to strength-matched alternatives. Fixed-tissue calculations show why source and recording coverage matter for a discriminating experiment.

The manuscript explicitly separates these mathematical and computational results from direct biological evidence. Its aggregate CA1 screening underestimates experimental uncoupling responses, and the article treats that discrepancy as a limitation rather than successful quantitative validation. The independent-geometry prediction is formulated for prospective experiments with independently constrained transport and local physiology, multiple source locations and withheld predictions.

[Before submitting: insert the final bounded shared-support comparison, its held-out separation and numerical refinement, with exact limits on the tested family; replace this note with the verified data DOI and code-archive identifiers.]

I believe that the combination of graph-based identifiability, electrical spatial buffering and experimentally motivated source–sensor design will be of interest to readers working on computational neuroscience and astroglial regulation of neuronal dynamics.

Best regards,
Andreu Ballús Santacana
Universitat Autònoma de Barcelona

## Final submission gates

- Inspect empirical bounded rival optimization, choose by training loss, report held-out errors and convergence status honestly; include frozen-candidate timestep comparison. Native exploratory unbounded-ish fits are secondary.
- Integrate edge-volume convention sensitivity and all relevant negative results. Check every numeric claim against the archived output, distinguish the illustrative whitened distance from statistical significance, and confirm no stale priority claims or conflicting table labels.
- Freeze final source and generate final typeset main article, supplement and figure files; inspect actual rendered pages. Confirm corresponding author metadata, affiliations, author declarations, funding, competing interests, and any fee-waiver request with the author.
- Complete Zenodo dataset deposit, confirm record DOI resolves and files match checksums. A draft GitHub release or reserved DOI alone is not a public deposit.
- Submit only the final version. eLife currently selects papers for expert review; reviewed papers are published as Reviewed Preprints and assessed publicly. Direct submission or transfer from bioRxiv/medRxiv is possible. Official workflow: https://elifesciences.org/about/submit-your-research .

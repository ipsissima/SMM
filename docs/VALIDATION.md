# Scientific interpretation and validation status

## Geometry versus observation

Version 3's parameter d rescales the intrinsic tissue graph: both diffusion operators contain d^-2, and a chord between nodes i and j retains its weight as the physical distance |i-j|d changes. Consequently the 5, 14.76 and 36.98 micrometre numerical comparisons are between geometrically scaled model tissues, not between electrode layouts on a fixed tissue. The fixed chord weights imply effective chord diffusivity w|i-j|^2 Dg: 12.8, 8 and 9.6 times the nominal glial diffusivity for the three four-step chords. This is a synthetic high-conductance conduit assumption.

Useful dimensionless controls are Dg/(d^2 kappa_a), De/(d^2 kappa_a), kappa_e/kappa_a, kappa_N/kappa_a, compartment fractions, mobility multiplier, and kappa_a times protocol duration. Source and sensor placement on a fixed tissue is a distinct design problem; cover both ends of candidate conduits and compare distances with the extent of extracellular spread during the actual forcing protocol. The short pulse and sustained step are different inputs, not just different observation durations.

The passive findings and their published-in-repository numerical values retain their conditional meaning. The manuscript files preserve the version-3 record; read this clarification alongside them.

## Independent full-physics checks

The pinned original EDNM solver is [Sætra–Mori 2024](https://doi.org/10.1371/journal.pcbi.1012114), source commit [826c211](https://github.com/martejulie/electrodiffusive-network-model/tree/826c211c70d9027a4acb02202ca91a60de79a15d). The [graph adapter](https://github.com/ipsissima/SMM/tree/d58918b6fef367f2d4676621eb643a5b0f82c3b2/validation/graph_extensions) and [pump/high-load workflow](https://github.com/ipsissima/SMM/tree/3798c749cdbed18e4222dc805e94f28a71cde5cb) are pinned. The two-unit high-load simulation reproduces a published nonlinear coupling-strength result, not a topology-specific positive control.

A prespecified full-model geometric-scaling comparison at 5, 15 and 37 micrometres and two timesteps is recorded at [541b642](https://github.com/ipsissima/SMM/tree/541b6429ea1cadbe9c5a83916d854f1e4c361318), [workflow run 45863596046](https://github.com/ipsissima/SMM/actions/runs/45863596046). All six jobs have reported successful execution; the quantitative outcomes must be checked against their saved arrays before interpreting them.

## Empirical scope

Anatomical and physiological studies support the motivation for SMM, but the topology-specific prediction has not been demonstrated in a biological preparation. The full-source CA1 large-load experimental calibration and a nonlinear seven-site topology-specific positive control remain separate research tasks. GitHub Actions artifacts are time-limited; a persistent external data deposit with a DOI has not been made.

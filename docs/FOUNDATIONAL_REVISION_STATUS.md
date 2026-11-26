# Foundational revision status — 26 November 2026

Canonical working source: [manuscript.md](../paper/manuscript.md), [supplement.md](../paper/supplement.md). Historical TeX remains unsynchronized until the final build. The manuscript now incorporates the completed fixed-tissue refinement, the empirically informed topology simulations (selected verified central/end-point cases), expanded single-source rival diagnostics, joint-source glial-only result, and explicitly incomplete quantitative agreement of the aggregate CA1 screen.

Completed:
- [32 empirical topology configurations](https://github.com/ipsissima/SMM/actions/runs/45874526639), 192 paired trajectories.
- [Fixed-tissue refinement](https://github.com/ipsissima/SMM/actions/runs/45874607462), [full diagnostics](../reproducibility/fixed_tissue_results.json).
- [Four single-source rival optimization jobs](https://github.com/ipsissima/SMM/actions/runs/45870291419). Glial-only fits converged; glial+ECS candidates exhausted evaluation limit. See [initial](../reproducibility/nonlinear_glial_only_fits.json) and [extended](../reproducibility/nonlinear_extended_fits.json) diagnostics.
- [CA1 aggregate screen](https://github.com/ipsissima/SMM/actions/runs/45872886590), three of four jobs complete at last check; fourth not yet used. Predictions underestimate the uncoupling effects. No raw-trace retrieval or direct biological topology validation.
- [Joint-source high-load fit](https://github.com/ipsissima/SMM/actions/runs/45874768150): glial-only case complete, two glial+ECS fits still running at last check.

Before submission: full machine-readable archive of new outcomes; inspect both remaining joint fits and final CA1 result; verify fitted candidates at refined timestep; final reference and text audit, regenerate PDF/TeX and supplement with visual QA; permanent data repository (no DOI claimed). No new experiments beyond these checks are needed unless a material result requires correction.

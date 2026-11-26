# Foundational revision status — 26 November 2026

Canonical working source: [manuscript.md](../paper/manuscript.md), [supplement.md](../paper/supplement.md). Historical TeX remains unsynchronized until the final build. The manuscript now incorporates the completed fixed-tissue refinement, the empirically informed topology simulations (all 32 cases audited, including regime-dependent nonlinear attenuation), expanded single-source rival diagnostics, joint-source glial-only result, and explicitly incomplete quantitative agreement of the aggregate CA1 screen.

Completed:
- [32 empirical topology configurations](https://github.com/ipsissima/SMM/actions/runs/45874526639), 192 paired trajectories.
- [Fixed-tissue refinement](https://github.com/ipsissima/SMM/actions/runs/45874607462), [full diagnostics](../reproducibility/fixed_tissue_results.json).
- [Four single-source rival optimization jobs](https://github.com/ipsissima/SMM/actions/runs/45870291419). Glial-only fits converged; glial+ECS candidates exhausted evaluation limit. See [initial](../reproducibility/nonlinear_glial_only_fits.json) and [extended](../reproducibility/nonlinear_extended_fits.json) diagnostics.
- [CA1 aggregate screen](https://github.com/ipsissima/SMM/actions/runs/45872886590), all four jobs complete and archived. Predictions underestimate the uncoupling effects. No raw-trace retrieval or direct biological topology validation.
- [Joint-source high-load fit](https://github.com/ipsissima/SMM/actions/runs/45874768150): glial-only case complete, two glial+ECS fits still running at last check.

Before submission: full machine-readable archive of new outcomes; inspect both remaining joint fits and final CA1 result; verify fitted candidates at refined timestep; final reference and text audit, regenerate PDF/TeX and supplement with visual QA; permanent data repository (no DOI claimed). No new experiments beyond these checks are needed unless a material result requires correction.

## Archive and build update

The [staging archive](https://github.com/ipsissima/SMM/actions/runs/45880525262) completed successfully: 17 assets, 934,069,451 bytes, including raw core trajectories, SHA-256 manifests and the full 32-case empirical topology comparison, in [a draft release](https://github.com/ipsissima/SMM/releases/tag/untagged-499c601e145e7c249bbe). This is not public permanent DOI archiving; append CA1 and joint-fit/refinement results before final release. The first manuscript build failed due to missing lmodern.sty; the dependency was added. A final visual check remains necessary. [Frozen candidate refinement workflow](../.github/workflows/refine-frozen-rivals.yml) has been configured to trigger after the joint-source workflow completes, with candidate selection based only on training cost; it has not yet run.

## Final referee revision

Explicit priority comparisons removed from the article (historical provenance retained separately); exact graph electrical buffering operator and two-extremes proposition added; onset scope corrected for neutral KCl; 5-µm comparison, two-site pump and native rival details moved to the supplement. Abstract now centers the empirical conductivity and conditional 30-µm scaling hypothesis. Main table order corrected. Zhang et al. verified directly in PubMed PMID 42285788 and publisher DOI 10.1016/j.tcb.2026.05.005.

New decisive tests: [empirical bounded joint-source rival fit](https://github.com/ipsissima/SMM/actions/runs/45881427345), four jobs at mobility 0.17988, edge caps 3 or 10 times the upper affine conductivity sensitivity endpoint and ECS 0.7–1.3; these are sensitivity assumptions, not direct per-edge measurements. Graceful evaluation deadline preserves the best completed candidate. [Targeted electrical peak refinement](https://github.com/ipsissima/SMM/actions/runs/45881475451), 43 configurations spanning migration-only peaks and joint Nernst–Planck scaling at the empirical central/endpoint values. Pending integration; no outcome claimed. The fast electrical onset is a conditional reduced-model prediction, not yet verified in EDNM. The archive is a staged GitHub release; Zenodo DOI remains pending. No submission before final integration, numerical refinement, archive completion, and review of rendered final article.


# SMM PSD Pipeline

Reproducible workflow for comparing Syncytial Mesh Model (SMM) simulation spectra with empirical EEG data.

**Contact:** Andreu.Ballus@uab.cat

## Steps

1. Preprocess EEG to PSD: `python preprocess_eeg_psd.py`
2. Simulate mesh PSD: `python simulate_mesh_psd.py`
3. Compare: `python compare_psd.py`
4. Plot/visualize: `python plot_psd_comparison_results.py`

- All scripts are set for the Google Colab/MNE workflow and expect your drive structure as used above.
- Example CSVs and `PSD_with_Coherence.csv` can be added as supplementary data.

## Requirements

```
pip install mne mne-bids numpy pandas matplotlib seaborn scipy
```

---

## Citation
If you use this code, please cite the associated preprint.


## Bifurcation probe

The mesh bifurcation probe projects the SMM dynamics onto the softest eigenmode
of the linearised operator and fits the resulting equilibria to the Thom cubic
normal form `u*A^3 + alpha*A + s*h = 0`.  This reveals the cusp wedge in the
`(kappa, h)` plane and quantifies how the external drive tilts the bifurcation.

### Running locally

Install the scientific dependencies once:

```bash
pip install -r requirements.txt
```

Then launch the default probe (32×32 grid, 13 coupling samples, 33 field
samples):

```bash
python bifurcation_probe.py --nx 32 --ny 32
```

The smoke-test shortcut `python bifurcation_probe.py --smoke` performs a small
deterministic scan suitable for CI or quick validation.

### Outputs

Numerical artifacts live in `bifurcation_results/`:

* `cusp_grid_summary.csv` summarises each `kappa` with the leading eigenvalue,
  hysteresis span, and fitted Thom parameters.
* `hysteresis_kappa_*.csv` and `mode_kappa_*.npy` store amplitude traces and
  eigenmodes.
* `discriminant_kappa_*.csv` tabulates the Thom discriminant `Δ(h)` for each
  coupling.
* `figures/` contains publication-ready PNGs of the eigenvalue, hysteresis, and
  parameter trends.

Regenerate plots from archived CSVs at any time with:

```bash
python plot_bifurcation_summary.py --outdir bifurcation_results --hysteresis
```

## Bifurcation probe (advanced)

The refined probe and analyzer introduced in this update implement the full
three-stage cusp interrogation requested for the mesh model:

1. **Phase-tracked projection** keeps the leading mesh eigenmode aligned across
   every field sample so that `A(h)` is continuous even when the eigenvector is
   nearly degenerate.
2. **Targeted densification** refines the `h` grid around the origin and the
   `kappa` grid around the softest eigenvalue, performs scaled SVD nullspace
   fits with bootstrap confidence intervals, and reports discriminant maps and
   cusp half-widths.
3. **Two-mode fallback** automatically fits a steady two-mode normal form when
   the spectral gap collapses and flags the result in the summary table.

### Running the refined probe

```bash
python bifurcation_probe_refined.py \
    --nx 32 --ny 32 --L 32 \
    --kappa_min 0.0 --kappa_max 0.6 --n_kappa_coarse 13 \
    --n_kappa_refined 13 --refine_factor 0.15 \
    --h_min -0.8 --h_max 0.8 --n_h_coarse 33 \
    --n_h_band 101 --h_band 0.2
```

Key CLI options include adaptive relaxation controls (`--dt`, `--tol`,
`--Tmax_min`, `--tau_factor`), bootstrap size (`--bootstrap_B`), and the
two-mode trigger threshold (`--gap_threshold`).  Deterministic smoke testing is
available with `python bifurcation_probe_refined.py --smoke`, which finishes in
under ten minutes on CI hardware.

### Analyzer and outputs

The probe stores all artifacts under `bifurcation_results_refined/`:

* `cusp_grid_summary_refined.csv` gathers eigenvalues, hysteresis metrics, Thom
  coefficients with confidence intervals, and two-mode diagnostics.
* `hysteresis_kappa_*.csv` record aligned up/down sweeps; corresponding
  `modes/modes_kappa_*.npz` files store the phase-tracked eigenmodes.
* `discriminant_kappa_*.csv` tabulate the discriminant `Δ(h)` and scaled drive
  `β = s h` for each coupling.
* `figures/` includes the λ–κ, hysteresis, α–κ, and per-κ discriminant plots.
* `bifurcation_verdict_refined.pdf` summarises whether a cusp was detected and
  reports the most reliable κ band and half-width estimates.

Re-run the post-processing on archived data at any point with:

```bash
python analyze_bifurcation_results_refined.py --indir bifurcation_results_refined
```

Use `--refresh` to recompute the summary from the saved hysteresis sweeps with a
custom bootstrap count.

## Two-mode coherence fitting (supplementary)

The supplementary two-mode analysis reconstructs the empirical probability of
observing significant coherence across spatial scales and fits a linear
decoherence-rate model with bootstrap confidence bands.  The implementation
falls back to a saturating probability link because the manuscript PDF could
not be parsed in this environment; the exact formula can be updated by passing
`--note-pdf` if a PDF parser is available locally.

### Quickstart

Install the scientific dependencies once:

```bash
pip install -r requirements.txt
```

Run the smoke test (fast, coarse grid):

```bash
bash RUN_TWO_MODE.sh
```

The script internally calls:

```bash
python supplementary_code/two_mode_fitting.py \
  --input PSD_with_Coherence.csv \
  --outdir supplementary_code/two_mode_results_smoke \
  --B 100 --seed 1234 --threshold 0.05 --assume-constant --smoke
```

Run the full analysis (dense grid, B=1000):

```bash
python supplementary_code/two_mode_fitting.py \
  --input PSD_with_Coherence.csv \
  --outdir supplementary_code/two_mode_results \
  --B 1000 --seed 1234 --threshold 0.05 --assume-constant
```

Outputs include figures of the empirical P(L) with model overlays
(`P_L_fit.png`), bootstrap parameter histograms (`parameter_bootstrap.png`),
CSV tables of parameter estimates and bootstrap samples, and a concise
`two_mode_fit_report.txt` with AIC/BIC comparison against a saturating-rate
alternative.  The `--assume-constant` flag reuses the per-subject coherence
value across scales when no per-scale CSV is available; provide
`PSD_with_Coherence_by_scale.csv` to use scale-specific measurements instead.

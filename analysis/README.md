# SMM → Thom diagnostic

This folder contains scripts to perform the numerical check that a Syncytial Mesh Model (SMM)
can be reduced to a Thom-style cusp normal form (i.e., shows bistability/hysteresis under slow control).

Files:
- `utils.py` : spatial Laplacian and eigenmode extraction
- `simulation_engine.py` : conservative RK4 mesh simulator (same stencil as repo)
- `eigenmode_projection.py` : modal projection & validation utilities
- `identify_coeffs.py` : numeric identification of μ, γ, δ from short runs
- `hysteresis_sweep.py` : stepwise sweep test and plotter for hysteresis
- `run_analysis.py` : orchestration script; saves results to `analysis_results/`

Run:
```bash
python3 analysis/run_analysis.py --outdir analysis_results
```

Outputs in analysis_results/:

e_star.npy, coeff_samples.csv, hysteresis_up.csv, hysteresis_down.csv, hysteresis.png, diagnostics.json

---

## 3) Run the analysis

You already saw the commands above. Run:

```bash
python3 analysis/run_analysis.py
```

This will:

compute eigenmode and save it,

run modal validation and produce modal_fraction,

identify μ,γ,δ by numeric experiments and save coeff_samples.csv,

perform a slow stepwise hysteresis sweep and save hysteresis_up.csv, hysteresis_down.csv, hysteresis.png,

place a diagnostics.json with eigenvalues and quick diagnostics.

Runtime & resources: with the default parameters the script is modest (a few minutes on CPU). If you run on Colab, the runtime is fine. If you need faster runs, reduce TperStep value in run_analysis.py or the sweep resolution.

4) How Codex should validate correctness & robustness

When the script finishes, open analysis_results/diagnostics.json and check:

"small_eigenvals": there should be a smallest positive eigenvalue separated from the next one. Conservative test: vals_sorted[1] / vals_sorted[0] > 1.5 indicates a usable spectral gap. If not, do a 2-mode reduction (see notes below).

"modal_fraction" (the frac): should be > 0.7 for single-mode slaving to be plausible. If frac < 0.5, the single-mode reduction is invalid.

"coefficients": gamma should be positive and stable across samples.

Check visually analysis_results/hysteresis.png: if the up- and down-curves show distinct jump points (clear loop), the cusp-type hysteresis is present under the chosen control (here, gamma_bg). If the curves overlap or show random jitter only, increase TperStep (slower ramp) and repeat, or reduce noise.

5) If Codex must implement a 2-mode fallback

If spectral gap is small or multiple modes are necessary, do the same steps but project onto two leading eigenvectors e1,e2, and use a 2D reduction. The code structure supports this extension: replace the scalar projection with vector projection and fit a quadratic/cubic vector polynomial for the reduced flow, then use continuation (numerical elimination) to locate swallowtail/umbilic singularities. I can produce this code on request.

6) Optional: create a runnable Colab notebook programmatically

If you prefer an .ipynb Colab, Codex can convert run_analysis.py into a notebook with the cells I provided earlier. A simple way is to create a notebook that shells out to run python3 analysis/run_analysis.py, or to copy cells into a notebook. If you want I can output a .ipynb JSON to commit.

7) Commit & PR instructions (repeated for convenience)
git add analysis
git commit -m "Add Thom–SMM diagnostic analysis"
git push -u origin analysis/thom-check
# Then open a PR via GitHub UI or via hub/gh CLI.

8) What success looks like (conservative, rigorous)

Mathematics success: the script runs without errors, identifies a stable positive gamma (cubic saturation), demonstrates frac > 0.7 and spectral gap > 1.5.

Neuroscience/empirical success: clear hysteresis loop in hysteresis.png that persists when ramp is slowed and when small perturbations/noise are added; fold points reproducible across repeats.

Failure diagnosis: if any of the diagnostic checks fail, the script’s output tells you why — and the next step is either (a) do 2-mode reduction (swallowtail) or (b) check for Hopf regime (complex pair) and analyze Stuart–Landau/Hopf codimension-2 phenomena instead.

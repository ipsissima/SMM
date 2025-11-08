
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

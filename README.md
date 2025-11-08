
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


## Thom cusp bifurcation probe

The repository now contains a mesh bifurcation probe that projects the SMM
dynamics onto the softest linear mode and verifies the appearance of a Thom
cusp-type catastrophe.  The workflow is fully scripted in `bifurcation_probe.py`
and produces numerical tables as well as diagnostic plots.

### Quick smoke run

The default parameters execute a small stability scan that finishes quickly on a
laptop.  Run the probe from the repository root:

```bash
python bifurcation_probe.py
```

Outputs are written to `bifurcation_results/` and include a
`cusp_grid_summary.csv` file plus one hysteresis CSV per sampled coupling
strength `kappa`.  Disable figures during automated runs by adding
`--no-plots`.

### Full cusp mapping

To explore the full cusp wedge suggested in the project prompt, increase the
resolution of the parameter sweeps.  Example:

```bash
python bifurcation_probe.py \
  --nx 32 --ny 32 --kappa-min 0.0 --kappa-max 0.6 --n-kappa 13 \
  --h-min -0.8 --h-max 0.8 --n-h 33 --tmax 120.0
```

The script reports the smallest eigenvalue of the effective operator, the peak
hysteresis amplitude difference `max |A_up - A_down|`, and the fitted Thom
normal-form coefficients.  The coefficient `alpha_fit(kappa)` traces how the
projected linear stiffness changes with coupling, while the field `h` plays the
role of `beta` in the reduced cubic equation `u*A^3 + alpha*A + beta = 0`.

### Plotting results separately

`plot_bifurcation_summary.py` regenerates the summary figures from saved CSVs.
It is useful when working from archival data:

```bash
python plot_bifurcation_summary.py --results-dir bifurcation_results --kappa 0.30
```

This command reproduces the cusp wedge and alpha trajectories, and generates a
hysteresis plot for the specified `kappa` if available.

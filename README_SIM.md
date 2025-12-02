# Improved Damped Wave PDE Model - Simulation Guide

## Overview

This module implements a reproducible 2D damped wave PDE simulation with:
- Perfectly Matched Layer (PML) absorbing boundaries
- Region-based source mapping with Gaussian spatial kernels
- Ensemble simulations with white noise
- Power spectral density (PSD) computation

## Quick Start

### Running the Simulation

The main simulation script reads parameters from a YAML file and runs an ensemble of simulations:

```bash
python simulate_mesh_psd.py --params params.yaml --ensemble 20
```

### Command-Line Options

- `--params <file>`: Path to YAML parameter file (default: `params.yaml`)
- `--ensemble <N>`: Override ensemble size from params file
- `--sim-duration <T>`: Override simulation duration in seconds
- `--output <dir>`: Output directory (default: `results`)
- `--quiet`: Suppress progress output

### Example Commands

```bash
# Run with default settings (50 ensemble members, 10s duration)
python simulate_mesh_psd.py

# Quick test with fewer ensemble members
python simulate_mesh_psd.py --ensemble 5 --sim-duration 3.0

# Save to custom directory
python simulate_mesh_psd.py --output my_results/
```

## Output Files

Results are saved to the specified output directory (default: `results/`):

| File | Description |
|------|-------------|
| `psd_median.csv` | Median PSD across ensemble (columns: `f`, `Pxx_median`) |
| `psd_all.npy` | All PSD arrays from ensemble (shape: `[ensemble_size, n_freq]`) |
| `psd_frequencies.npy` | Frequency array for PSD |
| `metadata.json` | Simulation parameters, git commit, and run information |

## Parameters

The simulation is configured via `params.yaml`:

```yaml
L_mm: 32.0              # Domain size (mm)
dx_mm: 0.5              # Spatial resolution (mm)
c_mm_per_s: 15.0        # Wave speed (mm/s)
gamma_s: 0.5            # Background damping (1/s)
dt_s: 0.001             # Time step (s)
sim_duration_s: 10.0    # Simulation duration (s)
pml_width_mm: 3.0       # PML layer width (mm)
pml_sigma_max: 50.0     # Maximum PML damping
sigma_kernel_mm: 1.0    # Gaussian source kernel width (mm)
N_regions: 16           # Number of source regions
ensemble_size: 50       # Number of ensemble members
psd_fs: 250             # PSD sampling rate (Hz)
psd_nperseg: 512        # Welch's method segment length
psd_noverlap: 256       # Welch's method overlap
noise_sigma_mesh: 1e-6  # White noise amplitude
seed: 42                # Random seed for reproducibility
```

### Key Physics Parameters

- **Grid**: Computed as `nx = ny = L_mm / dx_mm` (64×64 for defaults)
- **CFL Number**: For stability, `c*dt/dx < 0.5` (CFL = 0.03 for defaults)
- **Eigenfrequencies**: For mode n, `f_n ≈ n * c / (2 * L)` (e.g., f_17 ≈ 3.984 Hz)

## Implementation Details

### PDE Formulation

First-order form of damped wave equation:
```
du/dt = v
dv/dt = c² ∇²u - γ_total v + S(r,t) + η(r,t)
```

where:
- `u`: displacement field
- `v`: velocity field
- `∇²`: 9-point Laplacian stencil
- `γ_total = γ_background + σ_PML(r)`: total damping
- `S(r,t)`: external source (region-based)
- `η(r,t)`: white noise

### Numerical Methods

- **Integration**: 4th-order Runge-Kutta (RK4)
- **Laplacian**: 9-point stencil with vectorized implementation
- **Boundaries**: PML with quadratic profile `σ = σ_max(d/w)²`
- **Source**: Gaussian convolution from discrete regions
- **Noise**: White noise scaled as `σ/√dt` for proper statistics

### Source Mapping

Sources are placed on a coarse regular grid (√N_regions × √N_regions) and smoothed
with a Gaussian kernel. The current implementation uses a test pattern (4 Hz burst
at the center region for t ∈ [0,1]s), but the structure allows easy replacement
with neural mass model outputs.

## Analysis Tools

See `analysis/metrics.py` for signal analysis functions:

- `compute_welch_psd(signal, fs, nperseg, noverlap)`: PSD computation
- `compute_phase_gradient_coherence(signals, positions, fs, band, ...)`: Phase coherence metric

Example:
```python
from analysis.metrics import compute_welch_psd
import numpy as np

# Load PSD data
psd_all = np.load('results/psd_all.npy')
f = np.load('results/psd_frequencies.npy')

# Compute statistics
psd_mean = psd_all.mean(axis=0)
psd_std = psd_all.std(axis=0)
```

## Running Tests

Run all tests:
```bash
pytest -q
```

Run specific test modules:
```bash
# Test eigenfrequency prediction
pytest tests/test_eigenfreqs.py -v

# Test phase gradient coherence
pytest tests/test_phase_gradient_coherence.py -v
```

**Note**: `test_eigenfreqs.py` runs a reduced simulation (8 members, 3s duration)
and will be skipped on CI if the environment variable `CI=true` is set.

## Expected Results

With default parameters (`params.yaml` with `dx=0.5 mm`):

- **Grid**: 64×64
- **Peak frequency**: PSD should show elevation near f_17 ≈ 3.984 Hz
- **Runtime**: ~2-5 minutes for 50 ensemble members (depends on hardware)

The theoretical eigenfrequency is:
```
f_n = n * c / (2 * L) = 17 * 15 / (2 * 32) ≈ 3.984 Hz
```

## Troubleshooting

### Stability Issues

If you see `NaN` or exponentially growing values:
- Check CFL number: `c*dt/dx` should be < 0.5
- Reduce `dt_s` or increase `dx_mm`
- Increase `gamma_s` or `pml_sigma_max`

### Memory Issues

For large grids or ensembles:
- Reduce `ensemble_size`
- Reduce `sim_duration_s`
- Increase `dx_mm` (coarser grid)

### No Clear PSD Peaks

- Increase `sim_duration_s` for better frequency resolution
- Increase `ensemble_size` to reduce noise in median PSD
- Check `pml_width_mm` is sufficient (should be at least 3-5 grid points)

## Contact

For questions or issues, contact: Andreu.Ballus@uab.cat

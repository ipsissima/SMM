# Glial Field Mode Analysis and Ablation Studies

This document describes the new analysis scripts and ablation modes added to the SMM repository for studying telegraph dynamics and performing model comparisons.

## New Files

### 1. Analysis Scripts

#### `analysis/compute_thom_coefficient.py`
Computes the Thom cubic coefficient (α) for the glial telegraph operator by projecting the cubic nonlinearity u³ onto the adjoint eigenmode.

**Usage:**
```bash
# Using params.yaml
python analysis/compute_thom_coefficient.py --params params.yaml

# Or with explicit parameters
python analysis/compute_thom_coefficient.py --Nx 64 --Ny 64 --L_mm 32 --c 7.07 --gamma 1.25 --omega0 0.0
```

**Output:**
- Eigenvalue closest to 0
- Adjoint eigenvalue
- Thom coefficient α (complex and real parts)

**Notes:**
- Uses shift-invert method to find eigenpairs near σ=0
- For large grids, computation may take time (use smaller grids for prototyping)
- If α has significant imaginary part, the critical eigenpair is complex (Hopf-like)

#### `scripts/plot_dispersion_relation.py`
Computes and plots telegraph dispersion relation diagnostics.

**Usage:**
```bash
python scripts/plot_dispersion_relation.py
```

**Output:**
- `results/dispersion_summary.png` containing:
  - Re(ω(k)) and Im(ω(k)) vs wavenumber k
  - Group velocity v_g(k)
  - Attenuation length vs frequency
  - Phase wavenumber vs frequency
  - 4 mm wavelength scale marker

**Interpretation:**
- The attenuation plot shows which frequencies have long spatial reach
- Use for "Goldilocks" scale-selection diagnostics
- Customize parameters in the script or load from params.yaml

### 2. Glial Field Mode Ablations

The `GlialField` class now supports three different dynamics modes for ablation experiments:

#### Mode: `'telegraph'` (default)
Hyperbolic wave equation with finite propagation speed:
```
∂²ψ/∂t² + 2γ₀·∂ψ/∂t + ω₀²·ψ - c_eff²·Δψ = S(r,t) + ξ(r,t)
```

#### Mode: `'diffusion'`
Parabolic diffusion equation:
```
∂ψ/∂t = D·Δψ - γ·ψ + S(r,t) + ξ(r,t)
```

#### Mode: `'mean_field'`
Spatially uniform ODE (scalar telegraph equation for spatial mean):
```
∂²⟨ψ⟩/∂t² + 2γ₀·∂⟨ψ⟩/∂t + ω₀²·⟨ψ⟩ = ⟨S⟩ + ⟨ξ⟩
```

### 3. Configuration

Add to `GlialFieldConfig`:
```python
mode: str = 'telegraph'  # 'telegraph' | 'diffusion' | 'mean_field'
diffusion_D_mm2_per_s: float = 0.1  # Diffusion coefficient (only for 'diffusion' mode)
```

Or in `params.yaml`:
```yaml
mode: 'telegraph'
diffusion_D_mm2_per_s: 0.1
```

### 4. Demo Script

#### `scripts/demo_glial_modes.py`
Demonstrates all three modes with the same initial condition and source.

**Usage:**
```bash
python scripts/demo_glial_modes.py
```

**Output:**
- Visual comparison plot showing all three modes at different time points
- Console output with statistics for each mode

## Running Ablation Experiments

### Example: Compare modes with same parameters

```python
import numpy as np
from smm.glia import GlialFieldConfig, GlialField

# Telegraph mode
config_telegraph = GlialFieldConfig(
    Lx=32.0, Ly=32.0, Nx=64, Ny=64,
    mode='telegraph', c=15.0, gamma=0.5
)
glia_telegraph = GlialField(config_telegraph)

# Diffusion mode
config_diffusion = GlialFieldConfig(
    Lx=32.0, Ly=32.0, Nx=64, Ny=64,
    mode='diffusion', diffusion_D_mm2_per_s=0.5, gamma=0.5
)
glia_diffusion = GlialField(config_diffusion)

# Mean-field mode
config_mean_field = GlialFieldConfig(
    Lx=32.0, Ly=32.0, Nx=64, Ny=64,
    mode='mean_field', gamma=0.5, omega0=0.0
)
glia_mean_field = GlialField(config_mean_field)

# Run all modes and compare
# ... (add your analysis code)
```

### Expected Differences

**Power Spectral Density (PSD):**
- Telegraph: Peak near physiological band (3-6 Hz)
- Diffusion/Mean-field: No characteristic frequency peak

**Phase Gradient Coherence (PGC):**
- Telegraph: Frequency-dependent spatial coherence
- Diffusion: Smooth diffusive patterns
- Mean-field: No spatial gradients (spatially uniform)

**Plasticity:**
- Telegraph: Spatially-localized potentiation patterns
- Diffusion/Mean-field: Diffuse or uniform patterns

## Testing

New comprehensive tests are in `tests/test_glial_modes.py`:
```bash
# Run mode tests
python -m pytest tests/test_glial_modes.py -v

# Run all glial tests
python -m pytest tests/test_glial_field.py tests/test_glial_modes.py -v
```

## Notes

1. **Backward Compatibility**: Default mode is `'telegraph'`, so existing code works unchanged
2. **Micro-parameters**: All modes work with micro-derived or explicit parameters
3. **Numerical Stability**: 
   - Telegraph: RK4 (stable for reasonable dt)
   - Diffusion: Explicit Euler (stable for small dt, consider implicit for larger dt)
   - Mean-field: RK2 for scalar ODE (stable)

## References

The implementation follows the detailed specifications provided in the problem statement, including:
- 9-point Laplacian with Neumann BC via reflection
- Block operator construction for eigenpair computation
- Telegraph dispersion relation diagnostics
- Minimal-change design preserving existing functionality

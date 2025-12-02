# Telegraph Physics Refactoring Summary

## Overview
This document summarizes the refactoring of `bifurcation_probe_refined.py` to use Telegraph Equation physics from the glial kinetics model instead of the old massless Wave Equation.

## Motivation
The core simulation engine (`src/smm/glia.py`) was previously upgraded to use the Telegrapher's Equation derived from glial IP₃/Ca reaction-diffusion kinetics. This equation includes a "Mass Term" or "Characteristic Frequency" (ω₀²) which creates a restorative force and defines the intrinsic spatial correlation length of the mesh.

However, the bifurcation probe was still using the old massless Wave Equation for stability analysis, creating a discrepancy between simulated dynamics and bifurcation analysis.

## Physics Background

### Old (Massless Wave Equation)
```
L = κ * ∇² + r * I
RHS: κ*∇²u + r*u + u³ = h
```

### New (Telegraph Equation at Steady State)
```
L = c_eff² * ∇² + (r - ω₀²) * I
RHS: c_eff²*∇²u - ω₀²*u + r*u + u³ = h
```

Where:
- `ω₀² = α*δ - β*γ` (mass term from micro-parameters)
- `c_eff² = D*(δ-α)/2` (effective wave speed squared)
- `γ₀ = (α+δ)/2` (damping coefficient)

## Key Changes

### 1. Import Telegraph Physics (Line 20-21)
```python
from smm.glia import GliaMicroParams, GlialFieldConfig
```

### 2. Load Parameters from params.yaml (Lines 301-333)
```python
# Load Telegraph physics parameters from params.yaml
with open(params_path, 'r') as f:
    params = yaml.safe_load(f)

micro_params = GliaMicroParams(
    alpha=params.get('alpha', 1.0),
    beta=params.get('beta', 0.8),
    gamma=params.get('gamma', 0.9),
    delta=params.get('delta', 2.0),
    D_um2_per_s=params.get('D_um2_per_s', 100.0)
)

telegraph_params = micro_params.compute_telegraph_params()
omega0_sq = telegraph_params['omega0_squared']
```

### 3. Updated Linear Operator (Lines 68-80)
```python
def build_linear_operator(
    lap: csr_matrix, c_eff_sq: float, omega0_sq: float, r_param: float
) -> csr_matrix:
    """Return Telegraph steady-state operator: L = c_eff² * Laplacian - omega0² * Identity + r * Identity."""
    ident = sparse.identity(lap.shape[0], format="csr", dtype=float)
    return lap * float(c_eff_sq) + ident * float(r_param - omega0_sq)
```

### 4. Updated RHS in integrate_to_steady (Lines 139-142)
```python
def rhs(vec: np.ndarray) -> np.ndarray:
    lap_term = lap.dot(vec)
    # Telegraph steady-state: c_eff² ∇²u - ω₀² u + r u + u u³ = h
    return -(c_eff_sq * lap_term - omega0_sq * vec + r_param * vec + u_param * vec ** 3 - rhs_const)
```

### 5. Added ω₀² to Output (Line 512)
```python
record = {
    "kappa": float(kappa),
    "omega0_sq": float(omega0_sq),  # <-- NEW
    "lambda1": lambda1,
    ...
}
```

### 6. Eigenvalue Comparison Check (Lines 367-372)
```python
# Check eigenvalue shift due to mass term
if abs(lambda1 + omega0_sq) < abs(lambda1) * EIGENVALUE_SHIFT_THRESHOLD:
    LOG.info(
        "Mass term significantly shifts eigenvalue: λ₁=%.6f, -ω₀²=%.6f, λ₁+ω₀²=%.6f",
        lambda1, -omega0_sq, lambda1 + omega0_sq
    )
```

## Physics Implications

### 1. Zero Mode Suppression
The constant spatial field (zero mode of Laplacian) is no longer a valid eigenmode when ω₀² > 0, because the mass term `-ω₀²*u` penalizes non-zero constant offsets.

```
For constant field u = c:
L*c = c_eff² * 0 + (r - ω₀²) * c = (r - ω₀²) * c
```

### 2. Eigenvalue Shift
All eigenvalues shift by -ω₀²:
```
λ_new = λ_old - ω₀²
```

### 3. Stiffness
If ω₀² is large, the system becomes stiffer. The adaptive time step and relaxation factor in the probe handle this automatically.

### 4. Spatial Correlation Length
The mass term defines an intrinsic correlation length:
```
ξ ~ c_eff / √(ω₀²)
```

## Validation

### Tests Added (tests/test_bifurcation_telegraph.py)
1. **test_telegraph_linear_operator**: Verifies linear operator includes mass term correctly
2. **test_telegraph_rhs_includes_mass_term**: Verifies RHS uses Telegraph equation
3. **test_telegraph_parameters_from_yaml**: Validates parameter computation from yaml
4. **test_eigenvalue_shift_with_mass_term**: Confirms eigenvalues shift by -ω₀²
5. **test_zero_mode_suppressed_by_mass_term**: Validates constant field penalization

All tests pass ✓

### Smoke Test
```bash
python bifurcation_probe_refined.py --smoke --outdir test_results
```

Output confirms:
- Parameters loaded from params.yaml ✓
- ω₀² = 1.28 (rad/s)² ✓
- Hysteresis loops generated ✓
- omega0_sq column in CSV output ✓

## Example Output

From `cusp_grid_summary_refined.csv`:
```csv
kappa,omega0_sq,lambda1,lambda2,gap,...
0.450,1.280,-2.080,-2.074,0.006,...
0.488,1.280,-2.130,-2.124,0.006,...
```

## Backward Compatibility

The probe maintains backward compatibility:
- If `params.yaml` is not found, ω₀² defaults to 0.0 (massless)
- Parameter name `kappa` is retained for interface consistency (interpreted as c_eff²)
- Output file names unchanged
- All plots and analysis scripts work as before

## Performance Impact

Minimal - the mass term is just an additional diagonal matrix term:
- Same computational complexity O(N)
- No additional memory overhead
- Convergence may be slower for large ω₀² (handled by adaptive timestep)

## References

1. `src/smm/glia.py` - Telegraph equation implementation
2. `params.yaml` - Micro-parameter configuration
3. Original issue - "Refactor Bifurcation Probe to use Telegraph Physics"

## Author
Implemented by GitHub Copilot with guidance from problem statement.

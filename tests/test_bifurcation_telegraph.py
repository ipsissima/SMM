"""Test Telegraph physics implementation in bifurcation probe."""
import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from smm.glia import GliaMicroParams, GlialFieldConfig

# Import bifurcation probe functions
sys.path.insert(0, str(Path(__file__).parent.parent))
import bifurcation_probe_refined as bp


def test_telegraph_linear_operator():
    """Test that linear operator includes mass term omega0²."""
    # Create a simple 2x2 Laplacian matrix
    nx, ny = 4, 4
    dx = 1.0
    lap = bp.laplacian_9pt_matrix(nx, ny, dx)
    
    # Telegraph parameters
    c_eff_sq = 0.5
    omega0_sq = 1.28  # from params.yaml
    r_param = -0.2
    
    # Build linear operator
    linop = bp.build_linear_operator(lap, c_eff_sq, omega0_sq, r_param)
    
    # The diagonal elements should include the mass term
    # L = c_eff² * Laplacian + (r - omega0²) * Identity
    # For the Laplacian diagonal is -20/(6*dx²)
    expected_diag = c_eff_sq * (-20.0 / (6.0 * dx**2)) + (r_param - omega0_sq)
    
    # Check diagonal elements
    diagonal = linop.diagonal()
    np.testing.assert_allclose(diagonal, expected_diag, rtol=1e-10)


def test_telegraph_rhs_includes_mass_term():
    """Test that RHS in integrate_to_steady includes omega0² term."""
    # Create a simple test field
    nx, ny = 4, 4
    dx = 1.0
    lap = bp.laplacian_9pt_matrix(nx, ny, dx)
    
    # Parameters - use values where there's only one stable equilibrium
    c_eff_sq = 0.5
    omega0_sq_with = 1.28
    omega0_sq_without = 0.0
    h_value = 0.1  # Non-zero to break symmetry
    r_param = -0.2
    u_param = 1.0
    
    n_points = nx * ny
    dt = 0.01
    tol = 1e-8
    
    # Same initial condition for both
    phi_init = np.ones(n_points) * 0.01
    
    # Integrate with omega0²
    phi_with, meta_with = bp.integrate_to_steady(
        phi_init.copy(),
        lap,
        c_eff_sq,
        omega0_sq_with,
        h_value,
        r_param,
        u_param,
        dt,
        tol,
        Tmax_min=0.5,
        Tmax_max=2.0,
        tau_factor=1.0,
        lambda1_est=-1.0,
    )
    
    # Integrate without omega0² (massless)
    phi_without, meta_without = bp.integrate_to_steady(
        phi_init.copy(),
        lap,
        c_eff_sq,
        omega0_sq_without,
        h_value,
        r_param,
        u_param,
        dt,
        tol,
        Tmax_min=0.5,
        Tmax_max=2.0,
        tau_factor=1.0,
        lambda1_est=-1.0,
    )
    
    # The equilibria should be different when omega0² is included
    # The mass term changes the energy landscape
    diff = np.linalg.norm(phi_with - phi_without)
    assert diff > 0.01, f"Mass term should change equilibrium, but diff={diff}"


def test_telegraph_parameters_from_yaml():
    """Test that Telegraph parameters are correctly computed from params.yaml."""
    # These are the values from params.yaml
    params = {
        'alpha': 1.0,
        'beta': 0.8,
        'gamma': 0.9,
        'delta': 2.0,
        'D_um2_per_s': 100.0,
    }
    
    micro = GliaMicroParams(
        alpha=params['alpha'],
        beta=params['beta'],
        gamma=params['gamma'],
        delta=params['delta'],
        D_um2_per_s=params['D_um2_per_s']
    )
    
    telegraph = micro.compute_telegraph_params()
    
    # Verify the computed values
    # omega0² = α*δ - β*γ = 1.0*2.0 - 0.8*0.9 = 2.0 - 0.72 = 1.28
    expected_omega0_sq = params['alpha'] * params['delta'] - params['beta'] * params['gamma']
    assert abs(telegraph['omega0_squared'] - expected_omega0_sq) < 1e-10
    
    # c_eff² = D*(δ - α)/2 = 100*(2.0 - 1.0)/2 = 50
    # c_eff (μm/s) = sqrt(50) = 7.071...
    expected_c_eff_um = np.sqrt(params['D_um2_per_s'] * (params['delta'] - params['alpha']) / 2.0)
    assert abs(telegraph['c_eff_um_per_s'] - expected_c_eff_um) < 1e-6
    
    # Verify mm/s conversion: c_eff (mm/s) = 7.071.../1000 = 0.007071...
    expected_c_eff_mm = expected_c_eff_um / 1000.0
    assert abs(telegraph['c_eff_mm_per_s'] - expected_c_eff_mm) < 1e-9
    
    # γ₀ = (α + δ)/2 = (1.0 + 2.0)/2 = 1.5
    expected_gamma0 = (params['alpha'] + params['delta']) / 2.0
    assert abs(telegraph['gamma0'] - expected_gamma0) < 1e-10


def test_eigenvalue_shift_with_mass_term():
    """Test that eigenvalues shift by -omega0² with mass term."""
    # Create a simple mesh
    nx, ny = 8, 8
    dx = 1.0
    lap = bp.laplacian_9pt_matrix(nx, ny, dx)
    
    # First compute eigenvalues without mass term (omega0² = 0)
    c_eff_sq = 0.5
    omega0_sq_zero = 0.0
    r_param = -0.2
    
    linop_no_mass = bp.build_linear_operator(lap, c_eff_sq, omega0_sq_zero, r_param)
    vals_no_mass, _ = bp.smallest_eigenpairs(linop_no_mass, k=2)
    
    # Now compute with mass term
    omega0_sq = 1.28
    linop_with_mass = bp.build_linear_operator(lap, c_eff_sq, omega0_sq, r_param)
    vals_with_mass, _ = bp.smallest_eigenpairs(linop_with_mass, k=2)
    
    # The eigenvalues should shift by -omega0²
    # λ_new = λ_old - omega0²
    expected_shift = vals_no_mass - omega0_sq
    np.testing.assert_allclose(vals_with_mass, expected_shift, rtol=1e-8)


def test_zero_mode_suppressed_by_mass_term():
    """Test that constant field is penalized by mass term."""
    # The "zero mode" (constant spatial field) should no longer be a valid eigenmode
    # when omega0² > 0, because -omega0² * u penalizes non-zero constant offsets
    
    nx, ny = 8, 8
    dx = 1.0
    lap = bp.laplacian_9pt_matrix(nx, ny, dx)
    
    c_eff_sq = 0.5
    omega0_sq = 1.28
    r_param = -0.2
    
    linop = bp.build_linear_operator(lap, c_eff_sq, omega0_sq, r_param)
    
    # Create a constant field (would be zero mode of Laplacian)
    n_points = nx * ny
    constant_field = np.ones(n_points)
    
    # Apply the operator to the constant field
    result = linop.dot(constant_field)
    
    # For a constant field, Laplacian is zero, so:
    # L * 1 = c_eff² * 0 + (r - omega0²) * 1 = (r - omega0²) * 1
    expected = (r_param - omega0_sq) * constant_field
    np.testing.assert_allclose(result, expected, rtol=1e-10)
    
    # The constant field is NOT an eigenmode unless (r - omega0²) = eigenvalue
    # The mass term shifts the effective potential, suppressing constant offsets


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

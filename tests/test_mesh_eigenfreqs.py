"""
Test mesh eigenfrequency computation.

Verifies that the mesh PDE implementation produces the correct eigenfrequencies
for a 2D rectangular domain with Neumann boundary conditions.

For a domain [0,L]×[0,L] with Neumann BC, the eigenfrequencies are:
    f_{nx,ny} = (c/2π) √[(πnx/L)² + (πny/L)²]
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from smm.mesh import MeshField, MeshConfig
from smm.analysis import compute_psd, find_peaks_in_psd


def test_mesh_eigenfreq_1d():
    """Test that mesh shows correct eigenfrequency for 1D-like mode.
    
    For mode (nx=17, ny=0) on L=32mm domain with c=15mm/s:
        f_17 = 17 * c / (2*L) = 17 * 15 / 64 ≈ 3.98 Hz
    """
    # Configuration
    L = 32.0
    c = 15.0
    config = MeshConfig(
        Lx=L,
        Ly=L,
        Nx=64,
        Ny=64,
        c=c,
        gamma=0.1,  # Low damping to see peaks
        dt=0.001,
        pml_width=6,
        pml_sigma_max=50.0
    )
    
    # Check CFL
    assert config.check_cfl(), f"CFL {config.cfl} too large"
    
    # Create mesh
    rng = np.random.default_rng(42)
    mesh = MeshField(config, rng=rng)
    
    # Theoretical eigenfrequency for mode (17, 0)
    nx = 17
    f_theory = nx * c / (2 * L)
    
    print(f"\nTheoretical f_{nx} = {f_theory:.4f} Hz")
    
    # Run simulation with white noise
    T_sim = 5.0
    noise_amplitude = 1e-4
    record_interval = 4  # Record at ~250 Hz
    
    snapshots = mesh.run(
        T=T_sim,
        record_interval=record_interval,
        noise_amplitude=noise_amplitude
    )
    
    # Extract center trace
    center_y, center_x = config.Ny // 2, config.Nx // 2
    trace = snapshots[:, center_y, center_x]
    
    # Compute PSD
    fs = 1.0 / (config.dt * record_interval)
    f, Pxx = compute_psd(trace, fs=fs, nperseg=256, noverlap=128)
    
    # Find peaks
    peak_freqs, peak_powers = find_peaks_in_psd(f, Pxx, prominence_factor=0.05)
    
    print(f"Found {len(peak_freqs)} peaks in PSD")
    if len(peak_freqs) > 0:
        # Sort by power
        sorted_idx = np.argsort(peak_powers)[::-1]
        top_5_freqs = peak_freqs[sorted_idx[:min(5, len(peak_freqs))]]
        print(f"Top peak frequencies: {top_5_freqs}")
    
    # Check if there's a peak near f_theory
    tolerance = 0.5  # Hz
    if len(peak_freqs) > 0:
        errors = np.abs(peak_freqs - f_theory)
        min_error = np.min(errors)
        closest_peak = peak_freqs[np.argmin(errors)]
        
        print(f"Closest peak: {closest_peak:.4f} Hz (error: {min_error:.4f} Hz)")
        
        assert min_error < tolerance, \
            f"No peak within {tolerance} Hz of theoretical f_{nx}={f_theory:.4f} Hz"
    else:
        # Check if there's elevated power even without a sharp peak
        idx_theory = np.argmin(np.abs(f - f_theory))
        power_at_theory = Pxx[idx_theory]
        baseline = np.median(Pxx)
        
        ratio = power_at_theory / baseline
        print(f"Power at f_{nx}: {power_at_theory:.2e}, baseline: {baseline:.2e}, ratio: {ratio:.2f}")
        
        assert ratio > 2.0, \
            f"Expected elevated power at f_{nx}={f_theory:.4f} Hz, got ratio {ratio:.2f}"


def test_mesh_eigenfreq_theoretical_formula():
    """Test that MeshField.get_eigenfrequency() matches expected formula."""
    L = 32.0
    c = 15.0
    
    config = MeshConfig(Lx=L, Ly=L, Nx=64, Ny=64, c=c)
    mesh = MeshField(config)
    
    # Test several modes
    test_modes = [(1, 0), (0, 1), (1, 1), (2, 0), (3, 1)]
    
    for nx, ny in test_modes:
        # Theoretical eigenfrequency
        kx = np.pi * nx / L
        ky = np.pi * ny / L
        k = np.sqrt(kx**2 + ky**2)
        f_theory = c * k / (2 * np.pi)
        
        # Computed eigenfrequency
        f_computed = mesh.get_eigenfrequency(nx, ny)
        
        print(f"Mode ({nx},{ny}): theory={f_theory:.6f} Hz, computed={f_computed:.6f} Hz")
        
        assert np.abs(f_computed - f_theory) < 1e-10, \
            f"Eigenfrequency mismatch for mode ({nx},{ny})"


def test_mesh_cfl_check():
    """Test that CFL check works correctly."""
    # Safe configuration (CFL < 0.5)
    config_safe = MeshConfig(
        Lx=32.0, Ly=32.0, Nx=64, Ny=64,
        c=15.0, dt=0.001
    )
    assert config_safe.check_cfl(), "Safe configuration should pass CFL check"
    assert config_safe.cfl < 0.5, "CFL should be < 0.5"
    
    # Unsafe configuration (CFL > 0.5)
    config_unsafe = MeshConfig(
        Lx=32.0, Ly=32.0, Nx=64, Ny=64,
        c=15.0, dt=0.1  # Large time step
    )
    assert not config_unsafe.check_cfl(), "Unsafe configuration should fail CFL check"
    assert config_unsafe.cfl > 0.5, "CFL should be > 0.5"


def test_mesh_pml_damping():
    """Test that PML damping is created correctly."""
    config = MeshConfig(
        Lx=32.0, Ly=32.0, Nx=32, Ny=32,
        pml_width=4, pml_sigma_max=50.0, gamma=0.5
    )
    mesh = MeshField(config)
    
    # Check interior has only background damping
    interior_y, interior_x = config.Ny // 2, config.Nx // 2
    assert mesh.gamma_total[interior_y, interior_x] == config.gamma, \
        "Interior should have only background damping"
    
    # Check corners have maximum PML damping
    corner_gamma = mesh.gamma_total[0, 0]
    assert corner_gamma > config.gamma, \
        "Corners should have additional PML damping"
    
    # Check that damping increases towards edges
    edge_y = config.pml_width // 2
    center_y = config.Ny // 2
    assert mesh.gamma_total[edge_y, interior_x] > mesh.gamma_total[center_y, interior_x], \
        "Damping should increase towards edges"


def test_mesh_laplacian_homogeneous():
    """Test Laplacian on homogeneous field (should be zero)."""
    config = MeshConfig(Lx=32.0, Ly=32.0, Nx=32, Ny=32)
    mesh = MeshField(config)
    
    # Constant field
    u = np.ones((config.Ny, config.Nx))
    lap = mesh.laplacian_9pt(u)
    
    # Laplacian of constant should be ~zero (up to boundary effects)
    # Check interior only
    interior = lap[3:-3, 3:-3]
    assert np.max(np.abs(interior)) < 1e-10, \
        "Laplacian of constant field should be zero in interior"


def test_mesh_step_stability():
    """Test that mesh step doesn't produce NaNs or Infs."""
    config = MeshConfig(
        Lx=32.0, Ly=32.0, Nx=32, Ny=32,
        c=15.0, gamma=0.5, dt=0.001
    )
    rng = np.random.default_rng(42)
    mesh = MeshField(config, rng=rng)
    
    # Run for 100 steps
    for _ in range(100):
        mesh.step(noise_amplitude=1e-5)
    
    # Check for NaNs or Infs
    assert np.all(np.isfinite(mesh.u)), "Field u contains NaN or Inf"
    assert np.all(np.isfinite(mesh.v)), "Field v contains NaN or Inf"


if __name__ == '__main__':
    print("Testing mesh eigenfrequencies and numerics...")
    print("=" * 60)
    
    print("\n1. Testing eigenfrequency formula...")
    test_mesh_eigenfreq_theoretical_formula()
    print("PASSED")
    
    print("\n2. Testing CFL check...")
    test_mesh_cfl_check()
    print("PASSED")
    
    print("\n3. Testing PML damping...")
    test_mesh_pml_damping()
    print("PASSED")
    
    print("\n4. Testing Laplacian on homogeneous field...")
    test_mesh_laplacian_homogeneous()
    print("PASSED")
    
    print("\n5. Testing step stability...")
    test_mesh_step_stability()
    print("PASSED")
    
    print("\n6. Testing eigenfrequency peak (may take a few seconds)...")
    test_mesh_eigenfreq_1d()
    print("PASSED")
    
    print("\n" + "=" * 60)
    print("All mesh tests passed!")

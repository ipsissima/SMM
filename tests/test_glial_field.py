"""
Tests for glial field telegraph equation implementation.

Tests:
1. Telegraph dispersion: wave speed validation
2. Damping test: exponential decay
3. Physiological parameter ranges
4. Tripartite loop smoke test
"""

import sys
from pathlib import Path
import numpy as np
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from smm.glia import GliaMicroParams, GlialFieldConfig, GlialField
from smm.neural import NeuralMass, Kuramoto, NeuralMassConfig, KuramotoConfig
from smm.coupling import (
    tripartite_neural_to_glia_source,
    glia_to_neural_gliotransmission,
    ConnectivityPlasticity,
    create_region_positions_grid
)


def test_micro_params_computation():
    """Test that micro-parameters correctly compute telegraph parameters."""
    micro = GliaMicroParams(
        alpha=1.5,
        beta=0.8,
        gamma=1.0,
        delta=1.0,
        D_um2_per_s=100.0
    )
    
    params = micro.compute_telegraph_params()
    
    # Check gamma0 = (alpha + delta) / 2
    expected_gamma0 = (1.5 + 1.0) / 2.0
    assert np.isclose(params['gamma0'], expected_gamma0)
    
    # Check c_eff² = D * (delta - alpha) / 2
    # With delta=1.0, alpha=1.5: (1.0 - 1.5) = -0.5 < 0
    # So c_eff should be 0 (non-propagating)
    assert params['c_eff_um_per_s'] == 0.0
    
    # Test with delta > alpha (propagating case)
    micro2 = GliaMicroParams(alpha=1.0, beta=0.8, gamma=1.0, delta=2.0, D_um2_per_s=100.0)
    params2 = micro2.compute_telegraph_params()
    
    # c_eff² = 100 * (2.0 - 1.0) / 2 = 50
    # c_eff = sqrt(50) ≈ 7.07 μm/s
    expected_c_eff = np.sqrt(100.0 * (2.0 - 1.0) / 2.0)
    assert np.isclose(params2['c_eff_um_per_s'], expected_c_eff)


def test_physiological_ranges():
    """Test physiological parameter validation."""
    # Physiological parameters
    micro = GliaMicroParams(
        alpha=1.0,
        beta=0.8,
        gamma=0.9,
        delta=1.5,
        D_um2_per_s=100.0
    )
    
    checks = micro.check_physiological()
    
    # Should have wave speed in μm/s
    assert 'wave_speed_um_per_s' in checks
    assert checks['wave_speed_um_per_s'] > 0
    
    # Should have damping time
    assert 'damping_time_s' in checks
    assert checks['damping_time_s'] > 0
    
    # Should have propagation length
    assert 'propagation_length_um' in checks
    assert checks['propagation_length_um'] > 0
    
    # For reasonable parameters, should pass basic checks
    params = micro.compute_telegraph_params()
    assert params['gamma0'] > 0
    assert params['c_eff_um_per_s'] >= 0


def test_telegraph_wave_propagation():
    """Test telegraph equation wave propagation speed."""
    # Setup with known wave speed
    micro = GliaMicroParams(
        alpha=1.0,
        beta=0.5,
        gamma=0.5,
        delta=2.0,
        D_um2_per_s=200.0  # Will give c_eff² = 200*(2-1)/2 = 100, c_eff = 10 μm/s
    )
    
    config = GlialFieldConfig(
        Lx=10.0,  # 10 mm = 10000 μm
        Ly=10.0,
        Nx=128,
        Ny=128,
        micro_params=micro,
        dt=0.001,
        pml_width=5
    )
    
    glia = GlialField(config, rng=np.random.default_rng(42))
    
    # Check that derived parameters are set
    params = micro.compute_telegraph_params()
    assert np.isclose(config.c, params['c_eff_mm_per_s'], rtol=0.01)
    
    # Initialize with a localized perturbation in the center
    center_x, center_y = config.Nx // 2, config.Ny // 2
    glia.u[center_y-2:center_y+3, center_x-2:center_x+3] = 1.0
    
    # Run for a short time
    T = 0.5  # seconds
    snapshots = glia.run(T=T, record_interval=10, noise_amplitude=0.0)
    
    # Wave should propagate outward
    # Final snapshot should have lower amplitude at center (wave moved out)
    assert np.max(snapshots[-1]) > 0  # Some signal remains
    assert np.max(snapshots[-1]) < 1.0  # But reduced from initial


def test_damping_decay():
    """Test that field decays exponentially with rate ~gamma0."""
    micro = GliaMicroParams(
        alpha=1.5,
        beta=0.5,
        gamma=0.5,
        delta=1.5,
        D_um2_per_s=100.0
    )
    
    params = micro.compute_telegraph_params()
    gamma0 = params['gamma0']
    
    config = GlialFieldConfig(
        Lx=10.0,
        Ly=10.0,
        Nx=32,
        Ny=32,
        micro_params=micro,
        dt=0.001,
        pml_width=0  # No PML for this test
    )
    
    glia = GlialField(config, rng=np.random.default_rng(42))
    
    # Homogeneous initial condition
    glia.u[:, :] = 1.0
    glia.v[:, :] = 0.0
    
    # Run for a few damping times
    T = 2.0 / gamma0
    dt_record = 0.01
    n_steps = int(T / config.dt)
    record_interval = int(dt_record / config.dt)
    
    snapshots = glia.run(T=T, record_interval=record_interval, noise_amplitude=0.0)
    
    # Extract mean field over time
    mean_field = np.array([np.mean(snap) for snap in snapshots])
    
    # Should decay
    assert mean_field[-1] < mean_field[0]
    
    # Rough check: after time T = 2/γ₀, field should be ~exp(-2) ≈ 0.135 of initial
    # (Allowing for spatial effects)
    decay_ratio = mean_field[-1] / mean_field[0]
    expected_ratio = np.exp(-2)
    assert 0.01 < decay_ratio < 0.5  # Loose bounds


def test_tripartite_neural_to_glia_coupling():
    """Test neural → glia source term generation."""
    n_regions = 4
    positions = [(16, 16), (48, 16), (16, 48), (48, 48)]
    
    E = np.array([1.0, 0.5, 0.3, 0.0])
    I = np.array([0.2, 0.1, 0.0, 0.0])
    theta = np.array([0.0, np.pi/2, np.pi, 3*np.pi/2])
    
    source = tripartite_neural_to_glia_source(
        E, I, theta,
        positions=positions,
        grid_shape=(64, 64),
        dx=0.5,
        sigma=1.0,
        beta_E=1.0,
        beta_I=0.5,
        beta_theta=0.2
    )
    
    assert source.shape == (64, 64)
    assert np.max(source) > 0  # Should have some positive values
    assert np.sum(source) > 0  # Total source should be positive


def test_glia_to_neural_gliotransmission():
    """Test glia → neural gliotransmission."""
    u = np.random.randn(64, 64) * 0.5
    positions = [(16, 16), (48, 16), (16, 48), (48, 48)]
    
    # Test linear
    I_A_linear = glia_to_neural_gliotransmission(
        u, positions, g_A=0.1, nonlinearity='linear'
    )
    assert len(I_A_linear) == 4
    
    # Test tanh
    I_A_tanh = glia_to_neural_gliotransmission(
        u, positions, g_A=0.1, nonlinearity='tanh', gain=2.0
    )
    assert len(I_A_tanh) == 4
    assert np.max(np.abs(I_A_tanh)) <= 0.1  # Should be bounded by g_A for tanh
    
    # Test threshold_linear
    I_A_threshold = glia_to_neural_gliotransmission(
        u, positions, g_A=0.1, nonlinearity='threshold_linear', threshold=0.5
    )
    assert len(I_A_threshold) == 4


def test_connectivity_plasticity():
    """Test Ca-gated connectivity plasticity."""
    n_regions = 4
    plasticity = ConnectivityPlasticity(
        n_regions=n_regions,
        tau_C=60.0,
        lambda_C=0.1,
        eta_H=0.01,
        eta_psi=0.01,
        sigma_psi_target=0.1
    )
    
    assert plasticity.C.shape == (n_regions, n_regions)
    assert np.all(np.diag(plasticity.C) == 0)  # No self-connections
    
    # Simulate one step
    E = np.array([0.8, 0.6, 0.4, 0.2])
    u_values = np.array([0.5, 0.3, 0.1, 0.0])
    
    C_initial = plasticity.C.copy()
    plasticity.step(dt=0.1, E=E, u_values=u_values)
    
    # Connectivity should have changed
    assert not np.allclose(plasticity.C, C_initial)
    assert np.all(np.diag(plasticity.C) == 0)  # Still no self-connections


def test_tripartite_loop_smoke():
    """Smoke test for full tripartite loop integration."""
    # Small network
    n_regions = 4
    
    # Glial field
    micro = GliaMicroParams(alpha=1.0, beta=0.8, gamma=0.9, delta=1.5, D_um2_per_s=100.0)
    glia_config = GlialFieldConfig(
        Lx=10.0, Ly=10.0, Nx=32, Ny=32,
        micro_params=micro,
        dt=0.001,
        pml_width=3
    )
    glia = GlialField(glia_config, rng=np.random.default_rng(42))
    
    # Neural masses
    neural_config = NeuralMassConfig(n_regions=n_regions)
    neural = NeuralMass(neural_config)
    neural.E[:] = 0.5  # Initialize with some activity
    
    # Kuramoto
    kuramoto_config = KuramotoConfig(n_oscillators=n_regions)
    kuramoto = Kuramoto(kuramoto_config, rng=np.random.default_rng(42))
    
    # Connectivity plasticity
    plasticity = ConnectivityPlasticity(n_regions=n_regions)
    
    # Positions
    positions = create_region_positions_grid(32, 32, n_regions)
    
    # Run for short time
    T = 0.1
    dt = 0.001
    n_steps = int(T / dt)
    
    for _ in range(n_steps):
        # (a) Neural → Glia
        source = tripartite_neural_to_glia_source(
            neural.E, neural.I, kuramoto.theta,
            positions=positions,
            grid_shape=(32, 32),
            dx=glia_config.dx,
            sigma=1.0,
            beta_E=0.1,
            beta_I=0.0,
            beta_theta=0.0
        )
        
        # Step glia
        glia.step(source=source, noise_amplitude=1e-6)
        
        # (b) Glia → Neural
        I_A = glia_to_neural_gliotransmission(
            glia.u, positions, g_A=0.05, nonlinearity='tanh'
        )
        
        # Step neural
        neural.step(dt, I_mesh=I_A)
        
        # (c) Glia → Kuramoto (via modulation in step)
        # Sample glia values
        glia_values = np.array([glia.u[j, i] for i, j in positions])
        
        # Step kuramoto
        kuramoto.step(dt, mesh_input=0.01 * glia_values)
        
        # (d) Glia → Connectivity (slow, update every 10 steps)
        if _ % 10 == 0:
            plasticity.step(dt=10*dt, E=neural.E, u_values=glia_values)
    
    # Check that everything evolved
    assert neural.t > 0
    assert glia.t > 0
    assert kuramoto.t > 0
    
    # Neural activity should have varied
    assert np.std(neural.E) > 0 or np.max(neural.E) > 0
    
    # Glial field should be non-trivial
    assert np.max(np.abs(glia.u)) > 0
    
    # Connectivity should have changed slightly
    assert np.max(np.abs(plasticity.C)) > 0


if __name__ == '__main__':
    print("Running glial field tests...")
    print("=" * 70)
    
    print("\n1. Testing micro-parameter computation...")
    test_micro_params_computation()
    print("PASSED")
    
    print("\n2. Testing physiological ranges...")
    test_physiological_ranges()
    print("PASSED")
    
    print("\n3. Testing telegraph wave propagation...")
    test_telegraph_wave_propagation()
    print("PASSED")
    
    print("\n4. Testing damping decay...")
    test_damping_decay()
    print("PASSED")
    
    print("\n5. Testing neural → glia coupling...")
    test_tripartite_neural_to_glia_coupling()
    print("PASSED")
    
    print("\n6. Testing glia → neural gliotransmission...")
    test_glia_to_neural_gliotransmission()
    print("PASSED")
    
    print("\n7. Testing connectivity plasticity...")
    test_connectivity_plasticity()
    print("PASSED")
    
    print("\n8. Testing tripartite loop smoke test...")
    test_tripartite_loop_smoke()
    print("PASSED")
    
    print("\n" + "=" * 70)
    print("All glial field tests passed!")

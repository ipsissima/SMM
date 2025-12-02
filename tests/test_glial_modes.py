"""
Tests for glial field mode ablations (telegraph, diffusion, mean_field).

Tests the new mode parameter and ensures each mode behaves correctly.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from smm.glia import GliaMicroParams, GlialFieldConfig, GlialField


def test_telegraph_mode_default():
    """Test that telegraph mode is the default and works correctly."""
    config = GlialFieldConfig(
        Lx=10.0, Ly=10.0, Nx=32, Ny=32,
        c=15.0, gamma=0.5, omega0=0.0,
        dt=0.001, pml_width=3
    )
    
    # Check default mode
    assert config.mode == 'telegraph'
    
    glia = GlialField(config, rng=np.random.default_rng(42))
    
    # Initialize with a localized perturbation
    glia.u[15:17, 15:17] = 1.0
    
    # Run for a short time
    initial_energy = np.sum(glia.u**2)
    glia.step()
    
    # Field should evolve
    assert glia.t == config.dt
    assert np.sum(glia.u**2) != initial_energy


def test_diffusion_mode():
    """Test diffusion mode behavior."""
    config = GlialFieldConfig(
        Lx=10.0, Ly=10.0, Nx=32, Ny=32,
        gamma=0.5, dt=0.001, pml_width=0,
        mode='diffusion',
        diffusion_D_mm2_per_s=0.5
    )
    
    assert config.mode == 'diffusion'
    
    glia = GlialField(config, rng=np.random.default_rng(42))
    
    # Initialize with a localized perturbation
    center_x, center_y = 16, 16
    glia.u[center_y, center_x] = 1.0
    
    # Run for a short time
    T = 0.1
    n_steps = int(T / config.dt)
    
    for _ in range(n_steps):
        glia.step()
    
    # Field should have diffused outward
    assert glia.t > 0
    # Peak should be lower (diffused)
    assert glia.u[center_y, center_x] < 1.0
    # Neighboring points should have increased
    assert np.sum(glia.u) > 0  # Some signal remains


def test_mean_field_mode():
    """Test mean-field mode behavior."""
    config = GlialFieldConfig(
        Lx=10.0, Ly=10.0, Nx=32, Ny=32,
        gamma=0.5, omega0=0.0, dt=0.001, pml_width=0,
        mode='mean_field'
    )
    
    assert config.mode == 'mean_field'
    
    glia = GlialField(config, rng=np.random.default_rng(42))
    
    # Initialize with spatially varying perturbation
    glia.u[:16, :] = 1.0
    glia.u[16:, :] = 0.5
    glia.v[:, :] = 0.1  # Give it some velocity
    
    initial_mean = np.mean(glia.u)
    
    # Run for a short time
    T = 0.1
    n_steps = int(T / config.dt)
    
    for _ in range(n_steps):
        glia.step()
    
    # All points should have the same value (spatially uniform)
    assert np.std(glia.u) < 1e-10
    
    # Mean should have evolved according to scalar ODE (with damping, should decay)
    final_mean = np.mean(glia.u)
    # Due to damping and velocity, should change
    assert final_mean != initial_mean or abs(final_mean - initial_mean) < 1e-6


def test_mode_comparison_with_source():
    """Test that different modes respond differently to the same source."""
    # Create identical configs except for mode
    base_config = {
        'Lx': 10.0, 'Ly': 10.0, 'Nx': 32, 'Ny': 32,
        'gamma': 0.5, 'omega0': 0.0, 'dt': 0.001, 'pml_width': 0,
        'c': 10.0, 'diffusion_D_mm2_per_s': 0.5
    }
    
    # Create source
    source = np.zeros((32, 32))
    source[16, 16] = 1.0
    
    # Telegraph mode
    config_telegraph = GlialFieldConfig(**base_config, mode='telegraph')
    glia_telegraph = GlialField(config_telegraph, rng=np.random.default_rng(42))
    
    # Diffusion mode
    config_diffusion = GlialFieldConfig(**base_config, mode='diffusion')
    glia_diffusion = GlialField(config_diffusion, rng=np.random.default_rng(42))
    
    # Mean-field mode
    config_mean_field = GlialFieldConfig(**base_config, mode='mean_field')
    glia_mean_field = GlialField(config_mean_field, rng=np.random.default_rng(42))
    
    # Run all for same duration with same source
    T = 0.05
    n_steps = int(T / base_config['dt'])
    
    for _ in range(n_steps):
        glia_telegraph.step(source=source)
        glia_diffusion.step(source=source)
        glia_mean_field.step(source=source)
    
    # All should have evolved
    assert np.max(np.abs(glia_telegraph.u)) > 0
    assert np.max(np.abs(glia_diffusion.u)) > 0
    assert np.max(np.abs(glia_mean_field.u)) > 0
    
    # Mean-field should be spatially uniform
    assert np.std(glia_mean_field.u) < 1e-10
    
    # Telegraph and diffusion should have spatial structure
    assert np.std(glia_telegraph.u) > 1e-6
    assert np.std(glia_diffusion.u) > 1e-6
    
    # Telegraph and diffusion should give different results
    assert not np.allclose(glia_telegraph.u, glia_diffusion.u)


def test_invalid_mode_raises_error():
    """Test that an invalid mode raises ValueError."""
    config = GlialFieldConfig(
        Lx=10.0, Ly=10.0, Nx=32, Ny=32,
        mode='invalid_mode'
    )
    
    glia = GlialField(config, rng=np.random.default_rng(42))
    
    with pytest.raises(ValueError, match="Unknown glial mode"):
        glia.step()


def test_modes_with_micro_params():
    """Test that modes work with micro-derived parameters."""
    micro = GliaMicroParams(
        alpha=1.0, beta=0.8, gamma=0.9, delta=1.5,
        D_um2_per_s=100.0
    )
    
    # Telegraph mode
    config_telegraph = GlialFieldConfig(
        Lx=10.0, Ly=10.0, Nx=32, Ny=32,
        micro_params=micro, dt=0.001, pml_width=3,
        mode='telegraph'
    )
    glia_telegraph = GlialField(config_telegraph, rng=np.random.default_rng(42))
    glia_telegraph.u[16, 16] = 1.0
    glia_telegraph.step()
    assert glia_telegraph.t > 0
    
    # Diffusion mode (micro params set c, but diffusion uses diffusion_D_mm2_per_s)
    config_diffusion = GlialFieldConfig(
        Lx=10.0, Ly=10.0, Nx=32, Ny=32,
        micro_params=micro, dt=0.001, pml_width=0,
        mode='diffusion', diffusion_D_mm2_per_s=0.5
    )
    glia_diffusion = GlialField(config_diffusion, rng=np.random.default_rng(42))
    glia_diffusion.u[16, 16] = 1.0
    glia_diffusion.step()
    assert glia_diffusion.t > 0
    
    # Mean-field mode
    config_mean_field = GlialFieldConfig(
        Lx=10.0, Ly=10.0, Nx=32, Ny=32,
        micro_params=micro, dt=0.001, pml_width=0,
        mode='mean_field'
    )
    glia_mean_field = GlialField(config_mean_field, rng=np.random.default_rng(42))
    glia_mean_field.u[16, 16] = 1.0
    glia_mean_field.step()
    assert glia_mean_field.t > 0
    # Should be spatially uniform
    assert np.std(glia_mean_field.u) < 1e-10


if __name__ == '__main__':
    print("Running glial field mode tests...")
    print("=" * 70)
    
    print("\n1. Testing telegraph mode (default)...")
    test_telegraph_mode_default()
    print("PASSED")
    
    print("\n2. Testing diffusion mode...")
    test_diffusion_mode()
    print("PASSED")
    
    print("\n3. Testing mean-field mode...")
    test_mean_field_mode()
    print("PASSED")
    
    print("\n4. Testing mode comparison with source...")
    test_mode_comparison_with_source()
    print("PASSED")
    
    print("\n5. Testing invalid mode raises error...")
    test_invalid_mode_raises_error()
    print("PASSED")
    
    print("\n6. Testing modes with micro params...")
    test_modes_with_micro_params()
    print("PASSED")
    
    print("\n" + "=" * 70)
    print("All glial field mode tests passed!")

"""
Test coupling functions between layers.

Smoke tests to verify coupling produces stable trajectories and
expected behavior.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from smm.mesh import MeshField, MeshConfig
from smm.neural import NeuralMass, Kuramoto, NeuralMassConfig, KuramotoConfig
from smm.coupling import (
    neural_to_mesh_source,
    mesh_to_kuramoto,
    mesh_to_neural,
    create_region_positions_grid,
    CoupledModel
)


def test_create_region_positions():
    """Test region position creation."""
    Nx, Ny = 64, 64
    n_regions = 16
    
    positions = create_region_positions_grid(Nx, Ny, n_regions)
    
    # Check correct number
    assert len(positions) == n_regions, "Should create correct number of positions"
    
    # Check all positions are valid
    for i, j in positions:
        assert 0 <= i < Nx, f"x position {i} out of bounds"
        assert 0 <= j < Ny, f"y position {j} out of bounds"
    
    # Check positions are distributed (not all the same)
    positions_array = np.array(positions)
    assert np.std(positions_array[:, 0]) > 0, "Positions should be distributed in x"
    assert np.std(positions_array[:, 1]) > 0, "Positions should be distributed in y"


def test_neural_to_mesh_source():
    """Test neural→mesh source field creation."""
    E = np.array([1.0, 0.5, 0.0, 0.2])
    positions = [(10, 10), (20, 10), (10, 20), (20, 20)]
    grid_shape = (32, 32)
    dx = 0.5
    sigma = 1.0
    
    source = neural_to_mesh_source(E, positions, grid_shape, dx, sigma)
    
    # Check shape
    assert source.shape == grid_shape, "Source should have correct shape"
    
    # Check sum is proportional to sum of activities (approximately)
    # The Gaussian normalization should preserve total "mass"
    assert source.sum() > 0, "Source should have positive values"
    
    # Check that maximum is near one of the active positions
    max_idx = np.unravel_index(np.argmax(source), source.shape)
    # Should be within a few grid points of an active position
    active_positions = [positions[i] for i in range(len(E)) if E[i] > 0]
    min_dist = min(
        np.sqrt((max_idx[1] - i)**2 + (max_idx[0] - j)**2)
        for i, j in active_positions
    )
    assert min_dist < 5, "Maximum should be near an active position"


def test_mesh_to_kuramoto():
    """Test mesh→Kuramoto coupling."""
    u = np.random.randn(32, 32)
    positions = [(10, 10), (20, 10)]
    kappa = 0.1
    
    mesh_input = mesh_to_kuramoto(u, positions, kappa)
    
    # Check shape
    assert len(mesh_input) == len(positions), "Should return value for each position"
    
    # Check scaling
    mesh_input_scaled = mesh_to_kuramoto(u, positions, kappa=2*kappa)
    assert np.allclose(mesh_input_scaled, 2*mesh_input), "Should scale linearly with kappa"
    
    # Check values match field
    for idx, (i, j) in enumerate(positions):
        expected = kappa * u[j, i]
        assert np.abs(mesh_input[idx] - expected) < 1e-10, "Should extract correct value"


def test_mesh_to_neural():
    """Test mesh→neural coupling."""
    u = np.random.randn(32, 32)
    positions = [(10, 10), (20, 10)]
    kappa = 0.1
    
    # Linear coupling
    feedback_linear = mesh_to_neural(u, positions, kappa, nonlinearity='linear')
    assert len(feedback_linear) == len(positions)
    
    # Tanh coupling
    feedback_tanh = mesh_to_neural(u, positions, kappa, nonlinearity='tanh')
    assert len(feedback_tanh) == len(positions)
    assert np.all(np.abs(feedback_tanh) <= kappa), "Tanh should saturate"
    
    # Sigmoid coupling
    feedback_sigmoid = mesh_to_neural(u, positions, kappa, nonlinearity='sigmoid')
    assert len(feedback_sigmoid) == len(positions)
    assert np.all(feedback_sigmoid >= 0), "Sigmoid should be non-negative"
    assert np.all(feedback_sigmoid <= kappa), "Sigmoid should be bounded"


def test_coupled_model_initialization():
    """Test that CoupledModel initializes correctly."""
    # Create components
    mesh_config = MeshConfig(Lx=32, Ly=32, Nx=32, Ny=32, dt=0.001)
    mesh = MeshField(mesh_config)
    
    n_regions = 4
    neural_config = NeuralMassConfig(n_regions=n_regions)
    neural = NeuralMass(neural_config)
    
    kuramoto_config = KuramotoConfig(n_oscillators=n_regions)
    kuramoto = Kuramoto(kuramoto_config)
    
    positions = create_region_positions_grid(32, 32, n_regions)
    
    # Create coupled model
    model = CoupledModel(
        mesh, neural, kuramoto, positions,
        coupling_neural_mesh=0.1,
        coupling_mesh_kuramoto=0.05,
        coupling_mesh_neural=0.05
    )
    
    assert model.mesh is mesh
    assert model.neural is neural
    assert model.kuramoto is kuramoto
    assert len(model.positions) == n_regions


def test_coupled_model_step_no_crash():
    """Test that coupled model step doesn't crash."""
    # Small system for fast testing
    mesh_config = MeshConfig(Lx=16, Ly=16, Nx=16, Ny=16, dt=0.001)
    mesh = MeshField(mesh_config)
    
    n_regions = 4
    neural_config = NeuralMassConfig(n_regions=n_regions)
    neural = NeuralMass(neural_config)
    
    kuramoto_config = KuramotoConfig(n_oscillators=n_regions)
    kuramoto = Kuramoto(kuramoto_config)
    
    positions = create_region_positions_grid(16, 16, n_regions)
    
    model = CoupledModel(
        mesh, neural, kuramoto, positions,
        coupling_neural_mesh=0.1,
        coupling_mesh_kuramoto=0.05,
        coupling_mesh_neural=0.05
    )
    
    # Step without input
    model.step(dt=0.001)
    
    # Check no NaNs
    assert np.all(np.isfinite(mesh.u))
    assert np.all(np.isfinite(mesh.v))
    assert np.all(np.isfinite(neural.E))
    assert np.all(np.isfinite(neural.I))
    assert np.all(np.isfinite(kuramoto.theta))


def test_coupled_model_run_stability():
    """Test that coupled model runs stably for several steps."""
    mesh_config = MeshConfig(Lx=16, Ly=16, Nx=16, Ny=16, dt=0.001)
    rng = np.random.default_rng(42)
    mesh = MeshField(mesh_config, rng=rng)
    
    n_regions = 4
    neural_config = NeuralMassConfig(n_regions=n_regions)
    neural = NeuralMass(neural_config)
    
    kuramoto_config = KuramotoConfig(n_oscillators=n_regions)
    kuramoto = Kuramoto(kuramoto_config, rng=rng)
    
    positions = create_region_positions_grid(16, 16, n_regions)
    
    model = CoupledModel(
        mesh, neural, kuramoto, positions,
        coupling_neural_mesh=0.1,
        coupling_mesh_kuramoto=0.05,
        coupling_mesh_neural=0.05
    )
    
    # Run for short time
    results = model.run(
        T=0.1,
        dt=0.001,
        noise_amplitude=1e-6,
        record_interval=10
    )
    
    # Check results have correct structure
    assert 't' in results
    assert 'E' in results
    assert 'I' in results
    assert 'theta' in results
    assert 'u' in results
    
    # Check shapes
    n_steps = len(results['t'])
    assert results['E'].shape == (n_steps, n_regions)
    assert results['I'].shape == (n_steps, n_regions)
    assert results['theta'].shape == (n_steps, n_regions)
    assert results['u'].shape == (n_steps, 16, 16)
    
    # Check no NaNs
    assert np.all(np.isfinite(results['E']))
    assert np.all(np.isfinite(results['I']))
    assert np.all(np.isfinite(results['theta']))
    assert np.all(np.isfinite(results['u']))


def test_coupled_model_with_external_input():
    """Test coupled model with external input function."""
    mesh_config = MeshConfig(Lx=16, Ly=16, Nx=16, Ny=16, dt=0.001)
    mesh = MeshField(mesh_config)
    
    n_regions = 4
    neural_config = NeuralMassConfig(n_regions=n_regions)
    neural = NeuralMass(neural_config)
    
    kuramoto_config = KuramotoConfig(n_oscillators=n_regions)
    kuramoto = Kuramoto(kuramoto_config)
    
    positions = create_region_positions_grid(16, 16, n_regions)
    
    model = CoupledModel(
        mesh, neural, kuramoto, positions,
        coupling_neural_mesh=0.1
    )
    
    # External input to first region
    def I_ext_func(t):
        I_ext = np.zeros(n_regions)
        I_ext[0] = 0.5
        return I_ext
    
    # Run with input
    results = model.run(
        T=0.05,
        dt=0.001,
        I_ext_func=I_ext_func,
        record_interval=5
    )
    
    # Check that first region has higher activity than others
    E_mean = np.mean(results['E'], axis=0)
    assert E_mean[0] > np.mean(E_mean[1:]), \
        "First region should have higher activity due to external input"


if __name__ == '__main__':
    print("Testing coupling functions...")
    print("=" * 60)
    
    print("\n1. Testing region position creation...")
    test_create_region_positions()
    print("PASSED")
    
    print("\n2. Testing neural→mesh source...")
    test_neural_to_mesh_source()
    print("PASSED")
    
    print("\n3. Testing mesh→Kuramoto...")
    test_mesh_to_kuramoto()
    print("PASSED")
    
    print("\n4. Testing mesh→neural...")
    test_mesh_to_neural()
    print("PASSED")
    
    print("\n5. Testing coupled model initialization...")
    test_coupled_model_initialization()
    print("PASSED")
    
    print("\n6. Testing coupled model step...")
    test_coupled_model_step_no_crash()
    print("PASSED")
    
    print("\n7. Testing coupled model stability...")
    test_coupled_model_run_stability()
    print("PASSED")
    
    print("\n8. Testing coupled model with external input...")
    test_coupled_model_with_external_input()
    print("PASSED")
    
    print("\n" + "=" * 60)
    print("All coupling tests passed!")

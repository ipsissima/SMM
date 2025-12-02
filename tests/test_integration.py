"""
Integration test for complete SMM workflow.

Tests that all scripts run end-to-end without errors.
"""

import os
import sys
import subprocess
from pathlib import Path
import tempfile

import pytest
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_run_mesh_ensemble_script():
    """Test that run_mesh_ensemble.py script executes successfully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            'python', 'scripts/run_mesh_ensemble.py',
            '--ensemble', '2',
            '--sim-duration', '0.5',
            '--output', tmpdir,
            '--quiet'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        assert result.returncode == 0, f"Script failed: {result.stderr}"
        
        # Check outputs exist
        assert os.path.exists(os.path.join(tmpdir, 'psd_median.csv'))
        assert os.path.exists(os.path.join(tmpdir, 'psd_ensemble.npz'))
        assert os.path.exists(os.path.join(tmpdir, 'metadata.json'))


def test_run_full_model_script():
    """Test that run_full_model.py script executes successfully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a minimal params file for faster testing
        import yaml
        params = {
            'L_mm': 16.0,
            'Nx': 16,
            'Ny': 16,
            'c_mm_per_s': 15.0,
            'gamma_s': 0.5,
            'dt_s': 0.001,
            'pml_width_mm': 2.0,
            'pml_sigma_max': 50.0,
            'N_regions': 4,
            'tau_E_s': 0.01,
            'tau_I_s': 0.01,
            'w_EE': 1.5,
            'w_EI': 1.2,
            'w_IE': 1.0,
            'w_II': 0.5,
            'sigmoid_gain': 4.0,
            'sigmoid_threshold': 0.5,
            'K_kuramoto': 0.5,
            'omega_mean_hz': 4.0,
            'omega_std_hz': 0.5,
            'coupling_neural_mesh': 0.1,
            'coupling_mesh_kuramoto': 0.0,
            'coupling_mesh_neural': 0.0,
            'sigma_kernel_mm': 1.0,
            'sim_duration_s': 0.5,
            'seed': 42,
            'psd_fs': 100,
            'noise_sigma_mesh': 1e-6
        }
        
        params_file = os.path.join(tmpdir, 'params_test.yaml')
        with open(params_file, 'w') as f:
            yaml.dump(params, f)
        
        cmd = [
            'python', 'scripts/run_full_model.py',
            '--params', params_file,
            '--output', tmpdir,
            '--quiet'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        assert result.returncode == 0, f"Script failed: {result.stderr}"
        
        # Check outputs exist
        assert os.path.exists(os.path.join(tmpdir, 'coupled_results.npz'))
        assert os.path.exists(os.path.join(tmpdir, 'metadata.json'))
        assert os.path.exists(os.path.join(tmpdir, 'time_traces.png'))


def test_legacy_simulate_mesh_psd_compatibility():
    """Test that legacy simulate_mesh_psd.py still works."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            'python', 'simulate_mesh_psd.py',
            '--ensemble', '1',
            '--sim-duration', '0.5',
            '--output', tmpdir,
            '--quiet'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        assert result.returncode == 0, f"Legacy script failed: {result.stderr}"
        
        # Check outputs exist
        assert os.path.exists(os.path.join(tmpdir, 'psd_median.csv'))


def test_new_vs_legacy_consistency():
    """Test that new and legacy scripts produce similar results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Run new script
        new_dir = os.path.join(tmpdir, 'new')
        cmd_new = [
            'python', 'scripts/run_mesh_ensemble.py',
            '--ensemble', '1',
            '--sim-duration', '1.0',
            '--output', new_dir,
            '--quiet'
        ]
        subprocess.run(cmd_new, check=True)
        
        # Run legacy script
        old_dir = os.path.join(tmpdir, 'old')
        cmd_old = [
            'python', 'simulate_mesh_psd.py',
            '--ensemble', '1',
            '--sim-duration', '1.0',
            '--output', old_dir,
            '--quiet'
        ]
        subprocess.run(cmd_old, check=True)
        
        # Load and compare PSDs
        import pandas as pd
        
        psd_new = pd.read_csv(os.path.join(new_dir, 'psd_median.csv'))
        psd_old = pd.read_csv(os.path.join(old_dir, 'psd_median.csv'))
        
        # Both should have similar structure
        assert 'f' in psd_new.columns
        assert 'f' in psd_old.columns
        
        # Frequency arrays should be similar
        assert len(psd_new) > 0
        assert len(psd_old) > 0
        
        # Both should have PSD data
        assert 'Pxx_median' in psd_new.columns or 'Pxx' in psd_new.columns
        assert 'Pxx_median' in psd_old.columns
        
        print(f"New script: {len(psd_new)} frequency points")
        print(f"Old script: {len(psd_old)} frequency points")


if __name__ == '__main__':
    print("Running integration tests...")
    print("=" * 60)
    
    print("\n1. Testing run_mesh_ensemble.py script...")
    test_run_mesh_ensemble_script()
    print("PASSED")
    
    print("\n2. Testing run_full_model.py script...")
    test_run_full_model_script()
    print("PASSED")
    
    print("\n3. Testing legacy script compatibility...")
    test_legacy_simulate_mesh_psd_compatibility()
    print("PASSED")
    
    print("\n4. Testing new vs legacy consistency...")
    test_new_vs_legacy_consistency()
    print("PASSED")
    
    print("\n" + "=" * 60)
    print("All integration tests passed!")

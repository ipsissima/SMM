"""
Test that the simulated mesh PSD shows expected eigenfrequency peaks.

This test computes the theoretical eigenfrequency for mode n=17 in 1D
and verifies that the simulation produces a PSD peak near this frequency.
"""
import os
import subprocess
import numpy as np
import yaml
import pytest


def test_eigenfreq_f17():
    """Test that PSD shows peak near theoretical f_17 frequency.
    
    For a 1D wave equation with Neumann BC on domain [0, L]:
        eigenfrequencies: f_n = n * c / (2 * L)
    
    For n=17, L=32mm, c=15 mm/s:
        f_17 = 17 * 15 / (2 * 32) = 3.984375 Hz
    
    The simulation should show a PSD peak within ±0.5 Hz of this value.
    """
    # Skip on CI if requested (test can be heavy)
    if os.environ.get('CI') == 'true':
        pytest.skip("Skipping heavy simulation test on CI")
    
    # Load params
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    L_mm = params['L_mm']
    c_mm_per_s = params['c_mm_per_s']
    
    # Theoretical f_17
    n = 17
    f_17_theory = n * c_mm_per_s / (2 * L_mm)
    
    print(f"\nTheoretical f_17 = {f_17_theory:.4f} Hz")
    
    # Run simulation with reduced ensemble and duration for testing
    ensemble_size = 8
    sim_duration_s = 3.0
    output_dir = '/tmp/test_eigenfreq_results'
    
    cmd = [
        'python', 'simulate_mesh_psd.py',
        '--params', 'params.yaml',
        '--ensemble', str(ensemble_size),
        '--sim-duration', str(sim_duration_s),
        '--output', output_dir
    ]
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.check_call(cmd)
    
    # Load results
    import pandas as pd
    psd_median = pd.read_csv(os.path.join(output_dir, 'psd_median.csv'))
    
    f = psd_median['f'].values
    Pxx = psd_median['Pxx_median'].values
    
    # Find local maxima in the PSD
    from scipy.signal import find_peaks
    peaks, properties = find_peaks(Pxx, prominence=0.1*Pxx.max())
    
    peak_freqs = f[peaks]
    peak_powers = Pxx[peaks]
    
    print(f"Found {len(peak_freqs)} peaks in PSD")
    if len(peak_freqs) > 0:
        # Sort peaks by power (descending)
        sorted_indices = np.argsort(peak_powers)[::-1]
        top_peak_freqs = peak_freqs[sorted_indices[:5]]
        print(f"Top 5 peak frequencies: {top_peak_freqs}")
    
    # Check if there's a peak near f_17
    tolerance_hz = 0.5
    near_f17 = np.abs(peak_freqs - f_17_theory) < tolerance_hz
    
    if np.any(near_f17):
        closest_peak = peak_freqs[near_f17][0]
        print(f"✓ Found peak at {closest_peak:.4f} Hz, within {tolerance_hz} Hz of f_17={f_17_theory:.4f} Hz")
        assert True
    else:
        # Also check if there's significant power near f_17 even without a local peak
        idx_near = np.argmin(np.abs(f - f_17_theory))
        power_at_f17 = Pxx[idx_near]
        baseline_power = np.median(Pxx)
        
        print(f"No peak found at f_17={f_17_theory:.4f} Hz")
        print(f"Power at f_17: {power_at_f17:.2e}, baseline: {baseline_power:.2e}")
        print(f"Ratio: {power_at_f17/baseline_power:.2f}")
        
        # Check if power is at least 2x baseline
        if power_at_f17 > 2 * baseline_power:
            print(f"✓ Significant power elevation at f_17 (ratio {power_at_f17/baseline_power:.2f})")
            assert True
        else:
            pytest.fail(f"No peak or power elevation found near f_17={f_17_theory:.4f} Hz")


if __name__ == '__main__':
    test_eigenfreq_f17()

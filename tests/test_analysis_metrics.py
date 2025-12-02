"""
Test analysis metrics on synthetic data.

Verifies that PSD and phase gradient coherence functions work correctly
on known synthetic signals.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from smm.analysis import (
    compute_psd,
    compute_phase_gradient_coherence,
    compute_median_and_ci,
    compute_ensemble_psd,
    find_peaks_in_psd
)


def test_compute_psd_sine_wave():
    """Test PSD computation on pure sine wave."""
    # Generate 5 Hz sine wave
    fs = 100.0
    T = 10.0
    t = np.linspace(0, T, int(T * fs))
    f_signal = 5.0
    signal_data = np.sin(2 * np.pi * f_signal * t)
    
    # Compute PSD
    f, Pxx = compute_psd(signal_data, fs)
    
    # Find peak
    peak_idx = np.argmax(Pxx)
    peak_freq = f[peak_idx]
    
    print(f"Peak frequency: {peak_freq:.2f} Hz (expected: {f_signal:.2f} Hz)")
    
    # Check peak is near 5 Hz
    assert np.abs(peak_freq - f_signal) < 0.5, \
        f"Peak should be near {f_signal} Hz, got {peak_freq:.2f} Hz"


def test_compute_psd_defaults():
    """Test that PSD handles default parameters correctly."""
    # Short signal
    signal_short = np.random.randn(100)
    f, Pxx = compute_psd(signal_short, fs=100)
    
    assert len(f) > 0, "Should return frequency array"
    assert len(Pxx) > 0, "Should return PSD array"
    assert len(f) == len(Pxx), "Frequency and PSD should have same length"


def test_find_peaks_in_psd():
    """Test peak finding in PSD."""
    # Create synthetic PSD with known peaks
    f = np.linspace(0, 50, 1000)
    Pxx = np.exp(-f/10)  # Decaying background
    
    # Add peaks at 5, 10, 15 Hz
    peak_freqs_true = [5.0, 10.0, 15.0]
    for f_peak in peak_freqs_true:
        idx = np.argmin(np.abs(f - f_peak))
        Pxx[idx] += 10.0
    
    # Find peaks
    peak_freqs, peak_powers = find_peaks_in_psd(f, Pxx, prominence_factor=0.5)
    
    print(f"Found peaks at: {peak_freqs}")
    print(f"Expected peaks at: {peak_freqs_true}")
    
    # Should find 3 peaks
    assert len(peak_freqs) == 3, f"Should find 3 peaks, found {len(peak_freqs)}"
    
    # Check each peak is close to expected
    for f_true in peak_freqs_true:
        min_dist = np.min(np.abs(peak_freqs - f_true))
        assert min_dist < 0.5, f"Should find peak near {f_true} Hz"


def test_compute_median_and_ci():
    """Test median and confidence interval computation."""
    # Create ensemble with known statistics
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, (100, 50))  # 100 samples, 50 points each
    
    median, ci_lower, ci_upper = compute_median_and_ci(data, axis=0)
    
    # Check shapes
    assert median.shape == (50,), "Median should have correct shape"
    assert ci_lower.shape == (50,), "Lower CI should have correct shape"
    assert ci_upper.shape == (50,), "Upper CI should have correct shape"
    
    # Check ordering
    assert np.all(ci_lower <= median), "Lower CI should be <= median"
    assert np.all(median <= ci_upper), "Median should be <= upper CI"
    
    # For normal distribution, ~95% should be within CI
    # Check a few points
    for i in [0, 10, 20, 30, 40]:
        within_ci = np.sum((data[:, i] >= ci_lower[i]) & (data[:, i] <= ci_upper[i]))
        fraction = within_ci / len(data[:, i])
        assert fraction > 0.90, f"At least 90% should be within CI, got {fraction:.2f}"


def test_compute_ensemble_psd():
    """Test ensemble PSD computation."""
    # Create ensemble of sine waves with noise
    fs = 100.0
    T = 5.0
    t = np.linspace(0, T, int(T * fs))
    n_ensemble = 10
    f_signal = 5.0
    
    signals = np.zeros((n_ensemble, len(t)))
    rng = np.random.default_rng(42)
    for i in range(n_ensemble):
        signals[i] = np.sin(2 * np.pi * f_signal * t) + 0.1 * rng.normal(0, 1, len(t))
    
    # Compute ensemble PSD
    results = compute_ensemble_psd(signals, fs)
    
    # Check structure
    assert 'f' in results
    assert 'Pxx_median' in results
    assert 'Pxx_lower' in results
    assert 'Pxx_upper' in results
    assert 'Pxx_all' in results
    
    # Check shapes
    n_freqs = len(results['f'])
    assert results['Pxx_median'].shape == (n_freqs,)
    assert results['Pxx_all'].shape == (n_ensemble, n_freqs)
    
    # Check peak is near 5 Hz in median
    peak_idx = np.argmax(results['Pxx_median'])
    peak_freq = results['f'][peak_idx]
    assert np.abs(peak_freq - f_signal) < 1.0, \
        f"Median PSD peak should be near {f_signal} Hz"


def test_phase_gradient_coherence_traveling_wave():
    """Test phase gradient coherence on synthetic traveling wave."""
    # Create 8x8 grid
    nx, ny = 8, 8
    n_time = 1000
    fs = 100.0
    t = np.linspace(0, n_time/fs, n_time)
    
    # Grid positions
    positions = np.array([[i, j] for j in range(ny) for i in range(nx)])
    
    # Generate traveling wave in +x direction
    signals = np.zeros((nx*ny, n_time))
    wave_freq = 4.0  # Hz
    wave_speed = 2.0  # grid units/s
    
    for idx, (i, j) in enumerate(positions):
        phase_offset = 2*np.pi*wave_freq*i/wave_speed
        signals[idx, :] = np.sin(2*np.pi*wave_freq*t - phase_offset)
    
    # Compute coherence
    C = compute_phase_gradient_coherence(
        signals, positions, fs,
        band=(2, 8), window_s=0.5, step_s=0.25
    )
    
    mean_coherence = C.mean()
    print(f"Mean coherence for traveling wave: {mean_coherence:.3f}")
    
    # For perfect traveling wave, coherence should be high
    assert mean_coherence > 0.5, \
        f"Expected coherence >0.5 for traveling wave, got {mean_coherence:.3f}"


def test_phase_gradient_coherence_random_noise():
    """Test phase gradient coherence on random noise."""
    # Create 8x8 grid
    nx, ny = 8, 8
    n_time = 1000
    fs = 100.0
    
    # Grid positions
    positions = np.array([[i, j] for j in range(ny) for i in range(nx)])
    
    # Generate white noise
    rng = np.random.default_rng(42)
    signals = rng.normal(0, 1, (nx*ny, n_time))
    
    # Compute coherence
    C = compute_phase_gradient_coherence(
        signals, positions, fs,
        band=(2, 8), window_s=0.5, step_s=0.25
    )
    
    mean_coherence = C.mean()
    print(f"Mean coherence for random noise: {mean_coherence:.3f}")
    
    # For random noise, coherence should be low
    assert mean_coherence < 0.3, \
        f"Expected coherence <0.3 for noise, got {mean_coherence:.3f}"


def test_phase_gradient_coherence_standing_wave():
    """Test phase gradient coherence on standing wave."""
    # Create 8x8 grid
    nx, ny = 8, 8
    n_time = 1000
    fs = 100.0
    t = np.linspace(0, n_time/fs, n_time)
    
    # Grid positions
    positions = np.array([[i, j] for j in range(ny) for i in range(nx)])
    
    # Generate standing wave (amplitude varies in space, phase constant)
    signals = np.zeros((nx*ny, n_time))
    wave_freq = 4.0  # Hz
    
    for idx, (i, j) in enumerate(positions):
        # Spatial amplitude modulation
        amplitude = np.sin(np.pi * i / nx)
        signals[idx, :] = amplitude * np.sin(2*np.pi*wave_freq*t)
    
    # Compute coherence
    C = compute_phase_gradient_coherence(
        signals, positions, fs,
        band=(2, 8), window_s=0.5, step_s=0.25
    )
    
    mean_coherence = C.mean()
    print(f"Mean coherence for standing wave: {mean_coherence:.3f}")
    
    # Standing wave should have lower coherence than traveling wave
    # but higher than noise (phases are synchronized, but no propagation)
    # This is a bit ambiguous, so we just check it's computed without error
    assert 0 <= mean_coherence <= 1, "Coherence should be in [0, 1]"


def test_phase_gradient_coherence_shapes():
    """Test that phase gradient coherence returns correct shapes."""
    n_sensors = 16
    n_time = 500
    fs = 100.0
    
    # Random signals and positions
    rng = np.random.default_rng(42)
    signals = rng.normal(0, 1, (n_sensors, n_time))
    positions = rng.uniform(0, 10, (n_sensors, 2))
    
    # Compute coherence
    C = compute_phase_gradient_coherence(signals, positions, fs)
    
    # Check shape
    assert C.shape == (n_sensors,), \
        f"Coherence should have shape ({n_sensors},), got {C.shape}"
    
    # Check range
    assert np.all(C >= 0) and np.all(C <= 1), \
        "Coherence should be in [0, 1]"


if __name__ == '__main__':
    print("Testing analysis metrics...")
    print("=" * 60)
    
    print("\n1. Testing PSD on sine wave...")
    test_compute_psd_sine_wave()
    print("PASSED")
    
    print("\n2. Testing PSD with defaults...")
    test_compute_psd_defaults()
    print("PASSED")
    
    print("\n3. Testing peak finding...")
    test_find_peaks_in_psd()
    print("PASSED")
    
    print("\n4. Testing median and CI...")
    test_compute_median_and_ci()
    print("PASSED")
    
    print("\n5. Testing ensemble PSD...")
    test_compute_ensemble_psd()
    print("PASSED")
    
    print("\n6. Testing phase gradient coherence on traveling wave...")
    test_phase_gradient_coherence_traveling_wave()
    print("PASSED")
    
    print("\n7. Testing phase gradient coherence on noise...")
    test_phase_gradient_coherence_random_noise()
    print("PASSED")
    
    print("\n8. Testing phase gradient coherence on standing wave...")
    test_phase_gradient_coherence_standing_wave()
    print("PASSED")
    
    print("\n9. Testing phase gradient coherence shapes...")
    test_phase_gradient_coherence_shapes()
    print("PASSED")
    
    print("\n" + "=" * 60)
    print("All analysis tests passed!")

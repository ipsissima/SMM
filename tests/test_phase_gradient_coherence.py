"""
Test phase gradient coherence computation on synthetic data.

Creates traveling wave and random noise signals to validate that the
coherence metric correctly distinguishes coherent from incoherent patterns.
"""
import numpy as np
import pytest
from analysis.metrics import compute_phase_gradient_coherence


def test_phase_gradient_coherence_traveling_wave():
    """Test that coherence is high for a traveling wave pattern."""
    # Create 8x8 grid
    nx, ny = 8, 8
    n_time = 1000
    fs = 100.0
    t = np.linspace(0, n_time/fs, n_time)
    
    # Grid positions
    positions = []
    for j in range(ny):
        for i in range(nx):
            positions.append((i, j))
    
    # Generate traveling wave moving in +x direction
    # Phase varies linearly with x position
    signals = np.zeros((nx*ny, n_time))
    wave_freq = 4.0  # Hz
    wave_speed = 2.0  # grid units per second
    
    for idx, (i, j) in enumerate(positions):
        # Phase offset proportional to x position
        phase_offset = 2*np.pi*wave_freq*i/wave_speed
        signals[idx, :] = np.sin(2*np.pi*wave_freq*t - phase_offset)
    
    # Compute coherence
    C = compute_phase_gradient_coherence(
        signals, positions, fs, 
        band=(2, 8), window_s=0.5, step_s=0.25
    )
    
    mean_coherence = C.mean()
    print(f"Mean coherence for traveling wave: {mean_coherence:.3f}")
    
    # For a perfect traveling wave, coherence should be high
    assert mean_coherence > 0.6, \
        f"Expected coherence >0.6 for traveling wave, got {mean_coherence:.3f}"


def test_phase_gradient_coherence_noise():
    """Test that coherence is low for random noise."""
    # Create 8x8 grid
    nx, ny = 8, 8
    n_time = 1000
    fs = 100.0
    
    # Grid positions
    positions = []
    for j in range(ny):
        for i in range(nx):
            positions.append((i, j))
    
    # Generate white noise
    np.random.seed(42)
    signals_noise = np.random.randn(nx*ny, n_time)
    
    # Compute coherence
    C_noise = compute_phase_gradient_coherence(
        signals_noise, positions, fs,
        band=(2, 8), window_s=0.5, step_s=0.25
    )
    
    mean_coherence_noise = C_noise.mean()
    print(f"Mean coherence for white noise: {mean_coherence_noise:.3f}")
    
    # For random noise, coherence should be low
    assert mean_coherence_noise < 0.2, \
        f"Expected coherence <0.2 for noise, got {mean_coherence_noise:.3f}"


def test_phase_gradient_coherence_mixed():
    """Test coherence on mixed signal (coherent region + noise)."""
    # Create 8x8 grid
    nx, ny = 8, 8
    n_time = 1000
    fs = 100.0
    t = np.linspace(0, n_time/fs, n_time)
    
    # Grid positions
    positions = []
    for j in range(ny):
        for i in range(nx):
            positions.append((i, j))
    
    signals = np.zeros((nx*ny, n_time))
    
    # Left half: traveling wave
    # Right half: noise
    wave_freq = 4.0
    wave_speed = 2.0
    np.random.seed(42)
    
    for idx, (i, j) in enumerate(positions):
        if i < nx // 2:
            # Coherent wave
            phase_offset = 2*np.pi*wave_freq*i/wave_speed
            signals[idx, :] = np.sin(2*np.pi*wave_freq*t - phase_offset)
        else:
            # Random noise
            signals[idx, :] = np.random.randn(n_time)
    
    # Compute coherence
    C = compute_phase_gradient_coherence(
        signals, positions, fs,
        band=(2, 8), window_s=0.5, step_s=0.25
    )
    
    # Split coherence by region
    C_left = []
    C_right = []
    for idx, (i, j) in enumerate(positions):
        if i < nx // 2:
            C_left.append(C[idx])
        else:
            C_right.append(C[idx])
    
    mean_C_left = np.mean(C_left)
    mean_C_right = np.mean(C_right)
    
    print(f"Mean coherence (left, wave): {mean_C_left:.3f}")
    print(f"Mean coherence (right, noise): {mean_C_right:.3f}")
    
    # Left (wave) should have higher coherence than right (noise)
    assert mean_C_left > mean_C_right, \
        "Wave region should have higher coherence than noise region"
    
    # And left should be reasonably high
    assert mean_C_left > 0.4, \
        f"Wave region coherence should be >0.4, got {mean_C_left:.3f}"


if __name__ == '__main__':
    print("Testing phase gradient coherence...")
    print("=" * 60)
    
    print("\n1. Traveling wave test:")
    test_phase_gradient_coherence_traveling_wave()
    print("PASSED")
    
    print("\n2. White noise test:")
    test_phase_gradient_coherence_noise()
    print("PASSED")
    
    print("\n3. Mixed signal test:")
    test_phase_gradient_coherence_mixed()
    print("PASSED")
    
    print("\n" + "=" * 60)
    print("All tests passed!")

"""
Analysis metrics for spatial-temporal signal processing.

Provides functions for computing phase gradient coherence and power spectral
density from neural/mesh signals.
"""
import numpy as np
from scipy import signal


def compute_welch_psd(signal_data, fs, nperseg, noverlap):
    """Compute power spectral density using Welch's method.
    
    Parameters
    ----------
    signal_data : ndarray
        1D time series signal
    fs : float
        Sampling frequency in Hz
    nperseg : int
        Length of each segment for Welch's method
    noverlap : int
        Number of points to overlap between segments
        
    Returns
    -------
    f : ndarray
        Array of sample frequencies
    Pxx : ndarray
        Power spectral density
        
    Examples
    --------
    >>> import numpy as np
    >>> t = np.linspace(0, 10, 1000)
    >>> sig = np.sin(2*np.pi*5*t) + 0.1*np.random.randn(1000)
    >>> f, Pxx = compute_welch_psd(sig, fs=100, nperseg=256, noverlap=128)
    >>> peak_freq = f[np.argmax(Pxx)]
    >>> print(f"Peak at {peak_freq:.1f} Hz")
    """
    f, Pxx = signal.welch(signal_data, fs=fs, 
                          nperseg=nperseg, 
                          noverlap=noverlap,
                          window='hamming')
    return f, Pxx


def compute_phase_gradient_coherence(signals, positions, fs, band=(1, 8), 
                                    window_s=0.5, step_s=0.25):
    """Compute phase gradient coherence metric for spatial signals.
    
    This implements the phase gradient coherence algorithm:
    1. Bandpass filter signals in the specified frequency band
    2. Compute analytic phase via Hilbert transform
    3. Compute local spatial phase gradients using neighbor differences
    4. Compute coherence metric c(r,t) and time-average to get C_i
    
    The coherence at each location measures how well the instantaneous phase
    gradients align with the mean phase gradient direction (i.e., whether
    there is a consistent traveling wave pattern).
    
    Parameters
    ----------
    signals : ndarray
        2D array of shape (n_sensors, n_timepoints) containing signals
    positions : list of tuples or ndarray
        Either list of (x, y) coordinates or 2D array of shape (n_sensors, 2).
        For grid data, positions can also be a list of lists indicating
        neighbors for each sensor.
    fs : float
        Sampling frequency in Hz
    band : tuple
        Frequency band (low, high) in Hz for bandpass filter
    window_s : float
        Window duration in seconds for coherence computation
    step_s : float
        Step size in seconds for sliding window
        
    Returns
    -------
    C : ndarray
        Time-averaged coherence for each sensor/position (n_sensors,)
        
    Notes
    -----
    For a traveling wave with consistent direction, C_i should be high (close to 1).
    For random phase patterns or standing waves, C_i will be low (close to 0).
    
    Examples
    --------
    >>> # Create synthetic traveling wave on 8x8 grid
    >>> nx, ny = 8, 8
    >>> n_time = 1000
    >>> fs = 100.0
    >>> t = np.linspace(0, n_time/fs, n_time)
    >>> 
    >>> # Positions on grid
    >>> positions = []
    >>> for j in range(ny):
    >>>     for i in range(nx):
    >>>         positions.append((i, j))
    >>> 
    >>> # Generate traveling wave (wave moving in +x direction)
    >>> signals = np.zeros((nx*ny, n_time))
    >>> wave_freq = 4.0  # Hz
    >>> wave_speed = 2.0  # grid units per second
    >>> for idx, (i, j) in enumerate(positions):
    >>>     phase_offset = 2*np.pi*wave_freq*i/wave_speed
    >>>     signals[idx, :] = np.sin(2*np.pi*wave_freq*t - phase_offset)
    >>> 
    >>> C = compute_phase_gradient_coherence(signals, positions, fs, 
    >>>                                      band=(2, 8), window_s=0.5)
    >>> print(f"Mean coherence: {C.mean():.2f}")  # Should be high
    """
    n_sensors, n_timepoints = signals.shape
    
    # Design bandpass filter
    nyq = fs / 2
    low, high = band
    sos = signal.butter(4, [low/nyq, high/nyq], btype='band', output='sos')
    
    # Filter all signals and compute analytic phase
    phases = np.zeros_like(signals)
    for i in range(n_sensors):
        # Bandpass filter
        filtered = signal.sosfiltfilt(sos, signals[i, :])
        # Analytic signal via Hilbert transform
        analytic = signal.hilbert(filtered)
        phases[i, :] = np.angle(analytic)
    
    # Build neighbor lists if positions are coordinates
    if isinstance(positions, np.ndarray) and positions.shape[1] == 2:
        # Convert to list of coordinates
        positions = [tuple(p) for p in positions]
    
    # For grid positions, build neighbor list based on proximity
    neighbors = []
    if isinstance(positions[0], tuple) and len(positions[0]) == 2:
        # Build neighbors based on spatial proximity
        for i, (xi, yi) in enumerate(positions):
            neighbor_list = []
            for j, (xj, yj) in enumerate(positions):
                if i != j:
                    # Consider as neighbor if within distance ~1.5 grid units
                    dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                    if dist < 1.5:
                        neighbor_list.append(j)
            neighbors.append(neighbor_list)
    else:
        # Assume positions is already a neighbor list
        neighbors = positions
    
    # Compute windowed coherence
    window_samples = int(window_s * fs)
    step_samples = int(step_s * fs)
    
    n_windows = (n_timepoints - window_samples) // step_samples + 1
    
    coherence_time_series = np.zeros((n_sensors, n_windows))
    
    for win_idx in range(n_windows):
        start = win_idx * step_samples
        end = start + window_samples
        
        # For each sensor, compute phase gradient coherence in this window
        for i in range(n_sensors):
            if len(neighbors[i]) == 0:
                coherence_time_series[i, win_idx] = 0.0
                continue
            
            # Compute phase differences with neighbors
            phase_diffs = []
            for j in neighbors[i]:
                # Phase difference (unwrapped within window)
                delta_phi = phases[i, start:end] - phases[j, start:end]
                # Wrap to [-pi, pi]
                delta_phi = np.arctan2(np.sin(delta_phi), np.cos(delta_phi))
                phase_diffs.append(delta_phi)
            
            phase_diffs = np.array(phase_diffs)  # (n_neighbors, window_samples)
            
            # Compute coherence as consistency of phase gradient direction
            # Use circular mean and variance
            mean_grad = np.mean(phase_diffs, axis=0)  # Average over neighbors
            
            # Coherence measure: how consistent is the phase gradient?
            # Use 1 - circular variance
            # circ_var = 1 - |mean(exp(i*phase_grad))|
            complex_grads = np.exp(1j * phase_diffs)
            mean_complex = np.mean(complex_grads, axis=(0, 1))  # Mean over neighbors and time
            coherence = np.abs(mean_complex)
            
            coherence_time_series[i, win_idx] = coherence
    
    # Time-average coherence for each sensor
    C = np.mean(coherence_time_series, axis=1)
    
    return C


# Example usage demonstration
if __name__ == '__main__':
    print("Example: Computing PSD of a noisy sinusoid")
    print("-" * 50)
    
    # Generate test signal: 5 Hz sine wave with noise
    fs = 100.0
    t = np.linspace(0, 10, int(10*fs))
    sig = np.sin(2*np.pi*5*t) + 0.1*np.random.randn(len(t))
    
    f, Pxx = compute_welch_psd(sig, fs=fs, nperseg=256, noverlap=128)
    peak_freq = f[np.argmax(Pxx)]
    print(f"Peak frequency: {peak_freq:.1f} Hz (expected: 5.0 Hz)")
    
    print("\nExample: Phase gradient coherence on traveling wave")
    print("-" * 50)
    
    # Create 4x4 grid
    nx, ny = 4, 4
    n_time = 500
    t = np.linspace(0, 5, n_time)
    
    # Grid positions
    positions = []
    for j in range(ny):
        for i in range(nx):
            positions.append((i, j))
    
    # Traveling wave moving in +x direction
    signals = np.zeros((nx*ny, n_time))
    wave_freq = 4.0
    wave_speed = 2.0
    
    for idx, (i, j) in enumerate(positions):
        phase_offset = 2*np.pi*wave_freq*i/wave_speed
        signals[idx, :] = np.sin(2*np.pi*wave_freq*t - phase_offset)
    
    C = compute_phase_gradient_coherence(signals, positions, fs=fs,
                                        band=(2, 8), window_s=0.5, step_s=0.25)
    
    print(f"Mean coherence for traveling wave: {C.mean():.2f} (expected: >0.6)")
    
    # Random noise for comparison
    signals_noise = np.random.randn(nx*ny, n_time)
    C_noise = compute_phase_gradient_coherence(signals_noise, positions, fs=fs,
                                               band=(2, 8), window_s=0.5, step_s=0.25)
    
    print(f"Mean coherence for random noise: {C_noise.mean():.2f} (expected: <0.3)")

"""
Analysis metrics for SMM model.

Provides clean, pure functions for:
- Power spectral density (PSD) computation
- Phase gradient coherence
- Statistical utilities

All functions are pure (no I/O), taking NumPy arrays as input and
returning NumPy arrays or scalars.

Contact: Andreu.Ballus@uab.cat
"""

import numpy as np
from scipy import signal
from typing import Optional, Tuple, Dict


def compute_psd(
    signal_data: np.ndarray,
    fs: float,
    nperseg: Optional[int] = None,
    noverlap: Optional[int] = None,
    window: str = 'hamming',
    detrend: str = 'constant'
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute power spectral density using Welch's method.
    
    Wrapper around scipy.signal.welch with sensible defaults.
    
    Parameters
    ----------
    signal_data : ndarray
        1D time series signal
    fs : float
        Sampling frequency in Hz
    nperseg : int, optional
        Length of each segment (default: 256 or len(signal)/8)
    noverlap : int, optional
        Number of points to overlap (default: nperseg // 2)
    window : str
        Window function (default: 'hamming')
    detrend : str
        Detrending method (default: 'constant')
        
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
    >>> f, Pxx = compute_psd(sig, fs=100)
    >>> peak_freq = f[np.argmax(Pxx)]
    >>> print(f"Peak at {peak_freq:.1f} Hz")
    """
    # Set defaults
    if nperseg is None:
        nperseg = min(256, len(signal_data) // 8)
    if noverlap is None:
        noverlap = nperseg // 2
    
    # Adjust if signal is too short
    nperseg = min(nperseg, len(signal_data))
    noverlap = min(noverlap, nperseg - 1)
    
    f, Pxx = signal.welch(
        signal_data,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        window=window,
        detrend=detrend
    )
    
    return f, Pxx


def compute_phase_gradient_coherence(
    signals: np.ndarray,
    positions: np.ndarray,
    fs: float,
    band: Tuple[float, float] = (1, 8),
    window_s: float = 0.5,
    step_s: float = 0.25,
    neighbor_distance: float = 1.5
) -> np.ndarray:
    """Compute phase gradient coherence for spatial signals.
    
    Implements the phase gradient coherence algorithm:
    1. Bandpass filter signals in specified frequency band
    2. Compute analytic phase via Hilbert transform
    3. Compute local spatial phase gradients using neighbor differences
    4. Compute coherence as consistency of phase gradient direction
    
    The coherence at each location measures how well the instantaneous
    phase gradients align with the mean phase gradient direction.
    
    Parameters
    ----------
    signals : ndarray
        2D array of shape (n_sensors, n_timepoints)
    positions : ndarray
        Array of shape (n_sensors, 2) with (x, y) coordinates
    fs : float
        Sampling frequency in Hz
    band : tuple
        Frequency band (low, high) in Hz for bandpass filter
    window_s : float
        Window duration in seconds for coherence computation
    step_s : float
        Step size in seconds for sliding window
    neighbor_distance : float
        Maximum distance to consider as neighbor (in position units)
        
    Returns
    -------
    C : ndarray
        Time-averaged coherence for each sensor (n_sensors,)
        High values (close to 1) indicate coherent traveling waves.
        Low values (close to 0) indicate incoherent or random patterns.
        
    Examples
    --------
    >>> # Create synthetic traveling wave on 8x8 grid
    >>> nx, ny = 8, 8
    >>> n_time = 1000
    >>> fs = 100.0
    >>> t = np.linspace(0, n_time/fs, n_time)
    >>> 
    >>> # Grid positions
    >>> positions = []
    >>> for j in range(ny):
    >>>     for i in range(nx):
    >>>         positions.append([i, j])
    >>> positions = np.array(positions)
    >>> 
    >>> # Traveling wave moving in +x direction
    >>> signals = np.zeros((nx*ny, n_time))
    >>> wave_freq = 4.0  # Hz
    >>> wave_speed = 2.0  # grid units per second
    >>> for idx, (i, j) in enumerate(positions):
    >>>     phase_offset = 2*np.pi*wave_freq*i/wave_speed
    >>>     signals[idx, :] = np.sin(2*np.pi*wave_freq*t - phase_offset)
    >>> 
    >>> C = compute_phase_gradient_coherence(signals, positions, fs, band=(2, 8))
    >>> print(f"Mean coherence: {C.mean():.2f}")  # Should be high for traveling wave
    """
    n_sensors, n_timepoints = signals.shape
    
    # Ensure positions is a NumPy array
    if not isinstance(positions, np.ndarray):
        positions = np.array(positions)
    
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
    
    # Build neighbor lists based on spatial proximity
    neighbors = []
    for i in range(n_sensors):
        # Compute distances to all other sensors
        diffs = positions - positions[i]
        distances = np.sqrt(np.sum(diffs**2, axis=1))
        # Find neighbors within threshold (excluding self)
        neighbor_mask = (distances < neighbor_distance) & (distances > 0)
        neighbors.append(np.where(neighbor_mask)[0].tolist())
    
    # Compute windowed coherence
    window_samples = int(window_s * fs)
    step_samples = int(step_s * fs)
    
    n_windows = max(1, (n_timepoints - window_samples) // step_samples + 1)
    
    coherence_time_series = np.zeros((n_sensors, n_windows))
    
    for win_idx in range(n_windows):
        start = win_idx * step_samples
        end = min(start + window_samples, n_timepoints)
        
        # For each sensor, compute phase gradient coherence in this window
        for i in range(n_sensors):
            if len(neighbors[i]) == 0:
                coherence_time_series[i, win_idx] = 0.0
                continue
            
            # Compute phase differences with neighbors
            phase_diffs = []
            for j in neighbors[i]:
                # Phase difference (wrapped to [-π, π])
                delta_phi = phases[i, start:end] - phases[j, start:end]
                delta_phi = np.arctan2(np.sin(delta_phi), np.cos(delta_phi))
                phase_diffs.append(delta_phi)
            
            phase_diffs = np.array(phase_diffs)  # (n_neighbors, window_samples)
            
            # Compute coherence as consistency of phase gradient direction
            # Use circular mean: mean of exp(i·Δφ)
            complex_grads = np.exp(1j * phase_diffs)
            mean_complex = np.mean(complex_grads, axis=(0, 1))  # Mean over neighbors and time
            coherence = np.abs(mean_complex)
            
            coherence_time_series[i, win_idx] = coherence
    
    # Time-average coherence for each sensor
    C = np.mean(coherence_time_series, axis=1)
    
    return C


def compute_median_and_ci(
    data: np.ndarray,
    axis: int = 0,
    ci_level: float = 0.95
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute median and confidence interval.
    
    Parameters
    ----------
    data : ndarray
        Data array
    axis : int
        Axis along which to compute statistics
    ci_level : float
        Confidence interval level (default: 0.95 for 95% CI)
        
    Returns
    -------
    median : ndarray
        Median values
    ci_lower : ndarray
        Lower confidence bound
    ci_upper : ndarray
        Upper confidence bound
    """
    median = np.median(data, axis=axis)
    
    # Compute percentiles for CI
    alpha = 1 - ci_level
    lower_percentile = 100 * alpha / 2
    upper_percentile = 100 * (1 - alpha / 2)
    
    ci_lower = np.percentile(data, lower_percentile, axis=axis)
    ci_upper = np.percentile(data, upper_percentile, axis=axis)
    
    return median, ci_lower, ci_upper


def compute_ensemble_psd(
    signals: np.ndarray,
    fs: float,
    **psd_kwargs
) -> Dict[str, np.ndarray]:
    """Compute PSD for ensemble of signals.
    
    Parameters
    ----------
    signals : ndarray
        Array of shape (n_ensemble, n_timepoints)
    fs : float
        Sampling frequency in Hz
    **psd_kwargs
        Additional arguments passed to compute_psd
        
    Returns
    -------
    dict
        Dictionary with keys:
        - 'f': frequency array
        - 'Pxx_median': median PSD
        - 'Pxx_lower': lower 95% CI
        - 'Pxx_upper': upper 95% CI
        - 'Pxx_all': all PSDs (n_ensemble, n_freqs)
    """
    n_ensemble = signals.shape[0]
    
    # Compute PSD for first signal to get frequency array
    f, _ = compute_psd(signals[0], fs, **psd_kwargs)
    
    # Allocate storage
    Pxx_all = np.zeros((n_ensemble, len(f)))
    
    # Compute PSD for each ensemble member
    for i in range(n_ensemble):
        _, Pxx_all[i] = compute_psd(signals[i], fs, **psd_kwargs)
    
    # Compute statistics
    Pxx_median, Pxx_lower, Pxx_upper = compute_median_and_ci(Pxx_all, axis=0)
    
    return {
        'f': f,
        'Pxx_median': Pxx_median,
        'Pxx_lower': Pxx_lower,
        'Pxx_upper': Pxx_upper,
        'Pxx_all': Pxx_all
    }


def export_psd_to_csv(
    filename: str,
    f: np.ndarray,
    Pxx: np.ndarray,
    **additional_columns
) -> None:
    """Export PSD to CSV file.
    
    Parameters
    ----------
    filename : str
        Output CSV filename
    f : ndarray
        Frequency array
    Pxx : ndarray
        PSD array
    **additional_columns
        Additional columns to include (as keyword arguments)
        
    Examples
    --------
    >>> f = np.linspace(0, 50, 100)
    >>> Pxx = np.random.rand(100)
    >>> export_psd_to_csv('psd.csv', f, Pxx, Pxx_lower=Pxx*0.8, Pxx_upper=Pxx*1.2)
    """
    import pandas as pd
    
    data = {'f': f, 'Pxx': Pxx}
    data.update(additional_columns)
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)


def find_peaks_in_psd(
    f: np.ndarray,
    Pxx: np.ndarray,
    prominence_factor: float = 0.1,
    min_distance: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """Find peaks in power spectral density.
    
    Parameters
    ----------
    f : ndarray
        Frequency array
    Pxx : ndarray
        PSD array
    prominence_factor : float
        Minimum peak prominence as fraction of max PSD
    min_distance : float
        Minimum distance between peaks in Hz
        
    Returns
    -------
    peak_freqs : ndarray
        Frequencies of detected peaks
    peak_powers : ndarray
        PSD values at peaks
    """
    from scipy.signal import find_peaks
    
    # Convert min_distance to samples
    df = f[1] - f[0] if len(f) > 1 else 1.0
    min_samples = max(1, int(min_distance / df))
    
    # Find peaks
    peaks, properties = find_peaks(
        Pxx,
        prominence=prominence_factor * Pxx.max(),
        distance=min_samples
    )
    
    peak_freqs = f[peaks]
    peak_powers = Pxx[peaks]
    
    return peak_freqs, peak_powers

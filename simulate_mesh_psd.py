"""
Simulate Syncytial Mesh with improved damped wave PDE model.

This script implements a reproducible 2D damped wave equation simulation
with PML boundaries, region-based sources, and ensemble PSD computation.

Contact: Andreu.Ballus@uab.cat
"""
import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import yaml
from scipy import signal
from scipy.signal import convolve2d, fftconvolve, resample_poly


def load_params(params_file):
    """Load simulation parameters from YAML file.
    
    Parameters
    ----------
    params_file : str or Path
        Path to YAML parameter file
        
    Returns
    -------
    dict
        Dictionary of simulation parameters
    """
    with open(params_file, 'r') as f:
        params = yaml.safe_load(f)
    return params


def create_pml_sigma_map(nx, ny, pml_width, sigma_max):
    """Create PML damping coefficient map.
    
    Uses quadratic profile sigma = sigma_max * (d/pml_w)^2 where d is
    distance into the PML layer.
    
    Parameters
    ----------
    nx, ny : int
        Grid dimensions
    pml_width : int
        Width of PML layer in grid points
    sigma_max : float
        Maximum damping coefficient
        
    Returns
    -------
    ndarray
        2D array of damping coefficients
    """
    sigma_map = np.zeros((ny, nx))
    
    for i in range(nx):
        for j in range(ny):
            # Distance into PML from each edge
            d = 0
            if i < pml_width:
                d = max(d, pml_width - i)
            if i >= nx - pml_width:
                d = max(d, i - (nx - pml_width - 1))
            if j < pml_width:
                d = max(d, pml_width - j)
            if j >= ny - pml_width:
                d = max(d, j - (ny - pml_width - 1))
            
            if d > 0:
                sigma_map[j, i] = sigma_max * (d / pml_width) ** 2
    
    return sigma_map


def create_region_positions(nx, ny, N_regions):
    """Create positions for N_regions on a coarse subsampled grid.
    
    Parameters
    ----------
    nx, ny : int
        Grid dimensions
    N_regions : int
        Number of regions
        
    Returns
    -------
    list of tuple
        List of (i, j) positions for each region
    """
    # Distribute regions on a regular subgrid
    n_side = int(np.ceil(np.sqrt(N_regions)))
    positions = []
    
    for idx in range(N_regions):
        i_region = idx % n_side
        j_region = idx // n_side
        
        # Map to grid coordinates with some padding from edges
        i = int((i_region + 1) * nx / (n_side + 1))
        j = int((j_region + 1) * ny / (n_side + 1))
        
        positions.append((i, j))
    
    return positions


def gaussian_kernel_2d(sigma_mm, dx_mm, kernel_size=None):
    """Create 2D Gaussian kernel for spatial convolution.
    
    Parameters
    ----------
    sigma_mm : float
        Standard deviation in mm
    dx_mm : float
        Grid spacing in mm
    kernel_size : int, optional
        Size of kernel (will be odd). If None, auto-computed as 6*sigma
        
    Returns
    -------
    ndarray
        2D Gaussian kernel (normalized)
    """
    sigma_grid = sigma_mm / dx_mm
    
    if kernel_size is None:
        kernel_size = int(6 * sigma_grid)
        if kernel_size % 2 == 0:
            kernel_size += 1
    
    ax = np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma_grid**2))
    kernel /= kernel.sum()
    
    return kernel


def build_source_field(region_activities, region_positions, nx, ny, gaussian_kernel):
    """Build spatial source field from region activities.
    
    Parameters
    ----------
    region_activities : ndarray
        1D array of activities for each region
    region_positions : list of tuple
        List of (i, j) positions
    nx, ny : int
        Grid dimensions
    gaussian_kernel : ndarray
        2D Gaussian kernel for spatial smoothing
        
    Returns
    -------
    ndarray
        2D source field
    """
    # Create delta functions at region positions
    source_sparse = np.zeros((ny, nx))
    for (i, j), activity in zip(region_positions, region_activities):
        if 0 <= i < nx and 0 <= j < ny:
            source_sparse[j, i] += activity
    
    # Convolve with Gaussian kernel
    source_field = fftconvolve(source_sparse, gaussian_kernel, mode='same')
    
    return source_field


def laplacian_9pt_vectorized(u, dx):
    """Compute 9-point stencil Laplacian using vectorized operations.
    
    Uses the formula:
    lap = (4*(u[i+1,j]+u[i-1,j]+u[i,j+1]+u[i,j-1]) + 
           (u[i+1,j+1]+u[i+1,j-1]+u[i-1,j+1]+u[i-1,j-1]) - 20*u[i,j]) / (6*dx^2)
    
    Parameters
    ----------
    u : ndarray
        2D field
    dx : float
        Grid spacing
        
    Returns
    -------
    ndarray
        Laplacian of u
    """
    # Pad for boundary handling (Neumann BC approximation)
    u_pad = np.pad(u, 1, mode='edge')
    
    # 4-point neighbors
    up = u_pad[0:-2, 1:-1]
    down = u_pad[2:, 1:-1]
    left = u_pad[1:-1, 0:-2]
    right = u_pad[1:-1, 2:]
    
    # Diagonal neighbors
    ul = u_pad[0:-2, 0:-2]
    ur = u_pad[0:-2, 2:]
    dl = u_pad[2:, 0:-2]
    dr = u_pad[2:, 2:]
    
    lap = (4 * (up + down + left + right) + (ul + ur + dl + dr) - 20 * u) / (6 * dx**2)
    
    return lap


def rk4_step(u, v, dt, dx, c, gamma_total, source, noise):
    """Single RK4 integration step for first-order wave equation system.
    
    System:
        du/dt = v
        dv/dt = c^2 * lap(u) - gamma_total * v + source + noise
    
    Parameters
    ----------
    u, v : ndarray
        Current state (2D fields)
    dt : float
        Time step
    dx : float
        Spatial step
    c : float
        Wave speed
    gamma_total : ndarray
        Total damping (background + PML)
    source : ndarray
        External source term
    noise : ndarray
        Noise term
        
    Returns
    -------
    u_new, v_new : ndarray
        Updated state
    """
    def rhs(u_curr, v_curr):
        lap_u = laplacian_9pt_vectorized(u_curr, dx)
        du_dt = v_curr
        dv_dt = c**2 * lap_u - gamma_total * v_curr + source + noise
        return du_dt, dv_dt
    
    k1u, k1v = rhs(u, v)
    k2u, k2v = rhs(u + 0.5*dt*k1u, v + 0.5*dt*k1v)
    k3u, k3v = rhs(u + 0.5*dt*k2u, v + 0.5*dt*k2v)
    k4u, k4v = rhs(u + dt*k3u, v + dt*k3v)
    
    u_new = u + (dt/6) * (k1u + 2*k2u + 2*k3u + k4u)
    v_new = v + (dt/6) * (k1v + 2*k2v + 2*k3v + k4v)
    
    return u_new, v_new


def test_source_pattern(t, region_idx, N_regions, params):
    """Generate test source pattern: 4 Hz burst at center region for t in [0,1]s.
    
    This mimics the previous probe stimulus but is structured to be easily
    replaced by neural mass outputs later.
    
    Parameters
    ----------
    t : float
        Current time
    region_idx : int
        Index of region
    N_regions : int
        Total number of regions
    params : dict
        Parameter dictionary
        
    Returns
    -------
    float
        Activity level for this region
    """
    # Identify center region (middle of the grid)
    n_side = int(np.ceil(np.sqrt(N_regions)))
    center_idx = (n_side // 2) * n_side + (n_side // 2)
    
    if region_idx == center_idx and t <= 1.0:
        # 4 Hz oscillation with amplitude 1.0
        return 1.0 * np.cos(2 * np.pi * 4.0 * t)
    else:
        return 0.0


def run_single_simulation(params, seed, verbose=False):
    """Run a single simulation with given parameters and seed.
    
    Parameters
    ----------
    params : dict
        Simulation parameters
    seed : int
        Random seed for this run
    verbose : bool
        Print progress information
        
    Returns
    -------
    f : ndarray
        Frequency array
    Pxx : ndarray
        Power spectral density
    """
    # Set random seed
    rng = np.random.default_rng(seed)
    
    # Extract parameters
    L_mm = params['L_mm']
    dx_mm = params['dx_mm']
    c_mm_per_s = params['c_mm_per_s']
    gamma_s = params['gamma_s']
    dt_s = params['dt_s']
    sim_duration_s = params['sim_duration_s']
    pml_width_mm = params['pml_width_mm']
    pml_sigma_max = params['pml_sigma_max']
    sigma_kernel_mm = params['sigma_kernel_mm']
    N_regions = params['N_regions']
    psd_fs = params['psd_fs']
    psd_nperseg = params['psd_nperseg']
    psd_noverlap = params['psd_noverlap']
    noise_sigma_mesh = params['noise_sigma_mesh']
    
    # Compute grid
    nx = int(L_mm / dx_mm)
    ny = nx
    
    # CFL check (for stability, CFL = c*dt/dx should be < ~0.5 for explicit methods)
    cfl = c_mm_per_s * dt_s / dx_mm
    if verbose:
        print(f"Grid: {nx}x{ny}, CFL number: {cfl:.3f}")
    if cfl > 0.5:
        print(f"Warning: CFL = {cfl:.3f} may be too large for stability")
    
    # Setup PML
    pml_width = int(pml_width_mm / dx_mm)
    sigma_map = create_pml_sigma_map(nx, ny, pml_width, pml_sigma_max)
    gamma_total = gamma_s + sigma_map
    
    # Setup regions
    region_positions = create_region_positions(nx, ny, N_regions)
    gaussian_kernel = gaussian_kernel_2d(sigma_kernel_mm, dx_mm)
    
    # Initialize fields
    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
    
    # Time integration
    nt = int(sim_duration_s / dt_s)
    save_interval = int(1.0 / (psd_fs * dt_s))  # Save at psd_fs rate
    
    # Storage for downsampled trace (central node)
    u_trace = []
    
    for t_idx in range(nt):
        t = t_idx * dt_s
        
        # Build source from region activities
        region_activities = np.array([test_source_pattern(t, i, N_regions, params) 
                                      for i in range(N_regions)])
        source = build_source_field(region_activities, region_positions, nx, ny, gaussian_kernel)
        
        # Add white noise to v equation
        # noise ~ N(0, sigma^2/dt) so variance is independent of dt
        noise = rng.normal(0, noise_sigma_mesh / np.sqrt(dt_s), (ny, nx))
        
        # RK4 step
        u, v = rk4_step(u, v, dt_s, dx_mm, c_mm_per_s, gamma_total, source, noise)
        
        # Save trace at reduced rate
        if t_idx % save_interval == 0:
            u_trace.append(u[ny//2, nx//2])
    
    u_trace = np.array(u_trace)
    
    # Anti-alias filter and resample if needed
    # The trace is already at psd_fs due to save_interval
    # But we apply resample_poly for anti-aliasing as specified
    original_fs = 1.0 / dt_s
    actual_fs = original_fs / save_interval
    target_fs = psd_fs
    
    if abs(actual_fs - target_fs) > 0.01:
        # Need to resample - compute rational approximation
        # Round to nearest integers for ratio
        up = int(round(target_fs))
        down = int(round(actual_fs))
        from math import gcd
        g = gcd(up, down)
        up //= g
        down //= g
        u_trace = resample_poly(u_trace, up, down)
    
    # Compute PSD using Welch's method
    # Adjust nperseg/noverlap if trace is too short
    nperseg_actual = min(psd_nperseg, len(u_trace))
    noverlap_actual = min(psd_noverlap, nperseg_actual - 1)
    
    f, Pxx = signal.welch(u_trace, fs=psd_fs, 
                          nperseg=nperseg_actual, 
                          noverlap=noverlap_actual,
                          window='hamming')
    
    if verbose:
        print(f"Simulation complete. Trace length: {len(u_trace)}, PSD points: {len(f)}")
    
    return f, Pxx


def run_ensemble(params, ensemble_size=None, output_dir='results', verbose=True):
    """Run ensemble of simulations and compute median PSD.
    
    Parameters
    ----------
    params : dict
        Simulation parameters
    ensemble_size : int, optional
        Number of ensemble members (defaults to params['ensemble_size'])
    output_dir : str
        Directory for output files
    verbose : bool
        Print progress
        
    Returns
    -------
    None
        Writes psd_median.csv, psd_all.npy, and metadata.json to output_dir
    """
    if ensemble_size is None:
        ensemble_size = params['ensemble_size']
    
    base_seed = params['seed']
    
    # Storage for all PSDs
    all_psds = []
    f = None
    
    for i in range(ensemble_size):
        if verbose:
            print(f"Running ensemble member {i+1}/{ensemble_size}...")
        
        seed = base_seed + i
        f, Pxx = run_single_simulation(params, seed, verbose=(i == 0))
        all_psds.append(Pxx)
    
    all_psds = np.array(all_psds)
    
    # Compute median
    Pxx_median = np.median(all_psds, axis=0)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save median PSD as CSV
    import pandas as pd
    df_median = pd.DataFrame({'f': f, 'Pxx_median': Pxx_median})
    median_path = os.path.join(output_dir, 'psd_median.csv')
    df_median.to_csv(median_path, index=False)
    
    # Save all PSDs as numpy array
    all_path = os.path.join(output_dir, 'psd_all.npy')
    np.save(all_path, all_psds)
    
    # Save frequency array separately for convenience
    freq_path = os.path.join(output_dir, 'psd_frequencies.npy')
    np.save(freq_path, f)
    
    # Save metadata
    metadata = {
        'params': params,
        'ensemble_size': ensemble_size,
        'output_dir': output_dir,
    }
    
    # Try to get git commit
    try:
        git_commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], 
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.SubprocessError):
        git_commit = 'N/A'
    
    metadata['git_commit'] = git_commit
    
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    if verbose:
        print(f"\nEnsemble complete!")
        print(f"Results saved to {output_dir}/")
        print(f"  - psd_median.csv")
        print(f"  - psd_all.npy")
        print(f"  - psd_frequencies.npy")
        print(f"  - metadata.json")
        print(f"Git commit: {git_commit}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Simulate 2D damped wave PDE with ensemble PSD computation'
    )
    parser.add_argument('--params', type=str, default='params.yaml',
                        help='Path to YAML parameter file (default: params.yaml)')
    parser.add_argument('--ensemble', type=int, default=None,
                        help='Override ensemble size from params file')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory (default: results)')
    parser.add_argument('--sim-duration', type=float, default=None,
                        help='Override simulation duration (seconds)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress progress output')
    
    args = parser.parse_args()
    
    # Load parameters
    params = load_params(args.params)
    
    # Apply overrides
    if args.sim_duration is not None:
        params['sim_duration_s'] = args.sim_duration
    
    # Run ensemble
    run_ensemble(params, 
                 ensemble_size=args.ensemble,
                 output_dir=args.output,
                 verbose=not args.quiet)


if __name__ == '__main__':
    main()

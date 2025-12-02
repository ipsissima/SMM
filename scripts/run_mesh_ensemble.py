#!/usr/bin/env python
"""
Run ensemble of mesh simulations and compute PSD statistics.

This script:
1. Loads parameters from params.yaml
2. Runs an ensemble of mesh simulations with white noise
3. Computes power spectral density for each run
4. Saves median PSD and statistics to CSV/NPZ files

Usage:
    python scripts/run_mesh_ensemble.py [--params params.yaml] [--output results]

Contact: Andreu.Ballus@uab.cat
"""

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import yaml

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from smm.mesh import MeshField, MeshConfig
from smm.analysis import compute_psd, compute_ensemble_psd, export_psd_to_csv


def load_params(params_file: str) -> dict:
    """Load simulation parameters from YAML file.
    
    Parameters
    ----------
    params_file : str
        Path to YAML parameter file
        
    Returns
    -------
    dict
        Dictionary of simulation parameters
    """
    with open(params_file, 'r') as f:
        params = yaml.safe_load(f)
    return params


def run_single_simulation(config: MeshConfig, seed: int, 
                         sim_duration: float, psd_fs: float,
                         noise_amplitude: float,
                         verbose: bool = False) -> tuple:
    """Run a single mesh simulation and compute PSD.
    
    Parameters
    ----------
    config : MeshConfig
        Mesh configuration
    seed : int
        Random seed
    sim_duration : float
        Simulation duration in seconds
    psd_fs : float
        Sampling frequency for PSD computation
    noise_amplitude : float
        White noise amplitude
    verbose : bool
        Print progress
        
    Returns
    -------
    f : ndarray
        Frequency array
    Pxx : ndarray
        Power spectral density
    """
    # Create RNG with seed
    rng = np.random.default_rng(seed)
    
    # Create mesh field
    mesh = MeshField(config, rng=rng)
    
    if verbose:
        print(f"  Grid: {config.Nx}×{config.Ny}, CFL: {config.cfl:.3f}")
    
    # Determine recording interval to achieve psd_fs sampling rate
    record_interval = max(1, int(1.0 / (psd_fs * config.dt)))
    
    # Run simulation
    snapshots = mesh.run(
        T=sim_duration,
        record_interval=record_interval,
        noise_amplitude=noise_amplitude
    )
    
    # Extract central point trace
    center_y, center_x = config.Ny // 2, config.Nx // 2
    trace = snapshots[:, center_y, center_x]
    
    if verbose:
        print(f"  Trace length: {len(trace)} samples")
    
    # Compute PSD
    f, Pxx = compute_psd(trace, fs=psd_fs, nperseg=512, noverlap=256)
    
    return f, Pxx


def run_ensemble(params: dict, ensemble_size: int = None,
                output_dir: str = 'results', verbose: bool = True) -> None:
    """Run ensemble of simulations and save results.
    
    Parameters
    ----------
    params : dict
        Simulation parameters
    ensemble_size : int, optional
        Number of ensemble members (defaults to params['ensemble_size'])
    output_dir : str
        Output directory
    verbose : bool
        Print progress
    """
    if ensemble_size is None:
        ensemble_size = params['ensemble_size']
    
    # Create mesh configuration
    L_mm = params['L_mm']
    dx_mm = params.get('dx_mm', params['L_mm'] / params.get('Nx', 64))
    Nx = int(L_mm / dx_mm)
    Ny = Nx
    
    config = MeshConfig(
        Lx=L_mm,
        Ly=L_mm,
        Nx=Nx,
        Ny=Ny,
        c=params['c_mm_per_s'],
        gamma=params['gamma_s'],
        dt=params['dt_s'],
        pml_width=int(params['pml_width_mm'] / dx_mm),
        pml_sigma_max=params['pml_sigma_max']
    )
    
    # Check CFL
    if not config.check_cfl():
        print(f"WARNING: CFL number {config.cfl:.3f} exceeds recommended limit 0.5")
    
    # Extract parameters
    base_seed = params['seed']
    sim_duration = params['sim_duration_s']
    psd_fs = params['psd_fs']
    noise_amplitude = params['noise_sigma_mesh']
    
    # Run ensemble
    if verbose:
        print(f"Running ensemble of {ensemble_size} simulations...")
        print(f"  Duration: {sim_duration}s, dt: {config.dt}s")
        print(f"  Mesh: {config.Nx}×{config.Ny}, dx: {config.dx:.3f}mm")
        print(f"  Wave speed: {config.c} mm/s, damping: {config.gamma} 1/s")
    
    all_psds = []
    f = None
    
    for i in range(ensemble_size):
        if verbose:
            print(f"\n[{i+1}/{ensemble_size}] Running simulation...")
        
        seed = base_seed + i
        f, Pxx = run_single_simulation(
            config, seed, sim_duration, psd_fs, noise_amplitude,
            verbose=verbose and (i == 0)
        )
        all_psds.append(Pxx)
        
        if verbose and i == 0:
            print(f"  PSD computed: {len(f)} frequency points")
    
    all_psds = np.array(all_psds)
    
    # Compute statistics
    Pxx_median = np.median(all_psds, axis=0)
    Pxx_mean = np.mean(all_psds, axis=0)
    Pxx_std = np.std(all_psds, axis=0)
    Pxx_lower = np.percentile(all_psds, 2.5, axis=0)
    Pxx_upper = np.percentile(all_psds, 97.5, axis=0)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save median PSD as CSV
    export_psd_to_csv(
        os.path.join(output_dir, 'psd_median.csv'),
        f, Pxx_median,
        Pxx_mean=Pxx_mean,
        Pxx_std=Pxx_std,
        Pxx_lower=Pxx_lower,
        Pxx_upper=Pxx_upper
    )
    
    # Save all PSDs and frequencies as NPZ
    np.savez(
        os.path.join(output_dir, 'psd_ensemble.npz'),
        f=f,
        Pxx_all=all_psds,
        Pxx_median=Pxx_median,
        Pxx_mean=Pxx_mean,
        Pxx_std=Pxx_std,
        Pxx_lower=Pxx_lower,
        Pxx_upper=Pxx_upper
    )
    
    # Save metadata
    metadata = {
        'params': params,
        'ensemble_size': ensemble_size,
        'output_dir': output_dir,
        'mesh_config': {
            'Lx': config.Lx,
            'Ly': config.Ly,
            'Nx': config.Nx,
            'Ny': config.Ny,
            'dx': config.dx,
            'dy': config.dy,
            'c': config.c,
            'gamma': config.gamma,
            'dt': config.dt,
            'cfl': config.cfl
        }
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
    
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Ensemble complete!")
        print(f"Results saved to {output_dir}/")
        print(f"  - psd_median.csv (median PSD with confidence intervals)")
        print(f"  - psd_ensemble.npz (all PSDs and statistics)")
        print(f"  - metadata.json (simulation parameters and config)")
        print(f"Git commit: {git_commit}")
        print(f"{'='*60}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Run ensemble of mesh PDE simulations and compute PSD statistics',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--params', type=str, default='params.yaml',
        help='Path to YAML parameter file (default: params.yaml)'
    )
    parser.add_argument(
        '--ensemble', type=int, default=None,
        help='Override ensemble size from params file'
    )
    parser.add_argument(
        '--output', type=str, default='results',
        help='Output directory (default: results)'
    )
    parser.add_argument(
        '--sim-duration', type=float, default=None,
        help='Override simulation duration (seconds)'
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='Suppress progress output'
    )
    
    args = parser.parse_args()
    
    # Load parameters
    params = load_params(args.params)
    
    # Apply overrides
    if args.sim_duration is not None:
        params['sim_duration_s'] = args.sim_duration
    
    # Run ensemble
    run_ensemble(
        params,
        ensemble_size=args.ensemble,
        output_dir=args.output,
        verbose=not args.quiet
    )


if __name__ == '__main__':
    main()

#!/usr/bin/env python
"""
Run coupled 3-layer SMM (neural masses + mesh + Kuramoto).

This is an optional example demonstrating the full coupled model.
Most analysis uses the mesh layer alone or with minimal coupling.

Usage:
    python scripts/run_full_model.py [--params params.yaml] [--output results]

Contact: Andreu.Ballus@uab.cat
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import yaml
import matplotlib.pyplot as plt

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from smm.mesh import MeshField, MeshConfig
from smm.neural import NeuralMass, Kuramoto, NeuralMassConfig, KuramotoConfig
from smm.coupling import CoupledModel, create_region_positions_grid
from smm.analysis import compute_psd


def load_params(params_file: str) -> dict:
    """Load parameters from YAML file."""
    with open(params_file, 'r') as f:
        return yaml.safe_load(f)


def run_coupled_simulation(params: dict, output_dir: str = 'results_coupled',
                          verbose: bool = True) -> None:
    """Run coupled 3-layer simulation.
    
    Parameters
    ----------
    params : dict
        Simulation parameters
    output_dir : str
        Output directory
    verbose : bool
        Print progress
    """
    # Extract parameters
    L_mm = params['L_mm']
    dx_mm = params.get('dx_mm', params['L_mm'] / params.get('Nx', 64))
    Nx = int(L_mm / dx_mm)
    Ny = Nx
    
    # Create mesh configuration
    mesh_config = MeshConfig(
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
    
    # Create neural mass configuration
    n_regions = params['N_regions']
    neural_config = NeuralMassConfig(
        n_regions=n_regions,
        tau_E=params.get('tau_E_s', 0.01),
        tau_I=params.get('tau_I_s', 0.01),
        w_EE=params.get('w_EE', 1.5),
        w_EI=params.get('w_EI', 1.2),
        w_IE=params.get('w_IE', 1.0),
        w_II=params.get('w_II', 0.5),
        gain=params.get('sigmoid_gain', 4.0),
        threshold=params.get('sigmoid_threshold', 0.5)
    )
    
    # Create Kuramoto configuration
    kuramoto_config = KuramotoConfig(
        n_oscillators=n_regions,
        K=params.get('K_kuramoto', 0.5),
        omega_mean=params.get('omega_mean_hz', 4.0),
        omega_std=params.get('omega_std_hz', 0.5),
        kappa_mesh=0.0  # Will be set via coupling
    )
    
    # Create instances
    rng = np.random.default_rng(params['seed'])
    mesh = MeshField(mesh_config, rng=rng)
    neural = NeuralMass(neural_config)
    kuramoto = Kuramoto(kuramoto_config, rng=rng)
    
    # Create region positions
    positions = create_region_positions_grid(Nx, Ny, n_regions)
    
    if verbose:
        print(f"Coupled 3-layer model configuration:")
        print(f"  Mesh: {Nx}×{Ny} grid, {mesh_config.dx:.3f}mm spacing")
        print(f"  Regions: {n_regions} neural masses/oscillators")
        print(f"  CFL: {mesh_config.cfl:.3f}")
    
    # Create coupled model
    model = CoupledModel(
        mesh, neural, kuramoto, positions,
        coupling_neural_mesh=params.get('coupling_neural_mesh', 0.0),
        coupling_mesh_kuramoto=params.get('coupling_mesh_kuramoto', 0.0),
        coupling_mesh_neural=params.get('coupling_mesh_neural', 0.0),
        sigma_source=params.get('sigma_kernel_mm', 1.0)
    )
    
    # Define external input (example: periodic stimulation of center region)
    def I_ext_func(t):
        I_ext = np.zeros(n_regions)
        center_region = n_regions // 2
        if t < 1.0:
            # 4 Hz stimulation for first second
            I_ext[center_region] = 0.5 * np.cos(2 * np.pi * 4.0 * t)
        return I_ext
    
    # Run simulation
    if verbose:
        print(f"\nRunning simulation for {params['sim_duration_s']}s...")
    
    dt = mesh_config.dt
    results = model.run(
        T=params['sim_duration_s'],
        dt=dt,
        I_ext_func=I_ext_func,
        noise_amplitude=params['noise_sigma_mesh'],
        record_interval=10
    )
    
    if verbose:
        print(f"Simulation complete. Recorded {len(results['t'])} time points.")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results as NPZ
    np.savez(
        os.path.join(output_dir, 'coupled_results.npz'),
        **results
    )
    
    # Compute and save PSDs for neural activities
    psd_fs = params['psd_fs']
    downsample = max(1, int(1.0 / (psd_fs * dt * 10)))  # Account for record_interval
    
    # Example: PSD of first region's excitatory activity
    E0_trace = results['E'][:, 0]
    if len(E0_trace) > 100:  # Only if enough data
        f_neural, Pxx_neural = compute_psd(E0_trace, fs=psd_fs/10, nperseg=min(256, len(E0_trace)//4))
        
        # Save neural PSD
        np.savez(
            os.path.join(output_dir, 'psd_neural.npz'),
            f=f_neural,
            Pxx=Pxx_neural
        )
    
    # Compute Kuramoto order parameter over time
    R_array = []
    for theta_t in results['theta']:
        z = np.mean(np.exp(1j * theta_t))
        R_array.append(np.abs(z))
    R_array = np.array(R_array)
    
    # Save metadata
    metadata = {
        'params': params,
        'mesh_config': {
            'Lx': mesh_config.Lx,
            'Ly': mesh_config.Ly,
            'Nx': mesh_config.Nx,
            'Ny': mesh_config.Ny,
            'c': mesh_config.c,
            'gamma': mesh_config.gamma,
            'dt': mesh_config.dt,
            'cfl': mesh_config.cfl
        },
        'neural_config': {
            'n_regions': neural_config.n_regions,
            'tau_E': neural_config.tau_E,
            'tau_I': neural_config.tau_I
        },
        'kuramoto_config': {
            'n_oscillators': kuramoto_config.n_oscillators,
            'K': kuramoto_config.K,
            'omega_mean': kuramoto_config.omega_mean
        },
        'kuramoto_order_mean': float(np.mean(R_array)),
        'kuramoto_order_std': float(np.std(R_array))
    }
    
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Create simple visualization
    if verbose:
        print(f"\nCreating visualization...")
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    
    # Plot excitatory activities
    axes[0].plot(results['t'], results['E'][:, :4])
    axes[0].set_ylabel('E (first 4 regions)')
    axes[0].set_title('Neural Mass Activities')
    axes[0].grid(True, alpha=0.3)
    
    # Plot Kuramoto order parameter
    axes[1].plot(results['t'], R_array)
    axes[1].set_ylabel('Order parameter R')
    axes[1].set_ylim([0, 1])
    axes[1].set_title('Kuramoto Synchronization')
    axes[1].grid(True, alpha=0.3)
    
    # Plot mesh field at center
    center_y, center_x = mesh_config.Ny // 2, mesh_config.Nx // 2
    u_center = results['u'][:, center_y, center_x]
    axes[2].plot(results['t'], u_center)
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('u (center)')
    axes[2].set_title('Mesh Field at Center')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'time_traces.png'), dpi=150)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Results saved to {output_dir}/")
        print(f"  - coupled_results.npz (full state time series)")
        print(f"  - psd_neural.npz (PSD of neural activity)")
        print(f"  - time_traces.png (visualization)")
        print(f"  - metadata.json (configuration)")
        print(f"Kuramoto order: {np.mean(R_array):.3f} ± {np.std(R_array):.3f}")
        print(f"{'='*60}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Run coupled 3-layer SMM simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--params', type=str, default='params.yaml',
        help='Path to YAML parameter file (default: params.yaml)'
    )
    parser.add_argument(
        '--output', type=str, default='results_coupled',
        help='Output directory (default: results_coupled)'
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='Suppress progress output'
    )
    
    args = parser.parse_args()
    
    # Load parameters
    params = load_params(args.params)
    
    # Run coupled simulation
    run_coupled_simulation(
        params,
        output_dir=args.output,
        verbose=not args.quiet
    )


if __name__ == '__main__':
    main()

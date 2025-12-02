#!/usr/bin/env python
"""
Demonstration: Pure glial wave telegraph equation.

This script demonstrates the glial telegraph equation derived from IP₃/Ca dynamics,
showing physiological wave speed, damping, and propagation characteristics.

The simulation:
1. Sets up a glial field with physiological micro-parameters
2. Initializes with a localized perturbation or noise
3. Runs the simulation showing wave propagation and damping
4. Analyzes and visualizes:
   - Space-time plots
   - Wave speed estimation
   - Damping rate
   - Power spectral density

Usage:
    python scripts/run_glial_wave.py [--params params.yaml] [--output results]
"""

import argparse
import sys
from pathlib import Path
import yaml
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from smm.glia import GliaMicroParams, GlialFieldConfig, GlialField


def run_glial_wave_demo(params_file: str, output_dir: str, quiet: bool = False):
    """Run glial wave demonstration.
    
    Parameters
    ----------
    params_file : str
        Path to parameters YAML file
    output_dir : str
        Output directory for results
    quiet : bool
        Suppress output
    """
    # Load parameters
    with open(params_file, 'r') as f:
        params = yaml.safe_load(f)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Setup glial micro-parameters
    micro_params = GliaMicroParams(
        alpha=params.get('alpha', 1.0),
        beta=params.get('beta', 0.8),
        gamma=params.get('gamma', 1.0),
        delta=params.get('delta', 1.5),
        D_um2_per_s=params.get('D_um2_per_s', 100.0)
    )
    
    if not quiet:
        print("\n" + "="*70)
        print("GLIAL WAVE DEMONSTRATION")
        print("="*70)
        micro_params.print_summary()
    
    # Setup glial field
    Lx = params.get('L_mm', 32.0)
    Ly = params.get('L_mm', 32.0)
    Nx = params.get('Nx', 64)
    Ny = params.get('Ny', 64)
    
    config = GlialFieldConfig(
        Lx=Lx,
        Ly=Ly,
        Nx=Nx,
        Ny=Ny,
        micro_params=micro_params,
        dt=params.get('dt_s', 0.001),
        pml_width=params.get('pml_width_mm', 3.0) / (Lx / Nx),  # Convert mm to grid points
        pml_sigma_max=params.get('pml_sigma_max', 50.0)
    )
    
    glia = GlialField(config, rng=np.random.default_rng(params.get('seed', 42)))
    
    # Initialize with localized perturbation
    center_x, center_y = Nx // 2, Ny // 2
    sigma_init = 2  # grid points
    for i in range(Nx):
        for j in range(Ny):
            r2 = (i - center_x)**2 + (j - center_y)**2
            glia.u[j, i] = np.exp(-r2 / (2 * sigma_init**2))
    
    # Run simulation
    T = params.get('sim_duration_s', 2.0)
    noise_amp = params.get('noise_sigma_mesh', 1e-6)
    dt_record = 0.01  # Record every 10 ms
    record_interval = int(dt_record / config.dt)
    
    if not quiet:
        print(f"\nRunning simulation for {T:.1f} s...")
        print(f"Recording interval: {dt_record*1000:.1f} ms")
    
    snapshots = glia.run(T=T, record_interval=record_interval, noise_amplitude=noise_amp)
    
    if not quiet:
        print(f"Simulation complete. {len(snapshots)} snapshots recorded.")
    
    # Analysis: Space-time plot along central horizontal line
    central_line = Ny // 2
    spacetime = snapshots[:, central_line, :]
    
    # Time array
    t_array = np.arange(len(snapshots)) * dt_record
    x_array = np.arange(Nx) * config.dx
    
    # Estimate wave speed from space-time plot
    # Find first significant wave arrival at different positions
    threshold = 0.1 * np.max(spacetime)
    wave_speed_estimates = []
    
    for i in range(Nx // 2 + 10, Nx - 10):  # Sample points away from center
        times_above = np.where(spacetime[:, i] > threshold)[0]
        if len(times_above) > 0:
            arrival_time = t_array[times_above[0]]
            distance = x_array[i] - x_array[Nx // 2]
            if arrival_time > 0 and distance > 0:
                speed = distance / arrival_time  # mm/s
                wave_speed_estimates.append(speed)
    
    if len(wave_speed_estimates) > 0:
        measured_speed = np.median(wave_speed_estimates)
        if not quiet:
            print(f"\nMeasured wave speed: {measured_speed:.2f} mm/s")
            print(f"Expected wave speed: {config.c:.2f} mm/s")
    
    # Temporal PSD at center
    center_trace = snapshots[:, center_y, center_x]
    fs = 1.0 / dt_record  # Sampling frequency (Hz)
    freqs, psd = signal.welch(center_trace, fs=fs, nperseg=min(256, len(center_trace)//2))
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Space-time plot
    ax = axes[0, 0]
    im = ax.imshow(spacetime, aspect='auto', origin='lower', cmap='RdBu_r',
                   extent=[0, Lx, 0, T], interpolation='bilinear')
    ax.set_xlabel('Position x (mm)')
    ax.set_ylabel('Time (s)')
    ax.set_title('Space-Time Plot (Central Horizontal Slice)')
    plt.colorbar(im, ax=ax, label='Field amplitude')
    
    # 2. Snapshots at different times
    ax = axes[0, 1]
    times_to_plot = [0, len(snapshots)//4, len(snapshots)//2, 3*len(snapshots)//4]
    for idx, t_idx in enumerate(times_to_plot):
        if t_idx < len(snapshots):
            ax.plot(x_array, snapshots[t_idx, central_line, :], 
                   label=f't = {t_array[t_idx]:.2f} s', alpha=0.7)
    ax.set_xlabel('Position x (mm)')
    ax.set_ylabel('Field amplitude')
    ax.set_title('Spatial Profiles at Different Times')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Temporal trace at center
    ax = axes[1, 0]
    ax.plot(t_array, center_trace)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Field amplitude')
    ax.set_title(f'Temporal Evolution at Center')
    ax.grid(True, alpha=0.3)
    
    # 4. Power spectral density
    ax = axes[1, 1]
    ax.loglog(freqs[1:], psd[1:])  # Skip DC component
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power Spectral Density')
    ax.set_title('PSD at Center Point')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'glial_wave_analysis.png', dpi=150)
    if not quiet:
        print(f"\nSaved analysis plot to {output_path / 'glial_wave_analysis.png'}")
    
    # Save final snapshot as 2D image
    fig2, ax = plt.subplots(1, 1, figsize=(8, 7))
    im = ax.imshow(snapshots[-1], origin='lower', cmap='RdBu_r',
                   extent=[0, Lx, 0, Ly], interpolation='bilinear')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_title(f'Glial Field at t = {T:.2f} s')
    plt.colorbar(im, ax=ax, label='Field amplitude')
    plt.tight_layout()
    plt.savefig(output_path / 'glial_field_final.png', dpi=150)
    if not quiet:
        print(f"Saved final field snapshot to {output_path / 'glial_field_final.png'}")
    
    # Save data
    np.savez(
        output_path / 'glial_wave_data.npz',
        snapshots=snapshots,
        t_array=t_array,
        x_array=x_array,
        config={
            'Lx': Lx, 'Ly': Ly, 'Nx': Nx, 'Ny': Ny,
            'c': config.c, 'gamma': config.gamma, 'omega0': config.omega0,
            'alpha': micro_params.alpha, 'beta': micro_params.beta,
            'gamma_micro': micro_params.gamma, 'delta': micro_params.delta,
            'D_um2_per_s': micro_params.D_um2_per_s
        }
    )
    if not quiet:
        print(f"Saved simulation data to {output_path / 'glial_wave_data.npz'}")
    
    if not quiet:
        print("\n" + "="*70)
        print("Glial wave demonstration complete!")
        print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Demonstrate glial telegraph equation wave propagation'
    )
    parser.add_argument('--params', type=str, default='params.yaml',
                       help='Path to parameters file')
    parser.add_argument('--output', type=str, default='results_glial_wave',
                       help='Output directory')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress output')
    
    args = parser.parse_args()
    
    run_glial_wave_demo(args.params, args.output, args.quiet)


if __name__ == '__main__':
    main()

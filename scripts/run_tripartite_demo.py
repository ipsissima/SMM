#!/usr/bin/env python
"""
Demonstration: Full tripartite loop (neurons ⇄ glia ⇄ connectivity).

This script demonstrates the complete tripartite loop where:
1. Neuronal activity (E, I, θ) drives glial waves
2. Glial waves feed back to neuronal excitability
3. Glia modulates connectivity via Ca-gated plasticity
4. The whole system evolves dynamically

The simulation shows:
- Coupled dynamics of neural masses, Kuramoto phases, and glial field
- Feedback loops creating emergent spatiotemporal patterns
- Slow connectivity plasticity driven by glial coincidence detection

Usage:
    python scripts/run_tripartite_demo.py [--params params.yaml] [--output results]
"""

import argparse
import sys
from pathlib import Path
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from smm.glia import GliaMicroParams, GlialFieldConfig, GlialField
from smm.neural import NeuralMass, Kuramoto, NeuralMassConfig, KuramotoConfig
from smm.coupling import (
    tripartite_neural_to_glia_source,
    glia_to_neural_gliotransmission,
    ConnectivityPlasticity,
    create_region_positions_grid
)


def run_tripartite_demo(params_file: str, output_dir: str, quiet: bool = False):
    """Run tripartite loop demonstration.
    
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
    
    if not quiet:
        print("\n" + "="*70)
        print("TRIPARTITE LOOP DEMONSTRATION")
        print("="*70)
    
    # Setup components
    n_regions = params.get('N_regions', 8)
    
    # 1. Glial field
    micro_params = GliaMicroParams(
        alpha=params.get('alpha', 1.0),
        beta=params.get('beta', 0.8),
        gamma=params.get('gamma', 1.0),
        delta=params.get('delta', 1.5),
        D_um2_per_s=params.get('D_um2_per_s', 100.0)
    )
    
    Lx = params.get('L_mm', 32.0)
    Ly = params.get('L_mm', 32.0)
    Nx = params.get('Nx', 64)
    Ny = params.get('Ny', 64)
    
    glia_config = GlialFieldConfig(
        Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny,
        micro_params=micro_params,
        dt=params.get('dt_s', 0.001),
        pml_width=int(params.get('pml_width_mm', 3.0) / (Lx / Nx))  # Convert PML width from mm to grid points
    )
    
    glia = GlialField(glia_config, rng=np.random.default_rng(params.get('seed', 42)))
    
    if not quiet:
        print(f"\nGlial field: {Nx}x{Ny} grid, {Lx}x{Ly} mm")
        telegraph_params = micro_params.compute_telegraph_params()
        print(f"Wave speed: {telegraph_params['c_eff_um_per_s']:.2f} μm/s")
        print(f"Damping: γ₀ = {telegraph_params['gamma0']:.3f} s⁻¹")
    
    # 2. Neural masses
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
    neural = NeuralMass(neural_config)
    
    # Initialize with small random activity
    neural.E[:] = 0.3 + 0.1 * np.random.randn(n_regions)
    neural.I[:] = 0.2 + 0.05 * np.random.randn(n_regions)
    
    if not quiet:
        print(f"\nNeural masses: {n_regions} regions")
    
    # 3. Kuramoto oscillators
    kuramoto_config = KuramotoConfig(
        n_oscillators=n_regions,
        K=params.get('K_kuramoto', 0.5),
        omega_mean=params.get('omega_mean_hz', 4.0),
        omega_std=params.get('omega_std_hz', 0.5)
    )
    kuramoto = Kuramoto(kuramoto_config, rng=np.random.default_rng(params.get('seed', 42)))
    
    if not quiet:
        print(f"Kuramoto oscillators: {n_regions}, K = {kuramoto_config.K}")
    
    # 4. Connectivity plasticity (optional)
    enable_plasticity = params.get('enable_plasticity', False)
    plasticity = None
    if enable_plasticity:
        plasticity = ConnectivityPlasticity(
            n_regions=n_regions,
            tau_C=params.get('tau_C_s', 60.0),
            lambda_C=params.get('lambda_C', 0.1),
            eta_H=params.get('eta_H', 0.01),
            eta_psi=params.get('eta_psi', 0.01),
            sigma_psi_target=params.get('sigma_psi_target', 0.1)
        )
        if not quiet:
            print(f"Connectivity plasticity: ENABLED (τ_C = {plasticity.tau_C} s)")
    else:
        if not quiet:
            print("Connectivity plasticity: DISABLED")
    
    # 5. Region positions on grid
    positions = create_region_positions_grid(Nx, Ny, n_regions, edge_padding=0.15)
    
    if not quiet:
        print(f"\nRegion positions: {len(positions)} locations on grid")
    
    # Coupling parameters
    beta_E = params.get('coupling_E_to_glia', 0.1)
    beta_I = params.get('coupling_I_to_glia', 0.0)
    beta_theta = params.get('coupling_theta_to_glia', 0.0)
    g_A = params.get('coupling_glia_to_neural', 0.05)
    kappa_psi = params.get('coupling_glia_to_kuramoto', 0.01)
    sigma = params.get('sigma_kernel_mm', 1.5)
    
    if not quiet:
        print(f"\nCoupling strengths:")
        print(f"  E → glia:       β_E = {beta_E}")
        print(f"  I → glia:       β_I = {beta_I}")
        print(f"  θ → glia:       β_θ = {beta_theta}")
        print(f"  glia → neural:  g_A = {g_A}")
        print(f"  glia → θ:       κ_ψ = {kappa_psi}")
    
    # Simulation parameters
    T = params.get('sim_duration_s', 5.0)
    dt = glia_config.dt
    noise_amp = params.get('noise_sigma_mesh', 1e-6)
    
    # External input (weak constant + small noise)
    I_ext_mean = 0.5
    I_ext_noise = 0.1
    
    # Recording
    n_steps = int(T / dt)
    record_interval = int(0.01 / dt)  # Record every 10 ms
    n_records = n_steps // record_interval + 1
    
    t_rec = np.zeros(n_records)
    E_rec = np.zeros((n_records, n_regions))
    I_rec = np.zeros((n_records, n_regions))
    theta_rec = np.zeros((n_records, n_regions))
    glia_samples = np.zeros((n_records, n_regions))
    
    if plasticity is not None:
        C_rec = np.zeros((n_records, n_regions, n_regions))
    
    if not quiet:
        print(f"\nRunning simulation for {T:.1f} s ({n_steps} steps)...")
    
    # Integration loop
    rec_idx = 0
    for step in range(n_steps):
        # (a) Neural → Glia source term
        source = tripartite_neural_to_glia_source(
            neural.E, neural.I, kuramoto.theta,
            positions=positions,
            grid_shape=(Ny, Nx),
            dx=glia_config.dx,
            sigma=sigma,
            beta_E=beta_E,
            beta_I=beta_I,
            beta_theta=beta_theta
        )
        
        # Step glial field
        glia.step(source=source, noise_amplitude=noise_amp)
        
        # (b) Glia → Neural (gliotransmission)
        I_A = glia_to_neural_gliotransmission(
            glia.u, positions, g_A=g_A, 
            nonlinearity=params.get('glia_nonlinearity', 'tanh'),
            gain=params.get('glia_gain', 1.0)
        )
        
        # External input with noise
        I_ext = I_ext_mean + I_ext_noise * np.random.randn(n_regions)
        
        # Step neural masses
        neural.step(dt, I_ext=I_ext, I_mesh=I_A)
        
        # (c) Glia → Kuramoto modulation
        glia_values = np.array([glia.u[j, i] for i, j in positions])
        mesh_input = kappa_psi * glia_values
        
        # Step Kuramoto
        kuramoto.step(dt, mesh_input=mesh_input)
        
        # (d) Glia → Connectivity (slow plasticity, update every 100 steps)
        if plasticity is not None and step % 100 == 0:
            plasticity.step(
                dt=100*dt,
                E=neural.E,
                u_values=glia_values,
                sigmoid_gain=neural_config.gain,
                sigmoid_threshold=neural_config.threshold
            )
        
        # Record
        if step % record_interval == 0 and rec_idx < n_records:
            t_rec[rec_idx] = glia.t
            E_rec[rec_idx] = neural.E.copy()
            I_rec[rec_idx] = neural.I.copy()
            theta_rec[rec_idx] = kuramoto.theta.copy()
            glia_samples[rec_idx] = glia_values.copy()
            if plasticity is not None:
                C_rec[rec_idx] = plasticity.C.copy()
            rec_idx += 1
    
    if not quiet:
        print(f"Simulation complete. {rec_idx} records saved.")
    
    # Visualization
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Neural activity traces
    ax1 = fig.add_subplot(gs[0, :2])
    for i in range(min(4, n_regions)):
        ax1.plot(t_rec[:rec_idx], E_rec[:rec_idx, i], label=f'Region {i}', alpha=0.7)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Excitatory Activity')
    ax1.set_title('Neural Mass Activity (E)')
    ax1.legend(loc='upper right', ncol=2, fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 2. Kuramoto order parameter
    ax2 = fig.add_subplot(gs[0, 2])
    R_array = []
    for idx in range(rec_idx):
        z = np.mean(np.exp(1j * theta_rec[idx]))
        R_array.append(np.abs(z))
    ax2.plot(t_rec[:rec_idx], R_array)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Order Parameter R')
    ax2.set_title('Kuramoto Synchrony')
    ax2.set_ylim([0, 1])
    ax2.grid(True, alpha=0.3)
    
    # 3. Glial field samples at regions
    ax3 = fig.add_subplot(gs[1, :2])
    for i in range(min(4, n_regions)):
        ax3.plot(t_rec[:rec_idx], glia_samples[:rec_idx, i], label=f'Region {i}', alpha=0.7)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Glial Field u')
    ax3.set_title('Glial Field at Region Positions')
    ax3.legend(loc='upper right', ncol=2, fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # 4. Final glial field snapshot
    ax4 = fig.add_subplot(gs[1, 2])
    im = ax4.imshow(glia.u, origin='lower', cmap='RdBu_r',
                    extent=[0, Lx, 0, Ly], interpolation='bilinear')
    # Mark region positions
    for i, j in positions:
        x_pos = (i / Nx) * Lx
        y_pos = (j / Ny) * Ly
        ax4.plot(x_pos, y_pos, 'ko', markersize=4)
    ax4.set_xlabel('x (mm)')
    ax4.set_ylabel('y (mm)')
    ax4.set_title(f'Glial Field at t={T:.1f}s')
    plt.colorbar(im, ax=ax4, fraction=0.046)
    
    # 5. Connectivity evolution (if enabled)
    if plasticity is not None:
        ax5 = fig.add_subplot(gs[2, 0])
        # Mean connectivity strength over time
        C_mean = np.mean(np.abs(C_rec[:rec_idx]), axis=(1, 2))
        ax5.plot(t_rec[:rec_idx], C_mean)
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Mean |C|')
        ax5.set_title('Connectivity Strength Evolution')
        ax5.grid(True, alpha=0.3)
        
        # Final connectivity matrix
        ax6 = fig.add_subplot(gs[2, 1])
        im = ax6.imshow(plasticity.C, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
        ax6.set_xlabel('Region j')
        ax6.set_ylabel('Region i')
        ax6.set_title(f'Final Connectivity C_ij')
        plt.colorbar(im, ax=ax6, fraction=0.046)
    else:
        # Show cross-correlation between E and glia instead
        ax5 = fig.add_subplot(gs[2, :2])
        for i in range(min(2, n_regions)):
            xcorr = np.correlate(E_rec[:rec_idx, i] - np.mean(E_rec[:rec_idx, i]),
                               glia_samples[:rec_idx, i] - np.mean(glia_samples[:rec_idx, i]),
                               mode='same')
            lags = np.arange(-len(xcorr)//2, len(xcorr)//2) * (t_rec[1] - t_rec[0])
            ax5.plot(lags, xcorr / np.max(np.abs(xcorr)), label=f'Region {i}', alpha=0.7)
        ax5.set_xlabel('Lag (s)')
        ax5.set_ylabel('Cross-correlation (normalized)')
        ax5.set_title('E-Glia Cross-Correlation')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.set_xlim([-1, 1])
    
    # 6. Phase space: E vs glia
    ax7 = fig.add_subplot(gs[2, 2])
    for i in range(min(4, n_regions)):
        ax7.scatter(E_rec[:rec_idx, i], glia_samples[:rec_idx, i], 
                   s=1, alpha=0.3, label=f'Region {i}')
    ax7.set_xlabel('E')
    ax7.set_ylabel('Glial field u')
    ax7.set_title('E-Glia Phase Space')
    ax7.legend(markerscale=5, fontsize=8)
    ax7.grid(True, alpha=0.3)
    
    plt.savefig(output_path / 'tripartite_demo.png', dpi=150)
    if not quiet:
        print(f"\nSaved analysis plot to {output_path / 'tripartite_demo.png'}")
    
    # Save data
    save_dict = {
        't': t_rec[:rec_idx],
        'E': E_rec[:rec_idx],
        'I': I_rec[:rec_idx],
        'theta': theta_rec[:rec_idx],
        'glia_samples': glia_samples[:rec_idx],
        'positions': positions,
        'config': {
            'n_regions': n_regions,
            'beta_E': beta_E, 'beta_I': beta_I, 'beta_theta': beta_theta,
            'g_A': g_A, 'kappa_psi': kappa_psi,
            'enable_plasticity': enable_plasticity
        }
    }
    
    if plasticity is not None:
        save_dict['C'] = C_rec[:rec_idx]
    
    np.savez(output_path / 'tripartite_demo_data.npz', **save_dict)
    if not quiet:
        print(f"Saved simulation data to {output_path / 'tripartite_demo_data.npz'}")
    
    if not quiet:
        print("\n" + "="*70)
        print("Tripartite loop demonstration complete!")
        print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Demonstrate tripartite loop: neurons ⇄ glia ⇄ connectivity'
    )
    parser.add_argument('--params', type=str, default='params.yaml',
                       help='Path to parameters file')
    parser.add_argument('--output', type=str, default='results_tripartite',
                       help='Output directory')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress output')
    
    args = parser.parse_args()
    
    run_tripartite_demo(args.params, args.output, args.quiet)


if __name__ == '__main__':
    main()

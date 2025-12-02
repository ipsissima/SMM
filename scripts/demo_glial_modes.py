#!/usr/bin/env python3
"""
Demo script showing the three glial field modes in action.

This script demonstrates:
1. Telegraph mode (default) - hyperbolic wave equation
2. Diffusion mode - parabolic diffusion equation
3. Mean-field mode - spatially uniform ODE

Each mode is run with the same initial condition and source to show
how they behave differently.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import matplotlib.pyplot as plt
from smm.glia import GlialFieldConfig, GlialField

def run_mode_demo():
    """Run demonstration of all three modes."""
    
    # Common parameters
    Lx, Ly = 10.0, 10.0  # mm
    Nx, Ny = 64, 64
    dt = 0.001  # s
    T = 0.5  # s
    
    # Create source (localized pulse)
    source = np.zeros((Ny, Nx))
    source[Ny//2, Nx//2] = 1.0
    
    print("Running glial field mode comparison...")
    print("=" * 70)
    
    # 1. Telegraph mode
    print("\n1. Telegraph mode (hyperbolic wave equation)")
    config_telegraph = GlialFieldConfig(
        Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny, dt=dt,
        c=10.0, gamma=0.5, omega0=0.0,
        mode='telegraph', pml_width=5
    )
    glia_telegraph = GlialField(config_telegraph, rng=np.random.default_rng(42))
    
    snapshots_telegraph = []
    n_steps = int(T / dt)
    record_interval = 50
    
    for step in range(n_steps):
        glia_telegraph.step(source=source if step < 10 else None)
        if step % record_interval == 0:
            snapshots_telegraph.append(glia_telegraph.u.copy())
    
    print(f"   Final time: {glia_telegraph.t:.3f} s")
    print(f"   Max amplitude: {np.max(np.abs(glia_telegraph.u)):.6f}")
    print(f"   Spatial std: {np.std(glia_telegraph.u):.6f}")
    
    # 2. Diffusion mode
    print("\n2. Diffusion mode (parabolic diffusion equation)")
    config_diffusion = GlialFieldConfig(
        Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny, dt=dt,
        gamma=0.5, mode='diffusion',
        diffusion_D_mm2_per_s=1.0, pml_width=0
    )
    glia_diffusion = GlialField(config_diffusion, rng=np.random.default_rng(42))
    
    snapshots_diffusion = []
    for step in range(n_steps):
        glia_diffusion.step(source=source if step < 10 else None)
        if step % record_interval == 0:
            snapshots_diffusion.append(glia_diffusion.u.copy())
    
    print(f"   Final time: {glia_diffusion.t:.3f} s")
    print(f"   Max amplitude: {np.max(np.abs(glia_diffusion.u)):.6f}")
    print(f"   Spatial std: {np.std(glia_diffusion.u):.6f}")
    
    # 3. Mean-field mode
    print("\n3. Mean-field mode (spatially uniform ODE)")
    config_mean_field = GlialFieldConfig(
        Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny, dt=dt,
        gamma=0.5, omega0=0.0,
        mode='mean_field', pml_width=0
    )
    glia_mean_field = GlialField(config_mean_field, rng=np.random.default_rng(42))
    
    snapshots_mean_field = []
    for step in range(n_steps):
        glia_mean_field.step(source=source if step < 10 else None)
        if step % record_interval == 0:
            snapshots_mean_field.append(glia_mean_field.u.copy())
    
    print(f"   Final time: {glia_mean_field.t:.3f} s")
    print(f"   Mean amplitude: {np.mean(glia_mean_field.u):.6f}")
    print(f"   Spatial std: {np.std(glia_mean_field.u):.10e} (should be ~0)")
    
    # Create comparison plot
    print("\n" + "=" * 70)
    print("Creating comparison plot...")
    
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    
    # Time points to show
    time_indices = [0, len(snapshots_telegraph)//2, -1]
    time_labels = ['t=0', f't={T/2:.2f}s', f't={T:.2f}s']
    
    vmax = max(
        np.max(np.abs(snapshots_telegraph)),
        np.max(np.abs(snapshots_diffusion)),
        np.max(np.abs(snapshots_mean_field))
    )
    
    for col, (idx, label) in enumerate(zip(time_indices, time_labels)):
        # Telegraph
        im = axes[0, col].imshow(snapshots_telegraph[idx], cmap='RdBu_r', 
                                 vmin=-vmax, vmax=vmax, origin='lower')
        axes[0, col].set_title(f'Telegraph {label}')
        axes[0, col].axis('off')
        
        # Diffusion
        axes[1, col].imshow(snapshots_diffusion[idx], cmap='RdBu_r',
                           vmin=-vmax, vmax=vmax, origin='lower')
        axes[1, col].set_title(f'Diffusion {label}')
        axes[1, col].axis('off')
        
        # Mean-field
        axes[2, col].imshow(snapshots_mean_field[idx], cmap='RdBu_r',
                           vmin=-vmax, vmax=vmax, origin='lower')
        axes[2, col].set_title(f'Mean-field {label}')
        axes[2, col].axis('off')
    
    plt.colorbar(im, ax=axes, orientation='horizontal', 
                 fraction=0.05, pad=0.05, label='Field amplitude')
    plt.suptitle('Glial Field Mode Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = Path(__file__).parent / "results" / "mode_comparison.png"
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison plot to {output_path}")
    
    print("\n" + "=" * 70)
    print("Demo completed successfully!")
    print("\nKey observations:")
    print("  - Telegraph: Shows wave-like propagation with finite speed")
    print("  - Diffusion: Shows smooth diffusive spread")
    print("  - Mean-field: Spatially uniform, only temporal dynamics")
    
    return fig

if __name__ == "__main__":
    fig = run_mode_demo()
    plt.show()

"""
Glial (astrocytic) field solver implementing telegraph equation.

This module implements the mesoscopic glial field derived from a linearized
IP₃/Ca reaction-diffusion-ODE system in the astrocytic syncytium.

MICROSCOPIC MODEL (conceptual basis):
--------------------------------------
At each spatial point r, coarse-grained astrocyte dynamics:

    ∂ₜc = -α·c + β·i               (Ca²⁺ fluctuation around steady state)
    ∂ₜi = γ·c - δ·i + D·Δi         (IP₃ fluctuation around steady state)

where:
    c(r,t) ~ intracellular Ca²⁺ fluctuation
    i(r,t) ~ IP₃ fluctuation
    α > 0  ~ SERCA uptake, Ca²⁺ buffering (typical: 0.5-2.0 s⁻¹)
    β > 0  ~ IP₃R-mediated Ca²⁺ release (typical: 0.5-1.5 s⁻¹)
    γ > 0  ~ Ca-dependent IP₃ production via PLC (typical: 0.5-2.0 s⁻¹)
    δ > 0  ~ IP₃ degradation by phosphatase (typical: 0.5-2.0 s⁻¹)
    D      ~ IP₃ diffusion via gap junctions (typical: 50-300 μm²/s)

MESOSCOPIC TELEGRAPH EQUATION:
-------------------------------
The mesoscopic field ψ(r,t) (a linear combination of c and i, or simply c)
obeys, to leading order:

    ∂²ψ/∂t² + 2γ₀·∂ψ/∂t + ω₀²·ψ - c_eff²·Δψ = S_ψ(r,t) + ξ(r,t)

with parameters derived from the micro-model:

    γ₀ = (α + δ) / 2                   (damping rate, ~0.5-2 s⁻¹)
    c_eff² = D·(δ - α) / 2             (effective wave speed squared)
    ω₀² = α·δ - β·γ                    (local oscillation frequency squared)

PHYSIOLOGICAL RANGES:
---------------------
    - Ca wave speed c_eff: 5-30 μm/s (typical astrocyte Ca²⁺ waves)
    - Damping time 1/γ₀: 0.5-10 s (Ca²⁺ transient duration)
    - Propagation length L ~ c_eff/γ₀: 10-200 μm (astrocyte domain/syncytium)
    - Oscillation period (if ω₀ > 0): 1-10 s

NUMERICAL IMPLEMENTATION:
-------------------------
First-order system in time (u = ψ, v = ∂ₜψ):

    ∂ₜu = v
    ∂ₜv = c_eff²·∇²u - 2γ₀·v - ω₀²·u + S_ψ(r,t) + ξ(r,t)

Spatial discretization: 9-point stencil Laplacian
Time integration: RK4
Boundaries: PML (Perfectly Matched Layer) for absorption

Contact: Andreu.Ballus@uab.cat
"""

import numpy as np
import warnings
from typing import Optional, Callable, Tuple, Dict
from dataclasses import dataclass


@dataclass
class GliaMicroParams:
    """Microscopic parameters for the IP₃/Ca reaction-diffusion system.
    
    These are the linearized reaction rates around a homogeneous steady state.
    
    Parameters
    ----------
    alpha : float
        Ca²⁺ decay rate (1/s) - SERCA uptake, buffering
        Typical range: 0.5-2.0 s⁻¹
    beta : float
        IP₃ → Ca²⁺ coupling (1/s) - IP₃R open probability
        Typical range: 0.5-1.5 s⁻¹
    gamma : float
        Ca²⁺ → IP₃ production (1/s) - PLC activation
        Typical range: 0.5-2.0 s⁻¹
    delta : float
        IP₃ degradation rate (1/s) - IP₃ phosphatase
        Typical range: 0.5-2.0 s⁻¹
    D_um2_per_s : float
        IP₃ diffusion coefficient (μm²/s) - gap junction coupling
        Typical range: 50-300 μm²/s
    """
    alpha: float = 1.5
    beta: float = 0.8
    gamma: float = 1.0
    delta: float = 1.0
    D_um2_per_s: float = 100.0
    
    def compute_telegraph_params(self) -> Dict[str, float]:
        """Compute mesoscopic telegraph equation parameters.
        
        Returns
        -------
        dict
            Dictionary with keys:
            - 'gamma0': damping coefficient (1/s)
            - 'omega0': oscillation frequency (rad/s)
            - 'c_eff_um_per_s': effective wave speed (μm/s)
            - 'c_eff_mm_per_s': effective wave speed (mm/s)
        """
        # Damping coefficient
        gamma0 = (self.alpha + self.delta) / 2.0
        
        # Oscillation frequency (can be imaginary if system is overdamped)
        omega0_squared = self.alpha * self.delta - self.beta * self.gamma
        omega0 = np.sqrt(np.abs(omega0_squared)) if omega0_squared >= 0 else 0.0
        
        # Effective wave speed
        c_eff_squared_um = self.D_um2_per_s * (self.delta - self.alpha) / 2.0
        
        if c_eff_squared_um < 0:
            warnings.warn(
                f"Negative c_eff² = {c_eff_squared_um:.2f} μm²/s². "
                "This occurs when α > δ. The system may be unstable or non-propagating. "
                "Setting c_eff = 0.",
                RuntimeWarning
            )
            c_eff_um_per_s = 0.0
        else:
            c_eff_um_per_s = np.sqrt(c_eff_squared_um)
        
        c_eff_mm_per_s = c_eff_um_per_s / 1000.0  # Convert μm/s to mm/s
        
        return {
            'gamma0': gamma0,
            'omega0': omega0,
            'omega0_squared': omega0_squared if omega0_squared >= 0 else -np.abs(omega0_squared),
            'c_eff_um_per_s': c_eff_um_per_s,
            'c_eff_mm_per_s': c_eff_mm_per_s
        }
    
    def check_physiological(self) -> Dict[str, bool]:
        """Check if parameters fall within physiological ranges.
        
        Returns
        -------
        dict
            Dictionary with keys indicating various checks and their pass/fail status
        """
        params = self.compute_telegraph_params()
        
        checks = {}
        
        # Check wave speed (5-30 μm/s typical for Ca waves)
        c_eff = params['c_eff_um_per_s']
        checks['wave_speed_in_range'] = 1.0 <= c_eff <= 100.0
        checks['wave_speed_um_per_s'] = c_eff
        
        # Check damping time (0.1-30 s typical for Ca transients)
        gamma0 = params['gamma0']
        damping_time = 1.0 / gamma0 if gamma0 > 0 else np.inf
        checks['damping_time_in_range'] = 0.1 <= damping_time <= 30.0
        checks['damping_time_s'] = damping_time
        
        # Check propagation length (10-1000 μm typical)
        prop_length = c_eff / gamma0 if gamma0 > 0 else np.inf
        checks['propagation_length_in_range'] = 10.0 <= prop_length <= 1000.0
        checks['propagation_length_um'] = prop_length
        
        # Check if oscillatory (ω₀ > 0)
        checks['is_oscillatory'] = params['omega0'] > 0
        if checks['is_oscillatory']:
            period = 2 * np.pi / params['omega0']
            checks['oscillation_period_s'] = period
            checks['oscillation_period_in_range'] = 0.5 <= period <= 30.0
        
        # Overall check
        checks['all_physiological'] = (
            checks['wave_speed_in_range'] and
            checks['damping_time_in_range'] and
            checks['propagation_length_in_range']
        )
        
        return checks
    
    def print_summary(self):
        """Print a summary of micro and derived parameters."""
        print("=" * 70)
        print("GLIAL MICRO-MODEL PARAMETERS")
        print("=" * 70)
        print(f"Ca²⁺ decay rate (α):        {self.alpha:.3f} s⁻¹")
        print(f"IP₃ → Ca²⁺ coupling (β):    {self.beta:.3f} s⁻¹")
        print(f"Ca²⁺ → IP₃ production (γ):  {self.gamma:.3f} s⁻¹")
        print(f"IP₃ degradation (δ):        {self.delta:.3f} s⁻¹")
        print(f"IP₃ diffusion (D):          {self.D_um2_per_s:.1f} μm²/s")
        print()
        
        params = self.compute_telegraph_params()
        print("DERIVED TELEGRAPH PARAMETERS")
        print("-" * 70)
        print(f"Damping coefficient (γ₀):   {params['gamma0']:.3f} s⁻¹")
        print(f"Oscillation frequency (ω₀): {params['omega0']:.3f} rad/s")
        print(f"Wave speed (c_eff):         {params['c_eff_um_per_s']:.2f} μm/s "
              f"= {params['c_eff_mm_per_s']:.4f} mm/s")
        print()
        
        checks = self.check_physiological()
        print("PHYSIOLOGICAL VALIDATION")
        print("-" * 70)
        print(f"Ca wave speed:              {checks['wave_speed_um_per_s']:.2f} μm/s "
              f"[{'✓' if checks['wave_speed_in_range'] else '✗'} 1-100 μm/s]")
        print(f"Damping time (1/γ₀):        {checks['damping_time_s']:.2f} s "
              f"[{'✓' if checks['damping_time_in_range'] else '✗'} 0.1-30 s]")
        print(f"Propagation length:         {checks['propagation_length_um']:.1f} μm "
              f"[{'✓' if checks['propagation_length_in_range'] else '✗'} 10-1000 μm]")
        
        if checks['is_oscillatory']:
            print(f"Oscillation period:         {checks['oscillation_period_s']:.2f} s "
                  f"[{'✓' if checks['oscillation_period_in_range'] else '✗'} 0.5-30 s]")
        else:
            print("System is overdamped (no oscillations)")
        
        print()
        if checks['all_physiological']:
            print("✓ All parameters within physiological ranges")
        else:
            print("⚠ WARNING: Some parameters outside physiological ranges")
        print("=" * 70)


@dataclass
class GlialFieldConfig:
    """Configuration for 2D glial field.
    
    Parameters
    ----------
    Lx, Ly : float
        Domain size in mm
    Nx, Ny : int
        Number of grid points
    micro_params : GliaMicroParams, optional
        Microscopic reaction-diffusion parameters. If provided, telegraph
        parameters are derived. Otherwise, use c, gamma, omega0 directly.
    c : float, optional
        Wave speed in mm/s (used if micro_params is None)
    gamma : float, optional
        Background damping coefficient in 1/s (used if micro_params is None)
    omega0 : float, optional
        Oscillation frequency in rad/s (used if micro_params is None)
    dt : float
        Time step in seconds
    pml_width : int
        Width of PML layer in grid points
    pml_sigma_max : float
        Maximum PML damping coefficient
    """
    Lx: float = 32.0
    Ly: float = 32.0
    Nx: int = 64
    Ny: int = 64
    micro_params: Optional[GliaMicroParams] = None
    c: float = 15.0
    gamma: float = 0.5
    omega0: float = 0.0
    dt: float = 0.001
    pml_width: int = 6
    pml_sigma_max: float = 50.0
    
    def __post_init__(self):
        """Derive telegraph parameters from micro-params if provided."""
        if self.micro_params is not None:
            params = self.micro_params.compute_telegraph_params()
            self.c = params['c_eff_mm_per_s']
            self.gamma = params['gamma0']
            # omega0² can be negative (overdamped), store the signed value
            self.omega0_squared = params['omega0_squared']
            self.omega0 = params['omega0']
    
    @property
    def dx(self) -> float:
        """Grid spacing in x direction (mm)."""
        return self.Lx / self.Nx
    
    @property
    def dy(self) -> float:
        """Grid spacing in y direction (mm)."""
        return self.Ly / self.Ny
    
    @property
    def cfl(self) -> float:
        """CFL number for stability analysis."""
        return self.c * self.dt / min(self.dx, self.dy)
    
    def check_cfl(self, max_cfl: float = 0.5) -> bool:
        """Check if CFL condition is satisfied."""
        return self.cfl <= max_cfl


class GlialField:
    """2D glial (astrocytic) field solver for telegraph equation.
    
    Implements the mesoscopic telegraph equation derived from IP₃/Ca dynamics:
    
        ∂²ψ/∂t² + 2γ₀·∂ψ/∂t + ω₀²·ψ - c_eff²·Δψ = S_ψ(r,t) + ξ(r,t)
    
    as a first-order system:
    
        ∂ₜu = v
        ∂ₜv = c_eff²·∇²u - 2γ₀·v - ω₀²·u + S_ψ + ξ
    
    where:
        - u(r,t) is the glial field (Ca²⁺ or combined IP₃/Ca signal)
        - v(r,t) = ∂ₜu is the field velocity
        - c_eff is the effective wave speed
        - γ₀ is the damping coefficient
        - ω₀ is the oscillation frequency
        - S_ψ is the source term (from neuronal activity)
        - ξ is spatiotemporal noise
    
    Parameters
    ----------
    config : GlialFieldConfig
        Configuration parameters
    rng : np.random.Generator, optional
        Random number generator for noise
    
    Attributes
    ----------
    u : ndarray
        Glial field amplitude (Ny, Nx)
    v : ndarray
        Glial field velocity (Ny, Nx)
    gamma_total : ndarray
        Total damping coefficient including PML (Ny, Nx)
    t : float
        Current simulation time
    
    Examples
    --------
    >>> # Using micro-parameters
    >>> micro = GliaMicroParams(alpha=1.5, beta=0.8, gamma=1.0, delta=1.0, D_um2_per_s=100.0)
    >>> config = GlialFieldConfig(Lx=32.0, Ly=32.0, Nx=64, Ny=64, micro_params=micro)
    >>> glia = GlialField(config)
    >>> snapshots = glia.run(T=1.0, record_interval=10)
    """
    
    def __init__(self, config: GlialFieldConfig, rng: Optional[np.random.Generator] = None):
        self.config = config
        self.rng = rng if rng is not None else np.random.default_rng()
        
        # Initialize fields
        self.u = np.zeros((config.Ny, config.Nx))
        self.v = np.zeros((config.Ny, config.Nx))
        self.t = 0.0
        
        # Setup PML damping
        self.gamma_total = self._create_pml_damping()
        
        # Check CFL condition
        if not config.check_cfl():
            warnings.warn(
                f"CFL number {config.cfl:.3f} may be too large for stability. "
                f"Consider dt <= {0.5 * min(config.dx, config.dy) / config.c:.6f}",
                RuntimeWarning
            )
        
        # Print summary if using micro-parameters
        if config.micro_params is not None:
            config.micro_params.print_summary()
    
    def _create_pml_damping(self) -> np.ndarray:
        """Create PML damping coefficient map.
        
        Uses quadratic profile: σ = σ_max * (d/w)² where d is distance into PML.
        """
        sigma_map = np.zeros((self.config.Ny, self.config.Nx))
        
        for i in range(self.config.Nx):
            for j in range(self.config.Ny):
                # Distance into PML from each edge
                d = 0
                if i < self.config.pml_width:
                    d = max(d, self.config.pml_width - i)
                if i >= self.config.Nx - self.config.pml_width:
                    d = max(d, i - (self.config.Nx - self.config.pml_width - 1))
                if j < self.config.pml_width:
                    d = max(d, self.config.pml_width - j)
                if j >= self.config.Ny - self.config.pml_width:
                    d = max(d, j - (self.config.Ny - self.config.pml_width - 1))
                
                if d > 0:
                    sigma_map[j, i] = self.config.pml_sigma_max * (d / self.config.pml_width) ** 2
        
        return self.config.gamma + sigma_map
    
    def laplacian_9pt(self, u: np.ndarray) -> np.ndarray:
        """Compute 9-point stencil Laplacian with Neumann BC."""
        u_pad = np.pad(u, 1, mode='edge')
        
        # 4-point neighbors (cardinal directions)
        up = u_pad[0:-2, 1:-1]
        down = u_pad[2:, 1:-1]
        left = u_pad[1:-1, 0:-2]
        right = u_pad[1:-1, 2:]
        
        # 4-point diagonal neighbors
        ul = u_pad[0:-2, 0:-2]
        ur = u_pad[0:-2, 2:]
        dl = u_pad[2:, 0:-2]
        dr = u_pad[2:, 2:]
        
        dx = self.config.dx
        lap = (4 * (up + down + left + right) + (ul + ur + dl + dr) - 20 * u) / (6 * dx**2)
        
        return lap
    
    def rhs(self, u: np.ndarray, v: np.ndarray, 
            source: Optional[np.ndarray] = None,
            noise: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Compute right-hand side of the first-order system.
        
        System:
            du/dt = v
            dv/dt = c²∇²u - 2γ_total·v - ω₀²·u + source + noise
        
        Parameters
        ----------
        u, v : ndarray
            Current state (2D fields)
        source : ndarray, optional
            External source term S_ψ(r,t)
        noise : ndarray, optional
            Noise term ξ(r,t)
        
        Returns
        -------
        du_dt, dv_dt : ndarray
            Time derivatives
        """
        lap_u = self.laplacian_9pt(u)
        du_dt = v
        
        # Telegraph equation: includes oscillation term -ω₀²·u
        omega0_squared = getattr(self.config, 'omega0_squared', self.config.omega0**2)
        dv_dt = self.config.c**2 * lap_u - 2 * self.gamma_total * v - omega0_squared * u
        
        if source is not None:
            dv_dt += source
        if noise is not None:
            dv_dt += noise
        
        return du_dt, dv_dt
    
    def step(self, dt: Optional[float] = None,
             source: Optional[np.ndarray] = None,
             noise_amplitude: float = 0.0) -> None:
        """Advance the field by one time step using RK4."""
        if dt is None:
            dt = self.config.dt
        
        # Generate noise if needed
        noise = None
        if noise_amplitude > 0:
            noise = self.rng.normal(0, noise_amplitude / np.sqrt(dt), 
                                   (self.config.Ny, self.config.Nx))
        
        # RK4 integration
        k1u, k1v = self.rhs(self.u, self.v, source, noise)
        k2u, k2v = self.rhs(self.u + 0.5*dt*k1u, self.v + 0.5*dt*k1v, source, noise)
        k3u, k3v = self.rhs(self.u + 0.5*dt*k2u, self.v + 0.5*dt*k2v, source, noise)
        k4u, k4v = self.rhs(self.u + dt*k3u, self.v + dt*k3v, source, noise)
        
        self.u += (dt/6) * (k1u + 2*k2u + 2*k3u + k4u)
        self.v += (dt/6) * (k1v + 2*k2v + 2*k3v + k4v)
        self.t += dt
    
    def run(self, T: float, record_interval: int = 1,
            source_func: Optional[Callable[[float], np.ndarray]] = None,
            noise_amplitude: float = 0.0) -> np.ndarray:
        """Run simulation for duration T."""
        n_steps = int(T / self.config.dt)
        snapshots = []
        
        for step in range(n_steps):
            source = source_func(self.t) if source_func is not None else None
            self.step(source=source, noise_amplitude=noise_amplitude)
            
            if step % record_interval == 0:
                snapshots.append(self.u.copy())
        
        return np.array(snapshots)
    
    def reset(self) -> None:
        """Reset field to zero state."""
        self.u.fill(0.0)
        self.v.fill(0.0)
        self.t = 0.0

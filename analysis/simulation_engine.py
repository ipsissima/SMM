# analysis/simulation_engine.py
"""
Glial Mesh PDE Simulator - Telegraph-approximated Reaction-Diffusion System.

This module implements the Layer 3 mesoscopic field dynamics for the Syncytial
Mesh Model (SMM), strictly following the "Version 3" theoretical framework.

GOVERNING EQUATION (Telegraph-approximated Glial Reaction-Diffusion):
----------------------------------------------------------------------
The field u(r,t) evolves according to:

    ∂²u/∂t² + 2γ₀·∂u/∂t + ω₀²·u - c_eff²·∇²u = S(r,t)

where:
    - u(r,t): mesoscopic glial field (dimensionless or mV)
    - γ₀: damping coefficient (1/s), controls decay rate
    - ω₀²: characteristic frequency squared (rad²/s²), the MASS TERM creating
           the band-pass filter from glial micro-kinetics (α·δ - β·γ)
    - c_eff: effective wave speed (mm/s), ~15 mm/s, the group velocity of
             coherence envelopes (NOT microscopic Ca²⁺ wave speed ~20 μm/s)
    - S(r,t): phenomenological Gaussian-cosine source term approximating
              neural drive (Neural Masses and Kuramoto layers are conceptual)

NUMERICAL IMPLEMENTATION (First-order system):
-----------------------------------------------
State variables: u = field amplitude, v = ∂u/∂t

    ∂u/∂t = v
    ∂v/∂t = c_eff²·∇²u - 2γ₀·v - ω₀²·u + S(r,t)

Spatial discretization: 9-point stencil Laplacian (periodic boundaries)
Time integration: RK4
PML boundaries: Perfectly Matched Layer with enhanced damping at edges

CRITICAL PARAMETERS:
--------------------
    - c_eff (~15 mm/s): mesoscopic derived parameter, group velocity
    - ω₀² (omega0_sq): mass term from glial micro-kinetics
    - γ₀ (gamma): damping coefficient (2γ₀ factor in equation)

SCALE DEFENSE:
--------------
The 4 Hz resonance corresponds to the intrinsic correlation length λ* ~ 3.7 mm
selected by the medium, NOT the fundamental harmonic of the 32mm domain.

Contact: Andreu.Ballus@uab.cat
"""
import numpy as np
import warnings


class MeshSimulator:
    """
    Telegraph-approximated glial mesh field simulator.
    
    Implements the mesoscopic field dynamics with proper mass term (ω₀²).
    
    Parameters
    ----------
    nx : int, default=32
        Number of grid points in x direction
    ny : int, default=32
        Number of grid points in y direction
    dx : float, default=1.0
        Grid spacing (mm)
    c : float, default=0.015
        Effective wave speed c_eff (mm/s) - mesoscopic group velocity
    gamma_bg : float, default=0.1
        Background damping coefficient γ₀ (1/s) - note equation uses 2γ₀
    omega0_sq : float, default=0.0
        Characteristic frequency squared ω₀² (rad²/s²) - the MASS TERM
        from glial micro-kinetics (α·δ - β·γ)
    pml_width : int, default=4
        Width of PML (Perfectly Matched Layer) region in grid points
    gamma_pml : float, default=2.0
        Enhanced damping coefficient in PML region (1/s)
    
    Attributes
    ----------
    gamma : ndarray
        Spatially-varying damping field (background + PML)
    
    Notes
    -----
    The CFL condition for stability is: c_eff * dt / dx < 1
    For c_eff=15 mm/s, dx=1 mm, requires dt < 0.067 s
    Typical dt=0.001 s provides ample stability margin.
    """
    def __init__(self, nx=32, ny=32, dx=1.0, c=0.015, gamma_bg=0.1, 
                 omega0_sq=0.0, pml_width=4, gamma_pml=2.0):
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.c = c
        self.gamma_bg = gamma_bg
        self.omega0_sq = omega0_sq
        self.gamma_pml = gamma_pml
        self.pml_width = pml_width
        self._build_gamma()

    def _build_gamma(self):
        """
        Build spatially-varying damping field with PML boundaries.
        
        Uses vectorized operations (no explicit loops over spatial coordinates).
        PML region has enhanced damping to absorb outgoing waves.
        """
        self.gamma = self.gamma_bg * np.ones((self.ny, self.nx))
        
        # Vectorized PML mask construction
        i_indices = np.arange(self.nx)
        j_indices = np.arange(self.ny)
        i_grid, j_grid = np.meshgrid(i_indices, j_indices, indexing='xy')
        
        # Identify PML region (edges of domain)
        in_pml = (
            (i_grid < self.pml_width) |
            (i_grid >= self.nx - self.pml_width) |
            (j_grid < self.pml_width) |
            (j_grid >= self.ny - self.pml_width)
        )
        
        # Apply enhanced damping in PML region
        self.gamma[in_pml] = self.gamma_pml

    def laplacian_2d(self, u):
        """
        Compute 9-point stencil Laplacian with periodic boundaries.
        
        The 9-point stencil has the form:
            [1  4  1]
            [4 -20 4]  / (6·dx²)
            [1  4  1]
        
        Parameters
        ----------
        u : ndarray, shape (ny, nx)
            Field to differentiate
            
        Returns
        -------
        lap : ndarray, shape (ny, nx)
            Discrete Laplacian ∇²u
            
        Notes
        -----
        Uses np.roll for periodic boundary conditions.
        Fully vectorized - no explicit spatial loops.
        """
        lap = (
            -20.0 * u
            + 4.0
            * (
                np.roll(u, 1, axis=0)
                + np.roll(u, -1, axis=0)
                + np.roll(u, 1, axis=1)
                + np.roll(u, -1, axis=1)
            )
            + (
                np.roll(np.roll(u, 1, axis=0), 1, axis=1)
                + np.roll(np.roll(u, 1, axis=0), -1, axis=1)
                + np.roll(np.roll(u, -1, axis=0), 1, axis=1)
                + np.roll(np.roll(u, -1, axis=0), -1, axis=1)
            )
        ) / (6.0 * self.dx**2)
        return lap

    def rhs(self, u, v, S=None):
        """
        Compute right-hand side of the telegraph equation (first-order system).
        
        Equations:
            du/dt = v
            dv/dt = c_eff²·∇²u - 2γ·v - ω₀²·u + S(r,t)
        
        Parameters
        ----------
        u : ndarray, shape (ny, nx)
            Field amplitude at current time
        v : ndarray, shape (ny, nx)
            Time derivative ∂u/∂t at current time
        S : ndarray or None, shape (ny, nx)
            External source term S(r,t). If None, no forcing is applied.
            
        Returns
        -------
        du_dt : ndarray, shape (ny, nx)
            Time derivative of u (= v)
        dv_dt : ndarray, shape (ny, nx)
            Time derivative of v (= c²∇²u - 2γv - ω₀²u + S)
            
        Notes
        -----
        CRITICAL: This includes the mass term ω₀²·u which creates the band-pass
        filter effect from glial micro-kinetics (α·δ - β·γ).
        The gamma field already contains the factor of 2 from 2γ₀.
        """
        lap = self.laplacian_2d(u)
        # Telegraph equation: dv/dt = c²∇²u - 2γv - ω₀²u + S
        dv = self.c**2 * lap - 2.0 * self.gamma * v - self.omega0_sq * u
        if S is not None:
            dv = dv + S
        return v, dv

    def rk4_step(self, u, v, dt, t, stimulus_func=None):
        """
        Take one Runge-Kutta 4th order time step.
        
        Parameters
        ----------
        u : ndarray, shape (ny, nx)
            Field amplitude at time t
        v : ndarray, shape (ny, nx)
            Time derivative at time t
        dt : float
            Time step size (s)
        t : float
            Current time (s)
        stimulus_func : callable or None
            Optional function S(t) returning source term array of shape (ny, nx)
            
        Returns
        -------
        u_new : ndarray, shape (ny, nx)
            Field amplitude at time t + dt
        v_new : ndarray, shape (ny, nx)
            Time derivative at time t + dt
        """
        def rhs_uv(u_loc, v_loc, tloc):
            S = stimulus_func(tloc) if (stimulus_func is not None) else None
            return self.rhs(u_loc, v_loc, S)

        k1u, k1v = rhs_uv(u, v, t)
        k2u, k2v = rhs_uv(u + 0.5 * dt * k1u, v + 0.5 * dt * k1v, t + 0.5 * dt)
        k3u, k3v = rhs_uv(u + 0.5 * dt * k2u, v + 0.5 * dt * k2v, t + 0.5 * dt)
        k4u, k4v = rhs_uv(u + dt * k3u, v + dt * k3v, t + dt)
        u_new = u + (dt / 6.0) * (k1u + 2 * k2u + 2 * k3u + k4u)
        v_new = v + (dt / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
        return u_new, v_new

    def run(self, T=5.0, dt=0.001, u0=None, v0=None, stimulus_func=None, record_every=1):
        """
        Run the simulation from t=0 to t=T.
        
        Parameters
        ----------
        T : float, default=5.0
            Total simulation duration (s)
        dt : float, default=0.001
            Time step size (s)
        u0 : ndarray or None, shape (ny, nx)
            Initial field amplitude. If None, starts from zeros.
        v0 : ndarray or None, shape (ny, nx)
            Initial time derivative. If None, starts from zeros.
        stimulus_func : callable or None
            Optional function S(t) returning source term of shape (ny, nx)
        record_every : int, default=1
            Record field every N steps (for memory efficiency)
            
        Returns
        -------
        traces : ndarray, shape (n_snapshots, ny, nx)
            Recorded field snapshots
        u : ndarray, shape (ny, nx)
            Final field amplitude
        v : ndarray, shape (ny, nx)
            Final time derivative
            
        Warnings
        --------
        Checks CFL condition: c_eff * dt / dx < 1 for stability.
        """
        # Check CFL condition for stability
        cfl = self.c * dt / self.dx
        if cfl >= 1.0:
            warnings.warn(
                f"CFL condition violated: c*dt/dx = {cfl:.3f} >= 1. "
                f"Simulation may be unstable. Reduce dt or increase dx. "
                f"(c_eff={self.c} mm/s, dt={dt} s, dx={self.dx} mm)",
                RuntimeWarning
            )
        
        nt = int(T / dt)
        if u0 is None:
            u = np.zeros((self.ny, self.nx))
        else:
            u = u0.copy()
        if v0 is None:
            v = np.zeros((self.ny, self.nx))
        else:
            v = v0.copy()
        traces = []
        for step in range(nt):
            t = step * dt
            u, v = self.rk4_step(u, v, dt, t, stimulus_func=stimulus_func)
            if step % record_every == 0:
                traces.append(u.copy())
        return np.array(traces), u, v

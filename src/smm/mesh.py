"""
2D Mesh field solver for damped wave PDE.

Implements a first-order system for the damped wave (telegraph) equation:
    ∂ₜu = v
    ∂ₜv = c²∇²u - γv + S(r,t) + η(r,t)

where:
    - u(r,t) is the field amplitude
    - v(r,t) = ∂ₜu is the field velocity
    - c is the wave speed
    - γ is the damping coefficient (background + PML)
    - S(r,t) is an external source/forcing term
    - η(r,t) is white noise

The spatial domain is a 2D rectangular grid with Perfectly Matched Layer (PML)
boundaries implemented via position-dependent damping.

Contact: Andreu.Ballus@uab.cat
"""

import numpy as np
from typing import Optional, Callable, Tuple
from dataclasses import dataclass


@dataclass
class MeshConfig:
    """Configuration for 2D mesh field.
    
    Parameters
    ----------
    Lx, Ly : float
        Domain size in mm
    Nx, Ny : int
        Number of grid points
    c : float
        Wave speed in mm/s
    gamma : float
        Background damping coefficient in 1/s
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
    c: float = 15.0
    gamma: float = 0.5
    dt: float = 0.001
    pml_width: int = 6
    pml_sigma_max: float = 50.0
    
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
        """Check if CFL condition is satisfied.
        
        Parameters
        ----------
        max_cfl : float
            Maximum allowed CFL number
            
        Returns
        -------
        bool
            True if CFL condition is satisfied
        """
        return self.cfl <= max_cfl


class MeshField:
    """2D mesh field solver for damped wave equation.
    
    Implements a first-order system using RK4 time integration and
    9-point stencil for the Laplacian with PML boundaries.
    
    Parameters
    ----------
    config : MeshConfig
        Configuration parameters
    rng : np.random.Generator, optional
        Random number generator for noise
    
    Attributes
    ----------
    u : ndarray
        Field amplitude (Ny, Nx)
    v : ndarray
        Field velocity (Ny, Nx)
    gamma_total : ndarray
        Total damping coefficient including PML (Ny, Nx)
    t : float
        Current simulation time
        
    Examples
    --------
    >>> config = MeshConfig(Lx=32.0, Ly=32.0, Nx=64, Ny=64)
    >>> mesh = MeshField(config)
    >>> # Run for 1 second, recording every 10 steps
    >>> snapshots = mesh.run(T=1.0, record_interval=10)
    """
    
    def __init__(self, config: MeshConfig, rng: Optional[np.random.Generator] = None):
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
            import warnings
            warnings.warn(
                f"CFL number {config.cfl:.3f} may be too large for stability. "
                f"Consider dt <= {0.5 * min(config.dx, config.dy) / config.c:.6f}",
                RuntimeWarning
            )
    
    def _create_pml_damping(self) -> np.ndarray:
        """Create PML damping coefficient map.
        
        Uses quadratic profile: σ = σ_max * (d/w)² where d is distance into PML.
        
        Returns
        -------
        ndarray
            Total damping coefficient (background + PML) at each grid point
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
        """Compute 9-point stencil Laplacian.
        
        Uses the formula:
            ∇²u ≈ [4(u[i±1,j] + u[i,j±1]) + (u[i±1,j±1]) - 20u[i,j]] / (6·dx²)
        
        with Neumann boundary conditions (zero normal derivative).
        
        Parameters
        ----------
        u : ndarray
            2D field of shape (Ny, Nx)
            
        Returns
        -------
        ndarray
            Laplacian of u
        """
        # Pad for boundary handling (Neumann BC approximation)
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
        
        # Assume dx = dy for simplicity
        dx = self.config.dx
        lap = (4 * (up + down + left + right) + (ul + ur + dl + dr) - 20 * u) / (6 * dx**2)
        
        return lap
    
    def rhs(self, u: np.ndarray, v: np.ndarray, 
            source: Optional[np.ndarray] = None,
            noise: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Compute right-hand side of the first-order system.
        
        System:
            du/dt = v
            dv/dt = c²∇²u - γ_total·v + source + noise
        
        Parameters
        ----------
        u, v : ndarray
            Current state (2D fields)
        source : ndarray, optional
            External source term S(r,t)
        noise : ndarray, optional
            Noise term η(r,t)
            
        Returns
        -------
        du_dt, dv_dt : ndarray
            Time derivatives
        """
        lap_u = self.laplacian_9pt(u)
        du_dt = v
        dv_dt = self.config.c**2 * lap_u - self.gamma_total * v
        
        if source is not None:
            dv_dt += source
        if noise is not None:
            dv_dt += noise
        
        return du_dt, dv_dt
    
    def step(self, dt: Optional[float] = None,
             source: Optional[np.ndarray] = None,
             noise_amplitude: float = 0.0) -> None:
        """Advance the field by one time step using RK4.
        
        Parameters
        ----------
        dt : float, optional
            Time step (defaults to config.dt)
        source : ndarray, optional
            External source term S(r,t) for this step
        noise_amplitude : float
            Amplitude of white noise (noise ~ N(0, σ²/dt))
        """
        if dt is None:
            dt = self.config.dt
        
        # Generate noise if needed
        noise = None
        if noise_amplitude > 0:
            # Scale noise by 1/sqrt(dt) so variance is independent of dt
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
        """Run simulation for duration T.
        
        Parameters
        ----------
        T : float
            Total simulation time in seconds
        record_interval : int
            Record field every N steps
        source_func : callable, optional
            Function f(t) that returns source field S(r,t) at time t
        noise_amplitude : float
            Amplitude of white noise
            
        Returns
        -------
        ndarray
            Array of field snapshots, shape (n_snapshots, Ny, Nx)
        """
        n_steps = int(T / self.config.dt)
        snapshots = []
        
        for step in range(n_steps):
            # Get source term for current time
            source = source_func(self.t) if source_func is not None else None
            
            # Advance one step
            self.step(source=source, noise_amplitude=noise_amplitude)
            
            # Record if needed
            if step % record_interval == 0:
                snapshots.append(self.u.copy())
        
        return np.array(snapshots)
    
    def reset(self) -> None:
        """Reset field to zero state."""
        self.u.fill(0.0)
        self.v.fill(0.0)
        self.t = 0.0
    
    def get_spatial_fft(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute 2D spatial FFT of current field.
        
        Returns
        -------
        kx, ky : ndarray
            Wave number arrays
        U_k : ndarray
            2D FFT of field u
        """
        # Compute FFT
        U_k = np.fft.fft2(self.u)
        U_k = np.fft.fftshift(U_k)
        
        # Wave number arrays
        kx = np.fft.fftshift(np.fft.fftfreq(self.config.Nx, self.config.dx))
        ky = np.fft.fftshift(np.fft.fftfreq(self.config.Ny, self.config.dy))
        
        return kx, ky, U_k
    
    def get_mode_amplitude(self, nx: int, ny: int) -> float:
        """Get amplitude of spatial Fourier mode (nx, ny).
        
        Parameters
        ----------
        nx, ny : int
            Mode indices
            
        Returns
        -------
        float
            Mode amplitude |U_{nx,ny}|
        """
        kx, ky, U_k = self.get_spatial_fft()
        
        # Find closest wave numbers to the theoretical mode
        kx_mode = 2 * np.pi * nx / self.config.Lx
        ky_mode = 2 * np.pi * ny / self.config.Ly
        
        idx_x = np.argmin(np.abs(kx - kx_mode))
        idx_y = np.argmin(np.abs(ky - ky_mode))
        
        return np.abs(U_k[idx_y, idx_x])
    
    def get_eigenfrequency(self, nx: int, ny: int) -> float:
        """Get theoretical eigenfrequency for mode (nx, ny).
        
        For a 2D domain with Neumann BC:
            f_{nx,ny} = (c / 2π) √[(πnx/Lx)² + (πny/Ly)²]
        
        Parameters
        ----------
        nx, ny : int
            Mode indices
            
        Returns
        -------
        float
            Eigenfrequency in Hz
        """
        kx = np.pi * nx / self.config.Lx
        ky = np.pi * ny / self.config.Ly
        k = np.sqrt(kx**2 + ky**2)
        return self.config.c * k / (2 * np.pi)


def create_gaussian_source(positions: list, activities: np.ndarray,
                           grid_shape: Tuple[int, int], dx: float,
                           sigma: float) -> np.ndarray:
    """Create spatial source field from point sources with Gaussian kernels.
    
    This is a utility function for building S(r,t) from discrete sources
    (e.g., neural mass activities at specific locations).
    
    Parameters
    ----------
    positions : list of tuple
        List of (i, j) grid indices for each source
    activities : ndarray
        Activity values for each source
    grid_shape : tuple
        (Ny, Nx) shape of the grid
    dx : float
        Grid spacing in mm
    sigma : float
        Gaussian kernel width in mm
        
    Returns
    -------
    ndarray
        2D source field of shape (Ny, Nx)
    """
    from scipy.signal import fftconvolve
    
    Ny, Nx = grid_shape
    
    # Create delta functions at source positions
    source_sparse = np.zeros((Ny, Nx))
    for (i, j), activity in zip(positions, activities):
        if 0 <= i < Nx and 0 <= j < Ny:
            source_sparse[j, i] += activity
    
    # Create Gaussian kernel
    sigma_grid = sigma / dx
    kernel_size = int(6 * sigma_grid)
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    ax = np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma_grid**2))
    kernel /= kernel.sum()
    
    # Convolve with Gaussian kernel
    source_field = fftconvolve(source_sparse, kernel, mode='same')
    
    return source_field

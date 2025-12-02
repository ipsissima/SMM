"""
Coupling functions between neural, mesh, and Kuramoto layers.

Implements minimal, numerically consistent coupling:
1. Neural masses → mesh: excitatory activities create Gaussian source terms
2. Mesh → Kuramoto: mesh field modulates phase dynamics
3. Mesh → neural masses: optional feedback to excitatory input

Contact: Andreu.Ballus@uab.cat
"""

import numpy as np
from typing import Tuple, Optional
from .mesh import create_gaussian_source


def neural_to_mesh_source(
    E: np.ndarray,
    positions: list,
    grid_shape: Tuple[int, int],
    dx: float,
    sigma: float,
    amplitude: float = 1.0
) -> np.ndarray:
    """Convert neural mass activities to mesh source field.
    
    Creates a spatial source term S(r,t) as a sum of Gaussians centered
    at neural region positions, weighted by excitatory activities.
    
    Parameters
    ----------
    E : ndarray
        Excitatory activities (n_regions,)
    positions : list of tuple
        Grid positions (i, j) for each region
    grid_shape : tuple
        (Ny, Nx) shape of mesh grid
    dx : float
        Grid spacing in mm
    sigma : float
        Gaussian kernel width in mm
    amplitude : float
        Overall scaling factor
        
    Returns
    -------
    ndarray
        Source field S(r,t) of shape (Ny, Nx)
        
    Examples
    --------
    >>> E = np.array([1.0, 0.5, 0.0, 0.2])
    >>> positions = [(10, 10), (20, 10), (10, 20), (20, 20)]
    >>> source = neural_to_mesh_source(E, positions, (32, 32), 0.5, 1.0)
    >>> source.shape
    (32, 32)
    """
    # Scale activities by amplitude
    activities = amplitude * E
    
    # Create source field using Gaussian convolution
    source_field = create_gaussian_source(
        positions, activities, grid_shape, dx, sigma
    )
    
    return source_field


def mesh_to_kuramoto(
    u: np.ndarray,
    positions: list,
    kappa: float = 0.0
) -> np.ndarray:
    """Sample mesh field at region positions for Kuramoto modulation.
    
    Extracts mesh field values u(rᵢ,t) at specified positions and
    scales by coupling strength κ.
    
    Parameters
    ----------
    u : ndarray
        Mesh field of shape (Ny, Nx)
    positions : list of tuple
        Grid positions (i, j) for each region
    kappa : float
        Coupling strength (mesh → Kuramoto)
        
    Returns
    -------
    ndarray
        Mesh values at positions, scaled by kappa (n_regions,)
        
    Examples
    --------
    >>> u = np.random.randn(32, 32)
    >>> positions = [(10, 10), (20, 10)]
    >>> mesh_input = mesh_to_kuramoto(u, positions, kappa=0.1)
    >>> len(mesh_input)
    2
    """
    n_regions = len(positions)
    mesh_values = np.zeros(n_regions)
    
    for idx, (i, j) in enumerate(positions):
        # Extract value at position (with bounds checking)
        Ny, Nx = u.shape
        if 0 <= i < Nx and 0 <= j < Ny:
            mesh_values[idx] = u[j, i]
    
    return kappa * mesh_values


def mesh_to_neural(
    u: np.ndarray,
    positions: list,
    kappa: float = 0.0,
    nonlinearity: str = 'linear'
) -> np.ndarray:
    """Convert mesh field to neural mass feedback.
    
    Samples mesh field at region positions and optionally applies
    a nonlinearity before scaling.
    
    Parameters
    ----------
    u : ndarray
        Mesh field of shape (Ny, Nx)
    positions : list of tuple
        Grid positions (i, j) for each region
    kappa : float
        Coupling strength (mesh → neural)
    nonlinearity : str
        Type of nonlinearity: 'linear', 'tanh', 'sigmoid'
        
    Returns
    -------
    ndarray
        Feedback to neural masses (n_regions,)
        
    Examples
    --------
    >>> u = np.random.randn(32, 32)
    >>> positions = [(10, 10), (20, 10)]
    >>> feedback = mesh_to_neural(u, positions, kappa=0.1, nonlinearity='tanh')
    >>> len(feedback)
    2
    """
    n_regions = len(positions)
    mesh_values = np.zeros(n_regions)
    
    for idx, (i, j) in enumerate(positions):
        Ny, Nx = u.shape
        if 0 <= i < Nx and 0 <= j < Ny:
            mesh_values[idx] = u[j, i]
    
    # Apply nonlinearity
    if nonlinearity == 'tanh':
        mesh_values = np.tanh(mesh_values)
    elif nonlinearity == 'sigmoid':
        mesh_values = 1.0 / (1.0 + np.exp(-mesh_values))
    elif nonlinearity == 'linear':
        pass
    else:
        raise ValueError(f"Unknown nonlinearity: {nonlinearity}")
    
    return kappa * mesh_values


def create_region_positions_grid(
    Nx: int,
    Ny: int,
    n_regions: int,
    edge_padding: float = 0.1
) -> list:
    """Create evenly spaced region positions on a grid.
    
    Distributes n_regions points on a regular subgrid, avoiding edges.
    
    Parameters
    ----------
    Nx, Ny : int
        Grid dimensions
    n_regions : int
        Number of regions to place
    edge_padding : float
        Fraction of grid to avoid at edges (0.1 = avoid outer 10%)
        
    Returns
    -------
    list of tuple
        List of (i, j) grid positions
        
    Examples
    --------
    >>> positions = create_region_positions_grid(64, 64, 16)
    >>> len(positions)
    16
    >>> # All positions should be within grid bounds
    >>> all(0 <= i < 64 and 0 <= j < 64 for i, j in positions)
    True
    """
    n_side = int(np.ceil(np.sqrt(n_regions)))
    positions = []
    
    # Calculate padding in grid units
    pad_x = int(edge_padding * Nx)
    pad_y = int(edge_padding * Ny)
    
    for idx in range(n_regions):
        i_region = idx % n_side
        j_region = idx // n_side
        
        # Map to grid coordinates with padding
        i = int(pad_x + i_region * (Nx - 2*pad_x) / (n_side - 1)) if n_side > 1 else Nx // 2
        j = int(pad_y + j_region * (Ny - 2*pad_y) / (n_side - 1)) if n_side > 1 else Ny // 2
        
        # Ensure within bounds
        i = max(0, min(Nx - 1, i))
        j = max(0, min(Ny - 1, j))
        
        positions.append((i, j))
    
    return positions


class CoupledModel:
    """Coupled three-layer model (neural, mesh, Kuramoto).
    
    Integrates neural masses, mesh field, and Kuramoto oscillators with
    configurable coupling strengths.
    
    Parameters
    ----------
    mesh : MeshField
        Mesh field instance
    neural : NeuralMass
        Neural mass instance
    kuramoto : Kuramoto
        Kuramoto instance
    positions : list of tuple
        Region positions on mesh grid
    coupling_neural_mesh : float
        Coupling strength E → mesh
    coupling_mesh_kuramoto : float
        Coupling strength mesh → Kuramoto
    coupling_mesh_neural : float
        Coupling strength mesh → neural (feedback)
    sigma_source : float
        Width of Gaussian sources (mm)
        
    Examples
    --------
    >>> from smm.mesh import MeshField, MeshConfig
    >>> from smm.neural import NeuralMass, Kuramoto, NeuralMassConfig, KuramotoConfig
    >>> 
    >>> mesh_config = MeshConfig(Nx=32, Ny=32)
    >>> mesh = MeshField(mesh_config)
    >>> 
    >>> neural_config = NeuralMassConfig(n_regions=4)
    >>> neural = NeuralMass(neural_config)
    >>> 
    >>> kuramoto_config = KuramotoConfig(n_oscillators=4)
    >>> kuramoto = Kuramoto(kuramoto_config)
    >>> 
    >>> positions = create_region_positions_grid(32, 32, 4)
    >>> model = CoupledModel(mesh, neural, kuramoto, positions,
    ...                      coupling_neural_mesh=0.1)
    """
    
    def __init__(
        self,
        mesh,  # MeshField
        neural,  # NeuralMass
        kuramoto,  # Kuramoto
        positions: list,
        coupling_neural_mesh: float = 0.0,
        coupling_mesh_kuramoto: float = 0.0,
        coupling_mesh_neural: float = 0.0,
        sigma_source: float = 1.0
    ):
        self.mesh = mesh
        self.neural = neural
        self.kuramoto = kuramoto
        self.positions = positions
        
        self.coupling_neural_mesh = coupling_neural_mesh
        self.coupling_mesh_kuramoto = coupling_mesh_kuramoto
        self.coupling_mesh_neural = coupling_mesh_neural
        self.sigma_source = sigma_source
        
        # Check consistency
        n_regions = len(positions)
        assert neural.config.n_regions == n_regions, \
            "Neural mass and positions must have same number of regions"
        assert kuramoto.config.n_oscillators == n_regions, \
            "Kuramoto and positions must have same number of oscillators"
    
    def step(self, dt: float,
             I_ext: Optional[np.ndarray] = None,
             noise_amplitude: float = 0.0) -> None:
        """Advance all layers by one time step.
        
        Integration order:
        1. Compute mesh → neural and mesh → Kuramoto coupling
        2. Step neural masses with external input and mesh feedback
        3. Step Kuramoto with mesh modulation
        4. Compute neural → mesh coupling
        5. Step mesh with source term
        
        Parameters
        ----------
        dt : float
            Time step in seconds
        I_ext : ndarray, optional
            External input to neural masses (n_regions,)
        noise_amplitude : float
            Mesh noise amplitude
        """
        # Sample mesh field at region positions
        mesh_values = mesh_to_kuramoto(
            self.mesh.u, self.positions, kappa=1.0
        )  # Get unscaled values
        
        # Mesh → neural feedback
        I_mesh = None
        if self.coupling_mesh_neural != 0:
            I_mesh = self.coupling_mesh_neural * mesh_values
        
        # Step neural masses
        self.neural.step(dt, I_ext=I_ext, I_mesh=I_mesh)
        
        # Mesh → Kuramoto coupling
        kuramoto_input = None
        if self.coupling_mesh_kuramoto != 0:
            kuramoto_input = self.coupling_mesh_kuramoto * mesh_values
        
        # Step Kuramoto
        self.kuramoto.step(dt, mesh_input=kuramoto_input)
        
        # Neural → mesh source
        source = None
        if self.coupling_neural_mesh != 0:
            source = neural_to_mesh_source(
                self.neural.E,
                self.positions,
                (self.mesh.config.Ny, self.mesh.config.Nx),
                self.mesh.config.dx,
                self.sigma_source,
                amplitude=self.coupling_neural_mesh
            )
        
        # Step mesh
        self.mesh.step(dt, source=source, noise_amplitude=noise_amplitude)
    
    def run(self, T: float, dt: float,
            I_ext_func: Optional[callable] = None,
            noise_amplitude: float = 0.0,
            record_interval: int = 1) -> dict:
        """Run coupled model for duration T.
        
        Parameters
        ----------
        T : float
            Total simulation time in seconds
        dt : float
            Time step in seconds
        I_ext_func : callable, optional
            Function f(t) returning external input (n_regions,)
        noise_amplitude : float
            Mesh noise amplitude
        record_interval : int
            Record state every N steps
            
        Returns
        -------
        dict
            Dictionary with keys:
            - 't': time array
            - 'E': excitatory activities (n_steps, n_regions)
            - 'I': inhibitory activities (n_steps, n_regions)
            - 'theta': Kuramoto phases (n_steps, n_oscillators)
            - 'u': mesh field snapshots (n_steps, Ny, Nx)
        """
        n_steps = int(T / dt)
        n_record = n_steps // record_interval + 1
        
        # Allocate storage
        t_array = np.zeros(n_record)
        E_array = np.zeros((n_record, self.neural.config.n_regions))
        I_array = np.zeros((n_record, self.neural.config.n_regions))
        theta_array = np.zeros((n_record, self.kuramoto.config.n_oscillators))
        u_array = np.zeros((n_record, self.mesh.config.Ny, self.mesh.config.Nx))
        
        # Initial condition
        record_idx = 0
        t_array[0] = 0.0
        E_array[0] = self.neural.E.copy()
        I_array[0] = self.neural.I.copy()
        theta_array[0] = self.kuramoto.theta.copy()
        u_array[0] = self.mesh.u.copy()
        record_idx += 1
        
        # Time integration
        for step in range(n_steps):
            t = step * dt
            
            # Get external input
            I_ext = I_ext_func(t) if I_ext_func is not None else None
            
            # Advance one step
            self.step(dt, I_ext=I_ext, noise_amplitude=noise_amplitude)
            
            # Record
            if (step + 1) % record_interval == 0 and record_idx < n_record:
                t_array[record_idx] = self.mesh.t
                E_array[record_idx] = self.neural.E.copy()
                I_array[record_idx] = self.neural.I.copy()
                theta_array[record_idx] = self.kuramoto.theta.copy()
                u_array[record_idx] = self.mesh.u.copy()
                record_idx += 1
        
        return {
            't': t_array[:record_idx],
            'E': E_array[:record_idx],
            'I': I_array[:record_idx],
            'theta': theta_array[:record_idx],
            'u': u_array[:record_idx]
        }

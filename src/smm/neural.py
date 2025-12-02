"""
Neural mass and Kuramoto phase oscillator layers.

Implements minimal models for:
1. Wilson-Cowan-like neural mass dynamics
2. Kuramoto phase oscillators on a network

These are simplified versions focused on coupling with the mesh layer.
The biological interpretation is left to the paper; here we focus on
numerics and reproducibility.

Contact: Andreu.Ballus@uab.cat
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class NeuralMassConfig:
    """Configuration for Wilson-Cowan-like neural mass.
    
    Implements simplified dynamics:
        τE dE/dt = -E + S(wEE·E - wEI·I + I_ext + I_mesh)
        τI dI/dt = -I + S(wIE·E - wII·I)
    
    where S(x) is a sigmoidal activation function.
    
    Parameters
    ----------
    n_regions : int
        Number of neural regions
    tau_E : float
        Excitatory time constant (seconds)
    tau_I : float
        Inhibitory time constant (seconds)
    w_EE, w_EI, w_IE, w_II : float
        Synaptic weights
    gain : float
        Sigmoid gain parameter
    threshold : float
        Sigmoid threshold parameter
    """
    n_regions: int = 16
    tau_E: float = 0.01
    tau_I: float = 0.01
    w_EE: float = 1.5
    w_EI: float = 1.2
    w_IE: float = 1.0
    w_II: float = 0.5
    gain: float = 4.0
    threshold: float = 0.5


class NeuralMass:
    """Wilson-Cowan-like neural mass model.
    
    Simplified neural mass dynamics for coupling with mesh layer.
    
    Parameters
    ----------
    config : NeuralMassConfig
        Configuration parameters
    
    Attributes
    ----------
    E : ndarray
        Excitatory population activities (n_regions,)
    I : ndarray
        Inhibitory population activities (n_regions,)
    t : float
        Current simulation time
        
    Examples
    --------
    >>> config = NeuralMassConfig(n_regions=4)
    >>> neural = NeuralMass(config)
    >>> # Step with external input
    >>> I_ext = np.array([0.5, 0.5, 0.0, 0.0])
    >>> neural.step(dt=0.001, I_ext=I_ext)
    """
    
    def __init__(self, config: NeuralMassConfig):
        self.config = config
        self.E = np.zeros(config.n_regions)
        self.I = np.zeros(config.n_regions)
        self.t = 0.0
    
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoidal activation function.
        
        S(x) = 1 / (1 + exp(-gain * (x - threshold)))
        
        Parameters
        ----------
        x : ndarray
            Input values
            
        Returns
        -------
        ndarray
            Activated values
        """
        return 1.0 / (1.0 + np.exp(-self.config.gain * (x - self.config.threshold)))
    
    def rhs(self, E: np.ndarray, I: np.ndarray,
            I_ext: Optional[np.ndarray] = None,
            I_mesh: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Compute right-hand side of neural mass equations.
        
        Parameters
        ----------
        E, I : ndarray
            Current excitatory and inhibitory activities
        I_ext : ndarray, optional
            External input to excitatory population
        I_mesh : ndarray, optional
            Feedback from mesh field
            
        Returns
        -------
        dE_dt, dI_dt : ndarray
            Time derivatives
        """
        # Total input to E population
        input_E = self.config.w_EE * E - self.config.w_EI * I
        if I_ext is not None:
            input_E += I_ext
        if I_mesh is not None:
            input_E += I_mesh
        
        # Total input to I population
        input_I = self.config.w_IE * E - self.config.w_II * I
        
        # Dynamics
        dE_dt = (-E + self.sigmoid(input_E)) / self.config.tau_E
        dI_dt = (-I + self.sigmoid(input_I)) / self.config.tau_I
        
        return dE_dt, dI_dt
    
    def step(self, dt: float,
             I_ext: Optional[np.ndarray] = None,
             I_mesh: Optional[np.ndarray] = None) -> None:
        """Advance neural mass by one time step using Euler integration.
        
        Parameters
        ----------
        dt : float
            Time step in seconds
        I_ext : ndarray, optional
            External input (n_regions,)
        I_mesh : ndarray, optional
            Mesh feedback (n_regions,)
        """
        dE_dt, dI_dt = self.rhs(self.E, self.I, I_ext, I_mesh)
        self.E += dt * dE_dt
        self.I += dt * dI_dt
        self.t += dt
    
    def reset(self) -> None:
        """Reset to zero state."""
        self.E.fill(0.0)
        self.I.fill(0.0)
        self.t = 0.0


@dataclass
class KuramotoConfig:
    """Configuration for Kuramoto phase oscillators.
    
    Implements phase dynamics:
        dθᵢ/dt = ωᵢ + (K/N) Σⱼ sin(θⱼ - θᵢ) + κ·u(rᵢ,t)
    
    where:
        - θᵢ is the phase of oscillator i
        - ωᵢ is the natural frequency
        - K is the coupling strength
        - u(rᵢ,t) is the mesh field sampled at region i
        - κ is the mesh→Kuramoto coupling strength
    
    Parameters
    ----------
    n_oscillators : int
        Number of oscillators
    K : float
        Coupling strength
    omega_mean : float
        Mean natural frequency (Hz)
    omega_std : float
        Standard deviation of natural frequencies (Hz)
    kappa_mesh : float
        Coupling strength from mesh field
    """
    n_oscillators: int = 16
    K: float = 0.5
    omega_mean: float = 4.0
    omega_std: float = 0.5
    kappa_mesh: float = 0.0


class Kuramoto:
    """Kuramoto phase oscillator model.
    
    Simple phase oscillators with all-to-all coupling and optional
    mesh field modulation.
    
    Parameters
    ----------
    config : KuramotoConfig
        Configuration parameters
    rng : np.random.Generator, optional
        Random number generator for natural frequencies
    
    Attributes
    ----------
    theta : ndarray
        Phase of each oscillator (n_oscillators,)
    omega : ndarray
        Natural frequency of each oscillator (rad/s)
    t : float
        Current simulation time
        
    Examples
    --------
    >>> config = KuramotoConfig(n_oscillators=4, K=0.5)
    >>> kuramoto = Kuramoto(config)
    >>> # Step with mesh field modulation
    >>> mesh_values = np.array([0.1, -0.1, 0.0, 0.0])
    >>> kuramoto.step(dt=0.001, mesh_input=mesh_values)
    """
    
    def __init__(self, config: KuramotoConfig, 
                 rng: Optional[np.random.Generator] = None):
        self.config = config
        self.rng = rng if rng is not None else np.random.default_rng()
        
        # Initialize phases uniformly
        self.theta = self.rng.uniform(0, 2*np.pi, config.n_oscillators)
        
        # Sample natural frequencies
        self.omega = self.rng.normal(
            config.omega_mean * 2 * np.pi,  # Convert Hz to rad/s
            config.omega_std * 2 * np.pi,
            config.n_oscillators
        )
        
        self.t = 0.0
    
    def rhs(self, theta: np.ndarray,
            mesh_input: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute right-hand side of Kuramoto equations.
        
        Parameters
        ----------
        theta : ndarray
            Current phases
        mesh_input : ndarray, optional
            Mesh field values at oscillator positions
            
        Returns
        -------
        ndarray
            Time derivatives dθ/dt
        """
        N = self.config.n_oscillators
        
        # Compute coupling term: (K/N) Σⱼ sin(θⱼ - θᵢ)
        # Broadcasting: theta[None, :] - theta[:, None] gives (i, j) differences
        phase_diff = theta[None, :] - theta[:, None]  # (N, N)
        coupling = (self.config.K / N) * np.sum(np.sin(phase_diff), axis=1)
        
        # Base dynamics
        dtheta_dt = self.omega + coupling
        
        # Add mesh modulation if provided
        if mesh_input is not None and self.config.kappa_mesh != 0:
            dtheta_dt += self.config.kappa_mesh * mesh_input
        
        return dtheta_dt
    
    def step(self, dt: float,
             mesh_input: Optional[np.ndarray] = None) -> None:
        """Advance phases by one time step using Euler integration.
        
        Parameters
        ----------
        dt : float
            Time step in seconds
        mesh_input : ndarray, optional
            Mesh field values (n_oscillators,)
        """
        dtheta_dt = self.rhs(self.theta, mesh_input)
        self.theta += dt * dtheta_dt
        
        # Keep phases in [0, 2π)
        self.theta = np.mod(self.theta, 2*np.pi)
        
        self.t += dt
    
    def reset(self, seed: Optional[int] = None) -> None:
        """Reset to random initial phases.
        
        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        self.theta = self.rng.uniform(0, 2*np.pi, self.config.n_oscillators)
        self.t = 0.0
    
    def get_order_parameter(self) -> Tuple[float, float]:
        """Compute Kuramoto order parameter.
        
        The order parameter R·exp(iΨ) = (1/N) Σⱼ exp(iθⱼ) measures
        phase synchronization.
        
        Returns
        -------
        R : float
            Order parameter magnitude (0 = incoherent, 1 = fully synchronized)
        Psi : float
            Order parameter phase
        """
        z = np.mean(np.exp(1j * self.theta))
        R = np.abs(z)
        Psi = np.angle(z)
        return R, Psi

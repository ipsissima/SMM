"""Exact linear IP3/Ca reaction-diffusion model (not a wave model).

For c_t=-alpha*c+beta*i and i_t=gamma*c-delta*i+D*Laplace(i),
the eliminated c equation is
c_tt+(alpha+delta)c_t+(alpha*delta-beta*gamma)c
 -alpha*D*Laplace(c)-D*Laplace(c_t)=0.

The mixed space-time derivative is essential. For beta*gamma >= 0 all spatial
Fourier modes have real eigenvalues: this cooperative linear model does not
by itself generate oscillatory travelling waves. This implementation is
independent from historical phenomenological telegraph simulations and from
newer potassium-mediated SMM-K.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class IP3CaLinearParams:
    alpha: float = 1.0
    beta: float = 0.8
    gamma: float = 0.9
    delta: float = 2.0
    D_um2_per_s: float = 100.0

    def __post_init__(self):
        if self.alpha <= 0 or self.delta <= 0 or self.D_um2_per_s < 0:
            raise ValueError("alpha, delta must be positive and D nonnegative")
        if not np.all(np.isfinite([self.alpha,self.beta,self.gamma,self.delta,self.D_um2_per_s])):
            raise ValueError("All parameters must be finite")

    def modal_matrix(self, wavenumber_squared_um2: float) -> np.ndarray:
        k2 = float(wavenumber_squared_um2)
        if not np.isfinite(k2) or k2 < 0:
            raise ValueError("Squared wavenumber must be finite and nonnegative")
        return np.array([[-self.alpha, self.beta],
                         [self.gamma, -self.delta-self.D_um2_per_s*k2]], dtype=float)

    def modal_eigenvalues(self, wavenumber_squared_um2: float) -> np.ndarray:
        return np.linalg.eigvals(self.modal_matrix(wavenumber_squared_um2))

    def eliminated_characteristic_coefficients(self, wavenumber_squared_um2: float) -> np.ndarray:
        k2 = float(wavenumber_squared_um2)
        if not np.isfinite(k2) or k2 < 0:
            raise ValueError("Squared wavenumber must be finite and nonnegative")
        return np.array([1., self.alpha+self.delta+self.D_um2_per_s*k2,
                         self.alpha*self.delta-self.beta*self.gamma+self.alpha*self.D_um2_per_s*k2])

    def cooperative_discriminant(self, wavenumber_squared_um2: float) -> float:
        k2 = float(wavenumber_squared_um2)
        if not np.isfinite(k2) or k2 < 0:
            raise ValueError("Squared wavenumber must be finite and nonnegative")
        return (self.alpha-self.delta-self.D_um2_per_s*k2)**2+4*self.beta*self.gamma

    def is_locally_stable(self) -> bool:
        return self.alpha*self.delta > self.beta*self.gamma

    def characteristic_diffusive_length_um(self):
        """Not a wave-propagation distance; no universal equivalent exists."""
        return None

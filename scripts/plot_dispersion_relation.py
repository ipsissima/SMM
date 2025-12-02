#!/usr/bin/env python3
"""
scripts/plot_dispersion_relation.py

Plot telegraph dispersion diagnostics:
    - Re(omega(k)), Im(omega(k))
    - group velocity v_g(k)
    - attenuation length as function of frequency L_att(omega)
"""

import numpy as np
import matplotlib.pyplot as plt

def telegraph_omega(k, c, gamma0, omega0):
    disc = omega0**2 - gamma0**2 + (c**2) * (k**2)
    sqrt_disc = np.sqrt(disc.astype(complex))
    omega_plus = -1j * gamma0 + sqrt_disc
    omega_minus = -1j * gamma0 - sqrt_disc
    return omega_plus, omega_minus

def group_velocity_numeric(k, c, gamma0, omega0, branch=0):
    # numerical derivative of real(omega)
    omega_plus, omega_minus = telegraph_omega(k, c, gamma0, omega0)
    omega = omega_plus if branch == 0 else omega_minus
    re_omega = np.real(omega)
    dk = np.gradient(k)
    dre = np.gradient(re_omega)
    v_g = dre / dk
    return v_g

def k_of_omega(omega, c, gamma0, omega0):
    # Solve for k(omega): k = sqrt((omega^2 + 2 i gamma0 omega - omega0^2)/c^2)
    val = (omega**2 + 2j * gamma0 * omega - omega0**2) / (c**2)
    return np.sqrt(val.astype(complex))

def plot_dispersion(c, gamma0, omega0):
    # k in 1/mm
    k = np.logspace(-2, 2, 1000)  # 0.01 .. 100  1/mm
    omega_p, omega_m = telegraph_omega(k, c, gamma0, omega0)
    # use '+' branch (positive frequency)
    omega = omega_p
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.ravel()
    axs[0].loglog(k, np.abs(np.real(omega)), label=r'Re$\ \omega$')
    axs[0].loglog(k, np.abs(np.imag(omega)), label=r'Im$\ \omega$')
    axs[0].axvline(x=2*np.pi/4.0, color='r', ls='--', label='4 mm scale')
    axs[0].set_xlabel('k (1/mm)')
    axs[0].set_ylabel('Frequency (rad/s)')
    axs[0].legend()
    axs[0].set_title('Dispersion: Re/Im omega vs k')

    v_g = group_velocity_numeric(k, c, gamma0, omega0)
    axs[1].loglog(k, np.abs(v_g))
    axs[1].axvline(x=2*np.pi/4.0, color='r', ls='--')
    axs[1].set_xlabel('k (1/mm)')
    axs[1].set_ylabel('Group velocity (mm/s)')
    axs[1].set_title('Group velocity |v_g(k)|')

    # attenuation length vs frequency: choose frequency grid around predicted physiological band
    freqs_hz = np.linspace(0.1, 40.0, 400)  # Hz
    omegas = 2*np.pi*freqs_hz
    kvals = k_of_omega(omegas, c, gamma0, omega0)
    # attenuation length (1/Im(k))
    L_att = 1.0 / np.maximum(1e-12, np.abs(np.imag(kvals)))
    axs[2].semilogy(freqs_hz, L_att)
    axs[2].set_xlabel('frequency (Hz)')
    axs[2].set_ylabel('attenuation length (mm)')
    axs[2].set_title('Attenuation length vs frequency')

    # phase lag per mm: Re(k)
    axs[3].plot(freqs_hz, np.real(kvals))
    axs[3].set_xlabel('frequency (Hz)')
    axs[3].set_ylabel('Re(k) (rad/mm)')
    axs[3].set_title('Phase wavenumber vs frequency')

    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Example parameters (override with values from params.yaml)
    # c in mm/s; typically small for glial waves (e.g., c ~ 0.007 mm/s for 7 um/s)
    c = 0.00707  # mm/s  (example)
    gamma0 = 1.25  # 1/s
    omega0 = 0.0   # rad/s  (overdamped or zero-frequency)
    fig = plot_dispersion(c, gamma0, omega0)
    fig.suptitle(f"Telegraph dispersion (c={c:.5g} mm/s, gamma0={gamma0}, omega0={omega0})")
    fig.savefig("results/dispersion_summary.png", dpi=200)
    print("Saved results/dispersion_summary.png")
    plt.show()

# analysis/eigenmode_projection.py
import numpy as np
from .utils import compute_spatial_eigenmode


def project_onto_mode(traces, e_star):
    T = traces.shape[0]
    ny, nx = traces.shape[1], traces.shape[2]
    e = e_star.reshape((ny, nx))
    A = np.array([np.sum(trace * e) for trace in traces])
    return A


def compute_modal_eigenmode(nx=32, ny=32, dx=1.0):
    e_star, eigval = compute_spatial_eigenmode(nx, ny, dx)
    return e_star, eigval


def validate_single_mode(simulator, e_star, A0=1e-3, T=1.0, dt=0.001):
    u0 = (A0 * e_star).reshape(simulator.ny, simulator.nx)
    v0 = np.zeros_like(u0)
    traces = simulator.run(T=T, dt=dt, u0=u0, v0=v0, record_every=1)
    A = project_onto_mode(traces, e_star)
    energies = np.array([np.sum(tr**2) for tr in traces])
    modal = np.array([(np.sum(tr * e_star.reshape(tr.shape)) ** 2) for tr in traces])
    frac = float(np.mean(modal / (energies + 1e-12)))
    return A, frac

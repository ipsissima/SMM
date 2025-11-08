# analysis/identify_coeffs.py
import numpy as np
from .eigenmode_projection import project_onto_mode
from .simulation_engine import MeshSimulator


def fit_polynomial_dotA(A_series, dt):
    dA = np.gradient(A_series, dt)
    X = np.vstack([A_series, -A_series**3, np.ones_like(A_series)]).T
    coefs, *_ = np.linalg.lstsq(X, dA, rcond=None)
    mu, gamma, delta = coefs[0], coefs[1], coefs[2]
    return mu, gamma, delta


def numeric_identify_coeffs(simulator: MeshSimulator, e_star, A0_list=None, Tshort=1.0, dt=0.001):
    if A0_list is None:
        A0_list = np.linspace(1e-3, 5e-2, 7)
    samples = []
    for A0 in A0_list:
        u0 = (A0 * e_star).reshape(simulator.ny, simulator.nx)
        v0 = np.zeros_like(u0)
        traces = simulator.run(T=Tshort, dt=dt, u0=u0, v0=v0, record_every=1)
        A = project_onto_mode(traces, e_star)
        start = max(1, int(0.05 * len(A)))
        window = slice(start, len(A))
        mu, gamma, delta = fit_polynomial_dotA(A[window], dt)
        samples.append((A0, mu, gamma, delta))
    mu_med = float(np.median([s[1] for s in samples]))
    gamma_med = float(np.median([s[2] for s in samples]))
    delta_med = float(np.median([s[3] for s in samples]))
    return {"mu": mu_med, "gamma": gamma_med, "delta": delta_med, "samples": samples}

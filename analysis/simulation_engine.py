# analysis/simulation_engine.py
import numpy as np


class MeshSimulator:
    def __init__(self, nx=32, ny=32, dx=1.0, c=0.015, gamma_bg=0.1, pml_width=4, gamma_pml=2.0):
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.c = c
        self.gamma_bg = gamma_bg
        self.gamma_pml = gamma_pml
        self.pml_width = pml_width
        self._build_gamma()

    def _build_gamma(self):
        self.gamma = self.gamma_bg * np.ones((self.ny, self.nx))
        for i in range(self.nx):
            for j in range(self.ny):
                if (
                    i < self.pml_width
                    or i >= self.nx - self.pml_width
                    or j < self.pml_width
                    or j >= self.ny - self.pml_width
                ):
                    self.gamma[j, i] = self.gamma_pml

    def laplacian_2d(self, u):
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
        lap = self.laplacian_2d(u)
        dv = self.c**2 * lap - self.gamma * v
        if S is not None:
            dv = dv + S
        return v, dv

    def rk4_step(self, u, v, dt, t, stimulus_func=None):
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
        return np.array(traces)

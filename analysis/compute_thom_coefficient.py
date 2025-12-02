#!/usr/bin/env python3
"""
analysis/compute_thom_coefficient.py

Compute the Thom cubic coefficient alpha for the glial telegraph operator
by projecting the cubic nonlinearity u^3 onto the adjoint eigenmode.

Usage:
    python analysis/compute_thom_coefficient.py --params params.yaml
or
    python analysis/compute_thom_coefficient.py --Nx 64 --Ny 64 --L_mm 32 --c 7.07 --gamma 1.25 --omega0 0.0
"""

import argparse
import yaml
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import sys

def build_9pt_laplacian(Nx, Ny, dx):
    N = Nx * Ny
    rows = []
    cols = []
    data = []
    # weights per 9-point stencil (for -Delta)
    w_center = -20.0 / (6.0 * dx**2)
    w_card = 4.0 / (6.0 * dx**2)
    w_diag = 1.0 / (6.0 * dx**2)
    for j in range(Ny):
        for i in range(Nx):
            p = j * Nx + i
            # center
            rows.append(p); cols.append(p); data.append(w_center)
            # neighbors
            for (di, dj, w) in [ (0, -1, w_card), (0, 1, w_card), (-1, 0, w_card), (1, 0, w_card),
                                 (-1,-1, w_diag), (-1,1,w_diag), (1,-1,w_diag), (1,1,w_diag) ]:
                ii = i + di
                jj = j + dj
                # reflect for Neumann BC
                ii_ref = min(max(ii, 0), Nx - 1)
                jj_ref = min(max(jj, 0), Ny - 1)
                q = jj_ref * Nx + ii_ref
                rows.append(p); cols.append(q); data.append(w)
    L = sp.csr_matrix((data, (rows, cols)), shape=(N, N))
    return L

def build_block_L(Laplacian, c, omega0, gamma0):
    N = Laplacian.shape[0]
    I = sp.eye(N, format='csr')
    zero = sp.csr_matrix((N, N))
    A = c**2 * Laplacian - (omega0**2) * I
    B = -2.0 * gamma0 * I
    # Block matrix [[0, I], [A, B]]
    top = sp.hstack([zero, I], format='csr')
    bot = sp.hstack([A, B], format='csr')
    L = sp.vstack([top, bot], format='csr')
    return L

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--params', type=str, default=None)
    p.add_argument('--Nx', type=int, default=64)
    p.add_argument('--Ny', type=int, default=64)
    p.add_argument('--L_mm', type=float, default=32.0)
    p.add_argument('--c', type=float, default=None, help='c_eff in mm/s')
    p.add_argument('--gamma', type=float, default=None)
    p.add_argument('--omega0', type=float, default=None)
    p.add_argument('--k_eigs', type=int, default=1)
    p.add_argument('--sigma', type=float, default=0.0, help='target near eigenvalue (0)')
    return p.parse_args()

def main():
    args = parse_args()
    if args.params:
        with open(args.params, 'r') as f:
            params = yaml.safe_load(f)
        Nx = params.get('Nx', args.Nx)
        Ny = params.get('Ny', args.Ny)
        L_mm = params.get('L_mm', args.L_mm)
        dx = L_mm / Nx
        c = params.get('c_eff_mm_per_s', None)
        if c is None:
            # if micro params provided, user may want compute_telegraph_params; for now require explicit
            c = args.c if args.c is not None else params.get('c_mm_per_s', 15.0)
        gamma0 = params.get('gamma0', None) or params.get('gamma_s', args.gamma or 0.5)
        omega0 = params.get('omega0', None) or args.omega0 or 0.0
    else:
        Nx = args.Nx; Ny = args.Ny; L_mm = args.L_mm
        dx = L_mm / Nx
        c = args.c or 7.07
        gamma0 = args.gamma or 1.25
        omega0 = args.omega0 or 0.0

    print(f"Grid: {Nx}x{Ny}, L={L_mm} mm, dx={dx:.6f} mm")
    print(f"Parameters: c={c} mm/s, gamma0={gamma0}, omega0={omega0}")

    Lap = build_9pt_laplacian(Nx, Ny, dx)
    N = Nx * Ny
    L = build_block_L(Lap, c, omega0, gamma0)

    # Find eigenpair near 0 (sigma=0). Use shift-invert if needed.
    k = args.k_eigs
    print("Computing eigenpair near sigma=0 ... (this may take a moment)")
    eigvals, eigvecs = spla.eigs(L, k=k, sigma=args.sigma, which='LM', maxiter=2000)
    lam = eigvals[0]
    vec = eigvecs[:, 0]
    # compute adjoint eigenpair (left eigenvector) via eigenvectors of L.T
    eigvalsT, eigvecsT = spla.eigs(L.transpose().conj(), k=k, sigma=np.conj(args.sigma), which='LM', maxiter=2000)
    lamT = eigvalsT[0]
    vecT = eigvecsT[:, 0]

    # pick u-components (first N entries)
    Phi = vec[:N]
    Psi = vecT[:N]  # adjoint left eigenvector
    # normalize so denominator is real positive if possible
    dx_area = dx * (L_mm / Ny) if False else dx * dx  # dx*dy; here Lx/Ly == L_mm
    dy = dx
    area = dx * dy

    # compute inner products (complex)
    # ensure vectors are same orientation
    numerator = np.vdot(Psi, (Phi.real**3 + 3j*Phi.real**2*Phi.imag - 3*Phi.real*(Phi.imag**2) - 1j*(Phi.imag**3))) if False else np.vdot(Psi, Phi**3)
    denominator = np.vdot(Psi, Phi)
    alpha = numerator / denominator
    # scale by area element
    alpha *= area

    print("Eigenvalue (closest to 0):", lam)
    print("Adjoint eigenvalue:", lamT)
    print("Raw alpha (complex):", alpha)
    print("Real(alpha):", np.real(alpha))
    print("Imag(alpha):", np.imag(alpha))
    # Return real part as physically meaningful cubic coefficient
    print("Alpha (returned):", float(np.real(alpha)))

if __name__ == "__main__":
    main()

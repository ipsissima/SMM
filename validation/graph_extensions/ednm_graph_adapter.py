"""Graph-selective ionic transport correction for Sætra-Mori full EDNM Newton solver.

RESEARCH STATUS: transport adapter tested against finite differences; this
module is not yet wired to the original ahead-of-time compiled full solver.
The original model uses a closed path and local membrane terms. This module
returns the additional residual and Jacobian obtained by replacing the
soma/dendrite path transport with separately configured glial and ECS graphs.

Uses original source: D, upsilon, lambdas, psi, gamma_m, dxu, ionic residual
index maps and PREVIOUS volume fractions. Edge orientation is from lower to
higher node index; this reproduces the original source-edge-volume convention.
"""
import numpy as np
from scipy.sparse import coo_matrix


def _edges(W):
    W = np.asarray(W, float)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("Graph must be square")
    if np.any(W < 0) or not np.allclose(W, W.T):
        raise ValueError("Ionic graphs must be nonnegative and symmetric")
    if np.any(np.diag(W)):
        raise ValueError("No graph self-edges")
    return [(u, v, float(W[u, v]))
            for u in range(len(W)) for v in range(u + 1, len(W))
            if W[u, v] > 0]


def closed_path(n):
    W = np.zeros((n, n))
    for i in range(n - 1):
        W[i, i + 1] = W[i + 1, i] = 1.0
    return W


def correction(model, glial_graph, extracellular_graph, dt, alpha_previous,
               c_current, phi_current, layer="soma"):
    """Return sparse (delta residual, delta Jacobian) for the source solver.

    Shapes: alpha_previous=(3,N), c_current=(4,3,N), phi_current=(3,N).
    Layers are 'soma' and 'dendrite'. Source ionic coordinates and
    analytical Nernst-Planck derivatives are kept exactly.
    """
    n = model.N_units
    ap = np.asarray(alpha_previous, float)
    c = np.asarray(c_current, float)
    phi = np.asarray(phi_current, float)
    if ap.shape != (3, n) or c.shape != (4, 3, n) or phi.shape != (3, n):
        raise ValueError("Source-model field dimensions do not agree")
    if dt < 0 or not np.isfinite(dt):
        raise ValueError("Invalid timestep")
    if layer not in ("soma", "dendrite"):
        raise ValueError("Invalid layer")
    jc = model.jc_s if layer == "soma" else model.jc_d
    jphi = model.jphi_s if layer == "soma" else model.jphi_d
    stride = model.N_unknowns_unit
    dim = model.N_unknowns_tot
    original_edges = [(i, i + 1, 1.0) for i in range(n - 1)]
    delta = np.zeros(dim)
    rows, cols, values = [], [], []
    for r, W in ((1, glial_graph), (2, extracellular_graph)):
        new_edges = _edges(W)
        if new_edges == original_edges:
            continue
        for sign, edges in ((1.0, new_edges), (-1.0, original_edges)):
            for u, v, weight in edges:
                for ion in range(4):
                    f1 = (model.D[ion] * model.upsilon[ion, r] *
                          weight / (model.lambdas[r]**2 * model.dxu))
                    if f1 == 0:
                        continue
                    f2 = f1 * model.z[ion] / (2 * model.psi)
                    cu, cv = c[ion, r, u], c[ion, r, v]
                    dv = phi[r, v] - phi[r, u]
                    flux = f1*(cu-cv) - f2*(cu+cv)*dv
                    dcu, dcv = f1-f2*dv, -f1-f2*dv
                    dpu, dpv = f2*(cu+cv), -f2*(cu+cv)
                    coeff = sign * dt * ap[r, u] / (model.dxu * model.gamma_m)
                    dcols = (int(jc[ion, r]+stride*u),
                             int(jc[ion, r]+stride*v),
                             int(jphi[r]+stride*u),
                             int(jphi[r]+stride*v))
                    for node, orientation in ((u, -1.0), (v, +1.0)):
                        row = int(jc[ion, r]+stride*node)
                        delta[row] += orientation * coeff * flux
                        for column, derivative in zip(dcols, (dcu,dcv,dpu,dpv)):
                            rows.append(row)
                            cols.append(column)
                            values.append(orientation*coeff*derivative)
    return delta, coo_matrix((values, (rows, cols)),
                             shape=(dim, dim)).tocsr()

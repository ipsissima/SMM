# analysis/utils.py
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh


def laplacian_9pt_matrix(nx, ny, dx=1.0):
    N = nx * ny
    rows = []
    cols = []
    vals = []

    def idx(i, j):
        return (i % ny) * nx + (j % nx)

    for i in range(ny):
        for j in range(nx):
            center = idx(i, j)
            rows.append(center)
            cols.append(center)
            vals.append(-20.0)
            neigh = [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]
            for (ii, jj) in neigh:
                rows.append(center)
                cols.append(idx(ii, jj))
                vals.append(4.0)
            diags = [(i + 1, j + 1), (i + 1, j - 1), (i - 1, j + 1), (i - 1, j - 1)]
            for (ii, jj) in diags:
                rows.append(center)
                cols.append(idx(ii, jj))
                vals.append(1.0)
    data = np.array(vals) / (6.0 * dx**2)
    mat = coo_matrix((data, (rows, cols)), shape=(N, N))
    return mat.tocsr()


def compute_spatial_eigenmode(nx, ny, dx=1.0, k_eigs=6, zero_tol=1e-10):
    L = laplacian_9pt_matrix(nx, ny, dx)
    # compute smallest magnitude eigenpairs of -L (so positive eigenvalues)
    vals, vecs = eigsh(-L, k=k_eigs, which="SM")
    # sort by absolute value and skip the zero (uniform) eigenmode
    abs_vals = np.abs(vals)
    for idx in np.argsort(abs_vals):
        if abs_vals[idx] > zero_tol:
            pick = idx
            break
    else:
        raise RuntimeError("Unable to find a non-zero eigenvalue; increase k_eigs or adjust zero_tol")
    evec = np.real(vecs[:, pick])
    evec /= np.linalg.norm(evec)
    return evec, vals[pick]

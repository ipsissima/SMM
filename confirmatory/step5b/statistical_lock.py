#!/usr/bin/env python3
"""Frozen statistical primitives for SMM Step 5B."""
from __future__ import annotations
import numpy as np
SEED = 97
def project_csd(csd_64, U20):
    return U20.conj().T @ csd_64 @ U20
def regularize_csd(S, rel_floor=1e-6):
    S = (S + S.conj().T) / 2.0
    w, V = np.linalg.eigh(S)
    floor = rel_floor * max(float(np.max(w.real)), np.finfo(float).eps)
    w = np.maximum(w.real, floor)
    return (V * w) @ V.conj().T
def complex_whittle_score(empirical_csd, model_csd, dof=None, rel_floor=1e-6):
    E = np.asarray(empirical_csd); M = np.asarray(model_csd)
    if E.shape != M.shape or E.ndim != 3 or E.shape[1] != E.shape[2]:
        raise ValueError('CSD arrays must match with shape [frequency,p,p].')
    nu = np.ones(E.shape[0]) if dof is None else np.broadcast_to(np.asarray(dof, float), (E.shape[0],))
    total = 0.0
    for f in range(E.shape[0]):
        Sf = regularize_csd(M[f], rel_floor); Ef = regularize_csd(E[f], rel_floor)
        sign, logdet = np.linalg.slogdet(Sf)
        if sign <= 0: raise FloatingPointError('Regularized model CSD is not positive definite.')
        trace = np.trace(np.linalg.solve(Sf, Ef)).real
        total -= float(nu[f]) * (float(logdet.real) + float(trace))
    return total

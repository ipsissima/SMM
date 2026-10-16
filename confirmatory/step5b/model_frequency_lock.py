#!/usr/bin/env python3
"""Pre-EEG executable completion of the frozen M0--M3 frequency-domain models.

No EEG is read here.  The purpose of this module is to make the Step-5B model
hierarchy mathematically executable before any ds005385 signal is opened.

Key adversarial property
------------------------
M2 is a stable generic two-pole local slow controller driven by the same local
neural activity scalar J_N as M3.  With suitable pole locations and residues it
contains the *linearized* M3 K-homeostasis transfer exactly.  Thus M3 is not
protected by a weaker comparator.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

TAU_E = TAU_I = 0.008
DELTA_E = DELTA_I = 1.0
TAU_SE = 0.001
TAU_SI = 0.005
TAU_LR = 0.001
J_EI = J_IE = 13.0
R_E0 = 8.08904711
V_E0 = -2.45942045
R_I0 = 9.68671239
V_I0 = -2.05377914
KE0 = 3.5
C_E1 = 0.05210958
C_I1 = 0.04673428
Q_E = 4.736e-8
Q_I = 1.267e-8
RHO_TOTAL = 4.929e6
F_E = 0.75
PHI_E, PHI_A = 0.2, 0.4
KAPPA_E = 2.01315923
KAPPA_A = 0.48318016
KAPPA_N = 0.232

J_RE = RHO_TOTAL * F_E * Q_E
J_RI = RHO_TOTAL * (1.0 - F_E) * Q_I
ALPHA_E = C_E1 / KE0
ALPHA_I = C_I1 / KE0

GENERIC_TAU_MIN_S = 0.03
GENERIC_TAU_MAX_S = 30.0
GENERIC_GAIN_ABS_MAX = 0.5

@dataclass(frozen=True)
class M2Params:
    tau1_s: float
    tau2_s: float
    gE1: float
    gE2: float
    gI1: float
    gI2: float
    def validate(self):
        if not (GENERIC_TAU_MIN_S <= self.tau1_s <= self.tau2_s <= GENERIC_TAU_MAX_S):
            raise ValueError('M2 requires 0.03 <= tau1 <= tau2 <= 30 s')
        for x in (self.gE1, self.gE2, self.gI1, self.gI2):
            if abs(x) > GENERIC_GAIN_ABS_MAX:
                raise ValueError('M2 gain outside frozen [-0.5,0.5] bound')

@dataclass(frozen=True)
class M1Params:
    tau_s: float
    gE: float
    gI: float
    def validate(self):
        if not (GENERIC_TAU_MIN_S <= self.tau_s <= GENERIC_TAU_MAX_S):
            raise ValueError('M1 tau outside frozen [0.03,30] s')
        if abs(self.gE) > GENERIC_GAIN_ABS_MAX or abs(self.gI) > GENERIC_GAIN_ABS_MAX:
            raise ValueError('M1 gain outside frozen [-0.5,0.5] bound')

def m3_k_transfer(s: complex) -> complex:
    A = np.array([
        [-(KAPPA_E + KAPPA_N) / PHI_E, KAPPA_A / PHI_E],
        [KAPPA_E / PHI_A, -KAPPA_A / PHI_A],
    ], dtype=complex)
    B = np.array([1.0 / PHI_E, 0.0], dtype=complex)
    return np.linalg.solve(s * np.eye(2, dtype=complex) - A, B)[0]

def m3_poles_residues():
    A = np.array([
        [-(KAPPA_E + KAPPA_N) / PHI_E, KAPPA_A / PHI_E],
        [KAPPA_E / PHI_A, -KAPPA_A / PHI_A],
    ], dtype=float)
    poles = np.linalg.eigvals(A).real
    taus = -1.0 / poles
    num_s = 1.0 / PHI_E
    num_0 = (KAPPA_A / PHI_A) / PHI_E
    trA = np.trace(A)
    residues = np.array([(num_s * p + num_0) / (2.0 * p - trA) for p in poles])
    order = np.argsort(taus)
    return taus[order], residues[order]

def m2_exact_m3_params() -> M2Params:
    tau, residue = m3_poles_residues()
    p = M2Params(
        tau1_s=float(tau[0]), tau2_s=float(tau[1]),
        gE1=float(ALPHA_E * residue[0]), gE2=float(ALPHA_E * residue[1]),
        gI1=float(ALPHA_I * residue[0]), gI2=float(ALPHA_I * residue[1]),
    )
    p.validate()
    return p

def slow_feedback(model: str, s: complex, m1: M1Params | None = None,
                  m2: M2Params | None = None) -> tuple[complex, complex]:
    model = model.upper()
    if model == 'M0':
        return 0j, 0j
    if model == 'M3':
        h = m3_k_transfer(s)
        return ALPHA_E * h, ALPHA_I * h
    if model == 'M1':
        if m1 is None:
            raise ValueError('M1 parameters required')
        m1.validate()
        h = 1.0 / (s + 1.0 / m1.tau_s)
        return m1.gE * h, m1.gI * h
    if model == 'M2':
        if m2 is None:
            raise ValueError('M2 parameters required')
        m2.validate()
        h1 = 1.0 / (s + 1.0 / m2.tau1_s)
        h2 = 1.0 / (s + 1.0 / m2.tau2_s)
        return m2.gE1 * h1 + m2.gE2 * h2, m2.gI1 * h1 + m2.gI2 * h2
    raise ValueError(model)

def local_transfer(model: str, freq_hz: float, m1: M1Params | None = None,
                   m2: M2Params | None = None):
    s = 2j * np.pi * float(freq_hz)
    hE, hI = slow_feedback(model, s, m1=m1, m2=m2)
    A = np.zeros((6, 6), dtype=complex)
    A[0, 0] = 2.0 * V_E0 / TAU_E
    A[0, 1] = 2.0 * R_E0 / TAU_E
    A[1, 0] = -2.0 * np.pi**2 * TAU_E * R_E0
    A[1, 1] = 2.0 * V_E0 / TAU_E
    A[1, 4] = -1.0
    A[2, 2] = 2.0 * V_I0 / TAU_I
    A[2, 3] = 2.0 * R_I0 / TAU_I
    A[3, 2] = -2.0 * np.pi**2 * TAU_I * R_I0
    A[3, 3] = 2.0 * V_I0 / TAU_I
    A[3, 5] = 1.0
    A[4, 2] = J_EI / TAU_SI
    A[4, 4] = -1.0 / TAU_SI
    A[5, 0] = J_IE / TAU_SE
    A[5, 5] = -1.0 / TAU_SE
    A[1, 0] += (hE * J_RE) / TAU_E
    A[1, 2] += (hE * J_RI) / TAU_E
    A[3, 0] += (hI * J_RE) / TAU_I
    A[3, 2] += (hI * J_RI) / TAU_I
    B = np.zeros((6, 3), dtype=complex)
    B[1, 0] = 1.0 / TAU_E
    B[3, 1] = 1.0 / TAU_I
    B[1, 2] = 1.0
    H = np.linalg.solve(s * np.eye(6, dtype=complex) - A, B)
    h_rE = H[0, :]
    c = -TAU_E * H[4, :]
    c[0] += 1.0
    c[2] += TAU_E
    return h_rE, c

def load_network(folder: str | Path):
    folder = Path(folder)
    W = pd.read_csv(folder / 'W_ENIGMA_DK68_spectral_normalized.csv', index_col=0).to_numpy(float)
    edges = pd.read_csv(folder / 'DK68_edges_with_delay_proxy.csv')
    n = W.shape[0]
    dist = np.zeros((n, n), float)
    for row in edges.itertuples(index=False):
        i, j = int(row.i), int(row.j)
        d = float(row.euclidean_center_distance_mm)
        dist[i, j] = dist[j, i] = d
    if W.shape != (68, 68) or dist.shape != (68, 68):
        raise RuntimeError('Expected DK68 network')
    return W, dist

def regional_source_transfer(folder: str | Path, model: str, freq_hz: float,
                             G_N: float, velocity_m_s: float,
                             m1: M1Params | None = None,
                             m2: M2Params | None = None):
    if not (50.0 <= G_N <= 185.0):
        raise ValueError('G_N outside frozen [50,185]')
    if not (3.0 <= velocity_m_s <= 12.0):
        raise ValueError('velocity outside frozen [3,12] m/s')
    W, dist_mm = load_network(folder)
    omega = 2.0 * np.pi * float(freq_hz)
    delay_s = (dist_mm / 1000.0) / float(velocity_m_s)
    Wd = W * np.exp(-1j * omega * delay_s)
    Wd *= (W != 0.0)
    rcoef, scoef = local_transfer(model, freq_hz, m1=m1, m2=m2)
    hrE_uE, hrE_uI, hrE_sLR = rcoef
    c_uE, c_uI, c_sLR = scoef
    s = 2j * np.pi * float(freq_hz)
    lr_filter = float(G_N) / (1.0 + s * TAU_LR)
    A = np.eye(68, dtype=complex) - (hrE_sLR * lr_filter) * Wd
    Ainv = np.linalg.inv(A)
    R_E = Ainv * hrE_uE
    R_I = Ainv * hrE_uI
    K = c_sLR * lr_filter * Wd
    H_E = c_uE * np.eye(68, dtype=complex) + K @ R_E
    H_I = c_uI * np.eye(68, dtype=complex) + K @ R_I
    return H_E, H_I

def projected_model_csd(folder: str | Path, L20: np.ndarray, freqs_hz,
                        model: str, G_N: float, velocity_m_s: float,
                        pE: float, source_scale: float, sensor_floor: float,
                        m1: M1Params | None = None,
                        m2: M2Params | None = None):
    L20 = np.asarray(L20, float)
    if L20.shape != (20, 68):
        raise ValueError(f'Expected L20 shape (20,68), got {L20.shape}')
    if not (0.0 < pE < 1.0 and source_scale > 0.0 and sensor_floor > 0.0):
        raise ValueError('pE in (0,1), source_scale>0, sensor_floor>0 required')
    out = []
    I20 = np.eye(20)
    for f in np.asarray(freqs_hz, float):
        HE, HI = regional_source_transfer(folder, model, f, G_N, velocity_m_s, m1=m1, m2=m2)
        YE = L20 @ HE
        YI = L20 @ HI
        S = source_scale**2 * (pE * (YE @ YE.conj().T) + (1.0 - pE) * (YI @ YI.conj().T))
        S = S + sensor_floor * I20
        out.append((S + S.conj().T) / 2.0)
    return np.asarray(out)

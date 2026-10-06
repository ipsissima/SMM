#!/usr/bin/env python3
"""Development-only N1 normalized nested-comparator robustness probe.

The scientific models are unchanged. This probe changes only optimizer
coordinates/search and exploits the exact mathematical nesting M3 subset M2.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize
from scipy.stats import qmc

from empirical_csd_lock import (
    FREQS_HZ,
    chronological_two_block_indices,
    multitaper_csd,
    normalize_whittle_score,
)
from statistical_lock import complex_whittle_score
from model_frequency_lock import (
    GENERIC_GAIN_ABS_MAX,
    GENERIC_TAU_MAX_S,
    GENERIC_TAU_MIN_S,
    M2Params,
    m2_exact_m3_params,
    projected_model_csd,
    slow_feedback,
)

SEED = 97
SOBOL_M = 9
SOBOL_CANDIDATES = 2 ** SOBOL_M
POLISH_STARTS = 16
MAXITER = 300
FTOL = 1e-11
GTOL = 1e-7
MAXLS = 50
REL_FLOOR = 1e-6
TRAIN_TOL = 5e-4
NEST_TOL = 1e-8
TRANSFER_TOL = 1e-12
TRAIN_RANGE_MAX = 1e-3
CV_RANGE_MAX = 5e-3

HISTORICAL = {
    "M2": {"A": 535.8336527151650, "B": 534.0933137283831},
    "M3": {"A": 535.8430963300440, "B": 533.8829279760863},
}


def load_l20(forward_dir: Path):
    U20 = np.load(forward_dir / "leadfield_U20.npy")
    L = pd.read_csv(forward_dir / "leadfield_DK68_64x68.csv", index_col=0).to_numpy(float)
    if U20.shape != (64, 20) or L.shape != (64, 68):
        raise RuntimeError("Frozen forward shape drift")
    if np.max(np.abs(U20.T @ U20 - np.eye(20))) >= 1e-10:
        raise RuntimeError("Frozen U20 orthogonality failed")
    return U20, U20.T @ L


def amp_centers(empirical_csd, network_dir, L20, model):
    escale = float(np.median(np.real(np.trace(empirical_csd, axis1=1, axis2=2)) / empirical_csd.shape[1]))
    escale = max(escale, np.finfo(float).tiny)
    kwargs = {}
    if model == "M2":
        kwargs["m2"] = M2Params(0.1, 10.0, 0.0, 0.0, 0.0, 0.0)
    base = projected_model_csd(
        network_dir, L20, FREQS_HZ, model, 117.5, 7.5, 0.5, 1.0, escale * 1e-8, **kwargs
    )
    bscale = float(np.median(np.real(np.trace(base, axis1=1, axis2=2)) / base.shape[1]))
    bscale = max(bscale, np.finfo(float).tiny)
    return math.sqrt(escale / bscale), escale


def _lin(lo, hi, u):
    return lo + (hi - lo) * float(u)


def _invlin(lo, hi, x):
    return (float(x) - lo) / (hi - lo)


def decode_unit(model: str, u: np.ndarray, amp_center: float, noise_center: float):
    u = np.asarray(u, float)
    expected = 11 if model == "M2" else 5
    if u.shape != (expected,):
        raise ValueError(f"{model} expected {expected} unit coordinates")
    if np.any(u < -1e-12) or np.any(u > 1.0 + 1e-12):
        raise ValueError("unit coordinate outside [0,1]")
    u = np.clip(u, 0.0, 1.0)

    G = _lin(50.0, 185.0, u[0])
    v = _lin(3.0, 12.0, u[1])
    pE = _lin(1e-4, 1.0 - 1e-4, u[2])
    source_scale = amp_center * 10.0 ** _lin(-4.0, 4.0, u[3])
    sensor_floor = noise_center * 10.0 ** _lin(-8.0, 2.0, u[4])

    m2 = None
    if model == "M2":
        lmin = math.log(GENERIC_TAU_MIN_S)
        lmax = math.log(GENERIC_TAU_MAX_S)
        l1 = _lin(lmin, lmax, u[5])
        l2 = l1 + float(u[6]) * (lmax - l1)
        tau1 = math.exp(l1)
        tau2 = math.exp(l2)
        gains = [_lin(-GENERIC_GAIN_ABS_MAX, GENERIC_GAIN_ABS_MAX, q) for q in u[7:11]]
        m2 = M2Params(tau1, tau2, *gains)
        m2.validate()

    return dict(
        G_N=G,
        velocity_m_s=v,
        pE=pE,
        source_scale=source_scale,
        sensor_floor=sensor_floor,
        m2=m2,
    )


def encode_embedded_m3_as_m2(m3_params: dict, amp_center_m2: float, noise_center_m2: float):
    exact = m2_exact_m3_params()
    u = np.empty(11, float)
    u[0] = _invlin(50.0, 185.0, m3_params["G_N"])
    u[1] = _invlin(3.0, 12.0, m3_params["velocity_m_s"])
    u[2] = _invlin(1e-4, 1.0 - 1e-4, m3_params["pE"])
    u[3] = _invlin(-4.0, 4.0, math.log10(m3_params["source_scale"] / amp_center_m2))
    u[4] = _invlin(-8.0, 2.0, math.log10(m3_params["sensor_floor"] / noise_center_m2))

    lmin = math.log(GENERIC_TAU_MIN_S)
    lmax = math.log(GENERIC_TAU_MAX_S)
    l1 = math.log(exact.tau1_s)
    l2 = math.log(exact.tau2_s)
    u[5] = _invlin(lmin, lmax, l1)
    denom = lmax - l1
    u[6] = 0.0 if abs(denom) < 1e-15 else (l2 - l1) / denom
    for j, g in enumerate((exact.gE1, exact.gE2, exact.gI1, exact.gI2), start=7):
        u[j] = _invlin(-GENERIC_GAIN_ABS_MAX, GENERIC_GAIN_ABS_MAX, g)

    if np.any(u < -1e-10) or np.any(u > 1.0 + 1e-10):
        raise RuntimeError(f"Exact embedded M3 point lies outside frozen M2 search bounds: {u}")
    return np.clip(u, 0.0, 1.0)


def serialise_params(pars):
    out = {k: v for k, v in pars.items() if k != "m2"}
    if pars.get("m2") is not None:
        out["m2"] = pars["m2"].__dict__
    return out


def make_objective(empirical_csd, dof, model, network_dir, L20, amp_center, noise_center):
    def objective(u):
        try:
            p = decode_unit(model, u, amp_center, noise_center)
            kwargs = {k: v for k, v in p.items() if v is not None}
            S = projected_model_csd(network_dir, L20, FREQS_HZ, model, **kwargs)
            score = complex_whittle_score(empirical_csd, S, dof=dof, rel_floor=REL_FLOOR)
            val = -normalize_whittle_score(score, dof)
            return float(val) if np.isfinite(val) else 1e100
        except Exception:
            return 1e100
    return objective


def fit_unit(empirical_csd, dof, model, network_dir, L20, embedded_anchor=None):
    amp_center, noise_center = amp_centers(empirical_csd, network_dir, L20, model)
    dim = 11 if model == "M2" else 5
    objective = make_objective(empirical_csd, dof, model, network_dir, L20, amp_center, noise_center)

    sampler = qmc.Sobol(d=dim, scramble=True, seed=SEED)
    cand = sampler.random_base2(m=SOBOL_M)
    vals = np.asarray([objective(x) for x in cand], float)
    finite = np.isfinite(vals) & (vals < 1e99)
    if not np.any(finite):
        raise RuntimeError(f"{model}: no finite Sobol candidates")

    order = np.argsort(vals)[:POLISH_STARTS]
    successful = []
    all_local = []
    for k in order:
        res = minimize(
            objective,
            cand[k],
            method="L-BFGS-B",
            bounds=[(0.0, 1.0)] * dim,
            options={"maxiter": MAXITER, "ftol": FTOL, "gtol": GTOL, "maxls": MAXLS},
        )
        rec = dict(
            source=f"sobol-{int(k)}",
            success=bool(res.success),
            message=str(res.message),
            fun=float(res.fun),
            nit=int(res.nit),
            nfev=int(res.nfev),
            u=np.asarray(res.x, float),
        )
        all_local.append(rec)
        if rec["success"] and np.isfinite(rec["fun"]) and rec["fun"] < 1e99:
            successful.append(rec)

    anchor_raw_score = None
    anchor_local = None
    if embedded_anchor is not None:
        anchor = np.asarray(embedded_anchor, float)
        anchor_raw_fun = float(objective(anchor))
        if not np.isfinite(anchor_raw_fun) or anchor_raw_fun >= 1e99:
            raise RuntimeError("M2 embedded-M3 anchor is nonfinite")
        anchor_raw_score = -anchor_raw_fun
        res = minimize(
            objective,
            anchor,
            method="L-BFGS-B",
            bounds=[(0.0, 1.0)] * dim,
            options={"maxiter": MAXITER, "ftol": FTOL, "gtol": GTOL, "maxls": MAXLS},
        )
        anchor_local = dict(
            source="embedded-m3-polish",
            success=bool(res.success),
            message=str(res.message),
            fun=float(res.fun),
            nit=int(res.nit),
            nfev=int(res.nfev),
            u=np.asarray(res.x, float),
        )
        all_local.append(anchor_local)
        if anchor_local["success"] and np.isfinite(anchor_local["fun"]) and anchor_local["fun"] < 1e99:
            successful.append(anchor_local)

    if model == "M3":
        if not successful:
            raise RuntimeError("M3: no successful local polish")
        best = min(successful, key=lambda z: z["fun"])
        selected_source = best["source"]
        best_u = best["u"]
        best_fun = best["fun"]
        selected_success = True
    else:
        candidates = list(successful)
        # The unpolished exact embedding is deliberately retained as a valid
        # comparator solution. This is what enforces M3 subset M2 numerically.
        candidates.append(dict(
            source="embedded-m3-raw",
            success=True,
            message="exact nested feasible point retained",
            fun=-float(anchor_raw_score),
            nit=0,
            nfev=1,
            u=np.asarray(embedded_anchor, float),
        ))
        best = min(candidates, key=lambda z: z["fun"])
        selected_source = best["source"]
        best_u = best["u"]
        best_fun = best["fun"]
        selected_success = bool(best["success"])

    pars = decode_unit(model, best_u, amp_center, noise_center)
    return dict(
        params=pars,
        serial=serialise_params(pars),
        train_score=float(-best_fun),
        selected_source=selected_source,
        selected_success=selected_success,
        selected_message=str(best["message"]),
        selected_nit=int(best["nit"]),
        selected_nfev=int(best["nfev"]),
        sobol_raw_best_score=float(-np.min(vals)),
        anchor_raw_score=None if anchor_raw_score is None else float(anchor_raw_score),
        anchor_local=None if anchor_local is None else {
            k: v for k, v in anchor_local.items() if k != "u"
        },
        local_attempts=[
            {k: v for k, v in z.items() if k != "u"} for z in all_local
        ],
        amp_center=float(amp_center),
        noise_center=float(noise_center),
    )


def score_holdout(empirical_csd, dof, model, pars, network_dir, L20):
    kwargs = {k: v for k, v in pars.items() if v is not None}
    S = projected_model_csd(network_dir, L20, FREQS_HZ, model, **kwargs)
    return normalize_whittle_score(
        complex_whittle_score(empirical_csd, S, dof=dof, rel_floor=REL_FLOOR), dof
    )


def transfer_identity_error():
    exact = m2_exact_m3_params()
    err = 0.0
    for f in FREQS_HZ:
        s = 2j * np.pi * float(f)
        a = slow_feedback("M3", s)
        b = slow_feedback("M2", s, m2=exact)
        err = max(err, abs(a[0] - b[0]), abs(a[1] - b[1]))
    return float(err)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("epochs_fif", type=Path)
    ap.add_argument("--forward-dir", type=Path, required=True)
    ap.add_argument("--network-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    transfer_err = transfer_identity_error()
    if transfer_err > TRANSFER_TOL:
        raise RuntimeError(f"Analytic M3-in-M2 transfer identity failed: {transfer_err}")

    U20, L20 = load_l20(args.forward_dir)
    ep = mne.read_epochs(args.epochs_fif, preload=True, verbose="error")
    frozen_channels = [
        x.strip() for x in Path(__file__).with_name("channels_64.txt").read_text().splitlines() if x.strip()
    ]
    if ep.ch_names != frozen_channels:
        raise RuntimeError("Clean-epoch channel-order drift")
    x = ep.get_data(picks=frozen_channels)
    if x.shape[1:] != (64, 1000):
        raise RuntimeError(f"Expected clean epochs [n,64,1000], got {x.shape}")

    A, B = chronological_two_block_indices(len(ep))
    blocks = {}
    for name, idx in (("A", A), ("B", B)):
        f, S, nu = multitaper_csd(x[idx], sfreq=float(ep.info["sfreq"]), project=U20)
        if not np.array_equal(f, FREQS_HZ):
            raise RuntimeError("frequency lock drift")
        blocks[name] = (S, nu)

    out = {
        "protocol": "N1-normalized-nested-comparator",
        "seed": SEED,
        "profile": {
            "sobol_candidates": SOBOL_CANDIDATES,
            "sobol_m": SOBOL_M,
            "polish_starts": POLISH_STARTS,
            "method": "L-BFGS-B-unit-cube-plus-exact-nested-anchor",
            "maxiter": MAXITER,
            "ftol": FTOL,
            "gtol": GTOL,
            "maxls": MAXLS,
            "rel_floor": REL_FLOOR,
        },
        "transfer_identity_max_abs_error": transfer_err,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "mne": mne.__version__,
            "thread_env": {
                k: os.environ.get(k) for k in (
                    "OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","OPENBLAS_CORETYPE",
                    "MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS",
                    "OMP_DYNAMIC","PYTHONHASHSEED"
                )
            },
        },
        "directions": [],
    }

    for train_name, test_name in (("A", "B"), ("B", "A")):
        trainS, trainNu = blocks[train_name]
        testS, testNu = blocks[test_name]

        m3 = fit_unit(trainS, trainNu, "M3", args.network_dir, L20)
        m3_held = score_holdout(testS, testNu, "M3", m3["params"], args.network_dir, L20)

        amp_m2, noise_m2 = amp_centers(trainS, args.network_dir, L20, "M2")
        anchor_u = encode_embedded_m3_as_m2(m3["params"], amp_m2, noise_m2)
        anchor_pars = decode_unit("M2", anchor_u, amp_m2, noise_m2)
        anchor_score = -make_objective(
            trainS, trainNu, "M2", args.network_dir, L20, amp_m2, noise_m2
        )(anchor_u)
        embedding_score_error = abs(float(anchor_score) - float(m3["train_score"]))

        m2 = fit_unit(
            trainS, trainNu, "M2", args.network_dir, L20, embedded_anchor=anchor_u
        )
        m2_held = score_holdout(testS, testNu, "M2", m2["params"], args.network_dir, L20)

        nested_gap = float(m2["train_score"] - m3["train_score"])
        rec = {
            "train": train_name,
            "test": test_name,
            "embedding_score_error": embedding_score_error,
            "nested_training_gap_M2_minus_M3": nested_gap,
            "models": {
                "M3": {
                    **{k: v for k, v in m3.items() if k != "params"},
                    "heldout_score": float(m3_held),
                    "historical_target": HISTORICAL["M3"][train_name],
                    "historical_recovered": bool(
                        m3["train_score"] >= HISTORICAL["M3"][train_name] - TRAIN_TOL
                    ),
                },
                "M2": {
                    **{k: v for k, v in m2.items() if k != "params"},
                    "heldout_score": float(m2_held),
                    "historical_target": HISTORICAL["M2"][train_name],
                    "historical_recovered": bool(
                        m2["train_score"] >= HISTORICAL["M2"][train_name] - TRAIN_TOL
                    ),
                },
            },
        }
        rec["direction_pass"] = bool(
            embedding_score_error <= NEST_TOL
            and nested_gap >= -NEST_TOL
            and rec["models"]["M3"]["historical_recovered"]
            and rec["models"]["M2"]["historical_recovered"]
            and rec["models"]["M3"]["selected_success"]
            and rec["models"]["M2"]["selected_success"]
        )
        out["directions"].append(rec)

    for model in ("M2", "M3"):
        out[f"{model}_cv_elpd"] = float(np.mean([
            d["models"][model]["heldout_score"] for d in out["directions"]
        ]))
    out["delta_elpd_M3_minus_M2"] = out["M3_cv_elpd"] - out["M2_cv_elpd"]
    out["kernel_pass"] = bool(
        transfer_err <= TRANSFER_TOL and all(d["direction_pass"] for d in out["directions"])
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Development-only optimizer robustness probe.

This script exists because the original 32-Sobol/4-polish-start L-BFGS-B
procedure produced different optima for the same sub-001 data across two
GitHub runner-image releases despite pinned Python packages and one-thread
execution.

It must never be used on sub-044 or later.

Profile selection is based on training-objective convergence/recovery, not on
whether M3-M2 is favorable.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path

import mne
import numpy as np
import scipy

import fit_cv_subject as fit
from empirical_csd_lock import (
    FREQS_HZ,
    chronological_two_block_indices,
    multitaper_csd,
)

# Best training scores already observed across the two incompatible sub-001
# development executions. A robust optimizer should recover at least these
# basins (within numerical tolerance) rather than selecting one run because
# its held-out Delta favors either model.
REFERENCE_TRAIN_BEST = {
    "M2": {
        "A": 535.8336527151650,
        "B": 534.0933137283831,
    },
    "M3": {
        "A": 535.8430963300440,
        "B": 533.8829279760863,
    },
}


def env_fingerprint():
    keys = [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "OPENBLAS_CORETYPE",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OMP_DYNAMIC",
        "PYTHONHASHSEED",
        "ImageOS",
        "ImageVersion",
        "RUNNER_ARCH",
        "RUNNER_OS",
    ]
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "mne": mne.__version__,
        "env": {k: os.environ.get(k) for k in keys},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("epochs_fif", type=Path)
    ap.add_argument("--forward-dir", type=Path, required=True)
    ap.add_argument("--network-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sobol-m", type=int, default=8)
    ap.add_argument("--polish-starts", type=int, default=8)
    ap.add_argument("--train-tolerance", type=float, default=5e-4)
    args = ap.parse_args()

    # Hard information barrier.
    if args.epochs_fif.name.startswith("sub-"):
        try:
            n = int(args.epochs_fif.name.split("-")[1].split("_")[0])
        except Exception:
            n = None
        if n is not None and n >= 44:
            raise RuntimeError("HOLDOUT_BLOCKED: optimizer probe is development-only")

    fit.SOBOL_M = args.sobol_m
    fit.POLISH_STARTS = args.polish_starts

    U20, L20 = fit._load_L20(args.forward_dir)
    ep = mne.read_epochs(args.epochs_fif, preload=True, verbose="error")
    frozen_channels = [
        x.strip()
        for x in Path(fit.__file__).with_name("channels_64.txt").read_text().splitlines()
        if x.strip()
    ]
    if ep.ch_names != frozen_channels:
        raise RuntimeError("Clean-epoch channel order drift")
    x = ep.get_data(picks=frozen_channels)
    if x.shape[1:] != (64, 1000):
        raise RuntimeError(f"Expected [n,64,1000], got {x.shape}")

    A, B = chronological_two_block_indices(len(ep))
    blocks = {}
    for name, idx in [("A", A), ("B", B)]:
        f, S, nu = multitaper_csd(x[idx], sfreq=float(ep.info["sfreq"]), project=U20)
        if not np.array_equal(f, FREQS_HZ):
            raise RuntimeError("frequency lock drift")
        blocks[name] = (S, nu)

    result = {
        "development_only": True,
        "subject": "sub-001",
        "selection_basis": "training objective recovery/reproducibility only",
        "profile": {
            "sobol_m": args.sobol_m,
            "sobol_candidates": 2 ** args.sobol_m,
            "polish_starts": args.polish_starts,
            "method": "L-BFGS-B",
            "maxiter": fit.POLISH_MAXITER,
            "rel_floor": fit.REL_FLOOR,
            "seed": fit.SEED,
        },
        "environment": env_fingerprint(),
        "models": {},
    }

    all_recovered = True
    for model in ("M2", "M3"):
        dirs = []
        for train_name, test_name in [("A", "B"), ("B", "A")]:
            trainS, trainNu = blocks[train_name]
            testS, testNu = blocks[test_name]
            pars, serial, train_score, opt = fit.fit_block(
                trainS, trainNu, model, args.network_dir, L20
            )
            held = fit.score_holdout(testS, testNu, model, pars, args.network_dir, L20)
            target = REFERENCE_TRAIN_BEST[model][train_name]
            recovered = bool(train_score >= target - args.train_tolerance)
            all_recovered = all_recovered and recovered
            dirs.append(
                {
                    "train": train_name,
                    "test": test_name,
                    "train_score_normalized": train_score,
                    "heldout_score_normalized": held,
                    "previous_best_train_score": target,
                    "recovered_previous_best_within_tolerance": recovered,
                    "parameters": serial,
                    "optimizer": opt,
                }
            )
        result["models"][model] = {
            "directions": dirs,
            "cv_elpd": float(np.mean([d["heldout_score_normalized"] for d in dirs])),
        }

    result["delta_elpd_M3_minus_M2"] = (
        result["models"]["M3"]["cv_elpd"] - result["models"]["M2"]["cv_elpd"]
    )
    result["all_previous_best_training_basins_recovered"] = all_recovered

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

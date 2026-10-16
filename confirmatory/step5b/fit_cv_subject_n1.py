#!/usr/bin/env python3
"""Production Step 5B subject fit for the N1 normalized nested-comparator method.

This script is inert unless NUMERICAL_PROFILE.json is explicitly FROZEN as N1.
It imports the exact optimizer implementation used by the N1 development probe.
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

from empirical_csd_lock import (
    FREQS_HZ,
    chronological_two_block_indices,
    multitaper_csd,
)
from nested_optimizer_probe import (
    FTOL,
    GTOL,
    MAXITER,
    MAXLS,
    NEST_TOL,
    POLISH_STARTS,
    REL_FLOOR,
    SEED,
    SOBOL_CANDIDATES,
    amp_centers,
    encode_embedded_m3_as_m2,
    fit_unit,
    load_l20,
    make_objective,
    score_holdout,
    transfer_identity_error,
)

PROFILE_PATH = Path(__file__).with_name("NUMERICAL_PROFILE.json")


def require_frozen_n1():
    p=json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    if p.get("status")!="FROZEN":
        raise RuntimeError("N1 production fit requires NUMERICAL_PROFILE status FROZEN")
    if p.get("profile_name")!="N1":
        raise RuntimeError(f"N1 production fit refused non-N1 profile: {p.get('profile_name')}")
    expected={
        "seed":SEED,
        "sobol_candidates":SOBOL_CANDIDATES,
        "polish_starts":POLISH_STARTS,
        "polish_maxiter":MAXITER,
        "method":"N1-normalized-nested-comparator",
        "ftol":FTOL,
        "gtol":GTOL,
        "maxls":MAXLS,
        "rel_floor":REL_FLOOR,
        "openblas_coretype":"Haswell",
    }
    for k,v in expected.items():
        if p.get(k)!=v:
            raise RuntimeError(f"Frozen N1 profile drift for {k}: {p.get(k)} != {v}")
    if not p.get("robustness_run_id") or not p.get("final_freeze_commit"):
        raise RuntimeError("Frozen N1 profile lacks provenance")
    return p


def optimizer_summary(fit):
    return {
        "success":bool(fit["selected_success"]),
        "message":str(fit["selected_message"]),
        "nit":int(fit["selected_nit"]),
        "nfev":int(fit["selected_nfev"]),
        "selected_source":str(fit["selected_source"]),
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("epochs_fif",type=Path)
    ap.add_argument("--forward-dir",type=Path,required=True)
    ap.add_argument("--network-dir",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args()

    profile=require_frozen_n1()
    if os.environ.get("OPENBLAS_CORETYPE")!="Haswell":
        raise RuntimeError("N1 production fit requires OPENBLAS_CORETYPE=Haswell")

    transfer_err=transfer_identity_error()
    if transfer_err>1e-12:
        raise RuntimeError(f"N1 analytic embedding identity failed: {transfer_err}")

    U20,L20=load_l20(a.forward_dir)
    ep=mne.read_epochs(a.epochs_fif,preload=True,verbose="error")
    frozen_channels=[
        x.strip() for x in Path(__file__).with_name("channels_64.txt").read_text().splitlines() if x.strip()
    ]
    if ep.ch_names!=frozen_channels:
        raise RuntimeError(f"Clean-epoch channel order drift: got {ep.ch_names}")
    x=ep.get_data(picks=frozen_channels)
    if x.shape[1:]!=(64,1000):
        raise RuntimeError(f"Expected clean epochs [n,64,1000], got {x.shape}")

    A,B=chronological_two_block_indices(len(ep))
    blocks={}
    for name,idx in (("A",A),("B",B)):
        f,S,nu=multitaper_csd(x[idx],sfreq=float(ep.info["sfreq"]),project=U20)
        if not np.array_equal(f,FREQS_HZ):
            raise RuntimeError("frequency lock drift")
        blocks[name]=(S,nu)

    result={
        "seed":SEED,
        "n_epochs":int(len(ep)),
        "split_A":int(len(A)),
        "split_B":int(len(B)),
        "optimizer":{
            "sobol_candidates":SOBOL_CANDIDATES,
            "polish_starts":POLISH_STARTS,
            "polish_maxiter":MAXITER,
            "method":"N1-normalized-nested-comparator",
            "ftol":FTOL,
            "gtol":GTOL,
            "maxls":MAXLS,
            "rel_floor":REL_FLOOR,
        },
        "numerical_profile":profile,
        "transfer_identity_max_abs_error":float(transfer_err),
        "environment":{
            "platform":platform.platform(),
            "python":platform.python_version(),
            "numpy":np.__version__,
            "scipy":scipy.__version__,
            "mne":mne.__version__,
            "thread_env":{k:os.environ.get(k) for k in (
                "OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","OPENBLAS_CORETYPE",
                "MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS",
                "OMP_DYNAMIC","PYTHONHASHSEED"
            )},
            "runner_image":{
                "ImageOS":os.environ.get("ImageOS"),
                "ImageVersion":os.environ.get("ImageVersion"),
                "RUNNER_OS":os.environ.get("RUNNER_OS"),
                "RUNNER_ARCH":os.environ.get("RUNNER_ARCH"),
            },
        },
        "models":{"M2":{"directions":[]},"M3":{"directions":[]}},
        "nestedness":[],
    }

    for train_name,test_name in (("A","B"),("B","A")):
        trainS,trainNu=blocks[train_name]
        testS,testNu=blocks[test_name]

        m3=fit_unit(trainS,trainNu,"M3",a.network_dir,L20)
        m3_held=score_holdout(testS,testNu,"M3",m3["params"],a.network_dir,L20)

        amp_m2,noise_m2=amp_centers(trainS,a.network_dir,L20,"M2")
        anchor_u=encode_embedded_m3_as_m2(m3["params"],amp_m2,noise_m2)
        anchor_score=-make_objective(
            trainS,trainNu,"M2",a.network_dir,L20,amp_m2,noise_m2
        )(anchor_u)
        embedding_error=abs(float(anchor_score)-float(m3["train_score"]))
        if embedding_error>NEST_TOL:
            raise RuntimeError(
                f"{train_name}: exact M3-in-M2 score mismatch {embedding_error}"
            )

        m2=fit_unit(
            trainS,trainNu,"M2",a.network_dir,L20,embedded_anchor=anchor_u
        )
        m2_held=score_holdout(testS,testNu,"M2",m2["params"],a.network_dir,L20)
        nested_gap=float(m2["train_score"]-m3["train_score"])
        if nested_gap < -NEST_TOL:
            raise RuntimeError(
                f"{train_name}: M2 underoptimized relative to contained M3 point: {nested_gap}"
            )

        result["models"]["M3"]["directions"].append({
            "train":train_name,
            "test":test_name,
            "parameters":m3["serial"],
            "train_score_normalized":float(m3["train_score"]),
            "heldout_score_normalized":float(m3_held),
            "optimizer":optimizer_summary(m3),
        })
        result["models"]["M2"]["directions"].append({
            "train":train_name,
            "test":test_name,
            "parameters":m2["serial"],
            "train_score_normalized":float(m2["train_score"]),
            "heldout_score_normalized":float(m2_held),
            "optimizer":optimizer_summary(m2),
            "embedded_m3_raw_score":float(m2["anchor_raw_score"]),
        })
        result["nestedness"].append({
            "train":train_name,
            "embedding_score_error":float(embedding_error),
            "M2_minus_M3_training_gap":float(nested_gap),
        })

    for model in ("M2","M3"):
        result["models"][model]["cv_elpd"]=float(np.mean([
            d["heldout_score_normalized"] for d in result["models"][model]["directions"]
        ]))
    result["delta_elpd_M3_minus_M2"]=(
        result["models"]["M3"]["cv_elpd"]-result["models"]["M2"]["cv_elpd"]
    )

    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()

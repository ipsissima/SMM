#!/usr/bin/env python3
"""Aggregate the three-kernel N1 normalized nested-comparator probe."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

TRAIN_RANGE_MAX = 1e-3
CV_RANGE_MAX = 5e-3
NEST_TOL = 1e-8
TRANSFER_TOL = 1e-12
EXPECTED_KERNELS = {"Haswell", "Sandybridge", "Zen"}


def validate_serial(model: str, p: dict):
    for key, lo, hi in (
        ("G_N", 50.0, 185.0),
        ("velocity_m_s", 3.0, 12.0),
        ("pE", 1e-4, 1.0-1e-4),
    ):
        v=float(p[key])
        if not (math.isfinite(v) and lo-1e-12 <= v <= hi+1e-12):
            raise RuntimeError(f"{model}: decoded bound failure {key}={v}")
    for key in ("source_scale","sensor_floor"):
        v=float(p[key])
        if not (math.isfinite(v) and v>0.0):
            raise RuntimeError(f"{model}: invalid {key}={v}")
    if model=="M2":
        m=p["m2"]
        t1=float(m["tau1_s"]); t2=float(m["tau2_s"])
        if not (0.03-1e-12 <= t1 <= t2 <= 30.0+1e-12):
            raise RuntimeError(f"M2 tau bounds/order failure {t1},{t2}")
        for key in ("gE1","gE2","gI1","gI2"):
            v=float(m[key])
            if not (math.isfinite(v) and -0.5-1e-12 <= v <= 0.5+1e-12):
                raise RuntimeError(f"M2 gain bound failure {key}={v}")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("input_dir",type=Path)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()

    files=sorted(args.input_dir.rglob("sub-001-*.json"))
    if len(files)!=3:
        raise RuntimeError(f"Expected exactly three N1 kernel JSONs, found {files}")

    rows=[]
    by_kernel={}
    for p in files:
        d=json.loads(p.read_text())
        if d.get("protocol")!="N1-normalized-nested-comparator":
            raise RuntimeError(f"Wrong protocol in {p}")
        env=d["environment"]["thread_env"]
        kernel=env.get("OPENBLAS_CORETYPE")
        if kernel in by_kernel:
            raise RuntimeError(f"Duplicate kernel {kernel}")
        by_kernel[kernel]=d
        if int(d["profile"]["sobol_candidates"])!=512 or int(d["profile"]["polish_starts"])!=16:
            raise RuntimeError("N1 search-width drift")
        if int(d["profile"]["maxiter"])!=300:
            raise RuntimeError("N1 maxiter drift")
        if float(d["profile"]["ftol"])!=1e-11 or float(d["profile"]["gtol"])!=1e-7:
            raise RuntimeError("N1 stopping-rule drift")
        if int(d["profile"]["maxls"])!=50 or float(d["profile"]["rel_floor"])!=1e-6:
            raise RuntimeError("N1 numerical-rule drift")
        if float(d["transfer_identity_max_abs_error"])>TRANSFER_TOL:
            raise RuntimeError(f"{kernel}: transfer identity failed")
        for direction in d["directions"]:
            if float(direction["embedding_score_error"])>NEST_TOL:
                raise RuntimeError(f"{kernel} {direction['train']}: embedding-score equality failed")
            if float(direction["nested_training_gap_M2_minus_M3"]) < -NEST_TOL:
                raise RuntimeError(f"{kernel} {direction['train']}: nestedness inequality failed")
            if direction.get("direction_pass") is not True:
                raise RuntimeError(f"{kernel} {direction['train']}: direction did not pass")
            for model in ("M2","M3"):
                m=direction["models"][model]
                if m.get("historical_recovered") is not True:
                    raise RuntimeError(f"{kernel} {model} {direction['train']}: historical basin not recovered")
                if m.get("selected_success") is not True:
                    raise RuntimeError(f"{kernel} {model} {direction['train']}: selected optimizer result invalid")
                if not math.isfinite(float(m["train_score"])) or not math.isfinite(float(m["heldout_score"])):
                    raise RuntimeError("Nonfinite score")
                validate_serial(model,m["serial"])
        if d.get("kernel_pass") is not True:
            raise RuntimeError(f"{kernel}: kernel_pass false")
        by_kernel[kernel]=d

    if set(by_kernel)!=EXPECTED_KERNELS:
        raise RuntimeError(f"Kernel set mismatch: {set(by_kernel)}")

    train_ranges={}
    for model in ("M2","M3"):
        for train in ("A","B"):
            vals=[]
            for kernel,d in by_kernel.items():
                direction=next(x for x in d["directions"] if x["train"]==train)
                vals.append(float(direction["models"][model]["train_score"]))
            train_ranges[f"{model}_{train}"]=max(vals)-min(vals)

    cv_ranges={}
    for model in ("M2","M3"):
        vals=[float(d[f"{model}_cv_elpd"]) for d in by_kernel.values()]
        cv_ranges[model]=max(vals)-min(vals)

    transfer_max=max(float(d["transfer_identity_max_abs_error"]) for d in by_kernel.values())
    embedding_max=max(
        float(direction["embedding_score_error"])
        for d in by_kernel.values()
        for direction in d["directions"]
    )
    nested_gap_min=min(
        float(direction["nested_training_gap_M2_minus_M3"])
        for d in by_kernel.values()
        for direction in d["directions"]
    )

    result={
        "protocol":"N1-normalized-nested-comparator",
        "kernels":sorted(by_kernel),
        "profile":{
            "sobol_candidates":512,
            "polish_starts":16,
            "method":"L-BFGS-B-unit-cube-plus-exact-nested-anchor",
            "maxiter":300,
            "ftol":1e-11,
            "gtol":1e-7,
            "maxls":50,
            "rel_floor":1e-6,
            "seed":97,
        },
        "criteria":{
            "transfer_identity_max_abs_error":transfer_max,
            "transfer_identity_pass":transfer_max<=TRANSFER_TOL,
            "embedding_score_max_abs_error":embedding_max,
            "embedding_score_pass":embedding_max<=NEST_TOL,
            "minimum_nested_training_gap_M2_minus_M3":nested_gap_min,
            "nestedness_pass":nested_gap_min>=-NEST_TOL,
            "training_ranges":train_ranges,
            "training_range_pass":all(v<=TRAIN_RANGE_MAX for v in train_ranges.values()),
            "cv_ranges":cv_ranges,
            "cv_range_pass":all(v<=CV_RANGE_MAX for v in cv_ranges.values()),
            "all_kernel_internal_pass":all(d["kernel_pass"] for d in by_kernel.values()),
            "all_outputs_finite_and_in_bounds":True,
        },
        "kernel_results":{
            k:{
                "M2_cv_elpd":float(d["M2_cv_elpd"]),
                "M3_cv_elpd":float(d["M3_cv_elpd"]),
                "delta_elpd_M3_minus_M2":float(d["delta_elpd_M3_minus_M2"]),
                "directions":[{
                    "train":x["train"],
                    "M2_train":float(x["models"]["M2"]["train_score"]),
                    "M3_train":float(x["models"]["M3"]["train_score"]),
                    "nested_gap":float(x["nested_training_gap_M2_minus_M3"]),
                    "embedding_score_error":float(x["embedding_score_error"]),
                    "M2_selected_source":x["models"]["M2"]["selected_source"],
                    "M3_selected_source":x["models"]["M3"]["selected_source"],
                } for x in d["directions"]],
            } for k,d in by_kernel.items()
        },
        "selection_uses_delta_sign":False,
    }

    c=result["criteria"]
    result["probe_pass"]=bool(
        c["transfer_identity_pass"]
        and c["embedding_score_pass"]
        and c["nestedness_pass"]
        and c["training_range_pass"]
        and c["cv_range_pass"]
        and c["all_kernel_internal_pass"]
        and c["all_outputs_finite_and_in_bounds"]
    )

    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+"\n")

    md=[
        "# N1 normalized nested-comparator robustness aggregate",
        "",
        f"**Verdict: {'PASS' if result['probe_pass'] else 'FAIL'}**",
        "",
        f"- Kernels: {', '.join(result['kernels'])}",
        f"- Transfer identity max error: {transfer_max:.3e}",
        f"- Embedded-score max error: {embedding_max:.3e}",
        f"- Minimum M2-M3 training nestedness gap: {nested_gap_min:.9g}",
        f"- Training-score ranges: {train_ranges}",
        f"- CV ELPD ranges: {cv_ranges}",
        "",
        "DeltaELPD values are logged for audit only and are not used in the verdict.",
        "",
    ]
    args.out.with_suffix(".md").write_text("\n".join(md),encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()

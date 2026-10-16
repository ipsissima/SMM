#!/usr/bin/env python3
"""Prepare, but do not commit, the final frozen Step 5B numerical profile.

After P1 and P2 failed, the only currently admissible profile is N1:
normalized unit-cube coordinates plus the exact M3-in-M2 nested anchor.

The input aggregate must PASS the predeclared N1 criteria. DeltaELPD sign or
magnitude is never a profile-selection criterion.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("probe_aggregate",type=Path)
    ap.add_argument("--robustness-run-id",required=True)
    ap.add_argument("--freeze-commit",required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()

    agg=json.loads(args.probe_aggregate.read_text(encoding="utf-8"))
    if agg.get("protocol")!="N1-normalized-nested-comparator":
        raise RuntimeError("Only the predeclared N1 protocol may be frozen now")
    if agg.get("probe_pass") is not True:
        raise RuntimeError("Cannot freeze N1 from a failed robustness probe")
    if agg.get("selection_uses_delta_sign") is not False:
        raise RuntimeError("N1 selection must not use DeltaELPD favorability")
    if set(agg.get("kernels",[]))!={"Haswell","Sandybridge","Zen"}:
        raise RuntimeError("N1 cross-kernel set mismatch")

    p=agg["profile"]
    expected={
        "sobol_candidates":512,
        "polish_starts":16,
        "method":"L-BFGS-B-unit-cube-plus-exact-nested-anchor",
        "maxiter":300,
        "ftol":1e-11,
        "gtol":1e-7,
        "maxls":50,
        "rel_floor":1e-6,
        "seed":97,
    }
    for k,v in expected.items():
        if p.get(k)!=v:
            raise RuntimeError(f"N1 aggregate profile drift: {k}={p.get(k)} expected {v}")

    c=agg["criteria"]
    required_true=(
        "transfer_identity_pass",
        "embedding_score_pass",
        "nestedness_pass",
        "training_range_pass",
        "cv_range_pass",
        "all_kernel_internal_pass",
        "all_outputs_finite_and_in_bounds",
    )
    failed=[k for k in required_true if c.get(k) is not True]
    if failed:
        raise RuntimeError(f"N1 aggregate inconsistent with PASS: {failed}")

    current_path=Path(__file__).with_name("NUMERICAL_PROFILE.json")
    current=json.loads(current_path.read_text(encoding="utf-8"))
    if current.get("status")!="PENDING":
        raise RuntimeError("Refusing to prepare a second numerical freeze")

    out={
        "status":"FROZEN",
        "profile_name":"N1",
        "seed":97,
        "sobol_candidates":512,
        "polish_starts":16,
        "polish_maxiter":300,
        "method":"N1-normalized-nested-comparator",
        "ftol":1e-11,
        "gtol":1e-7,
        "maxls":50,
        "rel_floor":1e-6,
        "parameterization":"unit-cube; M2 ordered log-times; unchanged physical bounds",
        "exact_nested_anchor":True,
        "nested_training_tolerance":1e-8,
        "transfer_identity_tolerance":1e-12,
        "thread_policy":"single-thread BLAS/OpenMP",
        "robustness_requirement":"Passed N1_NESTED_OPTIMIZER_PROTOCOL.md",
        "robustness_run_id":str(args.robustness_run_id),
        "final_freeze_commit":str(args.freeze_commit),
        "openblas_coretype":"Haswell",
        "production_kernel_policy":(
            "Final development and holdout fits force OPENBLAS_CORETYPE=Haswell; "
            "N1 robustness must pass Haswell, Sandybridge and Zen."
        ),
        "robustness_training_ranges":c["training_ranges"],
        "robustness_cv_ranges":c["cv_ranges"],
        "robustness_transfer_identity_max_abs_error":c["transfer_identity_max_abs_error"],
        "robustness_embedding_score_max_abs_error":c["embedding_score_max_abs_error"],
        "robustness_min_nested_training_gap":c["minimum_nested_training_gap_M2_minus_M3"],
    }

    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(out,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(out,indent=2))


if __name__=="__main__":
    main()

#!/usr/bin/env python3
"""Prepare (but do not commit) the final frozen Step 5B numerical profile.

The input robustness aggregate must already PASS the predeclared criteria.
This script never inspects development DeltaELPD favorability when choosing
between P1 and P2; it simply translates the passing profile into the canonical
NUMERICAL_PROFILE schema.

Typical use after completing FINAL_NUMERICAL_FREEZE_TEMPLATE.md:
    python prepare_frozen_numerical_profile.py \
      optimizer_robustness.json \
      --robustness-run-id <run> \
      --freeze-commit <completed-freeze-doc-commit> \
      --out /tmp/NUMERICAL_PROFILE.frozen.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ALLOWED = {
    (256, 8): "P1",
    (512, 16): "P2",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("probe_aggregate", type=Path)
    ap.add_argument("--robustness-run-id", required=True)
    ap.add_argument("--freeze-commit", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    agg = json.loads(args.probe_aggregate.read_text(encoding="utf-8"))
    if agg.get("probe_pass") is not True:
        raise RuntimeError("Cannot freeze a numerical profile from a failed robustness probe")
    if agg.get("selection_uses_delta_sign") is not False:
        raise RuntimeError("Optimizer selection must not use DeltaELPD sign/favorability")

    profile = agg["profile"]
    pair = (int(profile["sobol_candidates"]), int(profile["polish_starts"]))
    if pair not in ALLOWED:
        raise RuntimeError(f"Robustness profile is not predeclared P1/P2: {pair}")

    criteria = agg["criteria"]
    required_true = (
        "all_optimizer_calls_success",
        "all_historical_best_training_basins_recovered",
        "all_scores_finite",
        "decoded_parameter_bounds_pass",
        "training_range_pass",
        "cv_range_pass",
    )
    failed = [k for k in required_true if criteria.get(k) is not True]
    if failed:
        raise RuntimeError(f"Robustness aggregate is internally inconsistent with PASS: {failed}")

    kernels = set(agg.get("kernels", []))
    if kernels != {"Haswell", "Sandybridge", "Zen"}:
        raise RuntimeError(f"Frozen robustness kernel set mismatch: {kernels}")

    current_path = Path(__file__).with_name("NUMERICAL_PROFILE.json")
    current = json.loads(current_path.read_text(encoding="utf-8"))
    if current.get("status") != "PENDING":
        raise RuntimeError("Refusing to prepare a second final numerical freeze")

    out = {
        "status": "FROZEN",
        "profile_name": ALLOWED[pair],
        "seed": 97,
        "sobol_candidates": pair[0],
        "polish_starts": pair[1],
        "polish_maxiter": 120,
        "method": "L-BFGS-B",
        "ftol": 1e-9,
        "gtol": 1e-6,
        "maxls": 30,
        "rel_floor": 1e-6,
        "thread_policy": "single-thread BLAS/OpenMP",
        "robustness_requirement": "Passed OPTIMIZER_ROBUSTNESS_DECISION_RULE.md",
        "robustness_run_id": str(args.robustness_run_id),
        "final_freeze_commit": str(args.freeze_commit),
        "openblas_coretype": "Haswell",
        "production_kernel_policy": (
            "Final development and holdout fits force OPENBLAS_CORETYPE=Haswell; "
            "robustness probe passed Haswell, Sandybridge and Zen."
        ),
        "robustness_training_ranges": criteria["training_ranges"],
        "robustness_cv_ranges": criteria["cv_ranges"],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

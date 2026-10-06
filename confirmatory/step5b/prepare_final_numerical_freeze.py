#!/usr/bin/env python3
"""Generate the final numerical-freeze document from a passing P1/P2 aggregate.

The output is evidence/provenance only. It never selects a profile using
DeltaELPD sign or magnitude.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ALLOWED = {(256,8):"P1",(512,16):"P2"}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("probe_aggregate",type=Path)
    ap.add_argument("--robustness-run-id",required=True)
    ap.add_argument("--aggregate-artifact-id",required=True)
    ap.add_argument("--aggregate-artifact-digest",required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()

    d=json.loads(args.probe_aggregate.read_text())
    if d.get("probe_pass") is not True:
        raise RuntimeError("Cannot generate final freeze from a failed robustness aggregate")
    if d.get("selection_uses_delta_sign") is not False:
        raise RuntimeError("Robustness selection must not use DeltaELPD sign")

    p=d["profile"]
    pair=(int(p["sobol_candidates"]),int(p["polish_starts"]))
    if pair not in ALLOWED:
        raise RuntimeError(f"Unrecognized predeclared profile: {pair}")
    name=ALLOWED[pair]
    crit=d["criteria"]

    required_true=[
        "all_optimizer_calls_success",
        "all_historical_best_training_basins_recovered",
        "all_scores_finite",
        "decoded_parameter_bounds_pass",
        "training_range_pass",
        "cv_range_pass",
    ]
    failed=[k for k in required_true if crit.get(k) is not True]
    if failed:
        raise RuntimeError(f"Aggregate claims PASS but criteria fail: {failed}")
    if set(d.get("kernels",[])) != {"Haswell","Sandybridge","Zen"}:
        raise RuntimeError("Cross-kernel set mismatch")

    digest=args.aggregate_artifact_digest
    if not (digest.startswith("sha256:") and len(digest)==71):
        raise RuntimeError("Artifact digest must be sha256:<64hex>")
    int(digest.split(":",1)[1],16)

    lines=[
        "# Step 5B final numerical freeze",
        "",
        "**Status: FROZEN FOR FINAL DEVELOPMENT AND CONFIRMATORY HOLDOUT**",
        "",
        "The holdout remained closed throughout numerical profile selection.",
        "",
        "## Selected predeclared profile",
        "",
        f"- Profile: {name}",
        f"- Sobol candidates: {pair[0]}",
        f"- L-BFGS-B polish starts: {pair[1]}",
        "- maxiter: 120",
        "- ftol: 1e-9",
        "- gtol: 1e-6",
        "- maxls: 30",
        "- CSD/model relative floor: 1e-6",
        "- seed: 97",
        "- production OpenBLAS kernel: Haswell",
        "- thread policy: one BLAS/OpenMP thread",
        "",
        "## Robustness provenance",
        "",
        f"- Robustness run ID: {args.robustness_run_id}",
        f"- Aggregate artifact ID: {args.aggregate_artifact_id}",
        f"- Aggregate artifact digest: {digest}",
        f"- Kernels tested: {', '.join(d['kernels'])}",
        "",
        "## Frozen pass criteria",
        "",
        f"- All optimizer calls successful: {crit['all_optimizer_calls_success']}",
        f"- Historical best training basins recovered: {crit['all_historical_best_training_basins_recovered']}",
        f"- All scores finite: {crit['all_scores_finite']}",
        f"- Decoded parameter bounds pass: {crit['decoded_parameter_bounds_pass']}",
        f"- Training-score cross-kernel range pass: {crit['training_range_pass']}",
        f"- Per-model CV cross-kernel range pass: {crit['cv_range_pass']}",
        f"- Training-score ranges: {json.dumps(crit['training_ranges'],sort_keys=True)}",
        f"- CV-score ranges: {json.dumps(crit['cv_ranges'],sort_keys=True)}",
        "",
        "## Selection firewall",
        "",
        "The numerical profile was selected exclusively by optimizer success, recovery of already observed best training basins, finite/bound checks, and cross-kernel numerical reproducibility.",
        "",
        "**The sign and magnitude of DeltaELPD(M3-M2) were not profile-selection criteria.**",
        "",
    ]
    if name=="P2":
        lines += [
            "P2 was reached only because the predeclared P1 profile failed its historical M2 training-basin recovery criterion. No intermediate profile was introduced.",
            "",
        ]
    lines += [
        "## Scientific invariants",
        "",
        "This freeze changes no biological mechanism, K-to-QIF mapping, astroglial topology, neuronal model, structural network, forward model, primary condition, frequency range, QC threshold, development/holdout boundary, M2 comparator strength, M3-vs-M2 primary contrast, likelihood, or confirmatory success criterion.",
        "",
        "The next permitted step is a fresh complete 34-subject development rerun under this profile. The confirmatory holdout may open only after that rerun is aggregated, permanently recorded and signed off.",
        "",
    ]

    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text("\n".join(lines),encoding="utf-8")
    print("FINAL_NUMERICAL_FREEZE_DOCUMENT_PASS")
    print(args.out.read_text())


if __name__=="__main__":
    main()

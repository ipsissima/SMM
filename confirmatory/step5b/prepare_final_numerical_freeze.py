#!/usr/bin/env python3
"""Generate the final Step 5B numerical-freeze document from a passing N1 aggregate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("probe_aggregate",type=Path)
    ap.add_argument("--robustness-run-id",required=True)
    ap.add_argument("--aggregate-artifact-id",required=True)
    ap.add_argument("--aggregate-artifact-digest",required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()

    d=json.loads(args.probe_aggregate.read_text())
    if d.get("protocol")!="N1-normalized-nested-comparator":
        raise RuntimeError("Final freeze generator now accepts N1 only")
    if d.get("probe_pass") is not True:
        raise RuntimeError("Cannot freeze a failed N1 aggregate")
    if d.get("selection_uses_delta_sign") is not False:
        raise RuntimeError("N1 selection must not use DeltaELPD sign")
    if set(d.get("kernels",[]))!={"Haswell","Sandybridge","Zen"}:
        raise RuntimeError("N1 kernel set mismatch")

    p=d["profile"]
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
            raise RuntimeError(f"N1 profile drift: {k}={p.get(k)} expected {v}")

    c=d["criteria"]
    required_true=[
        "transfer_identity_pass",
        "embedding_score_pass",
        "nestedness_pass",
        "training_range_pass",
        "cv_range_pass",
        "all_kernel_internal_pass",
        "all_outputs_finite_and_in_bounds",
    ]
    failed=[k for k in required_true if c.get(k) is not True]
    if failed:
        raise RuntimeError(f"N1 aggregate claims PASS but criteria fail: {failed}")

    digest=args.aggregate_artifact_digest
    if not (digest.startswith("sha256:") and len(digest)==71):
        raise RuntimeError("Artifact digest must be sha256:<64hex>")
    int(digest.split(":",1)[1],16)

    lines=[
        "# Step 5B final numerical freeze",
        "",
        "**Status: FROZEN FOR FINAL DEVELOPMENT AND CONFIRMATORY HOLDOUT**",
        "",
        "The confirmatory holdout remained closed throughout P1, P2 and N1 numerical-method development.",
        "",
        "## Selected method",
        "",
        "- Profile: N1",
        "- Search coordinates: normalized unit cube",
        "- M2 time constants: ordered logarithmic parameterization over the unchanged 0.03–30 s physical domain",
        "- Exact comparator safeguard: fitted M3 solution embedded exactly as a feasible M2 candidate",
        "- Sobol candidates: 512",
        "- L-BFGS-B Sobol polish starts: 16",
        "- Exact nested anchor: one additional deterministic M2 start and retained raw feasible candidate",
        "- maxiter: 300",
        "- ftol: 1e-11",
        "- gtol: 1e-7",
        "- maxls: 50",
        "- CSD/model relative floor: 1e-6",
        "- seed: 97",
        "- production OpenBLAS kernel: Haswell",
        "- thread policy: one BLAS/OpenMP thread",
        "",
        "## Why N1 replaced P1/P2",
        "",
        "P1 and P2 used the old physical-coordinate parameterization. Both failed the predeclared training-basin recovery criterion for M2. P2 further exposed the structural anomaly that M2 could return a training optimum below M3 even though the executable M2 family contains the linearized M3 transfer exactly.",
        "",
        "N1 corrects only the numerical method: it removes the nondifferentiable tau sorting fold, normalizes coordinate scales, and enforces the exact M3-in-M2 feasible anchor. It does not change either model class or any physical parameter bound.",
        "",
        "## Robustness provenance",
        "",
        f"- Robustness run ID: {args.robustness_run_id}",
        f"- Aggregate artifact ID: {args.aggregate_artifact_id}",
        f"- Aggregate artifact digest: {digest}",
        f"- Kernels tested: {', '.join(d['kernels'])}",
        "",
        "## Frozen N1 pass criteria",
        "",
        f"- Analytic transfer-identity pass: {c['transfer_identity_pass']}",
        f"- Max transfer identity error: {c['transfer_identity_max_abs_error']:.3e}",
        f"- Embedded M3-as-M2 score equality pass: {c['embedding_score_pass']}",
        f"- Max embedded score error: {c['embedding_score_max_abs_error']:.3e}",
        f"- M2 >= contained M3 training invariant pass: {c['nestedness_pass']}",
        f"- Minimum M2-M3 training gap: {c['minimum_nested_training_gap_M2_minus_M3']:.12g}",
        f"- Training-score cross-kernel range pass: {c['training_range_pass']}",
        f"- Per-model CV cross-kernel range pass: {c['cv_range_pass']}",
        f"- Training-score ranges: {json.dumps(c['training_ranges'],sort_keys=True)}",
        f"- CV-score ranges: {json.dumps(c['cv_ranges'],sort_keys=True)}",
        f"- Finite/bounds audit: {c['all_outputs_finite_and_in_bounds']}",
        "",
        "## Selection firewall",
        "",
        "**The sign and magnitude of DeltaELPD(M3-M2) were not N1 selection criteria.**",
        "",
        "## Scientific invariants",
        "",
        "N1 changes no biological mechanism, K-to-QIF mapping, astroglial topology or constants, neuronal equations, M2 flexibility or physical bounds, M3 constraints, structural network, EEG forward model, primary condition, preprocessing/QC threshold, 1–40-Hz endpoint, likelihood, development/holdout split, primary M3-vs-M2 contrast, or confirmatory success criterion.",
        "",
        "The next permitted operation after this freeze is a complete fresh 34-subject development rerun using the exact N1 production fitter. Holdout opening remains contingent on development aggregation and sign-off.",
        "",
    ]

    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text("\n".join(lines),encoding="utf-8")
    print("FINAL_N1_NUMERICAL_FREEZE_DOCUMENT_PASS")
    print(args.out.read_text())


if __name__=="__main__":
    main()

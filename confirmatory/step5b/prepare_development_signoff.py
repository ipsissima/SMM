#!/usr/bin/env python3
"""Prepare the final Step 5B development sign-off from permanent aggregate files.

This script performs no confirmatory inference and never reads holdout EEG.
It emits PASS only if the complete 34-subject development result satisfies the
frozen numerical/profile/accounting invariants.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

EXPECTED = [
    "sub-001","sub-002","sub-003","sub-004","sub-005","sub-006","sub-007",
    "sub-010","sub-011","sub-014","sub-015","sub-016","sub-017","sub-018",
    "sub-019","sub-020","sub-021","sub-022","sub-023","sub-024","sub-025",
    "sub-028","sub-029","sub-030","sub-031","sub-032","sub-033","sub-034",
    "sub-035","sub-036","sub-038","sub-039","sub-040","sub-042",
]


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("aggregate_dir",type=Path)
    ap.add_argument("--development-run-id",required=True)
    ap.add_argument("--development-commit-sha",required=True)
    ap.add_argument("--aggregate-artifact-id",required=True)
    ap.add_argument("--aggregate-artifact-digest",required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()

    here=Path(__file__).parent
    profile=json.loads((here/"NUMERICAL_PROFILE.json").read_text())
    if profile.get("status")!="FROZEN":
        raise RuntimeError("Development sign-off requires final numerical profile FROZEN")
    if not profile.get("final_freeze_commit") or not profile.get("robustness_run_id"):
        raise RuntimeError("Frozen numerical profile lacks final provenance")
    if profile.get("profile_name")!="N1" or profile.get("exact_nested_anchor") is not True:
        raise RuntimeError("Development sign-off now requires frozen N1 nested-comparator profile")

    summary_path=args.aggregate_dir/"development_summary.json"
    subjects_path=args.aggregate_dir/"development_subject_results.csv"
    opt_path=args.aggregate_dir/"development_optimizer_diagnostics.csv"
    md_path=args.aggregate_dir/"development_summary.md"
    for p in (summary_path,subjects_path,opt_path,md_path):
        if not p.is_file():
            raise RuntimeError(f"Missing permanent development aggregate file: {p}")

    summary=json.loads(summary_path.read_text())
    if summary.get("phase")!="development":
        raise RuntimeError("Wrong aggregate phase")
    if summary.get("confirmatory_inference_performed") is not False:
        raise RuntimeError("Confirmatory inference was performed on development")
    if summary.get("holdout_subjects_seen") is not False:
        raise RuntimeError("Development aggregate reports holdout access")
    if int(summary.get("n_subjects",0))!=34 or int(summary.get("expected_n_subjects",0))!=34:
        raise RuntimeError("Development n mismatch")
    if summary.get("all_optimizers_success") is not True:
        raise RuntimeError("Not all development optimizer calls succeeded")
    if int(summary.get("optimizer_calls",0))!=136:
        raise RuntimeError("Expected 136 model/direction fit results")
    if summary.get("all_nesting_checks_pass") is not True:
        raise RuntimeError("Development N1 nesting audit did not pass")
    if float(summary.get("max_embedding_score_error",1.0)) > float(profile["nested_training_tolerance"]):
        raise RuntimeError("Development embedded-score tolerance failed")
    if float(summary.get("minimum_M2_minus_M3_training_gap",-1.0)) < -float(profile["nested_training_tolerance"]):
        raise RuntimeError("Development M2>=M3 training nesting invariant failed")
    if float(summary.get("max_transfer_identity_abs_error",1.0)) > float(profile["transfer_identity_tolerance"]):
        raise RuntimeError("Development transfer-identity tolerance failed")
    if summary.get("numerical_profile") != profile:
        raise RuntimeError("Aggregate numerical profile differs from repository frozen profile")

    with subjects_path.open(newline="",encoding="utf-8") as f:
        rows=list(csv.DictReader(f))
    got=[r["subject"] for r in rows]
    if got!=EXPECTED:
        raise RuntimeError(f"Development subject/order mismatch: {got}")
    if any(int(s.split("-")[1])>=44 for s in got):
        raise RuntimeError("Holdout contamination in development subject table")

    with opt_path.open(newline="",encoding="utf-8") as f:
        opts=list(csv.DictReader(f))
    if len(opts)!=136:
        raise RuntimeError("Optimizer diagnostic row count mismatch")
    if any(str(r["success"]).lower() not in {"true","1"} for r in opts):
        raise RuntimeError("At least one development optimizer call is not successful")
    if set(r["subject"] for r in opts)!=set(EXPECTED):
        raise RuntimeError("Optimizer diagnostics subject-set mismatch")
    for s in EXPECTED:
        sr=[r for r in opts if r["subject"]==s]
        if len(sr)!=4:
            raise RuntimeError(f"{s}: expected four model/direction optimizer rows")
        if {(r["model"],r["train"],r["test"]) for r in sr} != {
            ("M2","A","B"),("M2","B","A"),("M3","A","B"),("M3","B","A")
        }:
            raise RuntimeError(f"{s}: optimizer direction/model schema mismatch")

    digest=args.aggregate_artifact_digest
    if not (digest.startswith("sha256:") and len(digest)==71):
        raise RuntimeError("Aggregate artifact digest must be sha256:<64hex>")
    int(digest.split(":",1)[1],16)

    rows_out=[
        "# Step 5B final development sign-off",
        "",
        "**PASS - HOLDOUT MAY OPEN**",
        "",
        "This PASS certifies only that the frozen development/numerical prerequisites for opening the confirmatory holdout are satisfied. It is not an SMM empirical-success verdict.",
        "",
        "## Provenance",
        "",
        f"- Canonical development run ID: {args.development_run_id}",
        f"- Canonical development commit SHA: {args.development_commit_sha}",
        f"- Aggregate artifact ID: {args.aggregate_artifact_id}",
        f"- Aggregate artifact digest: {digest}",
        f"- Final numerical profile: {profile['profile_name']}",
        f"- Robustness run ID: {profile['robustness_run_id']}",
        f"- Final numerical-freeze commit: {profile['final_freeze_commit']}",
        "",
        "## Completeness and information barrier",
        "",
        "- 34/34 expected frozen-QC-passed development subjects are present.",
        "- No subject sub-044 or later appears in the development aggregate.",
        "- confirmatory_inference_performed = false.",
        "- holdout_subjects_seen = false.",
        "- The subject table exactly matches the frozen development PASS set.",
        "",
        "## Numerical integrity",
        "",
        f"- Frozen profile: {profile['sobol_candidates']} Sobol candidates / {profile['polish_starts']} polish starts.",
        f"- Production OpenBLAS kernel: {profile['openblas_coretype']}.",
        "- 136/136 selected M2/M3 x two-direction fit results satisfy the N1 success schema.",
        "- Every development training block satisfies the exact M3-in-M2 nesting checks enforced by the aggregate.",
        f"- Max embedded-score error: {summary['max_embedding_score_error']:.3e}.",
        f"- Minimum M2-M3 training gap: {summary['minimum_M2_minus_M3_training_gap']:.12g}.",
        f"- Max transfer-identity error: {summary['max_transfer_identity_abs_error']:.3e}.",
        "- The aggregate embeds exactly the repository frozen numerical profile.",
        "",
        "## Descriptive development result",
        "",
        f"- Mean DeltaELPD(M3-M2): {summary['mean_delta_elpd_M3_minus_M2']:.12g}",
        f"- Median DeltaELPD(M3-M2): {summary['median_delta_elpd_M3_minus_M2']:.12g}",
        f"- SD: {summary['sd_delta_elpd_M3_minus_M2']:.12g}",
        f"- IQR: [{summary['q25_delta']:.12g}, {summary['q75_delta']:.12g}]",
        f"- Range: [{summary['min_delta']:.12g}, {summary['max_delta']:.12g}]",
        f"- M3 > M2: {summary['n_M3_better']}/34 ({summary['fraction_M3_better']:.6f})",
        "",
        "These development quantities are descriptive only and did not alter the model, comparator, endpoint, QC, frequency range, subject split, or confirmatory success criterion.",
        "",
        "## Permanent aggregate file hashes",
        "",
        f"- development_summary.json: sha256:{sha256(summary_path)}",
        f"- development_subject_results.csv: sha256:{sha256(subjects_path)}",
        f"- development_optimizer_diagnostics.csv: sha256:{sha256(opt_path)}",
        f"- development_summary.md: sha256:{sha256(md_path)}",
        "",
        "The next permitted operation is a provenance-only update of HOLDOUT_GATE.json, followed by the already frozen manual confirmatory workflow.",
        "",
    ]

    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text("\n".join(rows_out),encoding="utf-8")
    print("DEVELOPMENT_SIGNOFF_PASS")
    print(args.out.read_text())


if __name__=="__main__":
    main()

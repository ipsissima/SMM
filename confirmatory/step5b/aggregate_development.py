#!/usr/bin/env python3
"""Aggregate the final frozen 34-subject Step 5B development fits.

Development output is descriptive only. This script:
- accepts exactly the 34 prespecified QC-passed development subjects;
- requires the final NUMERICAL_PROFILE.json to be FROZEN;
- validates every subject fit against that profile;
- never computes the confirmatory bootstrap/sign-flip decision;
- fails closed if a holdout subject appears.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median, stdev

EXPECTED = [
    "sub-001","sub-002","sub-003","sub-004","sub-005","sub-006","sub-007",
    "sub-010","sub-011","sub-014","sub-015","sub-016","sub-017","sub-018",
    "sub-019","sub-020","sub-021","sub-022","sub-023","sub-024","sub-025",
    "sub-028","sub-029","sub-030","sub-031","sub-032","sub-033","sub-034",
    "sub-035","sub-036","sub-038","sub-039","sub-040","sub-042",
]

THREAD_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def load_profile() -> tuple[dict, dict]:
    path = Path(__file__).with_name("NUMERICAL_PROFILE.json")
    p = json.loads(path.read_text(encoding="utf-8"))
    if p.get("status") != "FROZEN":
        raise RuntimeError("Final development aggregation requires NUMERICAL_PROFILE status FROZEN")
    if not p.get("final_freeze_commit") or not p.get("robustness_run_id"):
        raise RuntimeError("Frozen numerical profile lacks provenance")
    if p.get("profile_name") != "N1" or p.get("exact_nested_anchor") is not True:
        raise RuntimeError("Final aggregation requires the frozen N1 nested-comparator profile")
    expected = {
        "sobol_candidates": int(p["sobol_candidates"]),
        "polish_starts": int(p["polish_starts"]),
        "polish_maxiter": int(p["polish_maxiter"]),
        "method": p["method"],
        "ftol": float(p["ftol"]),
        "gtol": float(p["gtol"]),
        "maxls": int(p["maxls"]),
        "rel_floor": float(p["rel_floor"]),
    }
    return p, expected


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def q(xs, p):
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    pos = (len(ys) - 1) * p
    lo = int(pos)
    hi = min(lo + 1, len(ys) - 1)
    frac = pos - lo
    return ys[lo] * (1 - frac) + ys[hi] * frac


def locate(input_dir: Path) -> dict[str, Path]:
    found: dict[str, list[Path]] = {}
    for p in input_dir.rglob("sub-*.json"):
        if p.name.endswith(".excluded.json"):
            raise RuntimeError(f"Unexpected exclusion record in final development fit set: {p}")
        stem = p.stem
        try:
            n = int(stem.split("-")[1])
        except Exception:
            continue
        if n >= 44:
            raise RuntimeError(f"HOLDOUT_CONTAMINATION_BLOCKED: {p}")
        found.setdefault(stem, []).append(p)

    duplicates = {k: v for k, v in found.items() if len(v) != 1}
    if duplicates:
        raise RuntimeError(f"Duplicate subject JSONs: {duplicates}")

    got = set(found)
    expected = set(EXPECTED)
    if got != expected:
        raise RuntimeError(
            f"Development subject set mismatch; missing={sorted(expected-got)}, "
            f"unexpected={sorted(got-expected)}"
        )
    return {k: found[k][0] for k in EXPECTED}


def validate_environment(subject: str, d: dict, profile: dict):
    env = d.get("environment", {})
    thread_env = env.get("thread_env", {})
    for key in THREAD_KEYS:
        if thread_env.get(key) != "1":
            raise RuntimeError(f"{subject}: non-single-thread environment for {key}")
    if thread_env.get("OMP_DYNAMIC") != "FALSE":
        raise RuntimeError(f"{subject}: OMP_DYNAMIC drift")
    if thread_env.get("PYTHONHASHSEED") != "0":
        raise RuntimeError(f"{subject}: PYTHONHASHSEED drift")
    if thread_env.get("OPENBLAS_CORETYPE") != profile["openblas_coretype"]:
        raise RuntimeError(f"{subject}: OPENBLAS_CORETYPE drift")
    if env.get("numpy") != "2.3.5" or env.get("scipy") != "1.17.0" or env.get("mne") != "1.13.2":
        raise RuntimeError(f"{subject}: pinned numerical package drift")


def validate(subject: str, d: dict, profile: dict, expected_opt: dict):
    if d.get("seed") != int(profile["seed"]):
        raise RuntimeError(f"{subject}: seed drift")
    if d.get("optimizer") != expected_opt:
        raise RuntimeError(f"{subject}: optimizer drift: {d.get('optimizer')}")
    validate_environment(subject, d, profile)

    if set(d.get("models", {})) != {"M2", "M3"}:
        raise RuntimeError(f"{subject}: model set drift")
    if int(d.get("n_epochs", 0)) < 30:
        raise RuntimeError(f"{subject}: fewer than 30 clean epochs")
    if int(d.get("split_A", 0)) + int(d.get("split_B", 0)) != int(d["n_epochs"]):
        raise RuntimeError(f"{subject}: block split mismatch")

    for model in ("M2", "M3"):
        md = d["models"][model]
        dirs = md.get("directions", [])
        if len(dirs) != 2:
            raise RuntimeError(f"{subject} {model}: expected two CV directions")
        if not math.isfinite(float(md["cv_elpd"])):
            raise RuntimeError(f"{subject} {model}: nonfinite CV ELPD")
        for direction in dirs:
            if direction["optimizer"].get("success") is not True:
                raise RuntimeError(f"{subject} {model}: optimizer failure")
            for key in ("train_score_normalized", "heldout_score_normalized"):
                if not math.isfinite(float(direction[key])):
                    raise RuntimeError(f"{subject} {model}: nonfinite {key}")

    if float(d.get("transfer_identity_max_abs_error", 1.0)) > float(profile["transfer_identity_tolerance"]):
        raise RuntimeError(f"{subject}: M3-in-M2 transfer identity drift")
    nested = d.get("nestedness", [])
    if len(nested) != 2:
        raise RuntimeError(f"{subject}: expected two N1 nestedness records")
    for rec in nested:
        if float(rec["embedding_score_error"]) > float(profile["nested_training_tolerance"]):
            raise RuntimeError(f"{subject}: embedded M3-as-M2 score mismatch")
        if float(rec["M2_minus_M3_training_gap"]) < -float(profile["nested_training_tolerance"]):
            raise RuntimeError(f"{subject}: M2 underoptimized relative to contained M3 point")

    delta = float(d["delta_elpd_M3_minus_M2"])
    calc = float(d["models"]["M3"]["cv_elpd"]) - float(d["models"]["M2"]["cv_elpd"])
    if not math.isfinite(delta) or abs(delta - calc) > 1e-10:
        raise RuntimeError(f"{subject}: delta mismatch")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir", type=Path)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    profile, expected_opt = load_profile()
    paths = locate(args.input_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    subject_rows = []
    opt_rows = []
    input_manifest = []

    for subject, path in paths.items():
        d = json.loads(path.read_text(encoding="utf-8"))
        validate(subject, d, profile, expected_opt)
        m2 = float(d["models"]["M2"]["cv_elpd"])
        m3 = float(d["models"]["M3"]["cv_elpd"])
        delta = float(d["delta_elpd_M3_minus_M2"])
        subject_rows.append({
            "subject": subject,
            "n_epochs": int(d["n_epochs"]),
            "split_A": int(d["split_A"]),
            "split_B": int(d["split_B"]),
            "M2_cv_elpd": m2,
            "M3_cv_elpd": m3,
            "delta_elpd_M3_minus_M2": delta,
            "M3_better": int(delta > 0.0),
        })
        for model in ("M2", "M3"):
            for direction in d["models"][model]["directions"]:
                opt = direction["optimizer"]
                opt_rows.append({
                    "subject": subject,
                    "model": model,
                    "train": direction["train"],
                    "test": direction["test"],
                    "success": bool(opt["success"]),
                    "message": str(opt["message"]),
                    "nit": int(opt["nit"]),
                    "nfev": int(opt["nfev"]),
                    "train_score_normalized": float(direction["train_score_normalized"]),
                    "heldout_score_normalized": float(direction["heldout_score_normalized"]),
                })
        input_manifest.append({
            "subject": subject,
            "path": str(path),
            "sha256": sha256(path),
        })

    deltas = [r["delta_elpd_M3_minus_M2"] for r in subject_rows]
    summary = {
        "phase": "development",
        "confirmatory_inference_performed": False,
        "holdout_subjects_seen": False,
        "n_subjects": len(deltas),
        "expected_n_subjects": 34,
        "mean_delta_elpd_M3_minus_M2": mean(deltas),
        "median_delta_elpd_M3_minus_M2": median(deltas),
        "sd_delta_elpd_M3_minus_M2": stdev(deltas),
        "min_delta": min(deltas),
        "q25_delta": q(deltas, 0.25),
        "q75_delta": q(deltas, 0.75),
        "max_delta": max(deltas),
        "n_M3_better": sum(x > 0 for x in deltas),
        "fraction_M3_better": sum(x > 0 for x in deltas) / len(deltas),
        "all_optimizers_success": all(r["success"] for r in opt_rows),
        "optimizer_calls": len(opt_rows),
        "frozen_optimizer": expected_opt,
        "numerical_profile": profile,
        "seed": int(profile["seed"]),
        "inputs": input_manifest,
        "interpretation_guardrail": (
            "Development statistics are descriptive only. They cannot change the mechanism, "
            "M2 comparator, primary endpoint, frequency range, exclusions, split, or "
            "confirmatory success criterion."
        ),
    }

    if summary["optimizer_calls"] != 34 * 2 * 2:
        raise RuntimeError("Expected exactly 136 successful model/direction optimizer calls")

    with (args.out_dir / "development_subject_results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(subject_rows[0]))
        w.writeheader()
        w.writerows(subject_rows)

    with (args.out_dir / "development_optimizer_diagnostics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(opt_rows[0]))
        w.writeheader()
        w.writerows(opt_rows)

    (args.out_dir / "development_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    md = [
        "# Step 5B final development aggregate",
        "",
        "**Status:** descriptive development audit only - not confirmatory inference.",
        "",
        f"- Subjects: {summary['n_subjects']}/34",
        f"- Mean DeltaELPD(M3-M2): {summary['mean_delta_elpd_M3_minus_M2']:.9f}",
        f"- Median DeltaELPD(M3-M2): {summary['median_delta_elpd_M3_minus_M2']:.9f}",
        f"- SD: {summary['sd_delta_elpd_M3_minus_M2']:.9f}",
        f"- IQR: [{summary['q25_delta']:.9f}, {summary['q75_delta']:.9f}]",
        f"- Range: [{summary['min_delta']:.9f}, {summary['max_delta']:.9f}]",
        f"- M3 > M2: {summary['n_M3_better']}/{summary['n_subjects']} "
        f"({summary['fraction_M3_better']:.3f})",
        f"- Optimizer calls successful: {summary['all_optimizers_success']} "
        f"({summary['optimizer_calls']} calls)",
        f"- Frozen profile: {profile['sobol_candidates']} Sobol candidates / "
        f"{profile['polish_starts']} polish starts / {profile['openblas_coretype']} kernel",
        "",
        "No bootstrap confidence interval, sign-flip p-value, band selection, or "
        "confirmatory decision is computed on development data.",
        "",
        "The holdout boundary is enforced in code: any sub-044 or later JSON aborts aggregation.",
        "",
    ]
    (args.out_dir / "development_summary.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "inputs"}, indent=2))


if __name__ == "__main__":
    main()

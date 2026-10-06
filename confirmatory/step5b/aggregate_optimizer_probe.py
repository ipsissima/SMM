#!/usr/bin/env python3
"""Aggregate development-only optimizer robustness probe outputs.

PASS/FAIL criteria are frozen in OPTIMIZER_ROBUSTNESS_DECISION_RULE.md.
The decision uses per-model training recovery and numerical reproducibility,
never the sign or favorability of M3-M2.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

KERNELS = {"Haswell", "Sandybridge", "Zen"}
TRAIN_RANGE_MAX = 0.001
CV_RANGE_MAX = 0.005


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    files = sorted(args.input_dir.rglob("sub-001-*.json"))
    if len(files) != 3:
        raise RuntimeError(f"Expected exactly 3 kernel JSONs, got {len(files)}")

    records = []
    seen = set()
    for p in files:
        d = json.loads(p.read_text(encoding="utf-8"))
        if not d.get("development_only"):
            raise RuntimeError(f"Not marked development-only: {p}")
        core = d["environment"]["env"].get("OPENBLAS_CORETYPE")
        if core not in KERNELS:
            raise RuntimeError(f"Unexpected core type {core!r}")
        if core in seen:
            raise RuntimeError(f"Duplicate core type {core}")
        seen.add(core)
        records.append((core, d))
    if seen != KERNELS:
        raise RuntimeError(f"Kernel set mismatch: {seen}")

    profile = records[0][1]["profile"]
    for _, d in records[1:]:
        if d["profile"] != profile:
            raise RuntimeError("Optimizer profile drift across kernels")

    checks = []
    all_success = True
    all_recovered = True
    train_ranges = {}
    cv_ranges = {}

    for model in ("M2", "M3"):
        for train in ("A", "B"):
            rows = []
            for core, d in records:
                direction = next(
                    x for x in d["models"][model]["directions"]
                    if x["train"] == train
                )
                success = direction["optimizer"]["success"] is True
                recovered = direction["recovered_previous_best_within_tolerance"] is True
                score = float(direction["train_score_normalized"])
                if not math.isfinite(score):
                    raise RuntimeError("Nonfinite training score")
                all_success &= success
                all_recovered &= recovered
                rows.append((core, score, success, recovered))
            spread = max(x[1] for x in rows) - min(x[1] for x in rows)
            train_ranges[f"{model}_{train}"] = spread
            checks.append({
                "kind": "training",
                "model": model,
                "direction": train,
                "values": {core: score for core, score, _, _ in rows},
                "range": spread,
                "range_pass": spread <= TRAIN_RANGE_MAX,
                "all_success": all(x[2] for x in rows),
                "all_historical_best_recovered": all(x[3] for x in rows),
            })

        vals = {core: float(d["models"][model]["cv_elpd"]) for core, d in records}
        spread = max(vals.values()) - min(vals.values())
        cv_ranges[model] = spread
        checks.append({
            "kind": "cv_reproducibility",
            "model": model,
            "values": vals,
            "range": spread,
            "range_pass": spread <= CV_RANGE_MAX,
        })

    train_range_pass = all(v <= TRAIN_RANGE_MAX for v in train_ranges.values())
    cv_range_pass = all(v <= CV_RANGE_MAX for v in cv_ranges.values())
    passed = bool(all_success and all_recovered and train_range_pass and cv_range_pass)

    summary = {
        "phase": "development_optimizer_robustness",
        "selection_uses_delta_sign": False,
        "profile": profile,
        "kernels": sorted(seen),
        "criteria": {
            "all_optimizer_calls_success": all_success,
            "all_historical_best_training_basins_recovered": all_recovered,
            "max_training_score_range": TRAIN_RANGE_MAX,
            "training_ranges": train_ranges,
            "training_range_pass": train_range_pass,
            "max_model_cv_elpd_range": CV_RANGE_MAX,
            "cv_ranges": cv_ranges,
            "cv_range_pass": cv_range_pass,
        },
        "probe_pass": passed,
        "delta_values_for_audit_only_not_selection": {
            core: float(d["delta_elpd_M3_minus_M2"]) for core, d in records
        },
        "next_action": (
            "Freeze this profile and rerun all 34 development subjects"
            if passed
            else "Escalate exactly to P2 per OPTIMIZER_ROBUSTNESS_DECISION_RULE.md"
        ),
        "checks": checks,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    md = [
        "# Step 5B optimizer robustness aggregate",
        "",
        f"**Verdict: {'PASS' if passed else 'FAIL'}**",
        "",
        f"- Profile: {profile}",
        f"- Kernels: {', '.join(sorted(seen))}",
        f"- All optimizer calls successful: {all_success}",
        f"- All historical best training basins recovered: {all_recovered}",
        f"- Training-score ranges: {train_ranges}",
        f"- Per-model CV ELPD ranges: {cv_ranges}",
        "",
        "DeltaELPD values are archived for transparency but are not used in profile selection.",
        f"Next action: {summary['next_action']}",
        "",
    ]
    args.out.with_suffix(".md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

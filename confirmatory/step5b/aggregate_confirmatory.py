#!/usr/bin/env python3
"""Aggregate the frozen Step 5B confirmatory holdout and apply the locked test.

All 565 assigned holdout subjects must be accounted for. Subjects that fail
the frozen primary QC criteria remain in the accounting table but have no
DeltaELPD and are excluded from the primary paired predictive analysis exactly
because the pre-EEG lock requires >=30 clean 4-s epochs and the other frozen QC
criteria.

The inferential rule is unchanged:
1) 10,000-resample subject bootstrap 95% percentile CI for mean Delta ELPD;
2) one-sided paired sign-flip test with 100,000 permutations;
3) success iff CI lower bound > 0 AND p_flip < 0.05.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

EXPECTED = [f"sub-{i:03d}" for i in range(44, 609)]
SEED = 97
N_BOOT = 10_000
N_FLIP = 100_000
THREAD_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def load_profile() -> tuple[dict, dict]:
    p = json.loads(Path(__file__).with_name("NUMERICAL_PROFILE.json").read_text())
    if p.get("status") != "FROZEN":
        raise RuntimeError("Confirmatory aggregation requires final NUMERICAL_PROFILE status FROZEN")
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


def subject_from_path(p: Path) -> tuple[str, bool]:
    if p.name.endswith(".excluded.json"):
        return p.name[:-len(".excluded.json")], True
    if p.name.startswith("sub-") and p.suffix == ".json":
        return p.stem, False
    raise ValueError(p)


def locate(input_dir: Path) -> dict[str, tuple[Path, bool]]:
    found: dict[str, list[tuple[Path, bool]]] = {}
    for p in input_dir.rglob("sub-*.json"):
        try:
            subject, excluded = subject_from_path(p)
            n = int(subject.split("-")[1])
        except Exception:
            continue
        if n < 44:
            raise RuntimeError(f"DEVELOPMENT_CONTAMINATION_BLOCKED: {p}")
        if n > 608:
            raise RuntimeError(f"OUT_OF_COHORT_SUBJECT_BLOCKED: {p}")
        found.setdefault(subject, []).append((p, excluded))

    duplicates = {k: v for k, v in found.items() if len(v) != 1}
    if duplicates:
        raise RuntimeError(f"Duplicate holdout subject records: {duplicates}")

    got = set(found)
    expected = set(EXPECTED)
    if got != expected:
        raise RuntimeError(
            f"Confirmatory accounting mismatch; missing={sorted(expected-got)}, "
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


def validate_fit(subject: str, d: dict, profile: dict, expected_opt: dict):
    if d.get("seed") != int(profile["seed"]):
        raise RuntimeError(f"{subject}: seed drift")
    if d.get("optimizer") != expected_opt:
        raise RuntimeError(f"{subject}: optimizer drift")
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


def validate_exclusion(subject: str, d: dict):
    if d.get("subject") != subject:
        raise RuntimeError(f"{subject}: exclusion subject mismatch")
    if d.get("included") is not False:
        raise RuntimeError(f"{subject}: malformed exclusion record")
    reason = str(d.get("reason"))
    if not reason.startswith("Primary QC failed:"):
        raise RuntimeError(f"{subject}: exclusion is not a frozen primary-QC failure")
    return reason


def bootstrap_mean_ci(x: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    vals = np.empty(N_BOOT, dtype=float)
    n = len(x)
    batch = 500
    pos = 0
    while pos < N_BOOT:
        b = min(batch, N_BOOT - pos)
        idx = rng.integers(0, n, size=(b, n))
        vals[pos:pos+b] = x[idx].mean(axis=1)
        pos += b
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return float(lo), float(hi)


def sign_flip_pvalue(x: np.ndarray) -> tuple[float, int]:
    rng = np.random.default_rng(SEED)
    obs = float(x.mean())
    ge = 0
    done = 0
    batch = 1000
    while done < N_FLIP:
        b = min(batch, N_FLIP - done)
        signs = np.where(rng.integers(0, 2, size=(b, len(x))) == 0, -1.0, 1.0)
        perm = (signs * x).mean(axis=1)
        ge += int(np.count_nonzero(perm >= obs))
        done += b
    p = (ge + 1.0) / (N_FLIP + 1.0)
    return float(p), ge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir", type=Path)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    profile, expected_opt = load_profile()
    records = locate(args.input_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    subject_rows = []
    opt_rows = []
    input_manifest = []
    included_deltas = []
    exclusion_reasons = []

    for subject, (path, excluded) in records.items():
        d = json.loads(path.read_text(encoding="utf-8"))
        input_manifest.append({
            "subject": subject,
            "record_type": "qc_excluded" if excluded else "fit",
            "path": str(path),
            "sha256": sha256(path),
        })

        if excluded:
            reason = validate_exclusion(subject, d)
            exclusion_reasons.append(reason)
            subject_rows.append({
                "subject": subject,
                "primary_included": 0,
                "qc_exclusion_reason": reason,
                "n_epochs": "",
                "split_A": "",
                "split_B": "",
                "M2_cv_elpd": "",
                "M3_cv_elpd": "",
                "delta_elpd_M3_minus_M2": "",
                "M3_better": "",
                "max_embedding_score_error": "",
                "min_M2_minus_M3_training_gap": "",
                "transfer_identity_max_abs_error": "",
            })
            continue

        validate_fit(subject, d, profile, expected_opt)
        m2 = float(d["models"]["M2"]["cv_elpd"])
        m3 = float(d["models"]["M3"]["cv_elpd"])
        delta = float(d["delta_elpd_M3_minus_M2"])
        included_deltas.append(delta)
        nested=d["nestedness"]
        subject_rows.append({
            "subject": subject,
            "primary_included": 1,
            "qc_exclusion_reason": "",
            "n_epochs": int(d["n_epochs"]),
            "split_A": int(d["split_A"]),
            "split_B": int(d["split_B"]),
            "M2_cv_elpd": m2,
            "M3_cv_elpd": m3,
            "delta_elpd_M3_minus_M2": delta,
            "M3_better": int(delta > 0.0),
            "max_embedding_score_error": max(float(q["embedding_score_error"]) for q in nested),
            "min_M2_minus_M3_training_gap": min(float(q["M2_minus_M3_training_gap"]) for q in nested),
            "transfer_identity_max_abs_error": float(d["transfer_identity_max_abs_error"]),
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

    x = np.asarray(included_deltas, dtype=float)
    if len(x) == 0:
        raise RuntimeError("No holdout recording passed the frozen primary QC criteria")
    if len(subject_rows) != 565:
        raise RuntimeError("Internal accounting error: expected 565 holdout assignments")
    if len(opt_rows) != len(x) * 4:
        raise RuntimeError("Optimizer accounting mismatch")

    obs = float(x.mean())
    ci_low, ci_high = bootstrap_mean_ci(x)
    p_flip, flips_ge = sign_flip_pvalue(x)
    success = bool(ci_low > 0.0 and p_flip < 0.05)

    result = {
        "phase": "confirmatory_holdout",
        "primary_condition": "ses-1 / EyesClosed / acq-pre",
        "n_holdout_assigned": 565,
        "n_primary_included_after_frozen_qc": int(len(x)),
        "n_primary_excluded_by_frozen_qc": int(565-len(x)),
        "qc_exclusion_reason_counts": dict(Counter(exclusion_reasons)),
        "contrast": "ELPD(M3)-ELPD(M2)",
        "mean_delta_elpd_M3_minus_M2": obs,
        "median_delta_elpd_M3_minus_M2": float(np.median(x)),
        "sd_delta_elpd_M3_minus_M2": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
        "bootstrap": {
            "seed": SEED,
            "resamples": N_BOOT,
            "method": "subject bootstrap, percentile 95% CI over frozen-QC-included subjects",
            "ci_low": ci_low,
            "ci_high": ci_high,
        },
        "sign_flip": {
            "seed": SEED,
            "permutations": N_FLIP,
            "alternative": "mean Delta ELPD > 0",
            "extreme_or_equal_count": flips_ge,
            "plus_one_correction": True,
            "p_value": p_flip,
        },
        "n_M3_better": int(np.count_nonzero(x > 0.0)),
        "fraction_M3_better": float(np.mean(x > 0.0)),
        "all_optimizers_success": all(r["success"] for r in opt_rows),
        "optimizer_calls": len(opt_rows),
        "max_embedding_score_error": max(
            float(r["max_embedding_score_error"]) for r in subject_rows if r["primary_included"] == 1
        ),
        "minimum_M2_minus_M3_training_gap": min(
            float(r["min_M2_minus_M3_training_gap"]) for r in subject_rows if r["primary_included"] == 1
        ),
        "max_transfer_identity_abs_error": max(
            float(r["transfer_identity_max_abs_error"]) for r in subject_rows if r["primary_included"] == 1
        ),
        "all_nesting_checks_pass": (
            max(float(r["max_embedding_score_error"]) for r in subject_rows if r["primary_included"] == 1)
                <= float(profile["nested_training_tolerance"])
            and min(float(r["min_M2_minus_M3_training_gap"]) for r in subject_rows if r["primary_included"] == 1)
                >= -float(profile["nested_training_tolerance"])
            and max(float(r["transfer_identity_max_abs_error"]) for r in subject_rows if r["primary_included"] == 1)
                <= float(profile["transfer_identity_tolerance"])
        ),
        "frozen_optimizer": expected_opt,
        "numerical_profile": profile,
        "success_rule": "CI_low > 0 AND one-sided sign-flip p < 0.05",
        "specific_SMM_predictive_success": success,
        "qc_rule": (
            "All 565 assigned holdout subjects are accounted for; inference includes only recordings "
            "meeting the frozen pre-EEG primary QC/inclusion criteria."
        ),
        "epistemic_guardrail": (
            "A positive result favors the physiologically constrained SMM conditional on the "
            "upstream biological derivation; EEG does not directly observe astrocytes. "
            "M3>M0 alone is not sufficient for SMM-specific success."
        ),
        "inputs": input_manifest,
    }

    with (args.out_dir / "confirmatory_subject_results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(subject_rows[0]))
        w.writeheader()
        w.writerows(subject_rows)

    with (args.out_dir / "confirmatory_optimizer_diagnostics.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "subject","model","train","test","success","message","nit","nfev",
            "train_score_normalized","heldout_score_normalized",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(opt_rows)

    (args.out_dir / "confirmatory_primary_result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )

    verdict = "PASS" if success else "FAIL"
    md = [
        "# Step 5B confirmatory primary result",
        "",
        f"**Frozen SMM-specific success verdict: {verdict}**",
        "",
        "- Holdout assignments: 565/565 accounted for",
        f"- Primary included after frozen QC: {len(x)}",
        f"- Frozen-QC exclusions: {565-len(x)}",
        f"- Mean DeltaELPD(M3-M2): {obs:.9f}",
        f"- 95% bootstrap CI: [{ci_low:.9f}, {ci_high:.9f}]",
        f"- One-sided sign-flip p: {p_flip:.8g}",
        f"- M3 > M2: {int(np.count_nonzero(x > 0))}/{len(x)} "
        f"({float(np.mean(x > 0)):.3f})",
        f"- All optimizer calls successful: {result['all_optimizers_success']}",
        "",
        "Success was defined before holdout inspection as CI_low > 0 AND p_flip < 0.05.",
        "The primary inclusion/QC rules were also frozen before holdout inspection.",
        "",
        "Interpretation guardrail: even a PASS means that unseen EEG favors the "
        "physiologically constrained SMM relative to the matched generic two-state slow-control "
        "model, conditional on the upstream mechanistic derivation. It is not direct observation "
        "of astrocytes by EEG.",
        "",
    ]
    (args.out_dir / "confirmatory_primary_result.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "inputs"}, indent=2))


if __name__ == "__main__":
    main()

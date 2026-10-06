#!/usr/bin/env python3
"""Aggregate the frozen Step 5B confirmatory holdout and apply the locked test.

This script accepts exactly sub-044..sub-608 and implements the pre-EEG rule:
1) 10,000-resample subject bootstrap 95% percentile CI for mean Delta ELPD;
2) one-sided paired sign-flip test with 100,000 permutations;
3) success iff CI lower bound > 0 AND p_flip < 0.05.

Both random procedures use seed 97.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

EXPECTED = [f"sub-{i:03d}" for i in range(44, 609)]
FROZEN_OPT = {
    "sobol_candidates": 32,
    "polish_starts": 4,
    "polish_maxiter": 120,
    "method": "L-BFGS-B",
    "rel_floor": 1e-6,
}
SEED = 97
N_BOOT = 10_000
N_FLIP = 100_000


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def locate(input_dir: Path) -> dict[str, Path]:
    found: dict[str, list[Path]] = {}
    for p in input_dir.rglob("sub-*.json"):
        stem = p.stem
        try:
            n = int(stem.split("-")[1])
        except Exception:
            continue
        if n < 44:
            raise RuntimeError(f"DEVELOPMENT_CONTAMINATION_BLOCKED: {p}")
        if n > 608:
            raise RuntimeError(f"OUT_OF_COHORT_SUBJECT_BLOCKED: {p}")
        found.setdefault(stem, []).append(p)

    duplicates = {k: v for k, v in found.items() if len(v) != 1}
    if duplicates:
        raise RuntimeError(f"Duplicate holdout JSONs: {duplicates}")

    got = set(found)
    expected = set(EXPECTED)
    if got != expected:
        raise RuntimeError(
            f"Confirmatory subject set mismatch; missing={sorted(expected-got)}, "
            f"unexpected={sorted(got-expected)}"
        )
    return {k: found[k][0] for k in EXPECTED}


def validate(subject: str, d: dict):
    if d.get("seed") != SEED:
        raise RuntimeError(f"{subject}: seed drift")
    if d.get("optimizer") != FROZEN_OPT:
        raise RuntimeError(f"{subject}: optimizer drift")
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

    delta = float(d["delta_elpd_M3_minus_M2"])
    calc = float(d["models"]["M3"]["cv_elpd"]) - float(d["models"]["M2"]["cv_elpd"])
    if not math.isfinite(delta) or abs(delta - calc) > 1e-10:
        raise RuntimeError(f"{subject}: delta mismatch")


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

    paths = locate(args.input_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    subject_rows = []
    opt_rows = []
    input_manifest = []

    for subject, path in paths.items():
        d = json.loads(path.read_text(encoding="utf-8"))
        validate(subject, d)
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

    x = np.asarray([r["delta_elpd_M3_minus_M2"] for r in subject_rows], dtype=float)
    obs = float(x.mean())
    ci_low, ci_high = bootstrap_mean_ci(x)
    p_flip, flips_ge = sign_flip_pvalue(x)
    success = bool(ci_low > 0.0 and p_flip < 0.05)

    result = {
        "phase": "confirmatory_holdout",
        "primary_condition": "ses-1 / EyesClosed / acq-pre",
        "n_subjects": len(x),
        "expected_n_subjects": 565,
        "contrast": "ELPD(M3)-ELPD(M2)",
        "mean_delta_elpd_M3_minus_M2": obs,
        "median_delta_elpd_M3_minus_M2": float(np.median(x)),
        "sd_delta_elpd_M3_minus_M2": float(np.std(x, ddof=1)),
        "bootstrap": {
            "seed": SEED,
            "resamples": N_BOOT,
            "method": "subject bootstrap, percentile 95% CI",
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
        "frozen_optimizer": FROZEN_OPT,
        "success_rule": "CI_low > 0 AND one-sided sign-flip p < 0.05",
        "specific_SMM_predictive_success": success,
        "epistemic_guardrail": (
            "A positive result favors the physiologically constrained SMM conditional on the "
            "upstream biological derivation; EEG does not directly observe astrocytes. "
            "M3>M0 alone is not sufficient for SMM-specific success."
        ),
        "inputs": input_manifest,
    }

    with (args.out_dir / "confirmatory_subject_results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(subject_rows[0]))
        w.writeheader(); w.writerows(subject_rows)

    with (args.out_dir / "confirmatory_optimizer_diagnostics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(opt_rows[0]))
        w.writeheader(); w.writerows(opt_rows)

    (args.out_dir / "confirmatory_primary_result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )

    verdict = "PASS" if success else "FAIL"
    md = [
        "# Step 5B confirmatory primary result",
        "",
        f"**Frozen SMM-specific success verdict: {verdict}**",
        "",
        f"- Subjects: {len(x)}/565",
        f"- Mean DeltaELPD(M3-M2): {obs:.9f}",
        f"- 95% bootstrap CI: [{ci_low:.9f}, {ci_high:.9f}]",
        f"- One-sided sign-flip p: {p_flip:.8g}",
        f"- M3 > M2: {int(np.count_nonzero(x > 0))}/{len(x)} "
        f"({float(np.mean(x > 0)):.3f})",
        f"- All optimizer calls successful: {result['all_optimizers_success']}",
        "",
        "Success was defined before holdout inspection as CI_low > 0 AND p_flip < 0.05.",
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

#!/usr/bin/env python3
"""Generate manuscript-ready primary-result figures from frozen aggregate tables.

This script never reads raw EEG or GitHub logs. It consumes only the permanent
machine-readable aggregate outputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save(fig, out: Path):
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("aggregate_dir", type=Path)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = args.aggregate_dir / "confirmatory_subject_results.csv"
    json_path = args.aggregate_dir / "confirmatory_primary_result.json"
    df = pd.read_csv(csv_path)
    result = json.loads(json_path.read_text(encoding="utf-8"))

    if len(df) != 565 or result.get("n_subjects") != 565:
        raise RuntimeError("Primary figure generation requires the complete 565-subject holdout")
    expected = [f"sub-{i:03d}" for i in range(44, 609)]
    if df["subject"].tolist() != expected:
        raise RuntimeError("Subject order/set drift in confirmatory table")

    delta = df["delta_elpd_M3_minus_M2"].to_numpy(float)

    # Figure 5A: ordered subject contrasts.
    order = np.argsort(delta)
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.plot(np.arange(len(delta)), delta[order], linewidth=1.0)
    ax.axhline(0.0, linewidth=1.0)
    ax.set_xlabel("Holdout subjects ordered by DeltaELPD")
    ax.set_ylabel("DeltaELPD (M3 - M2)")
    ax.set_title("Subject-level held-out predictive contrast")
    save(fig, args.out_dir / "figure_primary_delta_ordered.png")

    # Figure 5B: distribution.
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.hist(delta, bins="auto")
    ax.axvline(0.0, linewidth=1.0)
    ax.axvline(float(result["mean_delta_elpd_M3_minus_M2"]), linewidth=1.0)
    ax.set_xlabel("DeltaELPD (M3 - M2)")
    ax.set_ylabel("Subjects")
    ax.set_title("Distribution of primary predictive contrast")
    save(fig, args.out_dir / "figure_primary_delta_histogram.png")

    # Figure 5C: M2 vs M3 per subject.
    x = df["M2_cv_elpd"].to_numpy(float)
    y = df["M3_cv_elpd"].to_numpy(float)
    lo = float(min(x.min(), y.min()))
    hi = float(max(x.max(), y.max()))
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(x, y, s=10, alpha=0.7)
    ax.plot([lo, hi], [lo, hi], linewidth=1.0)
    ax.set_xlabel("M2 held-out CV ELPD")
    ax.set_ylabel("M3 held-out CV ELPD")
    ax.set_title("Matched generic slow control vs constrained SMM")
    save(fig, args.out_dir / "figure_primary_m2_vs_m3.png")

    # Figure 5D: frozen group estimate and interval.
    mu = float(result["mean_delta_elpd_M3_minus_M2"])
    ci = result["bootstrap"]
    low, high = float(ci["ci_low"]), float(ci["ci_high"])
    fig, ax = plt.subplots(figsize=(5.5, 2.8))
    ax.errorbar([mu], [0], xerr=[[mu-low], [high-mu]], fmt="o", capsize=5)
    ax.axvline(0.0, linewidth=1.0)
    ax.set_yticks([])
    ax.set_xlabel("Mean DeltaELPD (M3 - M2), 95% bootstrap CI")
    ax.set_title(
        f"Primary confirmatory result: "
        f"{'PASS' if result['specific_SMM_predictive_success'] else 'FAIL'}"
    )
    save(fig, args.out_dir / "figure_primary_group_estimate.png")

    print("PRIMARY_FIGURES_PASS", len(df))


if __name__ == "__main__":
    main()

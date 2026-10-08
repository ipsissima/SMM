#!/usr/bin/env python3
"""Generate manuscript-ready primary-result figures from frozen aggregate tables.

Consumes only permanent aggregate outputs. All 565 holdout assignments must be
present in the accounting table; plots use only frozen-QC-included subjects.
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

    df = pd.read_csv(args.aggregate_dir / "confirmatory_subject_results.csv")
    result = json.loads((args.aggregate_dir / "confirmatory_primary_result.json").read_text(encoding="utf-8"))

    if len(df) != 565 or result.get("n_holdout_assigned") != 565:
        raise RuntimeError("Primary figure generation requires complete 565-subject accounting")
    expected = [f"sub-{i:03d}" for i in range(44, 609)]
    if df["subject"].tolist() != expected:
        raise RuntimeError("Subject order/set drift in confirmatory table")

    inc = df.loc[df["primary_included"] == 1].copy()
    if len(inc) != int(result["n_primary_included_after_frozen_qc"]):
        raise RuntimeError("Frozen-QC included-n mismatch")
    if len(inc) == 0:
        raise RuntimeError("No frozen-QC-included subjects to plot")

    delta = inc["delta_elpd_M3_minus_M2"].to_numpy(float)

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    order = np.argsort(delta)
    ax.plot(np.arange(len(delta)), delta[order], linewidth=1.0)
    ax.axhline(0.0, linewidth=1.0)
    ax.set_xlabel("Frozen-QC-included holdout subjects ordered by DeltaELPD")
    ax.set_ylabel("DeltaELPD (M3 - M2)")
    ax.set_title("Subject-level held-out predictive contrast")
    save(fig, args.out_dir / "figure_primary_delta_ordered.png")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.hist(delta, bins="auto")
    ax.axvline(0.0, linewidth=1.0)
    ax.axvline(float(result["mean_delta_elpd_M3_minus_M2"]), linewidth=1.0)
    ax.set_xlabel("DeltaELPD (M3 - M2)")
    ax.set_ylabel("Frozen-QC-included subjects")
    ax.set_title("Distribution of primary predictive contrast")
    save(fig, args.out_dir / "figure_primary_delta_histogram.png")

    x = inc["M2_cv_elpd"].to_numpy(float)
    y = inc["M3_cv_elpd"].to_numpy(float)
    lo = float(min(x.min(), y.min()))
    hi = float(max(x.max(), y.max()))
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(x, y, s=10, alpha=0.7)
    ax.plot([lo, hi], [lo, hi], linewidth=1.0)
    ax.set_xlabel("M2 held-out CV ELPD")
    ax.set_ylabel("M3 held-out CV ELPD")
    ax.set_title("Matched generic slow control vs constrained SMM")
    save(fig, args.out_dir / "figure_primary_m2_vs_m3.png")

    mu = float(result["mean_delta_elpd_M3_minus_M2"])
    low = float(result["bootstrap"]["ci_low"])
    high = float(result["bootstrap"]["ci_high"])
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

    print("PRIMARY_FIGURES_PASS", "assigned=565", f"included={len(inc)}")


if __name__ == "__main__":
    main()

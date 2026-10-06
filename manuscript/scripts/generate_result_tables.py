#!/usr/bin/env python3
"""Generate manuscript result-table fragments from permanent aggregate JSON/CSV.

No raw EEG, Action logs, or transient text are parsed here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def f9(x):
    return f"{float(x):.9f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--development-dir", type=Path)
    ap.add_argument("--confirmatory-dir", type=Path)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.development_dir:
        d = json.loads((args.development_dir / "development_summary.json").read_text())
        rows = [
            ("Subjects", str(d["n_subjects"])),
            ("Mean DeltaELPD(M3-M2)", f9(d["mean_delta_elpd_M3_minus_M2"])),
            ("Median DeltaELPD(M3-M2)", f9(d["median_delta_elpd_M3_minus_M2"])),
            ("SD", f9(d["sd_delta_elpd_M3_minus_M2"])),
            ("IQR", f"[{f9(d['q25_delta'])}, {f9(d['q75_delta'])}]"),
            ("Range", f"[{f9(d['min_delta'])}, {f9(d['max_delta'])}]"),
            ("M3 > M2", f"{d['n_M3_better']}/{d['n_subjects']} ({d['fraction_M3_better']:.3f})"),
            ("All optimizer calls successful", str(d["all_optimizers_success"])),
        ]
        md=["| Development quantity | Value |","|---|---:|"]
        md += [f"| {k} | {v} |" for k,v in rows]
        md += ["","*Descriptive development quantities only; no confirmatory inference was performed.*",""]
        (args.out_dir / "table_development.md").write_text("\n".join(md),encoding="utf-8")

    if args.confirmatory_dir:
        d = json.loads((args.confirmatory_dir / "confirmatory_primary_result.json").read_text())
        rows = [
            ("Subjects", str(d["n_subjects"])),
            ("Mean DeltaELPD(M3-M2)", f9(d["mean_delta_elpd_M3_minus_M2"])),
            ("Median DeltaELPD(M3-M2)", f9(d["median_delta_elpd_M3_minus_M2"])),
            ("SD", f9(d["sd_delta_elpd_M3_minus_M2"])),
            ("Bootstrap 95% CI", f"[{f9(d['bootstrap']['ci_low'])}, {f9(d['bootstrap']['ci_high'])}]"),
            ("One-sided sign-flip p", f"{float(d['sign_flip']['p_value']):.8g}"),
            ("M3 > M2", f"{d['n_M3_better']}/{d['n_subjects']} ({d['fraction_M3_better']:.3f})"),
            ("Frozen criterion", d["success_rule"]),
            ("Verdict", "PASS" if d["specific_SMM_predictive_success"] else "FAIL"),
        ]
        md=["| Primary confirmatory quantity | Value |","|---|---:|"]
        md += [f"| {k} | {v} |" for k,v in rows]
        (args.out_dir / "table_primary_confirmatory.md").write_text("\n".join(md)+"\n",encoding="utf-8")

        subjects=pd.read_csv(args.confirmatory_dir / "confirmatory_subject_results.csv")
        if len(subjects) != 565:
            raise RuntimeError("Confirmatory table generation requires 565 subjects")

    print("RESULT_TABLES_PASS")


if __name__ == "__main__":
    main()

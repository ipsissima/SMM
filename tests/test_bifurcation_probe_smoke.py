"""Smoke-test the bifurcation probe on a tiny grid to ensure outputs exist."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_bifurcation_probe_smoke(tmp_path: Path) -> None:
    results_dir = tmp_path / "bifurcation_results"
    cmd = [
        sys.executable,
        "bifurcation_probe.py",
        "--nx",
        "8",
        "--ny",
        "8",
        "--n-kappa",
        "3",
        "--n-h",
        "5",
        "--dt",
        "0.05",
        "--tmax",
        "5.0",
        "--results-dir",
        str(results_dir),
        "--no-plots",
        "--log-level",
        "WARNING",
    ]
    subprocess.run(cmd, check=True)

    summary_path = results_dir / "cusp_grid_summary.csv"
    assert summary_path.exists(), "Expected cusp_grid_summary.csv to be created"

    summary_df = pd.read_csv(summary_path)
    assert not summary_df.empty
    assert len(summary_df) == 3
    assert {"kappa", "lambda_min", "u_fit", "alpha_fit"}.issubset(summary_df.columns)

    # Ensure at least one hysteresis CSV has been produced
    hysteresis_files = list(results_dir.glob("hysteresis_kappa_*.csv"))
    assert hysteresis_files, "Expected hysteresis CSV outputs"

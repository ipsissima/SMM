"""Smoke-test the bifurcation probe using the dedicated ``--smoke`` mode."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_bifurcation_probe_smoke(tmp_path: Path) -> None:
    """Run the probe with the smoke-test shortcut and validate core outputs."""

    results_dir = tmp_path / "bifurcation_smoke"
    cmd = [
        sys.executable,
        "bifurcation_probe.py",
        "--smoke",
        "--outdir",
        str(results_dir),
        "--log_level",
        "WARNING",
    ]
    subprocess.run(cmd, check=True, timeout=300)

    summary_path = results_dir / "cusp_grid_summary.csv"
    assert summary_path.exists(), "Expected cusp_grid_summary.csv to be created"

    summary_df = pd.read_csv(summary_path)
    assert not summary_df.empty
    assert {"kappa", "lambda_min", "u_fit", "alpha_fit", "s_fit", "rms"}.issubset(
        summary_df.columns
    )

    hysteresis_files = list(results_dir.glob("hysteresis_kappa_*.csv"))
    assert hysteresis_files, "Expected hysteresis CSV outputs"

    mode_files = list(results_dir.glob("mode_kappa_*.npy"))
    assert mode_files, "Expected eigenmode artifacts"

    discriminant_files = list(results_dir.glob("discriminant_kappa_*.csv"))
    assert discriminant_files, "Expected discriminant diagnostics"

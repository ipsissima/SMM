import subprocess
from pathlib import Path
import pandas as pd


def test_two_mode_fitting_smoke(tmp_path: Path):
    outdir = tmp_path / "two_mode"
    cmd = [
        "python",
        "supplementary_code/two_mode_fitting.py",
        "--input",
        "PSD_with_Coherence.csv",
        "--outdir",
        str(outdir),
        "--B",
        "20",
        "--seed",
        "42",
        "--threshold",
        "0.05",
        "--assume-constant",
    ]
    subprocess.check_call(cmd)

    summary = pd.read_csv(outdir / "two_mode_fit_summary.csv")
    grid = pd.read_csv(outdir / "P_L_empirical_and_fit.csv")

    assert {"parameter", "estimate", "2.5%", "97.5%"}.issubset(summary.columns)
    assert not grid.empty
    assert {"p_obs_mean", "p_model_median"}.issubset(grid.columns)

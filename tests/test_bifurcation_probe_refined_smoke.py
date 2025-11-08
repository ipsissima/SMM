import subprocess
import sys
from pathlib import Path


def test_bifurcation_probe_refined_smoke(tmp_path):
    outdir = tmp_path / "bifurcation_smoke"
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "bifurcation_probe_refined.py"),
        "--smoke",
        "--outdir",
        str(outdir),
    ]
    subprocess.run(cmd, check=True)
    assert (outdir / "cusp_grid_summary_refined.csv").exists()
    hysteresis_files = list(outdir.glob("hysteresis_kappa_*.csv"))
    assert hysteresis_files, "Expected at least one hysteresis CSV"

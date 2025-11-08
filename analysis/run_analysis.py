# analysis/run_analysis.py
import os
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analysis.utils import compute_spatial_eigenmode, laplacian_9pt_matrix
from analysis.simulation_engine import MeshSimulator
from analysis.eigenmode_projection import project_onto_mode, validate_single_mode
from analysis.identify_coeffs import numeric_identify_coeffs
from analysis.hysteresis_sweep import sweep_parameter_for_hysteresis, plot_hysteresis


def main(outdir="analysis_results", nx=32, ny=32, dx=1.0):
    os.makedirs(outdir, exist_ok=True)
    print("Building simulator")
    sim = MeshSimulator(nx=nx, ny=ny, dx=dx, c=0.015, gamma_bg=0.1)
    print("Computing eigenmode")
    e_star, eigval = compute_spatial_eigenmode(nx, ny, dx)
    np.save(os.path.join(outdir, "e_star.npy"), e_star)
    print("Eigenvalue (picked):", eigval)

    print("Validating single-mode slaving")
    A_tr, frac = validate_single_mode(sim, e_star, A0=1e-3, T=1.0, dt=0.001)
    print("Modal fraction:", frac)

    print("Identifying reduced coefficients")
    coeffs = numeric_identify_coeffs(
        sim, e_star, A0_list=np.linspace(1e-3, 5e-2, 7), Tshort=1.0, dt=0.001
    )
    print("Coefficients:", coeffs)
    pd.DataFrame(coeffs["samples"], columns=["A0", "mu", "gamma", "delta"]).to_csv(
        os.path.join(outdir, "coeff_samples.csv"), index=False
    )

    print("Running hysteresis sweep (this will take a few minutes)")
    alpha_up, A_up, alpha_down, A_down = sweep_parameter_for_hysteresis(
        param_name="gamma_bg",
        sweep_values_up=None,
        sweep_values_down=None,
        simulator=sim,
        e_star=e_star,
        dt=0.001,
        TperStep=0.5,
    )
    pd.DataFrame({"alpha_up": alpha_up, "A_up": A_up}).to_csv(
        os.path.join(outdir, "hysteresis_up.csv"), index=False
    )
    pd.DataFrame({"alpha_down": alpha_down, "A_down": A_down}).to_csv(
        os.path.join(outdir, "hysteresis_down.csv"), index=False
    )
    plot_hysteresis(alpha_up, A_up, alpha_down, A_down, outname=os.path.join(outdir, "hysteresis.png"))

    # Quick diagnostics
    Lmat = laplacian_9pt_matrix(nx, ny, dx)
    from scipy.sparse.linalg import eigsh

    vals, _ = eigsh(-Lmat, k=6, which="SM")
    vals_sorted = np.sort(np.abs(vals))
    diag = {
        "small_eigenvals": vals_sorted[:6].tolist(),
        "modal_fraction": float(frac),
        "coefficients": {
            "mu": coeffs["mu"],
            "gamma": coeffs["gamma"],
            "delta": coeffs["delta"],
        },
    }
    with open(os.path.join(outdir, "diagnostics.json"), "w") as f:
        json.dump(diag, f, indent=2)
    print("Saved results into", outdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="analysis_results")
    parser.add_argument("--nx", type=int, default=32)
    parser.add_argument("--ny", type=int, default=32)
    parser.add_argument("--dx", type=float, default=1.0)
    args = parser.parse_args()
    main(outdir=args.outdir, nx=args.nx, ny=args.ny, dx=args.dx)

#!/usr/bin/env python3
"""Advanced bifurcation probe with phase tracking, densification and analysis."""
from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

try:
    from analyze_bifurcation_results_refined import (
        compute_discriminant,
        compute_h_half,
        fit_cusp_parameters,
        format_interval,
        save_discriminant_csv,
        save_summary_pdf,
    )
except Exception:  # pragma: no cover - circular import fallback
    compute_discriminant = None  # type: ignore
    compute_h_half = None  # type: ignore
    fit_cusp_parameters = None  # type: ignore
    format_interval = None  # type: ignore
    save_discriminant_csv = None  # type: ignore
    save_summary_pdf = None  # type: ignore


LOG = logging.getLogger(__name__)


def laplacian_9pt_matrix(nx: int, ny: int, dx: float) -> csr_matrix:
    """Return the 9-point periodic Laplacian as a CSR matrix."""
    n_points = nx * ny
    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    for j in range(ny):
        for i in range(nx):
            idx = j * nx + i
            rows.append(idx)
            cols.append(idx)
            data.append(-20.0)
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ii = (i + di) % nx
                jj = (j + dj) % ny
                rows.append(idx)
                cols.append(jj * nx + ii)
                data.append(4.0)
            for di, dj in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                ii = (i + di) % nx
                jj = (j + dj) % ny
                rows.append(idx)
                cols.append(jj * nx + ii)
                data.append(1.0)
    lap = csr_matrix((data, (rows, cols)), shape=(n_points, n_points), dtype=float)
    lap /= 6.0 * dx ** 2
    return lap


def build_linear_operator(lap: csr_matrix, kappa: float, r_param: float) -> csr_matrix:
    """Return L_eff = kappa * Laplacian + r * Identity."""
    ident = sparse.identity(lap.shape[0], format="csr", dtype=float)
    return lap * float(kappa) + ident * float(r_param)


def smallest_eigenpairs(matrix: csr_matrix, k: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the smallest ``k`` eigenpairs with graceful fallback."""
    try:
        vals, vecs = eigsh(matrix, k=k, which="SA")
        order = np.argsort(vals)
        vals = vals[order]
        vecs = vecs[:, order]
        return vals, vecs
    except Exception as exc:  # pragma: no cover - rare fallback path
        LOG.warning("eigsh failed (%s); falling back to dense eigh", exc)
        dense = matrix.toarray()
        vals, vecs = np.linalg.eigh(dense)
        order = np.argsort(vals)
        vals = vals[order][:k]
        vecs = vecs[:, order][:, :k]
        return vals, vecs


def rk4_step(phi: np.ndarray, dt: float, rhs) -> np.ndarray:
    """Take one Runge-Kutta 4 step."""
    k1 = rhs(phi)
    k2 = rhs(phi + 0.5 * dt * k1)
    k3 = rhs(phi + 0.5 * dt * k2)
    k4 = rhs(phi + dt * k3)
    return phi + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate_to_steady(
    phi0: np.ndarray,
    lap: csr_matrix,
    kappa: float,
    h_value: float,
    r_param: float,
    u_param: float,
    dt: float,
    tol: float,
    Tmax_min: float,
    Tmax_max: float,
    tau_factor: float,
    lambda1_est: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Integrate gradient flow until convergence or time budget is exhausted."""
    phi = phi0.copy()
    n_points = phi.size
    rhs_const = np.ones(n_points, dtype=float) * float(h_value)

    def rhs(vec: np.ndarray) -> np.ndarray:
        lap_term = lap.dot(vec)
        return -(kappa * lap_term + r_param * vec + u_param * vec ** 3 - rhs_const)

    tau = 1.0 / max(1e-12, abs(lambda1_est))
    Tmax = max(float(Tmax_min), float(tau_factor) * tau)
    if math.isfinite(Tmax_max):
        Tmax = min(Tmax, float(Tmax_max))
    if tau_factor / max(1e-12, abs(lambda1_est)) > 1e5:
        LOG.warning(
            "Extremely slow relaxation detected (kappa=%.3f, h=%.3f, lambda1=%.3e)",
            kappa,
            h_value,
            lambda1_est,
        )
    nsteps = int(math.ceil(Tmax / dt))
    prev = phi.copy()
    for step in range(nsteps):
        phi = rk4_step(phi, dt, rhs)
        if step % 5 == 0:
            diff = np.linalg.norm(phi - prev)
            if diff < tol:
                return phi, {"steps": step + 1, "converged": 1, "diff": diff, "Tmax": Tmax}
            prev = phi.copy()
    return phi, {"steps": nsteps, "converged": 0, "diff": float(np.linalg.norm(phi - prev)), "Tmax": Tmax}


def align_vector(vec: np.ndarray, ref: np.ndarray | None) -> Tuple[np.ndarray, int]:
    """Align ``vec`` with ``ref`` to preserve phase continuity."""
    if ref is None:
        sign = 1 if np.sum(vec) >= 0 else -1
        return vec * sign, sign
    dot = float(np.dot(ref, vec))
    if dot < 0:
        return -vec, -1
    return vec, 1


def run_hysteresis_sweep(
    h_values: np.ndarray,
    indices: np.ndarray,
    lap: csr_matrix,
    kappa: float,
    r_param: float,
    u_param: float,
    dt: float,
    tol: float,
    Tmax_min: float,
    Tmax_max: float,
    tau_factor: float,
    lambda1_est: float,
    seed: int,
    v_refs: Dict[int, np.ndarray],
    w_refs: Dict[int, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """Run quasi-static sweep over ``h_values`` and project onto tracked eigenmodes."""
    rng = np.random.default_rng(seed)
    n_points = lap.shape[0]
    phi = 1e-3 * rng.standard_normal(n_points)
    amplitudes: List[float] = []
    second_amplitudes: List[float] = []
    v_signs: List[int] = []
    w_signs: List[int] = []
    v_store: Dict[int, np.ndarray] = {}
    w_store: Dict[int, np.ndarray] = {}
    v_prev: np.ndarray | None = None
    w_prev: np.ndarray | None = None
    for iter_idx, h_val in enumerate(h_values):
        grid_idx = int(indices[iter_idx])
        phi, meta = integrate_to_steady(
            phi,
            lap,
            kappa,
            h_val,
            r_param,
            u_param,
            dt,
            tol,
            Tmax_min,
            Tmax_max,
            tau_factor,
            lambda1_est,
        )
        if not meta.get("converged"):
            LOG.warning(
                "Sweep failed to converge within Tmax (kappa=%.3f, h=%.3f)",
                kappa,
                h_val,
            )
        linop = build_linear_operator(lap, kappa, r_param)
        vals, vecs = smallest_eigenpairs(linop, k=2)
        v1 = vecs[:, 0]
        v2 = vecs[:, 1]
        if grid_idx in v_refs:
            v1, sign1 = align_vector(v1, v_refs[grid_idx])
        else:
            v1, sign1 = align_vector(v1, v_prev)
            v_refs[grid_idx] = v1.copy()
        if grid_idx in w_refs:
            v2, sign2 = align_vector(v2, w_refs[grid_idx])
        else:
            v2, sign2 = align_vector(v2, w_prev)
            w_refs[grid_idx] = v2.copy()
        v_prev = v1.copy()
        w_prev = v2.copy()
        amplitudes.append(float(np.dot(phi, v1)))
        second_amplitudes.append(float(np.dot(phi, v2)))
        v_signs.append(sign1)
        w_signs.append(sign2)
    return (
        amplitudes,
        second_amplitudes,
        np.array(v_signs, dtype=int),
        np.array(w_signs, dtype=int),
        v_refs,
        w_refs,
    )


def prepare_h_grid(h_min: float, h_max: float, n_coarse: int, h_band: float, n_band: int) -> np.ndarray:
    """Return sorted unique combination of coarse and densified h values."""
    coarse = np.linspace(h_min, h_max, n_coarse)
    dense = np.linspace(-h_band, h_band, n_band)
    grid = np.unique(np.concatenate([coarse, dense]))
    return np.sort(grid)


def compute_loop_statistics(h: np.ndarray, up: np.ndarray, down: np.ndarray) -> Dict[str, float]:
    """Compute hysteresis metrics."""
    diff = up - down
    return {
        "maxdiff_signed": float(np.max(diff)),
        "maxdiff_abs": float(np.max(np.abs(diff))),
        "loop_area_signed": float(np.trapz(diff, h)),
        "loop_area_abs": float(np.trapz(np.abs(diff), h)),
        "noise_est": float(np.std(diff)),
    }


def ensure_analysis_helpers_available():
    """Ensure analysis helper functions are importable."""
    global compute_discriminant, compute_h_half, fit_cusp_parameters, format_interval
    global save_discriminant_csv, save_summary_pdf
    if fit_cusp_parameters is None:
        from analyze_bifurcation_results_refined import (
            compute_discriminant,
            compute_h_half,
            fit_cusp_parameters,
            format_interval,
            save_discriminant_csv,
            save_summary_pdf,
        )


def run_probe(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    ensure_analysis_helpers_available()
    if args.seed is not None:
        np.random.seed(args.seed)
    elif args.smoke:
        np.random.seed(0)
    outdir = Path(args.outdir)
    figures_dir = outdir / "figures"
    modes_dir = outdir / "modes"
    outdir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    modes_dir.mkdir(parents=True, exist_ok=True)
    nx = args.nx
    ny = args.ny
    dx = args.L / nx
    lap = laplacian_9pt_matrix(nx, ny, dx)
    coarse_kappas = np.linspace(args.kappa_min, args.kappa_max, args.n_kappa_coarse)
    lambda_data: List[Tuple[float, float, float]] = []
    for kappa in coarse_kappas:
        linop = build_linear_operator(lap, kappa, args.r)
        vals, _ = smallest_eigenpairs(linop, k=2)
        lambda_data.append((float(kappa), float(vals[0]), float(vals[1])))
    lambda_df = pd.DataFrame(lambda_data, columns=["kappa", "lambda1", "lambda2"])
    kappa0 = float(lambda_df.loc[lambda_df["lambda1"].idxmin(), "kappa"])
    delta = min(args.refine_factor, max(1e-6, args.kappa_max - args.kappa_min) / 2.0)
    refined_min = max(args.kappa_min, kappa0 - delta)
    refined_max = min(args.kappa_max, kappa0 + delta)
    refined_kappas = np.linspace(refined_min, refined_max, args.n_kappa_refined)
    summary_records: List[Dict[str, float | int | str]] = []
    lambda_plot_data: List[Tuple[float, float, float]] = []
    rng = np.random.default_rng(args.seed if args.seed is not None else (0 if args.smoke else None))
    h_grid = prepare_h_grid(args.h_min, args.h_max, args.n_h_coarse, args.h_band, args.n_h_band)
    for kappa in refined_kappas:
        linop = build_linear_operator(lap, kappa, args.r)
        lambdas, vecs = smallest_eigenpairs(linop, k=2)
        lambda1 = float(lambdas[0])
        lambda2 = float(lambdas[1])
        gap = float(lambda2 - lambda1)
        lambda_plot_data.append((kappa, lambda1, lambda2))
        v_refs: Dict[int, np.ndarray] = {}
        w_refs: Dict[int, np.ndarray] = {}
        base_indices = np.arange(h_grid.size)
        amplitudes_up, second_up, v_signs_up, w_signs_up, v_refs, w_refs = run_hysteresis_sweep(
            h_grid,
            base_indices,
            lap,
            kappa,
            args.r,
            args.u,
            args.dt,
            args.tol,
            args.Tmax_min,
            args.Tmax_max,
            args.tau_factor,
            lambda1,
            seed=rng.integers(0, 10_000_000),
            v_refs=v_refs,
            w_refs=w_refs,
        )
        amplitudes_up = np.array(amplitudes_up)
        second_up = np.array(second_up)
        amplitudes_down, second_down, v_signs_down, w_signs_down, _, _ = run_hysteresis_sweep(
            h_grid[::-1],
            base_indices[::-1],
            lap,
            kappa,
            args.r,
            args.u,
            args.dt,
            args.tol,
            args.Tmax_min,
            args.Tmax_max,
            args.tau_factor,
            lambda1,
            seed=rng.integers(0, 10_000_000),
            v_refs=v_refs,
            w_refs=w_refs,
        )
        amplitudes_down = np.array(amplitudes_down)[::-1]
        second_down = np.array(second_down)[::-1]
        hysteresis_df = pd.DataFrame(
            {
                "h": h_grid,
                "A_up": amplitudes_up,
                "A_down": amplitudes_down,
                "B_up": second_up,
                "B_down": second_down,
                "v1_sign_up": v_signs_up,
                "v2_sign_up": w_signs_up,
                "v1_sign_down": v_signs_down[::-1],
                "v2_sign_down": w_signs_down[::-1],
            }
        )
        hysteresis_path = outdir / f"hysteresis_kappa_{kappa:.3f}.csv"
        hysteresis_df.to_csv(hysteresis_path, index=False)
        modes_path = modes_dir / f"modes_kappa_{kappa:.3f}.npz"
        v_matrix = np.stack([v_refs[idx] for idx in range(h_grid.size)], axis=0)
        w_matrix = np.stack([w_refs[idx] for idx in range(h_grid.size)], axis=0)
        np.savez_compressed(
            modes_path,
            v1s=v_matrix,
            v2s=w_matrix,
            lambdas=np.tile(lambdas, (h_grid.size, 1)),
        )
        stats = compute_loop_statistics(h_grid, amplitudes_up, amplitudes_down)
        ensure_analysis_helpers_available()
        cusp_fit = fit_cusp_parameters(
            amplitudes_up,
            amplitudes_down,
            h_grid,
            bootstrap_B=args.bootstrap_B,
            seed=args.seed if args.seed is not None else (0 if args.smoke else None),
        )
        discriminant = compute_discriminant(h_grid, cusp_fit["alpha"], cusp_fit["s"])
        save_discriminant_csv(outdir, kappa, discriminant)
        plt.figure(figsize=(6, 4))
        plt.plot(h_grid, amplitudes_up, label="up", marker="o", ms=3)
        plt.plot(h_grid, amplitudes_down, label="down", marker="s", ms=3)
        plt.xlabel("h")
        plt.ylabel("A")
        plt.title(f"Hysteresis at kappa={kappa:.3f}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures_dir / f"hysteresis_kappa_{kappa:.3f}.png", dpi=200)
        plt.close()
        plt.figure(figsize=(6, 4))
        plt.plot(h_grid, discriminant["Delta"], label="Delta")
        plt.axhline(0, color="k", linestyle="--", linewidth=1)
        plt.xlabel("h")
        plt.ylabel("Delta(h)")
        plt.title(f"Discriminant at kappa={kappa:.3f}")
        plt.tight_layout()
        plt.savefig(figures_dir / f"discriminant_map_kappa_{kappa:.3f}.png", dpi=200)
        plt.close()
        dynamic_threshold = args.gap_threshold
        if dynamic_threshold is None:
            dynamic_threshold = 5.0 * abs(lambda1)
        two_mode_flag = int(gap <= dynamic_threshold)
        rms1 = math.nan
        rms2 = math.nan
        if two_mode_flag:
            X1 = np.column_stack(
                [
                    amplitudes_up ** 3,
                    amplitudes_up,
                    amplitudes_up * second_up,
                    second_up,
                    h_grid,
                ]
            )
            X2 = np.column_stack(
                [
                    second_up ** 3,
                    second_up,
                    amplitudes_up * second_up,
                    amplitudes_up,
                    h_grid,
                ]
            )
            coeff1, residuals1, _, _ = np.linalg.lstsq(X1, np.zeros_like(amplitudes_up), rcond=None)
            coeff2, residuals2, _, _ = np.linalg.lstsq(X2, np.zeros_like(second_up), rcond=None)
            pred1 = X1 @ coeff1
            pred2 = X2 @ coeff2
            rms1 = float(np.sqrt(np.mean(pred1 ** 2)))
            rms2 = float(np.sqrt(np.mean(pred2 ** 2)))
            if rms1 < 1e-4 and rms2 < 1e-4:
                LOG.info(
                    "Two-mode fit explains data at kappa=%.3f (rms=%.2e, %.2e)",
                    kappa,
                    rms1,
                    rms2,
                )
        record = {
            "kappa": float(kappa),
            "lambda1": lambda1,
            "lambda2": lambda2,
            "gap": gap,
            "meanA": float(np.mean(np.concatenate([amplitudes_up, amplitudes_down]))),
            "maxdiff_signed": stats["maxdiff_signed"],
            "maxdiff_abs": stats["maxdiff_abs"],
            "loop_area_signed": stats["loop_area_signed"],
            "loop_area_abs": stats["loop_area_abs"],
            "noise_est": stats["noise_est"],
            "u": cusp_fit["u"],
            "alpha": cusp_fit["alpha"],
            "s": cusp_fit["s"],
            "rms_fit": cusp_fit["rms_fit"],
            "sv_min": cusp_fit["sv_min"],
            "alpha_CI_low": cusp_fit["alpha_CI"][0],
            "alpha_CI_high": cusp_fit["alpha_CI"][1],
            "s_CI_low": cusp_fit["s_CI"][0],
            "s_CI_high": cusp_fit["s_CI"][1],
            "h_half": cusp_fit["h_half"],
            "h_half_CI_low": cusp_fit["h_half_CI"][0],
            "h_half_CI_high": cusp_fit["h_half_CI"][1],
            "two_mode_flag": two_mode_flag,
            "two_mode_rms1": rms1,
            "two_mode_rms2": rms2,
        }
        summary_records.append(record)
    summary_df = pd.DataFrame(summary_records)
    summary_df.sort_values("kappa", inplace=True)
    summary_path = outdir / "cusp_grid_summary_refined.csv"
    summary_df.to_csv(summary_path, index=False)
    lambda_df_refined = pd.DataFrame(lambda_plot_data, columns=["kappa", "lambda1", "lambda2"])
    lambda_df_refined.sort_values("kappa", inplace=True)
    plt.figure(figsize=(6, 4))
    plt.plot(lambda_df_refined["kappa"], lambda_df_refined["lambda1"], label="lambda1")
    plt.plot(lambda_df_refined["kappa"], lambda_df_refined["lambda2"], label="lambda2")
    plt.xlabel("kappa")
    plt.ylabel("lambda")
    plt.title("Eigenvalues vs kappa")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "lambda_vs_kappa.png", dpi=200)
    plt.close()
    plt.figure(figsize=(6, 4))
    plt.plot(summary_df["kappa"], summary_df["maxdiff_abs"], marker="o")
    plt.xlabel("kappa")
    plt.ylabel("max |A_up - A_down|")
    plt.title("Max hysteresis width")
    plt.tight_layout()
    plt.savefig(figures_dir / "maxdiff_vs_kappa.png", dpi=200)
    plt.close()
    plt.figure(figsize=(6, 4))
    kappas = summary_df["kappa"].to_numpy()
    alpha_vals = summary_df["alpha"].to_numpy()
    alpha_low = summary_df["alpha_CI_low"].to_numpy()
    alpha_high = summary_df["alpha_CI_high"].to_numpy()
    plt.plot(kappas, alpha_vals, marker="o", label="alpha")
    plt.fill_between(kappas, alpha_low, alpha_high, alpha=0.3, label="95% CI")
    plt.xlabel("kappa")
    plt.ylabel("alpha")
    plt.title("Cusp coefficient alpha")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "alpha_vs_kappa.png", dpi=200)
    plt.close()
    if save_summary_pdf is not None:
        save_summary_pdf(outdir, summary_df)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Advanced bifurcation probe")
    parser.add_argument("--nx", type=int, default=32)
    parser.add_argument("--ny", type=int, default=32)
    parser.add_argument("--L", type=float, default=32.0)
    parser.add_argument("--kappa_min", type=float, default=0.0)
    parser.add_argument("--kappa_max", type=float, default=0.6)
    parser.add_argument("--n_kappa_coarse", "--n-kappa-coarse", dest="n_kappa_coarse", type=int, default=13)
    parser.add_argument("--refine_factor", type=float, default=0.15)
    parser.add_argument("--n_kappa_refined", type=int, default=13)
    parser.add_argument("--h_min", type=float, default=-0.8)
    parser.add_argument("--h_max", type=float, default=0.8)
    parser.add_argument("--n_h_coarse", "--n-h-coarse", dest="n_h_coarse", type=int, default=33)
    parser.add_argument("--n_h_band", type=int, default=101)
    parser.add_argument("--h_band", type=float, default=0.2)
    parser.add_argument("--r", type=float, default=-0.2)
    parser.add_argument("--u", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--Tmax_min", "--tmax-min", dest="Tmax_min", type=float, default=20.0)
    parser.add_argument("--Tmax_max", "--tmax-max", dest="Tmax_max", type=float, default=120.0)
    parser.add_argument("--tau_factor", type=float, default=30.0)
    parser.add_argument("--gap_threshold", type=float, default=None)
    parser.add_argument("--bootstrap_B", "--bootstrap-b", dest="bootstrap_B", type=int, default=200)
    parser.add_argument("--outdir", "--results-dir", dest="outdir", type=str, default="bifurcation_results_refined")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="Run reduced smoke configuration")
    args = parser.parse_args(argv)
    if args.smoke:
        args.nx = 16
        args.ny = 16
        args.n_kappa_coarse = 5
        args.n_kappa_refined = 5
        args.n_h_coarse = 9
        args.n_h_band = 41
        args.Tmax_min = 10.0
        args.Tmax_max = min(args.Tmax_max, 60.0)
        args.bootstrap_B = min(args.bootstrap_B, 50)
        if args.gap_threshold is None:
            args.gap_threshold = 1.0
    return args


if __name__ == "__main__":
    run_probe(parse_args())

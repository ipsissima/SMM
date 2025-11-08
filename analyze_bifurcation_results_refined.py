#!/usr/bin/env python3
"""Analysis utilities for the advanced bifurcation probe."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def format_interval(interval: Tuple[float, float]) -> str:
    """Format a numeric interval for reporting."""
    low, high = interval
    if math.isnan(low) or math.isnan(high):
        return "[nan, nan]"
    return f"[{low:.4g}, {high:.4g}]"


def _fit_nullspace(A: np.ndarray, h: np.ndarray) -> Dict[str, float]:
    A_center = A - np.mean(A)
    h_center = h - np.mean(h)
    A_std = float(np.std(A_center)) or 1.0
    h_std = float(np.std(h_center)) or 1.0
    A_scaled = A_center / A_std
    h_scaled = h_center / h_std
    X = np.vstack([A_scaled ** 3, A_scaled, h_scaled]).T
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    p = Vt[-1, :]
    if p[0] < 0:
        p = -p
    residual = X @ p
    rms = float(np.sqrt(np.mean(residual ** 2)))
    sv_min = float(S[-1])
    ratio = float(S[-1] / S[0]) if S[0] > 0 else math.inf
    denom = A_std ** 3
    if abs(p[0]) < 1e-14 or denom == 0:
        u_phys = math.nan
    else:
        u_phys = p[0] / denom
    if u_phys == 0 or math.isnan(u_phys):
        alpha = math.nan
        s = math.nan
        u_norm = math.nan
    else:
        alpha = (p[1] / A_std) / u_phys
        s = (p[2] / h_std) / u_phys
        u_norm = 1.0
    return {
        "u": u_norm,
        "alpha": alpha,
        "s": s,
        "rms_fit": rms,
        "sv_min": sv_min,
        "sv_ratio": ratio,
        "meanA": float(np.mean(A)),
        "stdA": A_std,
        "stdh": h_std,
    }


def compute_h_half(alpha: float, s: float) -> float:
    """Return cusp half-width if defined."""
    if alpha is None or s is None:
        return math.nan
    if math.isnan(alpha) or math.isnan(s):
        return math.nan
    if alpha >= 0 or abs(s) < 1e-12:
        return math.nan
    value = -4.0 * (alpha ** 3) / (27.0 * (s ** 2))
    if value <= 0:
        return math.nan
    return math.sqrt(value)


def fit_cusp_parameters(
    A_up: np.ndarray,
    A_down: np.ndarray,
    h: np.ndarray,
    bootstrap_B: int = 200,
    seed: int | None = None,
) -> Dict[str, float | Tuple[float, float]]:
    """Fit the Thom cusp normal form coefficients with bootstrap confidence intervals."""
    A = np.concatenate([A_up, A_down])
    h_all = np.concatenate([h, h])
    fit = _fit_nullspace(A, h_all)
    rng = np.random.default_rng(seed)
    alphas: List[float] = []
    ss: List[float] = []
    half_widths: List[float] = []
    if bootstrap_B > 0:
        n = A.size
        for _ in range(bootstrap_B):
            idx = rng.integers(0, n, size=n)
            sample_fit = _fit_nullspace(A[idx], h_all[idx])
            alphas.append(sample_fit["alpha"])
            ss.append(sample_fit["s"])
            half_widths.append(compute_h_half(sample_fit["alpha"], sample_fit["s"]))
    alpha_ci = (math.nan, math.nan)
    s_ci = (math.nan, math.nan)
    hhalf_ci = (math.nan, math.nan)
    if alphas:
        alpha_ci = (float(np.nanquantile(alphas, 0.025)), float(np.nanquantile(alphas, 0.975)))
    if ss:
        s_ci = (float(np.nanquantile(ss, 0.025)), float(np.nanquantile(ss, 0.975)))
    if half_widths:
        hhalf_ci = (float(np.nanquantile(half_widths, 0.025)), float(np.nanquantile(half_widths, 0.975)))
    fit["alpha_CI"] = alpha_ci
    fit["s_CI"] = s_ci
    h_half = compute_h_half(fit["alpha"], fit["s"])
    fit["h_half"] = h_half
    fit["h_half_CI"] = hhalf_ci
    return fit


def compute_discriminant(h: np.ndarray, alpha: float, s: float) -> pd.DataFrame:
    """Return discriminant map for the fitted coefficients."""
    beta = s * h
    Delta = -4.0 * alpha ** 3 - 27.0 * beta ** 2
    return pd.DataFrame({"h": h, "beta": beta, "Delta": Delta})


def save_discriminant_csv(outdir: Path, kappa: float, df: pd.DataFrame) -> None:
    """Persist discriminant data for a specific kappa."""
    path = Path(outdir) / f"discriminant_kappa_{kappa:.3f}.csv"
    df.to_csv(path, index=False)


def _verdict(summary_df: pd.DataFrame) -> Tuple[str, Dict[str, float]]:
    valid = summary_df[np.isfinite(summary_df["h_half"])]
    if valid.empty:
        return "NO CUSP FOUND", {}
    best = valid.loc[valid["h_half"].idxmax()]
    info = {
        "kappa": float(best["kappa"]),
        "h_half": float(best["h_half"]),
        "alpha": float(best["alpha"]),
        "alpha_ci": (float(best["alpha_CI_low"]), float(best["alpha_CI_high"])),
        "s": float(best["s"]),
        "s_ci": (float(best["s_CI_low"]), float(best["s_CI_high"])),
        "h_ci": (
            float(best["h_half_CI_low"]),
            float(best["h_half_CI_high"]),
        ),
    }
    return "CUSP FOUND", info


def save_summary_pdf(outdir: Path, summary_df: pd.DataFrame) -> None:
    """Create a one-page verdict PDF with supporting figures."""
    outdir = Path(outdir)
    pdf_path = outdir / "bifurcation_verdict_refined.pdf"
    verdict, info = _verdict(summary_df)
    with PdfPages(pdf_path) as pdf:
        fig, ax = plt.subplots(figsize=(8.5, 6))
        ax.axis("off")
        text_lines = [f"Advanced bifurcation verdict: {verdict}"]
        if info:
            text_lines.append(f"Best kappa: {info['kappa']:.4f}")
            text_lines.append(f"Cusp half-width h*: {info['h_half']:.4g}")
            text_lines.append(f"h* 95% CI: {format_interval(info['h_ci'])}")
            text_lines.append(f"alpha: {info['alpha']:.4g} (CI {format_interval(info['alpha_ci'])})")
            text_lines.append(f"s: {info['s']:.4g} (CI {format_interval(info['s_ci'])})")
        else:
            text_lines.append("No reliable cusp signature detected within scanned parameters.")
        ax.text(0.02, 0.98, "\n".join(text_lines), va="top", ha="left", fontsize=12)
        pdf.savefig(fig)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(summary_df["kappa"], summary_df["lambda1"], label=r"$\lambda_1$")
        ax.plot(summary_df["kappa"], summary_df["lambda2"], label=r"$\lambda_2$")
        ax.set_xlabel("kappa")
        ax.set_ylabel("lambda")
        ax.set_title("Eigenvalues vs kappa")
        ax.legend()
        pdf.savefig(fig)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(summary_df["kappa"], summary_df["maxdiff_abs"], marker="o")
        ax.set_xlabel("kappa")
        ax.set_ylabel("max |A_up - A_down|")
        ax.set_title("Hysteresis width")
        pdf.savefig(fig)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(summary_df["kappa"], summary_df["alpha"], marker="o", label="alpha")
        ax.fill_between(
            summary_df["kappa"],
            summary_df["alpha_CI_low"],
            summary_df["alpha_CI_high"],
            alpha=0.3,
            label="95% CI",
        )
        ax.set_xlabel("kappa")
        ax.set_ylabel("alpha")
        ax.set_title("Cusp coefficient")
        ax.legend()
        pdf.savefig(fig)
        plt.close(fig)


def _collect_files(indir: Path) -> List[Tuple[float, Path, Path]]:
    hysteresis_files = sorted(Path(indir).glob("hysteresis_kappa_*.csv"))
    records = []
    for file in hysteresis_files:
        kappa_str = file.stem.split("_")[-1]
        try:
            kappa = float(kappa_str)
        except ValueError:
            continue
        mode_file = Path(indir) / "modes" / f"modes_kappa_{kappa:.3f}.npz"
        records.append((kappa, file, mode_file))
    return records


def analyze_directory(
    indir: Path,
    refresh: bool = False,
    bootstrap_B: int = 200,
    seed: int | None = None,
) -> pd.DataFrame:
    indir = Path(indir)
    summary_path = indir / "cusp_grid_summary_refined.csv"
    if not refresh and summary_path.exists():
        summary_df = pd.read_csv(summary_path)
        save_summary_pdf(indir, summary_df)
        return summary_df
    records = []
    for kappa, h_path, mode_path in _collect_files(indir):
        df = pd.read_csv(h_path)
        if not mode_path.exists():
            raise FileNotFoundError(f"Missing mode file for kappa={kappa:.3f}: {mode_path}")
        modes = np.load(mode_path)
        lambdas = modes["lambdas"][0]
        lambda1 = float(lambdas[0])
        lambda2 = float(lambdas[1])
        gap = lambda2 - lambda1
        cusp_fit = fit_cusp_parameters(
            df["A_up"].to_numpy(),
            df["A_down"].to_numpy(),
            df["h"].to_numpy(),
            bootstrap_B=bootstrap_B,
            seed=seed,
        )
        stats = {
            "kappa": kappa,
            "lambda1": lambda1,
            "lambda2": lambda2,
            "gap": gap,
            "maxdiff_signed": float(np.max(df["A_up"] - df["A_down"])),
            "maxdiff_abs": float(np.max(np.abs(df["A_up"] - df["A_down"]))),
            "loop_area_signed": float(np.trapz(df["A_up"] - df["A_down"], df["h"])),
            "loop_area_abs": float(np.trapz(np.abs(df["A_up"] - df["A_down"]), df["h"])),
            "meanA": float(np.mean(np.concatenate([df["A_up"], df["A_down"]]))),
            "noise_est": float(np.std(df["A_up"] - df["A_down"])),
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
            "two_mode_flag": int(gap <= 5.0 * abs(lambda1)),
            "two_mode_rms1": math.nan,
            "two_mode_rms2": math.nan,
        }
        records.append(stats)
        disc = compute_discriminant(df["h"].to_numpy(), cusp_fit["alpha"], cusp_fit["s"])
        save_discriminant_csv(indir, kappa, disc)
    summary_df = pd.DataFrame(records)
    summary_df.sort_values("kappa", inplace=True)
    summary_df.to_csv(summary_path, index=False)
    save_summary_pdf(indir, summary_df)
    return summary_df


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze refined bifurcation probe outputs")
    parser.add_argument("--indir", type=str, default="bifurcation_results_refined")
    parser.add_argument("--refresh", action="store_true", help="Recompute summary from hysteresis files")
    parser.add_argument("--bootstrap_B", type=int, default=200)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    analyze_directory(Path(args.indir), refresh=args.refresh, bootstrap_B=args.bootstrap_B, seed=args.seed)


if __name__ == "__main__":
    main()

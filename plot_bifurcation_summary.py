"""Plotting utilities for bifurcation probe outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_summary(results_dir: Path) -> pd.DataFrame:
    summary_path = results_dir / "cusp_grid_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary CSV not found at {summary_path}")
    return pd.read_csv(summary_path)


def collect_hysteresis_paths(results_dir: Path) -> List[Path]:
    return sorted(results_dir.glob("hysteresis_kappa_*.csv"))


def plot_summary_curves(summary: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(6, 4))
    sns.lineplot(data=summary, x="kappa", y="lambda_min", marker="o")
    plt.title("Leading eigenvalue vs. kappa")
    plt.tight_layout()
    plt.savefig(outdir / "lambda_vs_kappa.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.lineplot(data=summary, x="kappa", y="maxdiff", marker="s")
    plt.ylabel("max |A_up - A_down|")
    plt.title("Hysteresis extent")
    plt.tight_layout()
    plt.savefig(outdir / "maxdiff_vs_kappa.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.lineplot(data=summary, x="kappa", y="alpha_fit", marker="^")
    plt.title("Normal form alpha vs. kappa")
    plt.tight_layout()
    plt.savefig(outdir / "alpha_vs_kappa.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.lineplot(data=summary, x="kappa", y="s_fit", marker="D")
    plt.title("Drive coupling s vs. kappa")
    plt.tight_layout()
    plt.savefig(outdir / "s_vs_kappa.png", dpi=200)
    plt.close()


def plot_hysteresis_curves(hysteresis_paths: Iterable[Path], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    for csv_path in hysteresis_paths:
        df = pd.read_csv(csv_path)
        kappa = float(csv_path.stem.split("_")[-1])
        plt.figure(figsize=(6, 4))
        sns.lineplot(data=df, x="h", y="A_up", marker="o", label="A_up")
        sns.lineplot(data=df, x="h", y="A_down", marker="s", label="A_down")
        plt.title(f"Hysteresis for kappa={kappa:.3f}")
        plt.xlabel("h")
        plt.ylabel("Mode amplitude A")
        plt.tight_layout()
        plt.savefig(outdir / f"hysteresis_kappa_{kappa:.3f}.png", dpi=200)
        plt.close()


def plot_discriminant(results_dir: Path, outdir: Path) -> None:
    disc_paths = sorted(results_dir.glob("discriminant_kappa_*.csv"))
    if not disc_paths:
        return
    outdir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    for disc_path in disc_paths:
        df = pd.read_csv(disc_path)
        kappa = float(disc_path.stem.split("_")[-1])
        plt.figure(figsize=(6, 4))
        sns.lineplot(data=df, x="h", y="Delta")
        plt.title(f"Discriminant Δ(h) for kappa={kappa:.3f}")
        plt.xlabel("h")
        plt.ylabel("Δ")
        plt.tight_layout()
        plt.savefig(outdir / f"discriminant_kappa_{kappa:.3f}.png", dpi=200)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise bifurcation probe outputs")
    parser.add_argument("--outdir", type=Path, default=Path("bifurcation_results"), help="Results directory")
    parser.add_argument(
        "--hysteresis", action="store_true", help="Also render individual hysteresis traces"
    )
    args = parser.parse_args()

    results_dir = args.outdir
    figures_dir = results_dir / "figures"

    summary = load_summary(results_dir)
    plot_summary_curves(summary, figures_dir)

    if args.hysteresis:
        plot_hysteresis_curves(collect_hysteresis_paths(results_dir), figures_dir / "hysteresis")

    plot_discriminant(results_dir, figures_dir / "discriminant")


if __name__ == "__main__":  # pragma: no cover
    main()

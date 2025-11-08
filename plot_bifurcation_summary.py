"""Helper utilities to visualize results produced by ``bifurcation_probe.py``."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_summary(summary_path: Path, output_dir: Path) -> None:
    df = pd.read_csv(summary_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.plot(df["kappa"], df["max_hysteresis_A_diff"], marker="o")
    plt.xlabel("kappa")
    plt.ylabel("max |A_up - A_down|")
    plt.title("Cusp wedge extent")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_dir / "cusp_grid_summary.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(df["kappa"], df["alpha_fit"], marker="s")
    plt.xlabel("kappa")
    plt.ylabel("alpha_fit")
    plt.title("Normal form alpha(kappa)")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_dir / "alpha_vs_kappa.png", dpi=200)
    plt.close()


def plot_hysteresis(results_dir: Path, kappa: float, output_dir: Path) -> None:
    csv_path = results_dir / f"hysteresis_kappa_{kappa:.3f}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not locate hysteresis CSV for kappa={kappa:.3f} at {csv_path}")
    df = pd.read_csv(csv_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(df["h"], df["A_up"], label="A_up", marker="o")
    plt.plot(df["h"], df["A_down"], label="A_down", marker="s")
    plt.xlabel("h")
    plt.ylabel("Mode amplitude A")
    plt.title(f"Hysteresis for kappa={kappa:.3f}")
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_dir / f"hysteresis_kappa_{kappa:.3f}.png", dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot outputs from bifurcation_probe")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("bifurcation_results"),
        help="Directory containing cusp_grid_summary.csv",
    )
    parser.add_argument(
        "--kappa",
        type=float,
        default=None,
        help="If provided, generate a hysteresis plot for this kappa value",
    )
    args = parser.parse_args()

    summary_path = args.results_dir / "cusp_grid_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary CSV not found at {summary_path}")

    plot_summary(summary_path, args.results_dir)
    if args.kappa is not None:
        plot_hysteresis(args.results_dir, args.kappa, args.results_dir)


if __name__ == "__main__":  # pragma: no cover - manual execution path
    main()

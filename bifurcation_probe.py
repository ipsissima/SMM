"""bifurcation_probe
======================

Probe the Syncytial Mesh Model (SMM) for a Thom cusp-type catastrophe by
projecting the mesh dynamics onto the principal soft eigenmode of the linearized
operator.  The script constructs the same nine-point periodic Laplacian used by
``simulate_mesh_psd.py`` and performs quasi-static hysteresis ramps over applied
field strength ``h`` and coupling ``kappa``.  The reduced equilibria are fit to a
cubic Thom normal form, providing a decisive check for the existence of the cusp
wedge in ``(kappa, h)`` space.

The implementation follows the high-level requirements described in the project
prompt and saves all numerical and graphical outputs to a
``bifurcation_results/`` directory.
"""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.linalg import lstsq

matplotlib.use("Agg")  # ensure headless-friendly back-end
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is optional
    tqdm = None


LOGGER = logging.getLogger(__name__)


@dataclass
class ProbeConfig:
    """Container for runtime configuration parameters."""

    nx: int
    ny: int
    length: float
    kappa_min: float
    kappa_max: float
    n_kappa: int
    h_min: float
    h_max: float
    n_h: int
    r: float
    u: float
    dt: float
    tmax: float
    tol: float
    seed: int
    results_dir: Path
    save_plots: bool
    kappa_plot: float | None

    @property
    def dx(self) -> float:
        return self.length / float(self.nx)

    @property
    def total_nodes(self) -> int:
        return self.nx * self.ny


def laplacian_9pt_matrix(nx: int, ny: int, dx: float) -> sp.csr_matrix:
    """Construct the nine-point periodic Laplacian used by ``simulate_mesh_psd``.

    Parameters
    ----------
    nx, ny:
        Grid dimensions along ``x`` and ``y`` respectively.
    dx:
        Spatial step size (assumed identical along both axes).

    Returns
    -------
    scipy.sparse.csr_matrix
        Symmetric sparse matrix encoding the nine-point Laplacian with periodic
        wrap-around boundary conditions.
    """

    coeff_center = -20.0
    coeff_orth = 4.0
    coeff_diag = 1.0
    scale = 1.0 / (6.0 * dx * dx)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    def idx(ix: int, iy: int) -> int:
        return iy * nx + ix

    for iy in range(ny):
        for ix in range(nx):
            center = idx(ix, iy)
            rows.append(center)
            cols.append(center)
            data.append(coeff_center)

            orth_neighbors = [
                ((ix + 1) % nx, iy),
                ((ix - 1) % nx, iy),
                (ix, (iy + 1) % ny),
                (ix, (iy - 1) % ny),
            ]
            for jx, jy in orth_neighbors:
                rows.append(center)
                cols.append(idx(jx, jy))
                data.append(coeff_orth)

            diag_neighbors = [
                ((ix + 1) % nx, (iy + 1) % ny),
                ((ix + 1) % nx, (iy - 1) % ny),
                ((ix - 1) % nx, (iy + 1) % ny),
                ((ix - 1) % nx, (iy - 1) % ny),
            ]
            for jx, jy in diag_neighbors:
                rows.append(center)
                cols.append(idx(jx, jy))
                data.append(coeff_diag)

    matrix = sp.coo_matrix((np.array(data) * scale, (rows, cols)), shape=(nx * ny, nx * ny))
    return matrix.tocsr()


def effective_operator(lap: sp.csr_matrix, kappa: float, r: float) -> sp.csr_matrix:
    """Return the linearized operator ``L_eff = kappa * Lap + r * I``."""

    n = lap.shape[0]
    return kappa * lap + r * sp.eye(n, format="csr")


def leading_mode(l_eff: sp.csr_matrix) -> Tuple[float, np.ndarray]:
    """Compute the smallest eigenvalue and corresponding normalized eigenmode."""

    n = l_eff.shape[0]
    try:
        eigvals, eigvecs = spla.eigsh(l_eff, k=1, which="SA")
        eigenvalue = float(np.real(eigvals[0]))
        mode = np.real(eigvecs[:, 0])
    except Exception as exc:  # pragma: no cover - fallback path
        LOGGER.warning("eigsh failed (%s); switching to dense solver", exc)
        dense = l_eff.toarray()
        eigvals_dense, eigvecs_dense = np.linalg.eigh(dense)
        idx_min = int(np.argmin(eigvals_dense))
        eigenvalue = float(eigvals_dense[idx_min])
        mode = np.real(eigvecs_dense[:, idx_min])

    norm = np.linalg.norm(mode)
    if not math.isfinite(norm) or norm == 0.0:
        raise RuntimeError("Failed to obtain a finite eigenmode norm")
    mode /= norm
    return eigenvalue, mode


def project_onto_mode(phi: np.ndarray, mode: np.ndarray) -> float:
    """Return the scalar projection of ``phi`` onto the normalized ``mode``."""

    return float(np.dot(phi, mode))


def integrate_to_steady(
    phi0: np.ndarray,
    lap: sp.csr_matrix,
    kappa: float,
    r: float,
    u: float,
    h: float,
    dt: float,
    tmax: float,
    tol: float,
) -> np.ndarray:
    """Integrate the gradient-flow equation to a steady state using RK4."""

    phi = phi0.astype(float, copy=True)
    max_steps = max(1, int(math.ceil(tmax / dt)))

    def rhs(field: np.ndarray) -> np.ndarray:
        nonlinear = u * np.power(field, 3)
        linear = kappa * lap.dot(field) + r * field
        return -(linear + nonlinear - h)

    for _ in range(max_steps):
        k1 = rhs(phi)
        k2 = rhs(phi + 0.5 * dt * k1)
        k3 = rhs(phi + 0.5 * dt * k2)
        k4 = rhs(phi + dt * k3)
        phi_new = phi + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if np.linalg.norm(phi_new - phi) < tol:
            phi = phi_new
            break
        phi = phi_new
    return phi


def hysteresis_for_kappa(
    kappa: float,
    lap: sp.csr_matrix,
    h_values: np.ndarray,
    mode: np.ndarray,
    r: float,
    u: float,
    dt: float,
    tmax: float,
    tol: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Perform an up-down hysteresis sweep over ``h`` for a fixed ``kappa``."""

    n = lap.shape[0]
    phi = 0.01 * rng.standard_normal(n)
    amplitudes_up = []

    for h in h_values:
        phi = integrate_to_steady(phi, lap, kappa, r, u, h, dt, tmax, tol)
        amplitudes_up.append(project_onto_mode(phi, mode))

    amplitudes_down = []
    # Start from the final state reached at the largest ``h``
    for h in reversed(h_values):
        phi = integrate_to_steady(phi, lap, kappa, r, u, h, dt, tmax, tol)
        amplitudes_down.append(project_onto_mode(phi, mode))
    amplitudes_down.reverse()

    return np.array(amplitudes_up), np.array(amplitudes_down)


def fit_thom_normal_form(amplitudes: np.ndarray, h_values: np.ndarray) -> Tuple[float, float, float]:
    """Fit the cubic Thom normal form ``u * A^3 + alpha * A + h = 0``.

    Returns ``(u_fit, alpha_fit, rmse)``.
    """

    if amplitudes.size != h_values.size:
        raise ValueError("Amplitudes and field arrays must have identical sizes")

    X = np.column_stack((np.power(amplitudes, 3), amplitudes))
    y = -h_values
    solution, residuals, rank, _ = lstsq(X, y, rcond=None)
    u_fit, alpha_fit = solution
    if residuals.size > 0:
        rss = residuals[0]
        rmse = math.sqrt(rss / float(len(amplitudes)))
    else:
        residual = X @ solution - y
        rmse = math.sqrt(float(np.mean(np.square(residual))))
    return float(u_fit), float(alpha_fit), rmse


def save_hysteresis_csv(
    results_dir: Path,
    kappa: float,
    h_values: np.ndarray,
    up: np.ndarray,
    down: np.ndarray,
) -> Path:
    """Persist a per-kappa hysteresis sweep to CSV and return its path."""

    df = pd.DataFrame({"h": h_values, "A_up": up, "A_down": down})
    csv_path = results_dir / f"hysteresis_kappa_{kappa:.3f}.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def save_mode(results_dir: Path, kappa: float, mode: np.ndarray) -> Path:
    """Save a normalized eigenmode to disk."""

    mode_path = results_dir / f"mode_kappa_{kappa:.3f}.npy"
    np.save(mode_path, mode)
    return mode_path


def plot_hysteresis(results_dir: Path, kappa: float, h_values: np.ndarray, up: np.ndarray, down: np.ndarray) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(h_values, up, label="A_up", marker="o")
    plt.plot(h_values, down, label="A_down", marker="s")
    plt.xlabel("h")
    plt.ylabel("Mode amplitude A")
    plt.title(f"Hysteresis for kappa={kappa:.3f}")
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(results_dir / f"hysteresis_kappa_{kappa:.3f}.png", dpi=200)
    plt.close()


def plot_summary(results_dir: Path, summary_df: pd.DataFrame) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(summary_df["kappa"], summary_df["max_hysteresis_A_diff"], marker="o")
    plt.xlabel("kappa")
    plt.ylabel("max |A_up - A_down|")
    plt.title("Cusp wedge extent")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(results_dir / "cusp_grid_summary.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(summary_df["kappa"], summary_df["alpha_fit"], marker="s")
    plt.xlabel("kappa")
    plt.ylabel("alpha_fit")
    plt.title("Normal form alpha(kappa)")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(results_dir / "alpha_vs_kappa.png", dpi=200)
    plt.close()


def run_probe(config: ProbeConfig) -> pd.DataFrame:
    LOGGER.info("Constructing Laplacian for %dx%d grid (dx=%.3f)", config.nx, config.ny, config.dx)
    lap = laplacian_9pt_matrix(config.nx, config.ny, config.dx)
    h_values = np.linspace(config.h_min, config.h_max, config.n_h)
    kappas = np.linspace(config.kappa_min, config.kappa_max, config.n_kappa)

    results_dir = config.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(config.seed)

    rows = []
    iterator: Iterable[float]
    if tqdm is not None:
        iterator = tqdm(kappas, desc="kappa sweep")
    else:  # pragma: no cover - tqdm not installed
        iterator = kappas

    for kappa in iterator:
        l_eff = effective_operator(lap, kappa, config.r)
        lambda_min, mode = leading_mode(l_eff)
        rayleigh = float(mode @ (l_eff @ mode))

        up, down = hysteresis_for_kappa(
            kappa=kappa,
            lap=lap,
            h_values=h_values,
            mode=mode,
            r=config.r,
            u=config.u,
            dt=config.dt,
            tmax=config.tmax,
            tol=config.tol,
            rng=rng,
        )

        save_hysteresis_csv(results_dir, kappa, h_values, up, down)
        save_mode(results_dir, kappa, mode)

        # Use both branches for fitting the normal form
        amplitudes = np.concatenate((up, down))
        fields = np.concatenate((h_values, h_values))
        u_fit, alpha_fit, rmse = fit_thom_normal_form(amplitudes, fields)

        max_diff = float(np.max(np.abs(up - down)))
        rows.append(
            {
                "kappa": kappa,
                "lambda_min": lambda_min,
                "rayleigh": rayleigh,
                "max_hysteresis_A_diff": max_diff,
                "u_fit": u_fit,
                "alpha_fit": alpha_fit,
                "fit_rmse": rmse,
            }
        )

        if config.save_plots and (config.kappa_plot is None or math.isclose(kappa, config.kappa_plot, rel_tol=1e-6, abs_tol=1e-6)):
            plot_hysteresis(results_dir, kappa, h_values, up, down)

    summary_df = pd.DataFrame(rows)
    summary_path = results_dir / "cusp_grid_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    LOGGER.info("Saved summary to %s", summary_path)

    if config.save_plots:
        plot_summary(results_dir, summary_df)

    return summary_df


def parse_args() -> ProbeConfig:
    parser = argparse.ArgumentParser(description="Probe the SMM for a Thom cusp catastrophe")
    parser.add_argument("--nx", type=int, default=16, help="Number of grid points along x")
    parser.add_argument("--ny", type=int, default=16, help="Number of grid points along y")
    parser.add_argument("--length", type=float, default=32.0, help="Physical domain length")
    parser.add_argument("--kappa-min", type=float, default=0.0, help="Minimum coupling strength")
    parser.add_argument("--kappa-max", type=float, default=0.6, help="Maximum coupling strength")
    parser.add_argument("--n-kappa", type=int, default=5, help="Number of kappa samples")
    parser.add_argument("--h-min", type=float, default=-0.8, help="Minimum applied field h")
    parser.add_argument("--h-max", type=float, default=0.8, help="Maximum applied field h")
    parser.add_argument("--n-h", type=int, default=9, help="Number of h samples")
    parser.add_argument("--r", type=float, default=0.2, help="Linear growth coefficient r")
    parser.add_argument("--u", type=float, default=1.0, help="Cubic coefficient u")
    parser.add_argument("--dt", type=float, default=0.02, help="Time step for RK4 integrator")
    parser.add_argument("--tmax", type=float, default=20.0, help="Maximum simulated time per ramp step")
    parser.add_argument("--tol", type=float, default=1e-7, help="Absolute convergence tolerance")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for initial condition")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("bifurcation_results"),
        help="Directory to store CSVs and plots",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable generation of PNG plots (useful for automated tests)",
    )
    parser.add_argument(
        "--kappa-plot",
        type=float,
        default=None,
        help="Specific kappa value to plot hysteresis for (defaults to every kappa if omitted)",
    )
    parser.add_argument("--log-level", default="INFO", help="Python logging level")

    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="[%(levelname)s] %(message)s")

    return ProbeConfig(
        nx=args.nx,
        ny=args.ny,
        length=args.length,
        kappa_min=args.kappa_min,
        kappa_max=args.kappa_max,
        n_kappa=args.n_kappa,
        h_min=args.h_min,
        h_max=args.h_max,
        n_h=args.n_h,
        r=args.r,
        u=args.u,
        dt=args.dt,
        tmax=args.tmax,
        tol=args.tol,
        seed=args.seed,
        results_dir=args.results_dir,
        save_plots=not args.no_plots,
        kappa_plot=args.kappa_plot,
    )


def main() -> None:
    config = parse_args()
    LOGGER.info("Starting cusp probe with config: %s", config)
    summary = run_probe(config)
    LOGGER.info("Completed sweep over %d kappa values", len(summary))
    LOGGER.info("Minimum lambda_min: %.6f", float(summary["lambda_min"].min()))
    LOGGER.info("Maximum hysteresis amplitude difference: %.6f", float(summary["max_hysteresis_A_diff"].max()))


if __name__ == "__main__":  # pragma: no cover - manual execution path
    main()

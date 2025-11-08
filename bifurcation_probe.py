"""Bifurcation probe for the Syncytial Mesh Model (SMM).

This module sweeps the parameter plane spanned by the mesh coupling ``kappa``
and external drive ``h`` in order to detect cusp-type bifurcation structure.
The workflow matches the project specification:

* build the nine-point periodic Laplacian used by :mod:`simulate_mesh_psd`,
* compute the leading eigenmode of the linearized operator ``kappa * Lap + rI``,
* run quasi-static hysteresis ramps in ``h`` for each ``kappa`` using an RK4
  gradient-flow integrator,
* project equilibria onto the leading mode to obtain reduced amplitude traces,
* fit the Thom cubic normal form ``u A^3 + alpha A + s h = 0`` via least squares,
* emit diagnostic CSV files and summary figures in ``bifurcation_results/``.

The Thom parameters ``(u, alpha, s)`` reported here correspond to the original
SMM variables as follows: ``u`` inherits the cubic nonlinearity ``u`` from the
full dynamics, ``alpha`` captures the softening of the linear term as ``kappa``
varies, and ``s`` quantifies how the drive ``h`` tilts the cusp.  The parameters
are returned up to a common multiplicative factor; we normalise the solution so
that the coefficient vector has unit Euclidean norm with ``s >= 0`` to remove the
sign ambiguity.
"""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # headless-friendly backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import seaborn as sns

LOGGER = logging.getLogger(__name__)


@dataclass
class ProbeConfig:
    """Configuration parameters for the bifurcation probe."""

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
    outdir: Path
    smoke: bool
    seed: int = 0

    @property
    def dx(self) -> float:
        return self.length / float(self.nx)

    @property
    def grid_shape(self) -> Tuple[int, int]:
        return (self.nx, self.ny)

    @property
    def grid_size(self) -> int:
        return self.nx * self.ny


def laplacian_9pt_matrix(nx: int, ny: int, dx: float) -> sp.csr_matrix:
    """Construct a nine-point Laplacian with periodic boundaries.

    The stencil matches the implementation in :mod:`simulate_mesh_psd`:

    * centre weight ``-20``
    * orthogonal neighbours ``+4``
    * diagonal neighbours ``+1``

    All coefficients are scaled by ``1 / (6 * dx**2)``.
    """

    coeff_center = -20.0
    coeff_orth = 4.0
    coeff_diag = 1.0
    scale = 1.0 / (6.0 * dx * dx)

    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []

    def index(ix: int, iy: int) -> int:
        return iy * nx + ix

    for iy in range(ny):
        for ix in range(nx):
            center = index(ix, iy)
            rows.append(center)
            cols.append(center)
            data.append(coeff_center)

            orthogonal = [
                ((ix + 1) % nx, iy),
                ((ix - 1) % nx, iy),
                (ix, (iy + 1) % ny),
                (ix, (iy - 1) % ny),
            ]
            for jx, jy in orthogonal:
                rows.append(center)
                cols.append(index(jx, jy))
                data.append(coeff_orth)

            diagonal = [
                ((ix + 1) % nx, (iy + 1) % ny),
                ((ix + 1) % nx, (iy - 1) % ny),
                ((ix - 1) % nx, (iy + 1) % ny),
                ((ix - 1) % nx, (iy - 1) % ny),
            ]
            for jx, jy in diagonal:
                rows.append(center)
                cols.append(index(jx, jy))
                data.append(coeff_diag)

    lap = sp.coo_matrix((np.array(data) * scale, (rows, cols)), shape=(nx * ny, nx * ny))
    return lap.tocsr()


def effective_operator(lap: sp.csr_matrix, kappa: float, r: float) -> sp.csr_matrix:
    """Linear operator ``H = kappa * Lap + r * I`` in sparse CSR form."""

    n = lap.shape[0]
    return kappa * lap + r * sp.eye(n, format="csr")


def leading_mode(operator: sp.csr_matrix) -> Tuple[float, np.ndarray]:
    """Return the smallest eigenvalue and normalised eigenmode of ``operator``."""

    try:
        eigvals, eigvecs = spla.eigsh(operator, k=1, which="SA")
        lambda_min = float(np.real(eigvals[0]))
        mode = np.real(eigvecs[:, 0])
    except Exception as exc:  # pragma: no cover - dense fallback
        LOGGER.warning("eigsh failed (%s); falling back to dense eigh", exc)
        dense = operator.toarray()
        eigvals_dense, eigvecs_dense = np.linalg.eigh(dense)
        idx = int(np.argmin(eigvals_dense))
        lambda_min = float(eigvals_dense[idx])
        mode = np.real(eigvecs_dense[:, idx])

    norm = np.linalg.norm(mode)
    if not math.isfinite(norm) or norm == 0.0:
        raise RuntimeError("Non-finite eigenmode norm encountered")
    mode /= norm
    return lambda_min, mode


def rk4_step(phi: np.ndarray, dt: float, func) -> np.ndarray:
    """Perform a single RK4 step using callable ``func`` returning derivatives."""

    k1 = func(phi)
    k2 = func(phi + 0.5 * dt * k1)
    k3 = func(phi + 0.5 * dt * k2)
    k4 = func(phi + dt * k3)
    return phi + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


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
    """Integrate the gradient flow ``dphi/dt = -∂V/∂phi`` until steady state.

    The potential derivative is ``kappa * Lap @ phi + r * phi + u * phi**3 - h``.
    The routine stops when the Euclidean norm of successive iterates is smaller
    than ``tol`` or when the maximum number of RK4 steps implied by ``tmax`` is
    reached.
    """

    phi = phi0.astype(float, copy=True)
    max_steps = max(1, int(math.ceil(tmax / dt)))

    def rhs(field: np.ndarray) -> np.ndarray:
        nonlinear = u * np.power(field, 3)
        linear = kappa * lap.dot(field) + r * field
        return -(linear + nonlinear - h)

    for _ in range(max_steps):
        updated = rk4_step(phi, dt, rhs)
        if np.linalg.norm(updated - phi) < tol:
            phi = updated
            break
        phi = updated
    return phi


def hysteresis_sweep(
    lap: sp.csr_matrix,
    kappa: float,
    r: float,
    u: float,
    h_values: np.ndarray,
    mode: np.ndarray,
    dt: float,
    tmax: float,
    tol: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """Compute up/down hysteresis amplitudes for a fixed ``kappa``.

    Returns ``(A_up, A_down, states)`` where ``states`` contains the final field
    for each ``h`` in the up sweep followed by the mirrored down sweep.
    """

    n = lap.shape[0]
    phi = 1e-3 * rng.standard_normal(n)
    amplitudes_up: List[float] = []
    states: List[np.ndarray] = []

    for h in h_values:
        phi = integrate_to_steady(phi, lap, kappa, r, u, h, dt, tmax, tol)
        amplitudes_up.append(float(np.dot(phi, mode)))
        states.append(phi.copy())

    amplitudes_down: List[float] = []
    for h in reversed(h_values):
        phi = integrate_to_steady(phi, lap, kappa, r, u, h, dt, tmax, tol)
        amplitudes_down.append(float(np.dot(phi, mode)))
        states.append(phi.copy())
    amplitudes_down.reverse()

    return np.array(amplitudes_up), np.array(amplitudes_down), states


def fit_thom_normal_form(amplitudes: np.ndarray, h_values: np.ndarray) -> Tuple[float, float, float, float]:
    """Fit the Thom cubic normal form ``u A^3 + alpha A + s h = 0``.

    The design matrix is homogeneous; we extract the (unit-norm) null-space
    vector via singular value decomposition and report the corresponding RMS
    residual.
    """

    if amplitudes.shape != h_values.shape:
        raise ValueError("Amplitude and field arrays must share the same shape")

    X = np.column_stack((np.power(amplitudes, 3), amplitudes, h_values))
    # SVD returns right-singular vectors as rows of ``vh``; the vector associated
    # with the smallest singular value spans the null-space of X.
    _, svals, vh = np.linalg.svd(X, full_matrices=False)
    coef = vh[-1, :]
    norm = np.linalg.norm(coef)
    if norm == 0.0:
        raise RuntimeError("Degenerate null-space during Thom form fit")
    coef /= norm
    # Fix overall sign so that the drive coefficient is non-negative.
    if coef[2] < 0:
        coef = -coef
    residual = X @ coef
    rms = float(np.linalg.norm(residual) / math.sqrt(len(h_values)))
    u_fit, alpha_fit, s_fit = map(float, coef)
    return u_fit, alpha_fit, s_fit, rms


def save_hysteresis(results_dir: Path, kappa: float, h_values: Sequence[float], up: np.ndarray, down: np.ndarray) -> Path:
    df = pd.DataFrame({"h": h_values, "A_up": up, "A_down": down})
    csv_path = results_dir / f"hysteresis_kappa_{kappa:.3f}.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def save_mode(results_dir: Path, kappa: float, mode: np.ndarray) -> Path:
    path = results_dir / f"mode_kappa_{kappa:.3f}.npy"
    np.save(path, mode)
    return path


def save_discriminant(results_dir: Path, kappa: float, h_values: np.ndarray, s_coeff: float, alpha_coeff: float) -> Path:
    beta = s_coeff * h_values
    delta = -4.0 * (alpha_coeff ** 3) - 27.0 * np.square(beta)
    df = pd.DataFrame({"h": h_values, "beta": beta, "Delta": delta})
    path = results_dir / f"discriminant_kappa_{kappa:.3f}.csv"
    df.to_csv(path, index=False)
    return path


def ensure_figures_dir(base_dir: Path) -> Path:
    fig_dir = base_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


def plot_summary(results_dir: Path, summary_df: pd.DataFrame) -> None:
    """Generate diagnostic summary figures for the cusp grid sweep."""

    fig_dir = ensure_figures_dir(results_dir)
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(6, 4))
    sns.lineplot(x="kappa", y="lambda_min", data=summary_df, marker="o")
    plt.title("Leading eigenvalue vs. kappa")
    plt.tight_layout()
    plt.savefig(fig_dir / "lambda_vs_kappa.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.lineplot(x="kappa", y="maxdiff", data=summary_df, marker="s")
    plt.ylabel("max |A_up - A_down|")
    plt.title("Hysteresis extent")
    plt.tight_layout()
    plt.savefig(fig_dir / "maxdiff_vs_kappa.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.lineplot(x="kappa", y="alpha_fit", data=summary_df, marker="^")
    plt.title("Normal form alpha vs. kappa")
    plt.tight_layout()
    plt.savefig(fig_dir / "alpha_vs_kappa.png", dpi=200)
    plt.close()


def warn_if_slow_relaxation(lambda_min: float, tmax: float, kappa: float) -> None:
    if lambda_min == 0:
        return
    char_time = abs(1.0 / lambda_min)
    if char_time > 5.0 * tmax:
        LOGGER.warning(
            "Potential slow relaxation at kappa=%.3f: characteristic time %.2f exceeds Tmax %.2f.",
            kappa,
            char_time,
            tmax,
        )


def run_probe(config: ProbeConfig) -> pd.DataFrame:
    LOGGER.info(
        "Building 9-point Laplacian for grid %dx%d (dx=%.3f)", config.nx, config.ny, config.dx
    )
    lap = laplacian_9pt_matrix(config.nx, config.ny, config.dx)

    kappas = np.linspace(config.kappa_min, config.kappa_max, config.n_kappa)
    h_values = np.linspace(config.h_min, config.h_max, config.n_h)

    results_dir = config.outdir
    results_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(config.seed)

    rows = []
    iterator: Iterable[float]
    try:
        from tqdm import tqdm

        iterator = tqdm(kappas, desc="kappa sweep")
    except Exception:  # pragma: no cover - tqdm optional
        iterator = kappas

    mode_ref: np.ndarray | None = None
    for kappa in iterator:
        operator = effective_operator(lap, kappa, config.r)
        lambda_min, mode = leading_mode(operator)
        warn_if_slow_relaxation(lambda_min, config.tmax, kappa)

        if mode_ref is None:
            mode_ref = mode.copy()
        elif float(np.dot(mode_ref, mode)) < 0.0:
            mode = -mode
        mode_ref = mode_ref / np.linalg.norm(mode_ref)

        save_mode(results_dir, kappa, mode)

        up, down, _ = hysteresis_sweep(
            lap=lap,
            kappa=kappa,
            r=config.r,
            u=config.u,
            h_values=h_values,
            mode=mode,
            dt=config.dt,
            tmax=config.tmax,
            tol=config.tol,
            rng=rng,
        )

        save_hysteresis(results_dir, kappa, h_values, up, down)

        amplitudes = np.concatenate([up, down])
        fields = np.concatenate([h_values, h_values])
        u_fit, alpha_fit, s_fit, rms = fit_thom_normal_form(amplitudes, fields)

        save_discriminant(results_dir, kappa, h_values, s_fit, alpha_fit)

        maxdiff = float(np.max(np.abs(up - down)))
        rows.append(
            {
                "kappa": float(kappa),
                "lambda_min": lambda_min,
                "maxdiff": maxdiff,
                "u_fit": u_fit,
                "alpha_fit": alpha_fit,
                "s_fit": s_fit,
                "rms": rms,
            }
        )

    summary = pd.DataFrame(rows)
    summary_path = results_dir / "cusp_grid_summary.csv"
    summary.to_csv(summary_path, index=False)
    LOGGER.info("Wrote summary to %s", summary_path)

    plot_summary(results_dir, summary)

    # Report approximate cusp location if lambda crosses zero.
    lambda_vals = summary["lambda_min"].to_numpy()
    kappas_np = summary["kappa"].to_numpy()
    signs = np.sign(lambda_vals)
    crossings = np.where(np.diff(signs))[0]
    if crossings.size:
        idx = crossings[0]
        # Linear interpolation around zero crossing
        lam1, lam2 = lambda_vals[idx], lambda_vals[idx + 1]
        kap1, kap2 = kappas_np[idx], kappas_np[idx + 1]
        if lam2 != lam1:
            kappa_c = kap1 + (0 - lam1) * (kap2 - kap1) / (lam2 - lam1)
            LOGGER.info("Estimated kappa_c ≈ %.4f where lambda_min crosses zero", kappa_c)
        else:
            LOGGER.info("lambda_min changes sign near kappa=%.4f", kap1)
    else:
        LOGGER.info("No sign change detected in lambda_min over sampled kappa range")

    return summary


def parse_args(argv: Sequence[str] | None = None) -> ProbeConfig:
    parser = argparse.ArgumentParser(description="Run the SMM bifurcation probe")
    parser.add_argument("--nx", type=int, default=32, help="Number of grid points along x")
    parser.add_argument("--ny", type=int, default=32, help="Number of grid points along y")
    parser.add_argument("--L", type=float, default=32.0, help="Domain length (assumed square)")
    parser.add_argument("--kappa_min", type=float, default=0.0, help="Minimum coupling kappa")
    parser.add_argument("--kappa_max", type=float, default=0.6, help="Maximum coupling kappa")
    parser.add_argument("--n_kappa", type=int, default=13, help="Number of kappa samples")
    parser.add_argument("--h_min", type=float, default=-0.8, help="Minimum field h")
    parser.add_argument("--h_max", type=float, default=0.8, help="Maximum field h")
    parser.add_argument("--n_h", type=int, default=33, help="Number of h samples")
    parser.add_argument("--r", type=float, default=0.2, help="Linear coefficient r")
    parser.add_argument("--u", type=float, default=1.0, help="Cubic coefficient u")
    parser.add_argument("--dt", type=float, default=0.02, help="RK4 time step")
    parser.add_argument("--Tmax", type=float, default=120.0, help="Maximum integration time per h")
    parser.add_argument("--tol", type=float, default=1e-7, help="Convergence tolerance on ||phi_{n+1}-phi_n||")
    parser.add_argument("--outdir", type=Path, default=Path("bifurcation_results"), help="Output directory")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for initial perturbations")
    parser.add_argument("--smoke", action="store_true", help="Run a quick deterministic smoke test")
    parser.add_argument("--log_level", default="INFO", help="Logging level")

    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="[%(levelname)s] %(message)s")

    if args.smoke:
        LOGGER.info("Activating smoke-test settings")
        args.nx = args.ny = 16
        args.n_kappa = min(args.n_kappa, 5)
        args.n_h = min(args.n_h, 9)
        args.Tmax = min(args.Tmax, 10.0)
        args.dt = max(args.dt, 0.02)
        args.seed = 0

    return ProbeConfig(
        nx=args.nx,
        ny=args.ny,
        length=args.L,
        kappa_min=args.kappa_min,
        kappa_max=args.kappa_max,
        n_kappa=args.n_kappa,
        h_min=args.h_min,
        h_max=args.h_max,
        n_h=args.n_h,
        r=args.r,
        u=args.u,
        dt=args.dt,
        tmax=args.Tmax,
        tol=args.tol,
        outdir=args.outdir,
        smoke=args.smoke,
        seed=args.seed,
    )


def main(argv: Sequence[str] | None = None) -> None:
    config = parse_args(argv)
    LOGGER.info("Starting bifurcation probe with configuration: %s", config)
    run_probe(config)


if __name__ == "__main__":  # pragma: no cover
    main()

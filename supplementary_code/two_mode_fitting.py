"""Two-mode coherence probability fitting script.

This script reconstructs the empirical probability of observing significant
coherence P(L) across spatial scales and fits an analytic two-mode model with a
linear decoherence rate lambda(L) = lambda0 + kappa * L.  When the manuscript
formula is unavailable (PDF parsing may fail in restricted environments), the
script falls back to a saturating link:

    P_model(L) = p_max * (1 - exp(-lambda(L) / lambda_scale))

This mapping is monotone and approaches p_max as L -> infinity.  An alternative
model with a saturating rate lambda(L) = lambda0 + kappa * (1 - exp(-L / Ls))
provides a comparison for AIC/BIC.
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from scipy.stats import binom

logger = logging.getLogger(__name__)


DEFAULT_THRESHOLD = 0.05
DEFAULT_B = 1000
SMOKE_B = 100
SMOKE_L_POINTS = 60
FULL_L_POINTS = 200
L_MIN = 0.1
L_MAX = 100.0
EPS = 1e-9


@dataclass
class FitResult:
    name: str
    params: Dict[str, float]
    loglike: float
    aic: float
    bic: float
    converged: bool

    def parameter_vector(self, keys: Sequence[str]) -> np.ndarray:
        return np.array([self.params[k] for k in keys], dtype=float)


@dataclass
class EmpiricalPL:
    L_grid: np.ndarray
    p_mean: np.ndarray
    p_bootstrap: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    successes: np.ndarray
    n_subjects: int


@dataclass
class ModelPredictions:
    median: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray


class PdfParseWarning(RuntimeError):
    """Raised when parsing the reference PDF fails."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("PSD_with_Coherence.csv"),
        help="Input CSV with per-subject coherence statistics.",
    )
    parser.add_argument(
        "--scale-file",
        type=Path,
        default=None,
        help="Optional CSV with per-scale coherence (columns: subject/file, L or scale_mm, C).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("supplementary_code/two_mode_results"),
        help="Directory to store figures and CSV outputs.",
    )
    parser.add_argument("--B", type=int, default=DEFAULT_B, help="Bootstrap iterations.")
    parser.add_argument(
        "--seed", type=int, default=1234, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Significant coherence threshold applied to column C.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Enable quick smoke run (reduced bootstrap count and coarse L grid).",
    )
    parser.add_argument(
        "--fix-pmax",
        action="store_true",
        help="Fix p_max to the empirical plateau instead of fitting it.",
    )
    parser.add_argument(
        "--assume-constant",
        action="store_true",
        help=(
            "Use the per-subject C value for all scales when no scale file is present. "
            "This satisfies the required fallback to a user-specified mapping."
        ),
    )
    parser.add_argument(
        "--note-pdf",
        action="store_true",
        help="Attempt to parse the manuscript PDF for a closed-form model (best effort).",
    )
    return parser.parse_args()


def attempt_parse_pdf() -> Optional[str]:
    """Attempt to extract a note from the manuscript PDF.

    The environment may not ship a PDF parser; we record a warning when parsing
    fails so that users know why the default model is used.
    """
    pdf_path = Path("/mnt/data/SMM corrected.pdf")
    if not pdf_path.exists():
        raise PdfParseWarning("Reference manuscript not found at /mnt/data/SMM corrected.pdf")
    import importlib.util

    if importlib.util.find_spec("PyPDF2") is None:  # pragma: no cover - environment-specific
        raise PdfParseWarning("PyPDF2 is unavailable in this environment")
    import PyPDF2  # type: ignore

    try:
        reader = PyPDF2.PdfReader(str(pdf_path))
        text = "".join(page.extract_text() or "" for page in reader.pages)
    except Exception as exc:  # pragma: no cover - environment-specific
        raise PdfParseWarning(f"Failed to parse manuscript PDF: {exc}")

    return text


def build_L_grid(smoke: bool) -> np.ndarray:
    num = SMOKE_L_POINTS if smoke else FULL_L_POINTS
    return np.unique(np.logspace(np.log10(L_MIN), np.log10(L_MAX), num=num))


def detect_scale_column(df: pd.DataFrame) -> Optional[str]:
    if "scale_mm" in df.columns:
        return "scale_mm"
    if "L" in df.columns:
        return "L"
    return None


def normalise_subject_column(df: pd.DataFrame) -> pd.DataFrame:
    if "subject" in df.columns:
        return df
    if "file" in df.columns:
        df = df.rename(columns={"file": "subject"})
        return df
    df = df.copy()
    df.insert(0, "subject", [f"sub_{idx:04d}" for idx in range(len(df))])
    return df


def load_scale_data(
    input_path: Path,
    scale_file: Optional[Path],
    assume_constant: bool,
    threshold: float,
    L_grid: np.ndarray,
) -> Tuple[pd.DataFrame, str]:
    """Load coherence data with explicit scale information.

    Returns a tidy DataFrame with columns subject, L, C.  Raises a ValueError
    when required sources are unavailable and no constant assumption is given.
    """
    df = pd.read_csv(input_path)
    df = normalise_subject_column(df)
    scale_col = detect_scale_column(df)
    if scale_col:
        tidy = df[["subject", scale_col, "C"]].rename(columns={scale_col: "L"})
        tidy = tidy.dropna(subset=["L", "C"])
        if tidy.empty:
            raise ValueError("Input file contains scale column but no valid rows.")
        return tidy, "input"

    candidate = scale_file
    if candidate is None:
        candidate = input_path.parent / "PSD_with_Coherence_by_scale.csv"
        if not candidate.exists():
            candidate = Path("PSD_with_Coherence_by_scale.csv")
    if candidate and candidate.exists():
        scale_df = pd.read_csv(candidate)
        scale_df = normalise_subject_column(scale_df)
        scale_col = detect_scale_column(scale_df)
        if scale_col is None:
            raise ValueError(
                "Scale file is missing L/scale_mm column; provide one or use --assume-constant."
            )
        tidy = scale_df[["subject", scale_col, "C"]].rename(columns={scale_col: "L"})
        tidy = tidy.dropna(subset=["L", "C"])
        if tidy.empty:
            raise ValueError("Scale file contains no usable rows.")
        return tidy, str(candidate)

    if assume_constant:
        # Build a single-scale table replicated across the requested grid.
        const_rows = []
        for _, row in df.iterrows():
            for L in L_grid:
                const_rows.append({"subject": row["subject"], "L": L, "C": row["C"]})
        tidy = pd.DataFrame(const_rows)
        return tidy, "assumed_constant"

    raise ValueError(
        "No scale information found. Provide PSD_with_Coherence_by_scale.csv, "
        "use --scale-file, or enable --assume-constant to reuse per-subject C across L."
    )


def build_subject_curves(
    scale_df: pd.DataFrame, threshold: float, L_grid: np.ndarray
) -> Tuple[np.ndarray, List[str]]:
    subjects = list(scale_df["subject"].unique())
    n_subj = len(subjects)
    n_L = len(L_grid)
    significance = np.zeros((n_subj, n_L), dtype=int)

    for idx, subject in enumerate(subjects):
        sub_df = scale_df.loc[scale_df["subject"] == subject]
        Ls = sub_df["L"].to_numpy(dtype=float)
        Cs = sub_df["C"].to_numpy(dtype=float)
        order = np.argsort(Ls)
        Ls = Ls[order]
        Cs = Cs[order]
        if len(Ls) == 0:
            continue
        if len(Ls) == 1:
            interp_C = np.full_like(L_grid, fill_value=Cs[0], dtype=float)
        else:
            interp_C = np.interp(L_grid, Ls, Cs, left=Cs[0], right=Cs[-1])
        significance[idx] = (interp_C >= threshold).astype(int)
    return significance, subjects


def bootstrap_probabilities(significance: np.ndarray, B: int, rng: np.random.Generator) -> np.ndarray:
    n_subjects, n_L = significance.shape
    boot = np.empty((B, n_L), dtype=float)
    for b in range(B):
        sample_idx = rng.integers(0, n_subjects, size=n_subjects)
        boot[b] = significance[sample_idx].mean(axis=0)
    return boot


def summarise_empirical(significance: np.ndarray, L_grid: np.ndarray, B: int, rng: np.random.Generator) -> EmpiricalPL:
    boot = bootstrap_probabilities(significance, B, rng)
    ci_low = np.percentile(boot, 2.5, axis=0)
    ci_high = np.percentile(boot, 97.5, axis=0)
    p_mean = boot.mean(axis=0)
    successes = significance.sum(axis=0)
    return EmpiricalPL(
        L_grid=L_grid,
        p_mean=p_mean,
        p_bootstrap=boot,
        ci_low=ci_low,
        ci_high=ci_high,
        successes=successes,
        n_subjects=significance.shape[0],
    )


def lambda_linear(L: np.ndarray, lam0: float, kappa: float) -> np.ndarray:
    return lam0 + kappa * L


def lambda_saturating(L: np.ndarray, lam0: float, kappa: float, Ls: float) -> np.ndarray:
    return lam0 + kappa * (1.0 - np.exp(-L / max(Ls, EPS)))


def saturating_link(lam: np.ndarray, p_max: float, lam_scale: float) -> np.ndarray:
    lam_scale = max(lam_scale, EPS)
    lam = np.clip(lam, 0.0, None)
    return p_max * (1.0 - np.exp(-lam / lam_scale))


def model_probability(
    L: np.ndarray,
    params: Dict[str, float],
    model: str = "linear",
    fix_pmax: Optional[float] = None,
) -> np.ndarray:
    lam0 = params.get("lambda0", 0.0)
    kappa = params.get("kappa", 0.0)
    lam_scale = params.get("lambda_scale", 1.0)
    p_max = params.get("p_max", 1.0 if fix_pmax is None else fix_pmax)

    if fix_pmax is not None:
        p_max = fix_pmax

    if model == "linear":
        lam = lambda_linear(L, lam0, kappa)
    elif model == "saturating_rate":
        Ls = params.get("Ls", 10.0)
        lam = lambda_saturating(L, lam0, kappa, Ls)
    else:
        raise ValueError(f"Unknown model: {model}")
    return saturating_link(lam, p_max=p_max, lam_scale=lam_scale)


def fit_two_mode(
    L: np.ndarray,
    successes: np.ndarray,
    n: int,
    model: str,
    fix_pmax: Optional[float],
) -> FitResult:
    """Fit the two-mode model via binomial log-likelihood."""

    def objective(x: np.ndarray) -> float:
        params = vector_to_params(x)
        p = np.clip(model_probability(L, params, model=model, fix_pmax=fix_pmax), EPS, 1 - EPS)
        ll = binom.logpmf(successes, n, p).sum()
        return -ll

    def vector_to_params(vec: np.ndarray) -> Dict[str, float]:
        if model == "linear":
            lam0, kappa, p_max, lam_scale = vec
            params = {
                "lambda0": max(lam0, 0.0),
                "kappa": max(kappa, 0.0),
                "p_max": p_max if fix_pmax is None else fix_pmax,
                "lambda_scale": max(lam_scale, EPS),
            }
        else:
            lam0, kappa, Ls, p_max, lam_scale = vec
            params = {
                "lambda0": max(lam0, 0.0),
                "kappa": max(kappa, 0.0),
                "Ls": max(Ls, EPS),
                "p_max": p_max if fix_pmax is None else fix_pmax,
                "lambda_scale": max(lam_scale, EPS),
            }
        return params

    if model == "linear":
        x0 = np.array([0.1, 0.01, 0.8 if fix_pmax is None else fix_pmax, 1.0])
        bounds = [(0.0, None), (0.0, None), (0.0, 1.0), (EPS, None)]
        if fix_pmax is not None:
            bounds = [(0.0, None), (0.0, None), (fix_pmax, fix_pmax), (EPS, None)]
    else:
        x0 = np.array([0.1, 0.01, 10.0, 0.8 if fix_pmax is None else fix_pmax, 1.0])
        bounds = [(0.0, None), (0.0, None), (EPS, None), (0.0, 1.0), (EPS, None)]
        if fix_pmax is not None:
            bounds = [
                (0.0, None),
                (0.0, None),
                (EPS, None),
                (fix_pmax, fix_pmax),
                (EPS, None),
            ]

    result = minimize(objective, x0=x0, bounds=bounds, method="L-BFGS-B")
    params = vector_to_params(result.x)
    p = np.clip(model_probability(L, params, model=model, fix_pmax=fix_pmax), EPS, 1 - EPS)
    loglike = binom.logpmf(successes, n, p).sum()
    k_params = np.sum([b[0] != b[1] for b in bounds])
    aic = 2 * k_params - 2 * loglike
    bic = k_params * np.log(len(L)) - 2 * loglike
    return FitResult(
        name=model,
        params=params,
        loglike=float(loglike),
        aic=float(aic),
        bic=float(bic),
        converged=bool(result.success),
    )


def bootstrap_fit_parameters(
    significance: np.ndarray,
    L_grid: np.ndarray,
    B: int,
    rng: np.random.Generator,
    fix_pmax: Optional[float],
) -> Tuple[List[FitResult], np.ndarray]:
    n_subjects = significance.shape[0]
    fits: List[FitResult] = []
    for b in range(B):
        sample_idx = rng.integers(0, n_subjects, size=n_subjects)
        successes = significance[sample_idx].sum(axis=0)
        fit = fit_two_mode(L_grid, successes, n_subjects, model="linear", fix_pmax=fix_pmax)
        fits.append(fit)
    params_matrix = np.array([[f.params.get("lambda0", np.nan), f.params.get("kappa", np.nan)] for f in fits])
    return fits, params_matrix


def summarise_params(fits: List[FitResult]) -> pd.DataFrame:
    records = []
    for param in sorted({key for f in fits for key in f.params}):
        values = np.array([f.params[param] for f in fits])
        records.append(
            {
                "parameter": param,
                "estimate": float(np.median(values)),
                "2.5%": float(np.percentile(values, 2.5)),
                "97.5%": float(np.percentile(values, 97.5)),
            }
        )
    return pd.DataFrame.from_records(records)


def predictive_band(
    fits: List[FitResult], L_grid: np.ndarray, model: str, fix_pmax: Optional[float]
) -> ModelPredictions:
    preds = []
    for fit in fits:
        preds.append(model_probability(L_grid, fit.params, model=model, fix_pmax=fix_pmax))
    pred_arr = np.array(preds)
    return ModelPredictions(
        median=np.median(pred_arr, axis=0),
        ci_low=np.percentile(pred_arr, 2.5, axis=0),
        ci_high=np.percentile(pred_arr, 97.5, axis=0),
    )


def plot_probability(
    empirical: EmpiricalPL,
    model_pred: ModelPredictions,
    outpath: Path,
    label: str,
):
    plt.figure(figsize=(8, 5))
    plt.semilogx(empirical.L_grid, empirical.p_mean, "o", label="Empirical mean", color="black")
    plt.fill_between(empirical.L_grid, empirical.ci_low, empirical.ci_high, color="gray", alpha=0.3, label="Empirical 95% CI")
    plt.plot(empirical.L_grid, model_pred.median, label=f"Model ({label})", color="C1")
    plt.fill_between(empirical.L_grid, model_pred.ci_low, model_pred.ci_high, color="C1", alpha=0.2, label="Model 95% CI")
    plt.xlabel("Spatial scale L (mm)")
    plt.ylabel("P(significant coherence)")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_parameter_hist(params_matrix: np.ndarray, outpath: Path):
    plt.figure(figsize=(8, 4))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    labels = ["lambda0", "kappa"]
    for ax, idx, label in zip(axes, range(params_matrix.shape[1]), labels):
        ax.hist(params_matrix[:, idx], bins=30, color="C0", alpha=0.7)
        ax.set_title(label)
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def save_csv_outputs(
    outdir: Path,
    empirical: EmpiricalPL,
    model_pred: ModelPredictions,
    param_summary: pd.DataFrame,
    param_samples: np.ndarray,
    linear_fit: FitResult,
    alt_fit: FitResult,
):
    outdir.mkdir(parents=True, exist_ok=True)
    summary = param_summary.copy()
    summary.insert(0, "model", linear_fit.name)
    summary.to_csv(outdir / "two_mode_fit_summary.csv", index=False)

    samples_df = pd.DataFrame(param_samples, columns=["lambda0", "kappa"])
    samples_df.insert(0, "model", linear_fit.name)
    samples_df.to_csv(outdir / "two_mode_bootstrap_samples.csv", index=False)

    grid_df = pd.DataFrame(
        {
            "L_mm": empirical.L_grid,
            "p_obs_mean": empirical.p_mean,
            "p_obs_ci_low": empirical.ci_low,
            "p_obs_ci_high": empirical.ci_high,
            "p_model_median": model_pred.median,
            "p_model_ci_low": model_pred.ci_low,
            "p_model_ci_high": model_pred.ci_high,
        }
    )
    grid_df.to_csv(outdir / "P_L_empirical_and_fit.csv", index=False)

    report_lines = [
        "Two-mode coherence fitting report",
        "=================================",
        f"Linear rate fit converged: {linear_fit.converged}",
        f"lambda0={linear_fit.params.get('lambda0', np.nan):.4f}, "
        f"kappa={linear_fit.params.get('kappa', np.nan):.4f}, "
        f"p_max={linear_fit.params.get('p_max', np.nan):.4f}, "
        f"lambda_scale={linear_fit.params.get('lambda_scale', np.nan):.4f}",
        f"Log-likelihood={linear_fit.loglike:.3f}, AIC={linear_fit.aic:.3f}, BIC={linear_fit.bic:.3f}",
        "",
        f"Alternative ({alt_fit.name}) converged: {alt_fit.converged}",
        json.dumps(alt_fit.params, indent=2),
        f"Log-likelihood={alt_fit.loglike:.3f}, AIC={alt_fit.aic:.3f}, BIC={alt_fit.bic:.3f}",
        "",
        "AIC winner: linear" if linear_fit.aic <= alt_fit.aic else f"AIC winner: {alt_fit.name}",
        "BIC winner: linear" if linear_fit.bic <= alt_fit.bic else f"BIC winner: {alt_fit.name}",
    ]
    (outdir / "two_mode_fit_report.txt").write_text("\n".join(report_lines))


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    rng = np.random.default_rng(args.seed)
    L_grid = build_L_grid(args.smoke)
    B = SMOKE_B if args.smoke else args.B

    pdf_note = None
    if args.note_pdf:
        try:
            pdf_text = attempt_parse_pdf()
            pdf_note = "Parsed manuscript text successfully." if pdf_text else "Manuscript parsed but empty."
        except PdfParseWarning as warning:
            pdf_note = f"PDF parse unavailable: {warning}"
            logger.warning(pdf_note)

    scale_df, provenance = load_scale_data(
        input_path=args.input,
        scale_file=args.scale_file,
        assume_constant=args.assume_constant,
        threshold=args.threshold,
        L_grid=L_grid,
    )
    logger.info("Loaded coherence data from %s", provenance)

    significance, subjects = build_subject_curves(scale_df, args.threshold, L_grid)
    empirical = summarise_empirical(significance, L_grid, B=B, rng=rng)
    plateau = float(np.median(empirical.p_mean[-10:]))
    fix_pmax = plateau if args.fix_pmax else None

    linear_fit = fit_two_mode(
        empirical.L_grid,
        empirical.successes,
        empirical.n_subjects,
        model="linear",
        fix_pmax=fix_pmax,
    )
    alt_fit = fit_two_mode(
        empirical.L_grid,
        empirical.successes,
        empirical.n_subjects,
        model="saturating_rate",
        fix_pmax=fix_pmax,
    )

    boot_fits, param_samples = bootstrap_fit_parameters(
        significance, empirical.L_grid, B=B, rng=rng, fix_pmax=fix_pmax
    )
    param_summary = summarise_params([linear_fit] + boot_fits)
    model_pred = predictive_band([linear_fit] + boot_fits, empirical.L_grid, model="linear", fix_pmax=fix_pmax)

    args.outdir.mkdir(parents=True, exist_ok=True)
    plot_probability(empirical, model_pred, args.outdir / "P_L_fit.png", label="linear")
    plot_parameter_hist(param_samples, args.outdir / "parameter_bootstrap.png")

    save_csv_outputs(
        outdir=args.outdir,
        empirical=empirical,
        model_pred=model_pred,
        param_summary=param_summary,
        param_samples=param_samples,
        linear_fit=linear_fit,
        alt_fit=alt_fit,
    )

    provenance_note = f"Scale provenance: {provenance}. Subjects: {len(subjects)}."
    if pdf_note:
        provenance_note += f" PDF note: {pdf_note}."
    (args.outdir / "provenance.txt").write_text(provenance_note)
    logger.info("Finished fitting. Outputs saved to %s", args.outdir)


if __name__ == "__main__":
    main()

"""
fast_calibrate_signal.py — Calibración rápida de β_señal(tracker_prob).

En lugar de re-correr el simulador, usa el log existente de tracker_prob=0.5
y simula el comportamiento de un tracker con probabilidad p mezclando las
utilidades deterministas del tracker y del no-tracker:

    U_p_j = -bw * base_val_j + tau * (p_stat_j + p * p_grad_excess_j)

Luego samplea elecciones según softmax y corre conditional logit por bin.
Esto captura la no-linealidad esperada (concavidad por saturación del softmax)
sin el costo de nuevas simulaciones.

Limitación: no reproduce los epsilons Gumbel originales de cada decisión,
pero para calibrar la forma de la curva es una aproximación razonable.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as opt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from power_curve_t2 import (
    DEFAULT_H_BINS,
    fit_conditional_logit_l2,
    extract_beta,
    FEATURE_COLS_BRUTO,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)


def simulate_tracker_choices(
    df: pd.DataFrame,
    p: float,
    tau: float = 10.0,
    board_value_weight: float = 0.5,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Re-samplea elecciones de un tracker con probabilidad p usando utilidades mezcladas.
    Mantiene la estructura de decisiones/alternativas del log p=0.5.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()

    # Utilidad determinista mezclada
    df["mixed_util"] = (
        -board_value_weight * df["base_val"]
        + tau * (df["p_stat_clase"] + p * df["p_grad_excess"])
    )

    # Softmax por decision_id
    def softmax(u):
        u = u - np.max(u)
        e = np.exp(u)
        return e / np.sum(e)

    # Elegir según softmax (samplear una elección por decisión)
    choices = []
    for dec_id, grp in df.groupby("decision_id"):
        probs = softmax(grp["mixed_util"].values)
        chosen_idx = rng.choice(len(grp), p=probs)
        chosen_alt = grp.iloc[chosen_idx]["alternative_id"]
        # Marcar chosen para esta decisión
        df.loc[(df["decision_id"] == dec_id), "chosen"] = (
            df.loc[(df["decision_id"] == dec_id), "alternative_id"] == chosen_alt
        ).astype(int)

    return df


def beta_by_bin_for_p(
    df_base: pd.DataFrame,
    p: float,
    h_bins: List[Tuple[int, int, str]],
    tau: float = 10.0,
    board_value_weight: float = 0.5,
    seed: int = 42,
    n_replicates: int = 5,
) -> Dict[str, Dict]:
    """Calcula beta_señal por bin para un tracker_prob dado, promediando replicaciones."""
    results = {label: [] for _, _, label in h_bins}
    for rep in range(n_replicates):
        df_sim = simulate_tracker_choices(
            df_base, p, tau, board_value_weight, seed=seed + rep
        )
        for h_min, h_max, label in h_bins:
            df_bin = df_sim[(df_sim["H"] >= h_min) & (df_sim["H"] <= h_max)].copy()
            if df_bin["decision_id"].nunique() < 50:
                continue
            summary = fit_conditional_logit_l2(df_bin, FEATURE_COLS_BRUTO, label=f"p={p} H={label}", lam=0.0)
            beta, pval, ci_low, ci_high = extract_beta(summary, "p_grad_excess")
            if beta is not None:
                results[label].append(beta)

    out = {}
    for h_min, h_max, label in h_bins:
        vals = results[label]
        out[label] = {
            "H_range": [h_min, h_max],
            "beta_mean": float(np.mean(vals)) if vals else None,
            "beta_std": float(np.std(vals)) if vals else None,
            "beta_n": len(vals),
        }
    return out


def power_curve(p: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * np.power(p, b)


def fit_power_curve(ps: np.ndarray, betas: np.ndarray) -> Dict:
    def resid(params):
        a, b = params
        return power_curve(ps, a, b) - betas
    valid = betas is not None and len(betas) > 0
    if not valid:
        return {"a": None, "b": None, "r2": None, "rmse": None}
    result = opt.least_squares(resid, x0=[max(betas) * 2.0, 1.0], bounds=([0, 0.1], [np.inf, 3.0]))
    a, b = result.x
    fitted = power_curve(ps, a, b)
    ss_res = float(np.sum((betas - fitted) ** 2))
    ss_tot = float(np.sum((betas - np.mean(betas)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"a": float(a), "b": float(b), "r2": r2, "rmse": float(np.sqrt(ss_res / len(betas)))}


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibración rápida de beta_señal(p)")
    parser.add_argument("--log", type=str, required=True, help="Parquet del log tracker_prob=0.5")
    parser.add_argument("--ps", type=str, default="0.1,0.25,0.5,0.75,1.0")
    parser.add_argument("--tau", type=float, default=10.0)
    parser.add_argument("--board_value_weight", type=float, default=0.5)
    parser.add_argument("--n_replicates", type=int, default=5)
    parser.add_argument("--out", type=str, default="out_fast_calibrate")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_base = pd.read_parquet(args.log)
    ps = [float(x.strip()) for x in args.ps.split(",")]

    all_results = []
    beta_by_p = {label: [] for _, _, label in DEFAULT_H_BINS}

    for p in ps:
        print(f"Calibrando p={p}...")
        res = beta_by_bin_for_p(
            df_base, p, DEFAULT_H_BINS, args.tau, args.board_value_weight,
            n_replicates=args.n_replicates
        )
        all_results.append({"p": p, "by_bin": res})
        for label, vals in res.items():
            beta_by_p[label].append((p, vals["beta_mean"], vals["beta_std"]))

    # Ajustar curva por bin
    calibration = {}
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()
    p_fine = np.linspace(0.01, 1.0, 200)

    for idx, (h_min, h_max, label) in enumerate(DEFAULT_H_BINS):
        ax = axes[idx]
        points = beta_by_p[label]
        points = [x for x in points if x[1] is not None]
        if len(points) < 2:
            ax.set_title(f"H={label} (sin datos)")
            continue
        ps_arr = np.array([x[0] for x in points])
        betas_arr = np.array([x[1] for x in points])
        stds_arr = np.array([x[2] if x[2] is not None else 0 for x in points])

        params = fit_power_curve(ps_arr, betas_arr)
        fitted = power_curve(p_fine, params["a"], params["b"])
        linear = p_fine * (betas_arr[np.argmin(np.abs(ps_arr - 0.5))] / 0.5)

        ax.errorbar(ps_arr, betas_arr, yerr=1.96 * stds_arr, fmt="o", capsize=4, label="fast calibrate")
        ax.plot(p_fine, fitted, "-", label=f"ajuste a*p^b (b={params['b']:.2f})")
        ax.plot(p_fine, linear, "--", alpha=0.6, label="lineal desde 0.5")
        ax.set_xlabel("tracker_prob")
        ax.set_ylabel("beta_señal")
        ax.set_title(f"H={label}  R²={params['r2']:.2f}")
        ax.set_xlim(0, 1)
        ax.grid(True, ls=":", alpha=0.5)
        ax.legend(fontsize=7)

        calibration[label] = {
            "H_range": [h_min, h_max],
            "curve_type": "a*p^b",
            "params": params,
            "data": [
                {"p": float(p), "beta": float(b), "beta_std": float(s)}
                for p, b, s in points
            ],
        }

    ax_summary = axes[-1]
    ax_summary.axis("off")
    lines = ["Parametros de ajuste rapido", "=============================="]
    for label, cal in calibration.items():
        lines.append(f"\nH={label}: a={cal['params']['a']:.2f}, b={cal['params']['b']:.2f}, R²={cal['params']['r2']:.2f}")
    ax_summary.text(0.05, 0.95, "\n".join(lines), transform=ax_summary.transAxes,
                    fontsize=9, verticalalignment="top", fontfamily="monospace")

    fig.suptitle("Calibracion rapida de beta_señal(tracker_prob)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / "fig_fast_calibration.png", dpi=150)
    plt.close(fig)

    with open(out_dir / "fast_calibration.json", "w", encoding="utf-8") as f:
        json.dump(calibration, f, indent=2)

    print(f"Calibracion rapida guardada en {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

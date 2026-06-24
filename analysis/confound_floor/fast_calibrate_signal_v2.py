"""
fast_calibrate_signal_v2.py — Calibración rápida y normalizada de β_señal(p).

Usa el log de tracker_prob=0.5, simula elecciones para otros p con softmax determinista,
calcula beta por bin, ajusta beta(p)=a*p^b, y normaliza el resultado para que beta(0.5)
coincida con el valor real observado en la simulacion p=0.5.
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
from power_curve_t2 import DEFAULT_H_BINS, fit_conditional_logit_l2, extract_beta, FEATURE_COLS_BRUTO

warnings.filterwarnings("ignore", category=RuntimeWarning)


def simulate_and_estimate(
    df: pd.DataFrame,
    p: float,
    tau: float,
    bw: float,
    h_bins: List[Tuple[int, int, str]],
    seed: int,
) -> Dict[str, float]:
    """Simula elecciones para tracker_prob=p y estima beta por bin."""
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["mixed_util"] = -bw * df["base_val"] + tau * (df["p_stat_clase"] + p * df["p_grad_excess"])

    # Softmax vectorizado
    mx = df.groupby("decision_id")["mixed_util"].transform("max")
    df["eu"] = np.exp(df["mixed_util"] - mx)
    sum_eu = df.groupby("decision_id")["eu"].transform("sum")
    df["prob"] = df["eu"] / sum_eu

    # Samplear una elección por decisión de forma vectorizada
    n_dec = df["decision_id"].nunique()
    rns = rng.random(n_dec)
    rn_map = dict(zip(sorted(df["decision_id"].unique()), rns))
    df["rn"] = df["decision_id"].map(rn_map)
    df["cumprob"] = df.groupby("decision_id")["prob"].cumsum()
    df["chosen"] = (df["cumprob"] >= df["rn"]).astype(int)
    # Dejar solo la primera chosen por decision_id
    first_idx = df[df["chosen"] == 1].groupby("decision_id").head(1).index
    df["chosen"] = 0
    df.loc[first_idx, "chosen"] = 1

    betas = {}
    for h_min, h_max, label in h_bins:
        df_bin = df[(df["H"] >= h_min) & (df["H"] <= h_max)].copy()
        if df_bin["decision_id"].nunique() < 50:
            betas[label] = None
            continue
        summary = fit_conditional_logit_l2(df_bin, FEATURE_COLS_BRUTO, label=f"p={p} H={label}", lam=0.0)
        beta, _, _, _ = extract_beta(summary, "p_grad_excess")
        betas[label] = beta
    return betas


def fit_power_shape(ps: np.ndarray, betas: np.ndarray) -> Tuple[float, float, np.ndarray]:
    """Ajusta beta = a * p^b (log-lineal)."""
    valid = betas > 0
    ps_v = ps[valid]
    bs_v = betas[valid]
    if len(ps_v) < 2:
        return np.nan, np.nan, betas
    log_p = np.log(ps_v)
    log_b = np.log(bs_v)
    b, log_a = np.polyfit(log_p, log_b, 1)
    a = np.exp(log_a)
    fitted = a * np.power(ps, b)
    return float(a), float(b), fitted


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, required=True)
    parser.add_argument("--real_results", type=str, required=True,
                        help="JSON con resultados reales de tracker_prob=0.5 (para anclaje)")
    parser.add_argument("--ps", type=str, default="0.1,0.25,0.5,0.75,1.0")
    parser.add_argument("--tau", type=float, default=10.0)
    parser.add_argument("--board_value_weight", type=float, default=0.5)
    parser.add_argument("--n_replicates", type=int, default=3)
    parser.add_argument("--out", type=str, default="out_fast_calibrate_v2")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.log)
    ps = [float(x.strip()) for x in args.ps.split(",")]

    # Leer betas reales a p=0.5 por bin
    with open(args.real_results, encoding="utf-8") as f:
        real = json.load(f)
    real_beta_05 = {}
    # Asumimos que real_results es bin_results.json de power_curve_t2.py
    for label, res in real.items():
        if "tracker_bruto" in res:
            real_beta_05[label] = res["tracker_bruto"]["beta"]

    # Calcular betas aproximados para cada p, promediando replicaciones
    beta_by_p = {label: {} for _, _, label in DEFAULT_H_BINS}
    for p in ps:
        vals = {label: [] for _, _, label in DEFAULT_H_BINS}
        for rep in range(args.n_replicates):
            betas = simulate_and_estimate(df, p, args.tau, args.board_value_weight, DEFAULT_H_BINS, seed=100*int(p*100)+rep)
            for label, b in betas.items():
                if b is not None:
                    vals[label].append(b)
        for label in vals:
            if vals[label]:
                beta_by_p[label][p] = float(np.mean(vals[label]))

    # Ajustar forma y normalizar a beta(0.5) real
    calibration = {}
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()
    p_fine = np.linspace(0.01, 1.0, 200)

    for idx, (h_min, h_max, label) in enumerate(DEFAULT_H_BINS):
        ax = axes[idx]
        points = beta_by_p[label]
        if not points or 0.5 not in points:
            ax.set_title(f"H={label} (sin datos)")
            continue
        ps_arr = np.array(sorted(points.keys()))
        betas_arr = np.array([points[p] for p in ps_arr])

        a_raw, b_shape, fitted_raw = fit_power_shape(ps_arr, betas_arr)
        # Normalizar: beta_norm(p) = beta_real(0.5) * (p/0.5)^b_shape
        beta_real_05 = real_beta_05.get(label, betas_arr[np.argmin(np.abs(ps_arr - 0.5))])
        a_norm = beta_real_05 / (0.5 ** b_shape)
        fitted_norm = a_norm * np.power(p_fine, b_shape)
        linear = p_fine * (beta_real_05 / 0.5)

        ax.plot(ps_arr, betas_arr, "o", label="aproximacion rapida")
        ax.plot(p_fine, fitted_norm, "-", label=f"calibrado: beta(0.5)* (p/0.5)^{b_shape:.2f}")
        ax.plot(p_fine, linear, "--", alpha=0.6, label="lineal desde 0.5")
        ax.axvline(0.5, color="gray", ls=":", alpha=0.5)
        ax.axhline(beta_real_05, color="gray", ls=":", alpha=0.5)
        ax.set_xlabel("tracker_prob")
        ax.set_ylabel("beta_señal")
        ax.set_title(f"H={label}  forma b={b_shape:.2f}")
        ax.set_xlim(0, 1)
        ax.grid(True, ls=":", alpha=0.5)
        ax.legend(fontsize=7)

        calibration[label] = {
            "H_range": [h_min, h_max],
            "b_shape": float(b_shape),
            "beta_real_05": float(beta_real_05),
            "a_norm": float(a_norm),
            "curve": "beta_real_05 * (p / 0.5)^b_shape",
            "raw_data": {str(p): float(v) for p, v in points.items()},
        }

    ax_summary = axes[-1]
    ax_summary.axis("off")
    lines = ["Forma calibrada por bin", "========================"]
    for label, cal in calibration.items():
        lines.append(f"\nH={label}: b={cal['b_shape']:.2f}, beta(0.5)={cal['beta_real_05']:.2f}")
        lines.append(f"  curva: beta(p) = {cal['beta_real_05']:.2f} * (p/0.5)^{cal['b_shape']:.2f}")
    ax_summary.text(0.05, 0.95, "\n".join(lines), transform=ax_summary.transAxes,
                    fontsize=9, verticalalignment="top", fontfamily="monospace")

    fig.suptitle("Calibracion rapida (normalizada a beta(0.5) real)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / "fig_fast_calibration_v2.png", dpi=150)
    plt.close(fig)

    with open(out_dir / "fast_calibration_v2.json", "w", encoding="utf-8") as f:
        json.dump(calibration, f, indent=2)

    print(f"Calibracion guardada en {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

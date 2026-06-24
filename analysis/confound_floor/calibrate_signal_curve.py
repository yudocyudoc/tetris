"""
calibrate_signal_curve.py — Calibra la forma de β_señal(tracker_prob) por bin de H.

Corre las regresiones por bin sobre datos generados con varios tracker_prob
(0.10, 0.25, 0.50, 0.75) y ajusta una curva no lineal. El objetivo es no depender
 de la extrapolación lineal para el cálculo de potencia.
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
from power_curve_t2 import DEFAULT_H_BINS, regress_by_bin

warnings.filterwarnings("ignore", category=RuntimeWarning)


def sigmoid_curve(p: np.ndarray, beta_max: float, k: float, p50: float) -> np.ndarray:
    """Curva sigmoide: beta_max / (1 + exp(-k*(p - p50)))."""
    return beta_max / (1.0 + np.exp(-k * (p - p50)))


def power_curve(p: np.ndarray, a: float, b: float) -> np.ndarray:
    """Curva de potencia: a * p^b."""
    return a * np.power(p, b)


def fit_curve(
    ps: np.ndarray,
    betas: np.ndarray,
    weights: Optional[np.ndarray] = None,
    curve_type: str = "sigmoid",
) -> Tuple[Dict, np.ndarray]:
    """Ajusta curva no lineal a (p, beta). Devuelve (params, fitted)."""
    if weights is None:
        weights = np.ones_like(betas)

    if curve_type == "power":
        def resid(params):
            a, b = params
            return weights * (power_curve(ps, a, b) - betas)
        result = opt.least_squares(resid, x0=[betas[-1] * 2.0, 1.0], bounds=([0, 0.1], [np.inf, 3.0]))
        a, b = result.x
        fitted = power_curve(ps, a, b)
        params = {"a": float(a), "b": float(b)}
    elif curve_type == "sigmoid":
        def resid(params):
            beta_max, k, p50 = params
            return weights * (sigmoid_curve(ps, beta_max, k, p50) - betas)
        result = opt.least_squares(
            resid,
            x0=[betas[-1] * 1.5, 5.0, 0.3],
            bounds=([betas[-1], 0.1, 0.0], [betas[-1] * 5.0, 50.0, 0.8]),
        )
        beta_max, k, p50 = result.x
        fitted = sigmoid_curve(ps, beta_max, k, p50)
        params = {"beta_max": float(beta_max), "k": float(k), "p50": float(p50)}
    else:
        raise ValueError(f"curve_type desconocido: {curve_type}")

    ss_res = float(np.sum((betas - fitted) ** 2))
    ss_tot = float(np.sum((betas - np.mean(betas)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    params["r2"] = r2
    params["rmse"] = float(np.sqrt(ss_res / len(betas)))
    return params, fitted


def beta_signal_for_tracker_prob_calibrated(
    p: float,
    params: Dict,
    curve_type: str,
) -> float:
    """Evalúa la curva calibrada en p."""
    if curve_type == "power":
        return power_curve(np.array([p]), params["a"], params["b"])[0]
    elif curve_type == "sigmoid":
        return sigmoid_curve(np.array([p]), params["beta_max"], params["k"], params["p50"])[0]
    else:
        raise ValueError(curve_type)


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrar curva beta_señal(tracker_prob)")
    parser.add_argument("--results_dirs", type=str, required=True,
                        help="Comma-separated list de directorios con decisions_log_k5.parquet")
    parser.add_argument("--tracker_probs", type=str, required=True,
                        help="Comma-separated list de tracker_prob correspondientes")
    parser.add_argument("--curve_type", type=str, default="power",
                        choices=["power", "sigmoid"])
    parser.add_argument("--out", type=str, default="out_calibrate_signal")
    args = parser.parse_args()

    dirs = [Path(d.strip()) for d in args.results_dirs.split(",")]
    ps = [float(p.strip()) for p in args.tracker_probs.split(",")]
    if len(dirs) != len(ps):
        print("[error] results_dirs y tracker_probs deben tener la misma longitud")
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Recalcular beta_señal por bin para cada p
    data_by_bin: Dict[str, List[Tuple[float, float, float, float]]] = {label: [] for _, _, label in DEFAULT_H_BINS}

    for p, res_dir in zip(ps, dirs):
        parquet = res_dir / "decisions_log_k5.parquet"
        if not parquet.exists():
            print(f"[error] No se encontró {parquet}")
            return 1
        df = pd.read_parquet(parquet)
        # piezas por partida: asumimos no_censorship y max_pieces=500
        results_json = res_dir / "resultados_piso_k5.json"
        max_pieces = 500
        if results_json.exists():
            with open(results_json, encoding="utf-8") as f:
                meta = json.load(f)
            max_pieces = meta.get("params", {}).get("max_pieces", 500)
        bin_res = regress_by_bin(df, DEFAULT_H_BINS, pieces_per_game=float(max_pieces))
        for label, res in bin_res.items():
            if "error" in res:
                continue
            beta = res["tracker_bruto"]["beta"]
            ci_low = res["tracker_bruto"]["ci_low"]
            ci_high = res["tracker_bruto"]["ci_high"]
            se = (ci_high - ci_low) / (2 * 1.96) if ci_high is not None and ci_low is not None else np.nan
            data_by_bin[label].append((p, beta, se, res["n_decisions_unique"]))

    # Ajustar curva por bin
    calibration = {}
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    p_fine = np.linspace(0.01, 1.0, 200)
    for idx, (h_min, h_max, label) in enumerate(DEFAULT_H_BINS):
        ax = axes[idx]
        points = data_by_bin[label]
        if len(points) < 3:
            ax.set_title(f"H={label} (insuficientes puntos)")
            continue
        ps_arr = np.array([x[0] for x in points])
        betas_arr = np.array([x[1] for x in points])
        ses_arr = np.array([x[2] for x in points])
        weights = 1.0 / np.clip(ses_arr, 1e-6, np.inf)

        params, fitted = fit_curve(ps_arr, betas_arr, weights=weights, curve_type=args.curve_type)
        fitted_fine = np.array([
            beta_signal_for_tracker_prob_calibrated(p, params, args.curve_type)
            for p in p_fine
        ])

        calibration[label] = {
            "H_range": [h_min, h_max],
            "curve_type": args.curve_type,
            "params": params,
            "data": [
                {"p": float(p), "beta": float(b), "se": float(se), "n_dec": int(n)}
                for p, b, se, n in points
            ],
        }

        ax.errorbar(ps_arr, betas_arr, yerr=1.96 * ses_arr, fmt="o", capsize=4, label="datos")
        ax.plot(p_fine, fitted_fine, "-", label=f"ajuste {args.curve_type}")
        # lineal de referencia
        if 0.5 in ps_arr:
            beta_at_05 = betas_arr[np.argmin(np.abs(ps_arr - 0.5))]
            ax.plot(p_fine, p_fine * (beta_at_05 / 0.5), "--", alpha=0.6, label="lineal desde 0.5")
        ax.set_xlabel("tracker_prob")
        ax.set_ylabel("beta_señal")
        ax.set_title(f"H={label}  R²={params['r2']:.2f}")
        ax.set_xlim(0, 1)
        ax.grid(True, ls=":", alpha=0.5)
        ax.legend(fontsize=7)

    # Panel de resumen
    ax_summary = axes[-1]
    ax_summary.axis("off")
    summary_lines = ["Parametros de ajuste por bin", "=============================="]
    for label, cal in calibration.items():
        summary_lines.append(f"\nH={label} ({cal['curve_type']}):")
        for k, v in cal["params"].items():
            summary_lines.append(f"  {k}: {v:.4f}")
    ax_summary.text(0.05, 0.95, "\n".join(summary_lines), transform=ax_summary.transAxes,
                    fontsize=9, verticalalignment="top", fontfamily="monospace")

    fig.suptitle("Calibracion de beta_señal(tracker_prob) por bin de H", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / "fig_calibration_signal_curve.png", dpi=150)
    plt.close(fig)

    with open(out_dir / "calibration.json", "w", encoding="utf-8") as f:
        json.dump(calibration, f, indent=2)

    print(f"Calibracion guardada en {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

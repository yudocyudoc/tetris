"""
power_curve_t2.py — Curva de potencia para Fase 1A (acomodación t+2).

Construye la curva N(tracker_prob) por bin de H, usando:
1. Piso como intervalo: la separación efectiva se calcula contra el extremo del IC
   del piso que más reduce la señal (conservador).
2. N efectivo en decisiones útiles: escala con decisiones cuyo predictor intra-decisión
   no es degenerado, no con piezas totales.
3. Umbral de factibilidad: traduce N(decisiones) a sesiones humanas usando el ratio
   empírico decisiones/partida y un rango de piezas por sesión realista.

La pregunta explícita que responde:
    "¿Cuál es el tracker_prob mínimo detectable con el N que una campaña humana
     realista entrega, y está por debajo del tracking humano plausible?"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st

# Añadir el directorio del script al path para importar funciones del simulador
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from confound_floor_t2 import (
    BOARD_WIDTH,
    diagnostic_intra_decision_variance,
    extract_beta,
    fit_conditional_logit,
    fit_conditional_logit_l2,
    run_regressions,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Configuración por defecto
# ---------------------------------------------------------------------------
DEFAULT_H_BINS: List[Tuple[int, int, str]] = [
    (4, 6, "4-6"),
    (7, 8, "7-8"),
    (9, 10, "9-10"),
    (11, 12, "11-12"),
    (13, 15, "13-15"),
]

FEATURE_COLS_BRUTO = ["p_grad_excess", "p_stat_clase"]
FEATURE_COLS_ORACLE = FEATURE_COLS_BRUTO + [
    "base_val",
    "base_val_z",
    "base_val_z2",
    "p_stat_clase_z",
    "p_stat_z2",
    "base_val_x_pstat_z",
    "res_n_holes",
    "res_bumpiness",
    "res_full_h_max",
    "res_full_h_std",
    "res_full_h_var",
] + [f"res_col_h{c}" for c in range(BOARD_WIDTH)] + [f"res_col_holes{c}" for c in range(BOARD_WIDTH)]

# Rango de piezas por sesión humano, según blueprint y datos de campo.
PIECES_PER_SESSION_RANGE = (100, 470)

# Campaña realista: 10-15 sesiones por condición (blueprint instrumentado).
SESSIONS_PER_CAMPAIGN_RANGE = (10, 15)


# ---------------------------------------------------------------------------
# Regresiones por bin de H
# ---------------------------------------------------------------------------
def regress_by_bin(
    df: pd.DataFrame,
    h_bins: List[Tuple[int, int, str]],
    pieces_per_game: Optional[float] = None,
) -> Dict[str, Dict]:
    """Corre bruto/oráculo para tracker/no-tracker dentro de cada bin de H."""
    results = {}
    for h_min, h_max, label in h_bins:
        df_bin = df[(df["H"] >= h_min) & (df["H"] <= h_max)].copy()
        if df_bin["decision_id"].nunique() < 50:
            results[label] = {"error": "muy pocas decisiones"}
            continue

        res_t = run_regressions(df_bin, "tracker", FEATURE_COLS_BRUTO, FEATURE_COLS_ORACLE)
        res_nt = run_regressions(df_bin, "no_tracker", FEATURE_COLS_BRUTO, FEATURE_COLS_ORACLE)

        b_t, p_t, ci_t_l, ci_t_h = extract_beta(res_t["model_bruto"], "p_grad_excess")
        b_nt, p_nt, ci_nt_l, ci_nt_h = extract_beta(res_nt["model_oracle"], "p_grad_excess")

        diag = diagnostic_intra_decision_variance(df_bin)

        n_games = df_bin["game_id"].nunique()
        n_dec_unique = df_bin["decision_id"].nunique()
        ppg = pieces_per_game if pieces_per_game is not None else float(
            df.groupby("game_id")["piece_idx"].max().mean()
        )

        results[label] = {
            "H_range": [h_min, h_max],
            "n_games": int(n_games),
            "n_decisions_unique": int(n_dec_unique),
            "decisions_per_game": float(n_dec_unique / n_games) if n_games else np.nan,
            "pieces_per_game": float(ppg),
            "tracker_bruto": {
                "beta": b_t,
                "pvalue": p_t,
                "ci_low": ci_t_l,
                "ci_high": ci_t_h,
            },
            "no_tracker_oracle": {
                "beta": b_nt,
                "pvalue": p_nt,
                "ci_low": ci_nt_l,
                "ci_high": ci_nt_h,
            },
            "intra_decision": diag,
        }
    return results


# ---------------------------------------------------------------------------
# Cálculo de potencia
# ---------------------------------------------------------------------------
def conditional_logit_info_per_dec(
    mean_intra_std: float,
    k: int,
) -> float:
    """
    Información de Fisher por decisión (bajo H0) para conditional logit.
    I_dec = ((k-1)/k) * sigma^2, donde sigma^2 es la varianza intra-decisión del predictor.
    """
    sigma2 = mean_intra_std ** 2
    return ((k - 1) / k) * sigma2


def required_n_decisions(
    delta: float,
    mean_intra_std: float,
    k: int,
    alpha: float = 0.05,
    power: float = 0.80,
) -> Optional[float]:
    """Número de decisiones útiles requeridas para detectar delta con potencia dada."""
    if delta <= 0 or mean_intra_std <= 0:
        return np.inf
    z_alpha = st.norm.ppf(1 - alpha / 2)
    z_beta = st.norm.ppf(power)
    info_per_dec = conditional_logit_info_per_dec(mean_intra_std, k)
    n = ((z_alpha + z_beta) ** 2) / (delta ** 2 * info_per_dec)
    return n


def effective_separation(
    beta_signal: float,
    ci_low_floor: Optional[float],
    ci_high_floor: Optional[float],
) -> Optional[float]:
    """
    Separación efectiva conservadora.
    Como la señal es positiva, el piso más desfavorable es el extremo superior del IC
    (un piso positivo reduce la separación). Si no hay IC, se usa el punto del piso.
    """
    if beta_signal is None:
        return None
    if ci_high_floor is not None:
        floor_worst = ci_high_floor
    elif ci_low_floor is not None:
        floor_worst = ci_low_floor
    else:
        return None
    sep = beta_signal - floor_worst
    return sep if sep > 0 else 0.0


def beta_signal_for_tracker_prob(
    tracker_prob: float,
    beta_anchor: float,
    tracker_prob_anchor: float = 0.5,
) -> float:
    """
    Extrapola la señal a otros tracker_prob asumiendo linealidad desde el origen.
    Es una aproximación; el script documenta explícitamente que esto debe calibrarse.
    """
    if tracker_prob_anchor <= 0:
        raise ValueError("tracker_prob_anchor debe ser > 0")
    return tracker_prob * (beta_anchor / tracker_prob_anchor)


def build_power_curve(
    bin_result: Dict,
    tracker_prob_grid: np.ndarray,
    alpha: float = 0.05,
    power: float = 0.80,
) -> Dict[str, np.ndarray]:
    """Construye curva N vs tracker_prob para un bin dado."""
    beta_anchor = bin_result["tracker_bruto"]["beta"]
    ci_low_floor = bin_result["no_tracker_oracle"]["ci_low"]
    ci_high_floor = bin_result["no_tracker_oracle"]["ci_high"]
    mean_intra_std = bin_result["intra_decision"]["p_grad_excess"]["mean_intra_std"]
    k = 5  # default del experimento

    n_required = []
    separations = []
    for p in tracker_prob_grid:
        beta_sig = beta_signal_for_tracker_prob(p, beta_anchor)
        sep = effective_separation(beta_sig, ci_low_floor, ci_high_floor)
        separations.append(sep)
        if sep is None or sep <= 0:
            n_required.append(np.inf)
        else:
            n = required_n_decisions(sep, mean_intra_std, k, alpha, power)
            # Ajustar por decisiones con varianza intra nula (no aportan)
            frac_zero = bin_result["intra_decision"]["p_grad_excess"]["frac_zero_std"]
            n_required.append(n / (1 - frac_zero) if n is not None else np.inf)

    return {
        "tracker_prob": tracker_prob_grid,
        "beta_signal": beta_anchor * (tracker_prob_grid / 0.5),
        "separation": np.array(separations),
        "n_decisions_required": np.array(n_required),
    }


# ---------------------------------------------------------------------------
# Traducción a sesiones y umbral de factibilidad
# ---------------------------------------------------------------------------
def sessions_for_n_decisions(
    n_dec: float,
    decisions_per_game: float,
    pieces_per_game: float,
    pieces_per_session: float,
) -> float:
    """Convierte N(decisiones) a N(sesiones) dado el rendimiento empírico."""
    if decisions_per_game <= 0 or pieces_per_game <= 0 or pieces_per_session <= 0:
        return np.inf
    games = n_dec / decisions_per_game
    pieces = games * pieces_per_game
    return pieces / pieces_per_session


def min_detectable_tracker_prob(
    n_dec_available: float,
    beta_anchor: float,
    tracker_prob_anchor: float,
    ci_high_floor: float,
    mean_intra_std: float,
    k: int,
    alpha: float = 0.05,
    power: float = 0.80,
    frac_zero_std: float = 0.0,
) -> Optional[float]:
    """
    Resuelve para tracker_prob tal que N_requerido(n_dec_available) = disponible.
    delta = beta_signal(p) - ci_high_floor.
    """
    z_alpha = st.norm.ppf(1 - alpha / 2)
    z_beta = st.norm.ppf(power)
    info_per_dec = conditional_logit_info_per_dec(mean_intra_std, k)
    n_effective = n_dec_available * (1 - frac_zero_std)
    delta_min = np.sqrt((z_alpha + z_beta) ** 2 / (n_effective * info_per_dec))
    slope = beta_anchor / tracker_prob_anchor
    p_min = (delta_min + ci_high_floor) / slope
    return float(p_min) if p_min > 0 else 0.0


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------
def plot_power_curves(
    power_curves: Dict[str, Dict],
    bin_results: Dict[str, Dict],
    out_dir: Path,
    pieces_per_session_range: Tuple[float, float] = PIECES_PER_SESSION_RANGE,
    sessions_per_campaign_range: Tuple[float, float] = SESSIONS_PER_CAMPAIGN_RANGE,
    alpha: float = 0.05,
    power: float = 0.80,
) -> None:
    """Genera figura principal: N(sesiones) vs tracker_prob por bin, con umbrales."""
    n_bins = len(power_curves)
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    tracker_prob_grid = None
    for idx, (label, curve) in enumerate(power_curves.items()):
        ax = axes[idx]
        tracker_prob_grid = curve["tracker_prob"]
        br = bin_results[label]

        # Curvas de sesiones por bin
        for pieces_per_session in pieces_per_session_range:
            sessions = np.array([
                sessions_for_n_decisions(
                    n, br["decisions_per_game"], br["pieces_per_game"], pieces_per_session
                )
                if np.isfinite(n) else np.nan
                for n in curve["n_decisions_required"]
            ])
            ax.plot(tracker_prob_grid, sessions, alpha=0.8,
                    label=f"{pieces_per_session:.0f} piezas/sesión")

        # Banda de campaña realista
        sess_low, sess_high = sessions_per_campaign_range
        ax.axhspan(sess_low, sess_high, color="green", alpha=0.1,
                   label=f"campaña {sess_low:.0f}-{sess_high:.0f} sesiones")

        # Umbral de factibilidad: una sesión
        for pieces_per_session in pieces_per_session_range:
            n_dec_available = pieces_per_session / br["pieces_per_game"] * br["decisions_per_game"]
            p_min = min_detectable_tracker_prob(
                n_dec_available,
                br["tracker_bruto"]["beta"],
                0.5,
                br["no_tracker_oracle"]["ci_high"],
                br["intra_decision"]["p_grad_excess"]["mean_intra_std"],
                5,
                alpha,
                power,
                br["intra_decision"]["p_grad_excess"]["frac_zero_std"],
            )
            if p_min is not None and 0 <= p_min <= 1:
                ax.axvline(p_min, color="C3", linestyle="--", alpha=0.7,
                           label=f"1 sesión ({pieces_per_session:.0f} p) → p_min={p_min:.2f}")

        # Umbral de factibilidad: campaña (rango)
        for n_sessions in sessions_per_campaign_range:
            for pieces_per_session in pieces_per_session_range:
                total_pieces = n_sessions * pieces_per_session
                n_dec_available = total_pieces / br["pieces_per_game"] * br["decisions_per_game"]
                p_min = min_detectable_tracker_prob(
                    n_dec_available,
                    br["tracker_bruto"]["beta"],
                    0.5,
                    br["no_tracker_oracle"]["ci_high"],
                    br["intra_decision"]["p_grad_excess"]["mean_intra_std"],
                    5,
                    alpha,
                    power,
                    br["intra_decision"]["p_grad_excess"]["frac_zero_std"],
                )
                if p_min is not None and 0 <= p_min <= 1:
                    ax.axvline(p_min, color="C2", linestyle="-.", alpha=0.7,
                               label=f"{n_sessions:.0f} sesiones → p_min={p_min:.2f}")

        ax.set_xlabel("tracker_prob")
        ax.set_ylabel("Sesiones requeridas")
        ax.set_title(f"H={label}  (n_dec={br['n_decisions_unique']})")
        ax.set_yscale("log")
        ax.set_ylim(0.1, 500)
        ax.grid(True, which="both", ls=":", alpha=0.5)
        ax.legend(fontsize=7, loc="upper right")

    # Panel extra: resumen de p_min por escenario
    ax_summary = axes[-1]
    ax_summary.axis("off")
    summary_text = build_summary_table(power_curves, bin_results, alpha, power)
    ax_summary.text(0.05, 0.95, summary_text, transform=ax_summary.transAxes,
                    fontsize=9, verticalalignment="top", fontfamily="monospace")

    fig.suptitle(
        f"Potencia condicional logit (α={alpha}, power={power})\n"
        "Piso como extremo desfavorable del IC; N efectivo en decisiones útiles",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / "fig_power_curve_sessions.png", dpi=150)
    plt.close(fig)


def fmt_pmin(p: Optional[float]) -> str:
    if p is None:
        return "NA"
    if p >= 1.0:
        return ">1"
    return f"{p:.2f}"


def build_summary_table(
    power_curves: Dict[str, Dict],
    bin_results: Dict[str, Dict],
    alpha: float,
    power: float,
) -> str:
    """Texto resumen con p_min por bin y escenario."""
    lines = [
        "p_min detectable por bin y escenario",
        "====================================",
        "",
        f"{'Bin':<8} {'1 sesion':<14} {'Campana':<14} {'beta_senal(0.5)':<16} {'piso CI_high':<14}",
        "-" * 70,
    ]
    for label in power_curves:
        br = bin_results[label]
        beta_sig = br["tracker_bruto"]["beta"]
        ci_high = br["no_tracker_oracle"]["ci_high"]
        std = br["intra_decision"]["p_grad_excess"]["mean_intra_std"]
        frac_zero = br["intra_decision"]["p_grad_excess"]["frac_zero_std"]

        p_min_session = []
        for pieces_per_session in PIECES_PER_SESSION_RANGE:
            n_dec = pieces_per_session / br["pieces_per_game"] * br["decisions_per_game"]
            p = min_detectable_tracker_prob(n_dec, beta_sig, 0.5, ci_high, std, 5, alpha, power, frac_zero)
            if p is not None:
                p_min_session.append(min(p, 1.0))

        p_min_campaign = []
        for n_sessions in SESSIONS_PER_CAMPAIGN_RANGE:
            for pieces_per_session in PIECES_PER_SESSION_RANGE:
                total_pieces = n_sessions * pieces_per_session
                n_dec = total_pieces / br["pieces_per_game"] * br["decisions_per_game"]
                p = min_detectable_tracker_prob(n_dec, beta_sig, 0.5, ci_high, std, 5, alpha, power, frac_zero)
                if p is not None:
                    p_min_campaign.append(min(p, 1.0))

        session_str = f"[{fmt_pmin(min(p_min_session))}-{fmt_pmin(max(p_min_session))}]"
        campaign_str = f"[{fmt_pmin(min(p_min_campaign))}-{fmt_pmin(max(p_min_campaign))}]"
        lines.append(
            f"{label:<8} {session_str:<14} {campaign_str:<14} "
            f"{beta_sig:<16.2f} {ci_high:<14.2f}"
        )
    lines.append("")
    lines.append("Notas:")
    lines.append("- p_min = tracker_prob minimo detectable al 80% de potencia.")
    lines.append("- '>1' = ni tracker_prob=1.0 es detectable en ese escenario.")
    lines.append("- '1 sesion' usa 100-470 piezas/sesion.")
    lines.append("- 'Campana' usa 10-15 sesiones.")
    lines.append("- beta_senal extrapolado linealmente desde tracker_prob=0.5.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Curva de potencia Fase 1A t+2")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directorio con decisions_log_k5.parquet y resultados_piso_k5.json")
    parser.add_argument("--out", type=str, default="out_power_t2",
                        help="Directorio de salida")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--power", type=float, default=0.80)
    parser.add_argument("--pieces_per_session_min", type=float, default=100)
    parser.add_argument("--pieces_per_session_max", type=float, default=470)
    parser.add_argument("--sessions_min", type=float, default=10)
    parser.add_argument("--sessions_max", type=float, default=15)
    parser.add_argument("--tracker_prob_max", type=float, default=1.0)
    parser.add_argument("--n_grid", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results_dir = Path(args.results_dir)
    parquet_path = results_dir / "decisions_log_k5.parquet"
    if not parquet_path.exists():
        print(f"[error] No se encontró {parquet_path}")
        return 1

    print(f"Cargando datos de {parquet_path}...")
    df = pd.read_parquet(parquet_path)

    # Determinar piezas por partida de manera robusta
    results_json_path = results_dir / "resultados_piso_k5.json"
    max_pieces = 500
    no_censorship = False
    if results_json_path.exists():
        with open(results_json_path, encoding="utf-8") as f:
            res_meta = json.load(f)
        max_pieces = res_meta.get("params", {}).get("max_pieces", 500)
        no_censorship = res_meta.get("params", {}).get("no_censorship", False)
    if no_censorship:
        # Con no_censorship cada partida corre exactamente max_pieces piezas.
        pieces_per_game = float(max_pieces)
    else:
        # Sin no_censorship no tenemos contador fiable de piezas totales;
        # usar max_pieces como upper bound conservador.
        pieces_per_game = float(max_pieces)
    print(f"Piezas por partida usadas para conversion: {pieces_per_game} "
          f"(no_censorship={no_censorship}, max_pieces={max_pieces})")

    print("Calculando regresiones por bin de H...")
    bin_results = regress_by_bin(df, DEFAULT_H_BINS, pieces_per_game=pieces_per_game)

    # Guardar resultados por bin
    with open(out_dir / "bin_results.json", "w", encoding="utf-8") as f:
        json.dump(bin_results, f, indent=2, default=str)

    for label, res in bin_results.items():
        if "error" in res:
            print(f"  {label}: {res['error']}")
            continue
        b_t = res["tracker_bruto"]["beta"]
        ci_t = (res["tracker_bruto"]["ci_low"], res["tracker_bruto"]["ci_high"])
        b_nt = res["no_tracker_oracle"]["beta"]
        ci_nt = (res["no_tracker_oracle"]["ci_low"], res["no_tracker_oracle"]["ci_high"])
        print(f"  H={label}: n_dec={res['n_decisions_unique']}, "
              f"tracker_bruto beta={b_t:.2f} CI({ci_t[0]:.2f},{ci_t[1]:.2f}), "
              f"piso_oracle beta={b_nt:.2f} CI({ci_nt[0]:.2f},{ci_nt[1]:.2f})")

    print("\nConstruyendo curvas de potencia...")
    tracker_prob_grid = np.linspace(0.01, args.tracker_prob_max, args.n_grid)
    power_curves = {}
    for label, res in bin_results.items():
        if "error" in res:
            continue
        power_curves[label] = build_power_curve(res, tracker_prob_grid, args.alpha, args.power)

    with open(out_dir / "power_curves.json", "w", encoding="utf-8") as f:
        # Convertir arrays numpy a listas para JSON
        curves_serializable = {
            label: {k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in curve.items()}
            for label, curve in power_curves.items()
        }
        json.dump(curves_serializable, f, indent=2)

    print("Generando figuras...")
    pieces_range = (args.pieces_per_session_min, args.pieces_per_session_max)
    sessions_range = (args.sessions_min, args.sessions_max)
    plot_power_curves(
        power_curves,
        bin_results,
        out_dir,
        pieces_range,
        sessions_range,
        args.alpha,
        args.power,
    )

    # Respuesta explícita a la pregunta del blueprint
    answer = answer_question(power_curves, bin_results, pieces_range, sessions_range, args.alpha, args.power)
    print("\n" + "=" * 70)
    print(answer)
    print("=" * 70)
    (out_dir / "answer.txt").write_text(answer, encoding="utf-8")

    print(f"\nSalidas en: {out_dir.resolve()}")
    return 0


def answer_question(
    power_curves: Dict[str, Dict],
    bin_results: Dict[str, Dict],
    pieces_range: Tuple[float, float],
    sessions_range: Tuple[float, float],
    alpha: float,
    power: float,
) -> str:
    """Responde la pregunta explicita del usuario."""
    lines = [
        "PREGUNTA QUE LA CURVA RESPONDE:",
        "Cual es el tracker_prob minimo detectable con el N que una campaña",
        "humana realista entrega, y esta por debajo del tracking humano plausible?",
        "",
        "RESPUESTA:",
    ]

    # p_min para campaña (escenario mas relevante)
    campaign_p_mins = []
    for label in power_curves:
        br = bin_results[label]
        p_mins = []
        for n_sessions in sessions_range:
            for pieces_per_session in pieces_range:
                total_pieces = n_sessions * pieces_per_session
                n_dec = total_pieces / br["pieces_per_game"] * br["decisions_per_game"]
                p = min_detectable_tracker_prob(
                    n_dec,
                    br["tracker_bruto"]["beta"],
                    0.5,
                    br["no_tracker_oracle"]["ci_high"],
                    br["intra_decision"]["p_grad_excess"]["mean_intra_std"],
                    5,
                    alpha,
                    power,
                    br["intra_decision"]["p_grad_excess"]["frac_zero_std"],
                )
                if p is not None:
                    p_mins.append(min(p, 1.0))
        if p_mins:
            campaign_p_mins.append((label, min(p_mins), max(p_mins)))

    lines.append(f"Con una campaña de {sessions_range[0]:.0f}-{sessions_range[1]:.0f} sesiones")
    lines.append(f"y {pieces_range[0]:.0f}-{pieces_range[1]:.0f} piezas/sesion:")
    lines.append("")
    for label, pmin, pmax in campaign_p_mins:
        pmin_str = fmt_pmin(pmin)
        pmax_str = fmt_pmin(pmax)
        note = ""
        if pmin >= 1.0:
            note = "  (no detectable ni con tracking perfecto)"
        lines.append(f"  H={label}: tracker_prob minimo detectable = {pmin_str} - {pmax_str}{note}")

    lines.append("")
    lines.append("INTERPRETACION:")
    feasible = [x for x in campaign_p_mins if x[1] < 1.0]
    if feasible:
        overall_min = min(x[1] for x in feasible)
        lines.append(f"- El p_min mas bajo across bins es {overall_min:.2f}.")
        lines.append("- Eso significa que, en el mejor bin y escenario de campaña,")
        lines.append("  el humano debe trackear al menos esa fraccion de decisiones")
        lines.append("  para que el efecto sea detectable.")
    else:
        lines.append("- En ningun bin una campaña realista alcanza para detectar")
        lines.append("  ni siquiera tracker_prob=1.0 bajo los supuestos actuales.")
    lines.append("- Si el tracking humano real es menor que el p_min del bin relevante,")
    lines.append("  el estudio conductual puro no tendra potencia para detectarlo.")
    lines.append("- Eso no significa que el efecto no exista; significa que haria falta")
    lines.append("  un probe exogeno (manipulacion del preview/bag) para aislarlo.")
    lines.append("")
    lines.append("FRANQUEZA METODOLOGICA:")
    lines.append("- tracker_prob es el propio efecto que se quiere medir.")
    lines.append("- La curva dice 'para un humano que trackea p, necesito N(p)'.")
    lines.append("- El p_min no se conocera hasta recolectar datos humanos pilotos.")
    lines.append("- La extrapolacion lineal de beta desde tracker_prob=0.5 debe validarse")
    lines.append("  con simulaciones adicionales antes de dimensionar un estudio final.")

    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())

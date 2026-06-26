"""
confound_floor_stress_t2.py — Prueba de estres de featurizacion (Fase 1A §11.4-bis).

Dos piezas acopladas que activan el canal de confound A via profundidad de huecos:

1. Generador bag-en-relleno: pi_fill que inyecta S_t en la geometria del tablero
   variando la PROFUNDIDAD de los huecos segun n_JL_restantes (piezas J/L restantes
   en la bolsa actual). El oraculo L2 cuenta huecos por columna (res_col_holes{c})
   pero no su profundidad -> gap real del oraculo.

2. No-tracker sensible a profundidad: usa depth_sensitive_board_value en su decision
   t+2. Bag-ciego: solo mira geometria, no S_t. Pero como la profundidad covaría
   con S_t via el generador, su decision hereda la correlacion.

Criterio de fallo: piso del no-tracker oracle se aparta de cero en algun bin.
Oraculo: identico al de confound_floor_t2.py (sin cambios). Sin --extend_oracle.
Con --extend_oracle: añade res_col_hole_depth{c} al oraculo (test del remedio).

Flags adicionales vs confound_floor_t2.py:
  --p_bag_fill    Intensidad bag-en-relleno (0.0 = bag-ciego, 1.0 = siempre)
  --alpha_depth   Peso del termino de profundidad en depth_sensitive_board_value
  --extend_oracle Añade profundidad por columna al oraculo (test del remedio)
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.optimize as opt
import scipy.stats as st

# Importar primitivas del script principal (mismo directorio)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from confound_floor_t2 import (
    AgentParams,
    BAG,
    BOARD_HEIGHT,
    BOARD_WIDTH,
    SevenBagGenerator,
    board_full_features,
    bcts_features,
    choose_among_placements,
    clear_lines,
    column_heights,
    compatibility_class,
    current_bag_state,
    empty_board,
    extended_board_value,
    fit_conditional_logit,
    fit_conditional_logit_l2,
    p_favorable_given_class,
    p_stat_favorable_class,
    p_t2_distribution,
    pi_fill_base,
    place_piece,
    valid_placements,
)

# ---------------------------------------------------------------------------
# Funciones nuevas: profundidad de huecos y generador bag-en-relleno
# ---------------------------------------------------------------------------

def hole_depth_score(board: np.ndarray) -> float:
    """Suma de (fila_hueco - fila_superficie) para cada hueco en cada columna.

    Un hueco 'superficial' (cerca del tope de la columna) tiene profundidad baja.
    Un hueco 'enterrado' (lejos del tope) tiene profundidad alta.
    El oraculo captura res_col_holes{c} (conteo) pero no la distribucion vertical.
    """
    score = 0.0
    for c in range(BOARD_WIDTH):
        col = board[:, c]
        occ = np.where(col == 1)[0]
        if len(occ) == 0:
            continue
        surface_row = int(occ[0])
        for r in range(surface_row, BOARD_HEIGHT):
            if board[r, c] == 0:
                score += float(r - surface_row)
    return score


def hole_depth_per_col(board: np.ndarray) -> np.ndarray:
    """Profundidad maxima del hueco mas profundo por columna (para oraculo extendido)."""
    depths = np.zeros(BOARD_WIDTH, dtype=float)
    for c in range(BOARD_WIDTH):
        col = board[:, c]
        occ = np.where(col == 1)[0]
        if len(occ) == 0:
            continue
        surface_row = int(occ[0])
        max_depth = 0
        for r in range(surface_row, BOARD_HEIGHT):
            if board[r, c] == 0:
                depth = r - surface_row
                if depth > max_depth:
                    max_depth = depth
        depths[c] = float(max_depth)
    return depths


def depth_sensitive_board_value(board: np.ndarray, alpha_depth: float = 0.4) -> float:
    """extended_board_value + penalizacion por profundidad de huecos.

    alpha_depth=0.4 => ~20% del peso del termino de huecos (2.0).
    El no-tracker prefiere tableros con huecos superficiales vs enterrados.
    """
    base = extended_board_value(board)
    return base + alpha_depth * hole_depth_score(board)


def get_top_k_placements_depth(
    board: np.ndarray,
    piece: str,
    k: int,
    alpha_depth: float = 0.4,
) -> list:
    """Top-k colocaciones usando depth_sensitive_board_value.

    El no-tracker construye su conjunto de consideracion prefiriendo tableros
    donde los huecos son superficiales, no enterrados. Esto hace que su eleccion
    entre las k alternativas dependa de la profundidad heredada del tablero.
    """
    placements = valid_placements(board, piece)
    scored = []
    for shape, col, row in placements:
        new_board = place_piece(board, shape, col, row)
        if new_board is None:
            continue
        new_board, _ = clear_lines(new_board)
        val = depth_sensitive_board_value(new_board, alpha_depth)
        scored.append((shape, col, row, val))
    scored.sort(key=lambda x: x[3])
    return scored[:k]


def bag_en_relleno_fill(
    board: np.ndarray,
    piece: str,
    gen: SevenBagGenerator,
    p_bag_fill: float,
    rng: np.random.Generator,
) -> Tuple[Optional[np.ndarray], dict]:
    """Relleno que inyecta S_t en la profundidad de huecos del tablero.

    n_JL_restantes >= 1 -> maximizar profundidad de huecos (huecos enterrados).
    n_JL_restantes = 0  -> minimizar profundidad de huecos (tablero limpio).

    Con probabilidad p_bag_fill aplica la estrategia; con 1-p_bag_fill usa pi_fill_base.
    La inyeccion es bag-aware en el RELLENO pero no en la decision t+2.
    """
    if rng.random() >= p_bag_fill:
        return pi_fill_base(board, piece)

    # Calcular n_JL_restantes en la bolsa actual
    bag_remaining = current_bag_state(gen)
    n_jl = sum(1 for p in bag_remaining if p in {"J", "L"})

    placements = valid_placements(board, piece)
    if not placements:
        return None, {}

    scored = []
    for shape, col, row in placements:
        new_board = place_piece(board, shape, col, row)
        if new_board is None:
            continue
        new_board, _ = clear_lines(new_board)
        depth = hole_depth_score(new_board)
        scored.append((shape, col, row, new_board, depth))

    if not scored:
        return None, {}

    if n_jl >= 1:
        # JL en bolsa: crear huecos profundos (maximizar depth)
        scored.sort(key=lambda x: -x[4])
    else:
        # Sin JL: tablero limpio (minimizar depth)
        scored.sort(key=lambda x: x[4])

    return scored[0][3], {}


# ---------------------------------------------------------------------------
# Simulacion de partida (solo no-tracker, generador bag-en-relleno)
# ---------------------------------------------------------------------------

def simulate_stress_game(
    seed: int,
    params_no_tracker: AgentParams,
    k: int,
    H_min: int,
    H_max: int,
    no_censorship: bool,
    max_pieces: int,
    p_bag_fill: float,
    alpha_depth: float,
) -> List[Dict]:
    """Simula una partida de estres y devuelve decisiones del no-tracker."""
    gen = SevenBagGenerator(seed)
    rng_fill = np.random.default_rng(seed + 10)
    rng_nt = np.random.default_rng(seed + 2)

    board = empty_board()
    decisions: List[Dict] = []
    decision_id = 0

    for _ in range(max_pieces):
        piece = gen.advance()

        # Avanzar tablero con generador bag-en-relleno
        board_after, _ = bag_en_relleno_fill(board, piece, gen, p_bag_fill, rng_fill)
        if board_after is None:
            if no_censorship:
                board = empty_board()
                continue
            else:
                break

        heights = column_heights(board)
        H = int(np.max(heights))

        if H_min <= H <= H_max:
            S_t = current_bag_state(gen)
            t2_dist = p_t2_distribution(gen)

            top_placements = get_top_k_placements_depth(board, piece, k, alpha_depth)
            if len(top_placements) < 2:
                board = board_after
                continue

            def no_tracker_t2_term(res_board: np.ndarray) -> float:
                count, _ = compatibility_class(res_board)
                return p_stat_favorable_class(count)

            chosen_idx_nt, _ = choose_among_placements(
                board, piece, top_placements, no_tracker_t2_term, params_no_tracker, rng_nt
            )

            for idx, (shape, col, row, base_val) in enumerate(top_placements):
                res_board = place_piece(board, shape, col, row)
                res_board, _ = clear_lines(res_board)
                count, compatible = compatibility_class(res_board)
                p_stat = p_stat_favorable_class(count)
                p_tracker = p_favorable_given_class(compatible, t2_dist)
                p_grad_excess = p_tracker - p_stat

                res_features = bcts_features(res_board)
                res_full = board_full_features(res_board)
                res_heights = column_heights(res_board)
                res_holes_per_col = np.zeros(BOARD_WIDTH, dtype=int)
                for cc in range(BOARD_WIDTH):
                    c_col = res_board[:, cc]
                    occ = np.where(c_col == 1)[0]
                    if len(occ):
                        res_holes_per_col[cc] = int(np.sum(c_col[occ[0]:] == 0))
                res_depth_per_col = hole_depth_per_col(res_board)

                row_data = {
                    "game_id": seed,
                    "decision_id": decision_id,
                    "H": H,
                    "piece": piece,
                    "S_t_size": len(S_t),
                    "S_t": ",".join(sorted(S_t)),
                    "alternative_id": idx,
                    "base_val": base_val,
                    "compatible_count": count,
                    "p_stat_clase": p_stat,
                    "p_tracker_clase": p_tracker,
                    "p_grad_excess": p_grad_excess,
                    "chosen": 1 if idx == chosen_idx_nt else 0,
                }
                row_data.update({f"res_{kk}": v for kk, v in res_features.items()})
                row_data.update({f"res_full_{kk}": v for kk, v in res_full.items()})
                for cc in range(BOARD_WIDTH):
                    row_data[f"res_col_h{cc}"] = int(res_heights[cc])
                    row_data[f"res_col_holes{cc}"] = int(res_holes_per_col[cc])
                    row_data[f"res_col_hole_depth{cc}"] = float(res_depth_per_col[cc])

                decisions.append(row_data)
            decision_id += 1

        board = board_after

    return decisions


# ---------------------------------------------------------------------------
# Worker de nivel de modulo (necesario para pickle con ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _simulate_worker(args: tuple) -> List[Dict]:
    """Envuelve simulate_stress_game para que sea serializable por pickle."""
    seed, params_nt, k, H_min, H_max, max_pieces, p_bag_fill, alpha_depth = args
    return simulate_stress_game(
        seed, params_nt, k, H_min, H_max,
        no_censorship=True, max_pieces=max_pieces,
        p_bag_fill=p_bag_fill, alpha_depth=alpha_depth,
    )


# ---------------------------------------------------------------------------
# Regresion y output (espeja confound_floor_t2.py)
# ---------------------------------------------------------------------------

H_BIN_EDGES = [4, 7, 9, 11, 13, 16]
H_BIN_LABELS = ["4-6", "7-8", "9-10", "11-12", "13-15"]


def assign_bin(H: int) -> Optional[str]:
    for i, label in enumerate(H_BIN_LABELS):
        lo, hi = H_BIN_EDGES[i], H_BIN_EDGES[i + 1] - 1
        if lo <= H <= hi:
            return label
    return None


def run_stress_regressions(df: pd.DataFrame, extend_oracle: bool) -> Dict:
    """Corre bruto y oracle sobre el no-tracker para un bin."""
    df = df.copy()

    feature_cols_bruto = ["p_grad_excess", "p_stat_clase"]

    # No-linearidades de los controles del no-tracker
    for c in ["base_val", "p_stat_clase"]:
        mean = df.groupby("decision_id")[c].transform("mean")
        std = df.groupby("decision_id")[c].transform("std").replace(0, 1)
        df[f"{c}_z"] = (df[c] - mean) / std
    df["base_val_z2"] = df["base_val_z"] ** 2
    df["p_stat_z2"] = df["p_stat_clase_z"] ** 2
    df["base_val_x_pstat_z"] = df["base_val_z"] * df["p_stat_clase_z"]

    feature_cols_oracle = feature_cols_bruto + [
        "base_val", "base_val_z", "base_val_z2",
        "p_stat_clase_z", "p_stat_z2", "base_val_x_pstat_z",
        "res_n_holes", "res_bumpiness",
        "res_full_h_max", "res_full_h_std", "res_full_h_var",
    ] + [f"res_col_h{c}" for c in range(BOARD_WIDTH)] \
      + [f"res_col_holes{c}" for c in range(BOARD_WIDTH)]

    if extend_oracle:
        # Test del remedio: añadir profundidad por columna al oraculo
        feature_cols_oracle += [f"res_col_hole_depth{c}" for c in range(BOARD_WIDTH)]

    result_bruto = fit_conditional_logit(df, feature_cols_bruto, label="no_tracker bruto")
    result_oracle = fit_conditional_logit_l2(df, feature_cols_oracle, label="no_tracker oracle", lam=1.0)

    TARGET = "p_grad_excess"

    def extract(res: Optional[dict]) -> dict:
        if res is None or "error" in res:
            return {"beta": None, "pvalue": None, "ci_low": None, "ci_high": None}
        beta = res.get("params", {}).get(TARGET)
        pval = res.get("pvalues", {}).get(TARGET)
        ci_pair = res.get("ci", {}).get(TARGET)
        return {
            "beta": float(beta) if beta is not None else None,
            "pvalue": float(pval) if pval is not None else None,
            "ci_low": float(ci_pair[0]) if ci_pair else None,
            "ci_high": float(ci_pair[1]) if ci_pair else None,
        }

    return {
        "n_decisions": int(df["decision_id"].nunique()),
        "bruto": extract(result_bruto),
        "oracle": extract(result_oracle),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Prueba de estres de featurizacion §11.4-bis")
    parser.add_argument("--n_games", type=int, default=50)
    parser.add_argument("--max_pieces", type=int, default=500)
    parser.add_argument("--tau", type=float, default=10.0)
    parser.add_argument("--board_value_weight", type=float, default=0.5)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--H_min", type=int, default=4)
    parser.add_argument("--H_max", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--p_bag_fill", type=float, default=0.5,
                        help="Intensidad bag-en-relleno (0=bag-ciego, 1=siempre). "
                             "Barrer {0, 0.25, 0.5, 1.0} para curva piso-vs-intensidad.")
    parser.add_argument("--alpha_depth", type=float, default=0.4,
                        help="Peso del termino de profundidad en depth_sensitive_board_value. "
                             "Debe ser claramente no-trivial (default 0.4).")
    parser.add_argument("--extend_oracle", action="store_true",
                        help="Añade res_col_hole_depth{c} al oraculo (test del remedio).")
    parser.add_argument("--n_workers", type=int, default=1)
    parser.add_argument("--out", type=str, default="out_stress_t2")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    params_nt = AgentParams(
        tau=args.tau,
        board_value_weight=args.board_value_weight,
        tracker_prob=0.0,
    )
    k = args.k

    print(f"Prueba de estres de featurizacion")
    print(f"  p_bag_fill={args.p_bag_fill}, alpha_depth={args.alpha_depth}, "
          f"n_games={args.n_games}, max_pieces={args.max_pieces}")
    print(f"  extend_oracle={'SI' if args.extend_oracle else 'NO'}")
    print(f"  Simulando {args.n_games} partidas...")

    all_decisions = []
    seeds = range(args.seed, args.seed + args.n_games)

    if args.n_workers == 1:
        for s in seeds:
            decs = simulate_stress_game(
                s, params_nt, k, args.H_min, args.H_max,
                no_censorship=True, max_pieces=args.max_pieces,
                p_bag_fill=args.p_bag_fill, alpha_depth=args.alpha_depth,
            )
            all_decisions.extend(decs)
            if (s - args.seed + 1) % 10 == 0:
                print(f"  {s - args.seed + 1}/{args.n_games} partidas...")
    else:
        import concurrent.futures
        import multiprocessing

        worker_args = [
            (s, params_nt, k, args.H_min, args.H_max,
             args.max_pieces, args.p_bag_fill, args.alpha_depth)
            for s in seeds
        ]

        # Usar spawn para evitar deadlock de fork en Colab (numpy/scipy inician
        # threads en el proceso padre que se congelan al hacer fork).
        ctx = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.n_workers, mp_context=ctx
        ) as ex:
            futures = {ex.submit(_simulate_worker, wa): wa[0] for wa in worker_args}
            for i, fut in enumerate(concurrent.futures.as_completed(futures)):
                all_decisions.extend(fut.result())
                if (i + 1) % 10 == 0:
                    print(f"  {i + 1}/{args.n_games} partidas...")

    if not all_decisions:
        print("ERROR: sin decisiones. Revisar parametros H_min/H_max.")
        return 1

    df = pd.DataFrame(all_decisions)
    df["H_bin"] = df["H"].apply(assign_bin)
    df = df.dropna(subset=["H_bin"])

    # Guardar log
    df.to_csv(out_dir / f"decisions_stress_k{k}.csv", index=False)
    print(f"  {len(df)} filas ({df['decision_id'].nunique()} decisiones unicas) guardadas.")

    # Regresion por bin
    bin_results = {}
    print(f"\n{'Bin':<8} {'N_dec':>6}  {'bruto_beta':>11} {'bruto_p':>8}  "
          f"{'oracle_beta':>11} {'oracle_p':>8}  {'oracle_CI':>18}  estado")
    print("-" * 90)

    for label in H_BIN_LABELS:
        df_bin = df[df["H_bin"] == label]
        if len(df_bin) < 30 or df_bin["decision_id"].nunique() < 10:
            bin_results[label] = {"error": "insuficientes decisiones"}
            print(f"{label:<8} {'<10':>6}  — skip")
            continue

        try:
            res = run_stress_regressions(df_bin, args.extend_oracle)
        except Exception as e:
            bin_results[label] = {"error": str(e)}
            print(f"{label:<8}  ERROR: {e}")
            continue

        # Guardar siempre, incluso si bruto no convergio
        bin_results[label] = {
            "n_decisions": res["n_decisions"],
            "p_bag_fill": args.p_bag_fill,
            "alpha_depth": args.alpha_depth,
            "extend_oracle": args.extend_oracle,
            "bruto": res["bruto"],
            "oracle": res["oracle"],
        }

        # Diagnostico: piso se aparta de cero?
        ob = res["oracle"]["beta"]
        op = res["oracle"]["pvalue"]
        ci_lo = res["oracle"]["ci_low"]
        ci_hi = res["oracle"]["ci_high"]

        def _fmt(val, fmt_str: str, fallback: str = "        n/d") -> str:
            return format(val, fmt_str) if val is not None else fallback

        if ob is not None and op is not None:
            estado = "FALLO — piso se aparta de cero" if op < 0.05 else "ok"
            ci_str = f"({ci_lo:.3f}, {ci_hi:.3f})" if ci_lo is not None else "n/d"
            print(f"{label:<8} {res['n_decisions']:>6}  "
                  f"{_fmt(res['bruto']['beta'], '+11.4f')} {_fmt(res['bruto']['pvalue'], '8.4f')}  "
                  f"{ob:>+11.4f} {op:>8.4f}  {ci_str:>18}  {estado}")
        else:
            print(f"{label:<8} {res['n_decisions']:>6}  "
                  f"{_fmt(res['bruto']['beta'], '+11.4f')} {_fmt(res['bruto']['pvalue'], '8.4f')}  "
                  f"{'oracle no convergido':>38}")

    output = {
        "params": {
            "n_games": args.n_games,
            "max_pieces": args.max_pieces,
            "tau": args.tau,
            "board_value_weight": args.board_value_weight,
            "k": k,
            "H_min": args.H_min,
            "H_max": args.H_max,
            "p_bag_fill": args.p_bag_fill,
            "alpha_depth": args.alpha_depth,
            "extend_oracle": args.extend_oracle,
        },
        "bin_results": bin_results,
    }
    out_path = out_dir / f"stress_results_k{k}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nResultados guardados en {out_path}")

    # Veredicto global
    failures = [
        label for label, v in bin_results.items()
        if "oracle" in v and v["oracle"]["pvalue"] is not None and v["oracle"]["pvalue"] < 0.05
    ]
    if failures:
        print(f"\nVEREDICTO: FALLO en bins {failures}")
        print("  El oraculo no absorbe la correlacion profundidad<->S_t.")
        print("  Remedio: rerun con --extend_oracle para verificar que añadir")
        print("  res_col_hole_depth{{c}} cierra el piso.")
    else:
        print("\nVEREDICTO: piso ≈ 0 en todos los bins bajo oraculo actual.")
        print("  Featurizacion robusta a la tactica bag-en-relleno con esta intensidad.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

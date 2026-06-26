"""
confound_floor_natural_t2.py — Camino B: piso del confound A natural (§11.4-bis).

Pregunta: ¿existe un confound A natural (no inyectado) que el oráculo actual
no captura? La correlación profundidad↔S_t proviene solo de la mecánica del
juego (huella de historia compartida), sin amplificación artificial.

No-tracker: bag-ciego, sensible a profundidad de hueco (weight alpha_depth declarado).
Oráculo L2: cuenta huecos por columna (res_col_holes{c}), SIN profundidad
vertical — ese es el gap a testear. Remedio (--extend_oracle): añade
res_col_hole_depth{c} al oráculo y re-corre para ver si cierra el piso.

Diagnóstico pre-regresión (obligatorio, loguear antes de la regresión):
  corr(n_JL_restantes, profundidad_tablero_en_decision) — global y por bin de H.
  Si ≈0: no hay confound natural que cerrar (resultado válido).
  Si ≠0: el canal existe; ver IC del piso para juzgar suficiencia del oráculo.

Regla de lectura del piso (IC relativo a la escala de señal ~2–3):
  IC angosto ≈ 0  → suficiencia demostrada sobre el caso real.
  IC angosto lejos de 0 → remedio: añadir profundidad al oráculo.
  IC ancho (llega a magnitudes de la señal) → no concluido, falta potencia.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from confound_floor_t2 import (
    AgentParams,
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
# Funciones de profundidad (bag-ciegas)
# ---------------------------------------------------------------------------

def hole_depth_score(board: np.ndarray) -> float:
    """Suma de (fila_hueco - fila_superficie) por hueco en cada columna."""
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
    """Profundidad máxima del hueco más profundo por columna."""
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
    """extended_board_value + penalización por profundidad de huecos."""
    return extended_board_value(board) + alpha_depth * hole_depth_score(board)


def get_top_k_placements_depth(
    board: np.ndarray,
    piece: str,
    k: int,
    alpha_depth: float = 0.4,
) -> list:
    """Top-k colocaciones ordenadas por depth_sensitive_board_value (ascendente)."""
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


# ---------------------------------------------------------------------------
# Simulación de partida natural
# ---------------------------------------------------------------------------

def simulate_natural_game(
    seed: int,
    params_no_tracker: AgentParams,
    k: int,
    H_min: int,
    H_max: int,
    no_censorship: bool,
    max_pieces: int,
    alpha_depth: float,
) -> List[Dict]:
    """Simula una partida natural: fill pi_fill_base, no-tracker sensible a profundidad.

    Diagnóstico incluido:
    - n_jl_now: n_JL_restantes en la bolsa en el momento de la decisión (S_t).
    - board_depth_at_decision: hole_depth_score(board) antes de cualquier colocación.
    Estos dos campos permiten calcular la correlación natural profundidad↔S_t.
    """
    gen = SevenBagGenerator(seed)
    rng_nt = np.random.default_rng(seed + 2)

    board = empty_board()
    decisions: List[Dict] = []
    decision_id = 0

    for _ in range(max_pieces):
        piece = gen.advance()

        # S_t en el momento de la decisión
        bag_now = current_bag_state(gen)
        n_jl_now = sum(1 for p in bag_now if p in {"J", "L"})

        # Profundidad del tablero antes de colocar nada
        board_depth_now = hole_depth_score(board)

        # Fill natural: pi_fill_base, bag-ciego, sin inyección
        board_after, _ = pi_fill_base(board, piece)
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

            for idx, (shape, col, row, depth_val) in enumerate(top_placements):
                res_board = place_piece(board, shape, col, row)
                res_board, _ = clear_lines(res_board)
                count, compatible = compatibility_class(res_board)
                p_stat = p_stat_favorable_class(count)
                p_tracker = p_favorable_given_class(compatible, t2_dist)
                p_grad_excess = p_tracker - p_stat

                # base_val para el oráculo = extended_board_value (sin penalización depth)
                base_val_oracle = extended_board_value(res_board)

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

                row_data: Dict = {
                    "game_id": seed,
                    "decision_id": decision_id,
                    "H": H,
                    "piece": piece,
                    "S_t_size": len(S_t),
                    "S_t": ",".join(sorted(S_t)),
                    "alternative_id": idx,
                    "base_val": base_val_oracle,
                    "depth_val_agent": depth_val,
                    "n_jl_now": n_jl_now,
                    "board_depth_at_decision": board_depth_now,
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
# Worker de nivel de módulo (necesario para pickle con ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _simulate_worker(args: tuple) -> List[Dict]:
    seed, params_nt, k, H_min, H_max, max_pieces, alpha_depth = args
    return simulate_natural_game(
        seed, params_nt, k, H_min, H_max,
        no_censorship=True, max_pieces=max_pieces,
        alpha_depth=alpha_depth,
    )


# ---------------------------------------------------------------------------
# Regresión
# ---------------------------------------------------------------------------

H_BIN_EDGES = [4, 7, 9, 11, 13, 16]
H_BIN_LABELS = ["4-6", "7-8", "9-10", "11-12", "13-15"]


def assign_bin(H: int) -> Optional[str]:
    for i, label in enumerate(H_BIN_LABELS):
        lo, hi = H_BIN_EDGES[i], H_BIN_EDGES[i + 1] - 1
        if lo <= H <= hi:
            return label
    return None


def run_natural_regressions(df: pd.DataFrame, extend_oracle: bool) -> Dict:
    """Corre bruto y oracle (sin features de profundidad) para un bin."""
    df = df.copy()

    feature_cols_bruto = ["p_grad_excess", "p_stat_clase"]

    for c in ["base_val", "p_stat_clase"]:
        mean = df.groupby("decision_id")[c].transform("mean")
        std = df.groupby("decision_id")[c].transform("std").replace(0, 1)
        df[f"{c}_z"] = (df[c] - mean) / std
    df["base_val_z2"] = df["base_val_z"] ** 2
    df["p_stat_z2"] = df["p_stat_clase_z"] ** 2
    df["base_val_x_pstat_z"] = df["base_val_z"] * df["p_stat_clase_z"]

    # Oráculo: cuenta huecos por columna, alturas, features BCTS.
    # SIN profundidad vertical — ese es el gap que se mide.
    feature_cols_oracle = feature_cols_bruto + [
        "base_val", "base_val_z", "base_val_z2",
        "p_stat_clase_z", "p_stat_z2", "base_val_x_pstat_z",
        "res_n_holes", "res_bumpiness",
        "res_full_h_max", "res_full_h_std", "res_full_h_var",
    ] + [f"res_col_h{c}" for c in range(BOARD_WIDTH)] \
      + [f"res_col_holes{c}" for c in range(BOARD_WIDTH)]

    if extend_oracle:
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
    parser = argparse.ArgumentParser(
        description="Camino B: piso del confound A natural §11.4-bis"
    )
    parser.add_argument("--n_games", type=int, default=300)
    parser.add_argument("--max_pieces", type=int, default=500)
    parser.add_argument("--tau", type=float, default=10.0)
    parser.add_argument("--board_value_weight", type=float, default=0.5)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--H_min", type=int, default=4)
    parser.add_argument("--H_max", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha_depth", type=float, default=0.4,
                        help="Peso del término de profundidad en la política del no-tracker. "
                             "Debe ser claramente no-trivial (default 0.4).")
    parser.add_argument("--extend_oracle", action="store_true",
                        help="Añade res_col_hole_depth{c} al oráculo (test del remedio).")
    parser.add_argument("--n_workers", type=int, default=1)
    parser.add_argument("--out", type=str, default="out_natural_t2")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    params_nt = AgentParams(
        tau=args.tau,
        board_value_weight=args.board_value_weight,
        tracker_prob=0.0,
    )
    k = args.k

    print("Camino B — Piso del confound A natural")
    print(f"  alpha_depth={args.alpha_depth}, extend_oracle={'SI' if args.extend_oracle else 'NO'}")
    print(f"  n_games={args.n_games}, max_pieces={args.max_pieces}")
    print(f"  Simulando {args.n_games} partidas...")

    all_decisions = []
    seeds = range(args.seed, args.seed + args.n_games)

    if args.n_workers == 1:
        for s in seeds:
            decs = simulate_natural_game(
                s, params_nt, k, args.H_min, args.H_max,
                no_censorship=True, max_pieces=args.max_pieces,
                alpha_depth=args.alpha_depth,
            )
            all_decisions.extend(decs)
            if (s - args.seed + 1) % 50 == 0:
                print(f"  {s - args.seed + 1}/{args.n_games} partidas...")
    else:
        worker_args = [
            (s, params_nt, k, args.H_min, args.H_max,
             args.max_pieces, args.alpha_depth)
            for s in seeds
        ]
        ctx = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.n_workers, mp_context=ctx
        ) as ex:
            futures = {ex.submit(_simulate_worker, wa): wa[0] for wa in worker_args}
            for i, fut in enumerate(concurrent.futures.as_completed(futures)):
                all_decisions.extend(fut.result())
                if (i + 1) % 50 == 0:
                    print(f"  {i + 1}/{args.n_games} partidas...")

    if not all_decisions:
        print("ERROR: sin decisiones. Revisar parámetros H_min/H_max.")
        return 1

    df = pd.DataFrame(all_decisions)
    df["H_bin"] = df["H"].apply(assign_bin)
    df = df.dropna(subset=["H_bin"])

    csv_path = out_dir / f"decisions_natural_k{k}.csv"
    df.to_csv(csv_path, index=False)
    print(f"  {len(df)} filas ({df['decision_id'].nunique()} decisiones únicas) guardadas.")

    # ------------------------------------------------------------------
    # DIAGNÓSTICO: correlación natural profundidad↔S_t
    # ------------------------------------------------------------------
    # Una fila por decisión (alternative_id=0) para no duplicar
    df_d = df[df["alternative_id"] == 0].copy()

    print(f"\nDIAGNÓSTICO CONFOUND NATURAL (alpha_depth={args.alpha_depth}):")
    print(f"  (corr(n_JL_restantes, profundidad_tablero) — positivo si profundidad")
    print(f"   tiende a ser mayor cuando quedan más piezas J/L en la bolsa)")
    print()

    if len(df_d) > 10:
        import scipy.stats as _st
        r_global, p_global = _st.pearsonr(df_d["n_jl_now"], df_d["board_depth_at_decision"])
        print(f"  {'GLOBAL':<8}  n={len(df_d):>5}  "
              f"corr={r_global:+.4f}  (p={p_global:.4f})")

        for label in H_BIN_LABELS:
            sub = df_d[df_d["H_bin"] == label]
            if len(sub) < 20:
                print(f"  {label:<8}  n={len(sub):>5}  — insuficiente")
                continue
            r, p = _st.pearsonr(sub["n_jl_now"], sub["board_depth_at_decision"])
            print(f"  {label:<8}  n={len(sub):>5}  corr={r:+.4f}  (p={p:.4f})")
    print()

    # ------------------------------------------------------------------
    # Regresión por bin
    # ------------------------------------------------------------------
    bin_results = {}
    print(f"\n{'Bin':<8} {'N_dec':>6}  {'bruto_beta':>11} {'bruto_p':>8}  "
          f"{'oracle_beta':>11} {'oracle_CI_95':>22}")
    print("-" * 78)

    SIGNAL_SCALE = 2.5  # magnitud típica de β_señal en estos bins

    for label in H_BIN_LABELS:
        df_bin = df[df["H_bin"] == label]
        if len(df_bin) < 30 or df_bin["decision_id"].nunique() < 10:
            bin_results[label] = {"error": "insuficientes decisiones"}
            print(f"{label:<8} {'<10':>6}  — skip")
            continue

        try:
            res = run_natural_regressions(df_bin, args.extend_oracle)
        except Exception as e:
            bin_results[label] = {"error": str(e)}
            print(f"{label:<8}  ERROR: {e}")
            continue

        bin_results[label] = {
            "n_decisions": res["n_decisions"],
            "alpha_depth": args.alpha_depth,
            "extend_oracle": args.extend_oracle,
            "bruto": res["bruto"],
            "oracle": res["oracle"],
        }

        ob = res["oracle"]["beta"]
        op = res["oracle"]["pvalue"]
        ci_lo = res["oracle"]["ci_low"]
        ci_hi = res["oracle"]["ci_high"]

        def _fmt(val, fmt_str: str, fallback: str = "        n/d") -> str:
            return format(val, fmt_str) if val is not None else fallback

        if ob is not None and ci_lo is not None:
            # Veredicto por IC relativo a la escala de señal
            ci_width = ci_hi - ci_lo
            ci_str = f"({ci_lo:+.3f}, {ci_hi:+.3f})"
            if ci_width > SIGNAL_SCALE:
                verdict = "IC ancho — no concluido"
            elif abs(ob) < 0.3 and ci_width < SIGNAL_SCALE:
                verdict = "IC angosto ≈ 0 — suficiente"
            else:
                verdict = "IC angosto, lejos de 0 — añadir profundidad"
            print(f"{label:<8} {res['n_decisions']:>6}  "
                  f"{_fmt(res['bruto']['beta'], '+11.4f')} {_fmt(res['bruto']['pvalue'], '8.4f')}  "
                  f"{ob:>+11.4f} {ci_str:>22}  {verdict}")
        else:
            print(f"{label:<8} {res['n_decisions']:>6}  "
                  f"{_fmt(res['bruto']['beta'], '+11.4f')} {_fmt(res['bruto']['pvalue'], '8.4f')}  "
                  f"{'oracle no convergido':>36}")

    output = {
        "params": {
            "n_games": args.n_games,
            "max_pieces": args.max_pieces,
            "tau": args.tau,
            "board_value_weight": args.board_value_weight,
            "k": k,
            "H_min": args.H_min,
            "H_max": args.H_max,
            "alpha_depth": args.alpha_depth,
            "extend_oracle": args.extend_oracle,
        },
        "bin_results": bin_results,
    }
    out_path = out_dir / f"natural_results_k{k}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nResultados guardados en {out_path}")

    # Veredicto global
    n_ancho = sum(
        1 for v in bin_results.values()
        if "oracle" in v and v["oracle"]["ci_low"] is not None
        and (v["oracle"]["ci_high"] - v["oracle"]["ci_low"]) > SIGNAL_SCALE
    )
    n_lejos = sum(
        1 for v in bin_results.values()
        if "oracle" in v and v["oracle"]["beta"] is not None
        and v["oracle"]["ci_high"] is not None
        and abs(v["oracle"]["beta"]) >= 0.3
        and (v["oracle"]["ci_high"] - v["oracle"]["ci_low"]) < SIGNAL_SCALE
    )
    n_sufic = sum(
        1 for v in bin_results.values()
        if "oracle" in v and v["oracle"]["beta"] is not None
        and v["oracle"]["ci_high"] is not None
        and abs(v["oracle"]["beta"]) < 0.3
        and (v["oracle"]["ci_high"] - v["oracle"]["ci_low"]) < SIGNAL_SCALE
    )

    print(f"\nVEREDICTO GLOBAL ({args.n_games} partidas, alpha_depth={args.alpha_depth}):")
    print(f"  Bins 'IC angosto ≈ 0' (suficiencia): {n_sufic}")
    print(f"  Bins 'IC angosto lejos de 0' (añadir profundidad): {n_lejos}")
    print(f"  Bins 'IC ancho' (no concluido): {n_ancho}")
    if n_ancho > 0:
        print("  → Falta potencia en algunos bins. Considerar n_games mayor o")
        print("    amplificación controlada si el confound natural resulta ser débil.")
    elif n_lejos > 0:
        print("  → Re-correr con --extend_oracle para verificar que añadir")
        print("    res_col_hole_depth{c} cierra el piso.")
    else:
        print("  → Oráculo actual suficiente para el confound A natural.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

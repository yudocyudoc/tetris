"""
confound_floor_t2.py — Medición del piso de confound de Fase 1A en acomodación a t+2.

Escenario B del plan corregido:
- Partidas naturales con agente base bag-ciego.
- Horizonte fijado por preview=1 (decisión de colocación de t evaluada contra t+2).
- H moderado: se filtran decisiones por altura del stack para evitar survivorship.
- Clase de favorabilidad rica: conteo de tipos t+2 compatibles con el board_resultante.
- Predictor: P_tracker(fav | S_t, clase) - P_stat(fav | clase), controlando P_stat(fav|clase).
- Estimador: conditional logit sobre el conjunto de consideración (top-k por pi_fill base).
- Diagnóstico: censura on/off no debe mover el β del no-tracker a H moderado.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as opt
import scipy.stats as st
from statsmodels.discrete.conditional_models import ConditionalLogit

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Constantes del dominio
# ---------------------------------------------------------------------------
BOARD_WIDTH = 10
BOARD_HEIGHT = 20
BAG = ["I", "J", "L", "O", "S", "T", "Z"]

# Representación de piezas como matrices binarias (filas × columnas).
PIECE_SHAPES: Dict[str, List[np.ndarray]] = {}


def _register(piece: str, base: List[List[int]]) -> None:
    PIECE_SHAPES[piece] = [np.array(base, dtype=np.int8)]


_register("I", [[1, 1, 1, 1]])
_register("O", [[1, 1], [1, 1]])
_register("T", [[1, 1, 1], [0, 1, 0]])
_register("J", [[1, 0, 0], [1, 1, 1]])
_register("L", [[0, 0, 1], [1, 1, 1]])
_register("S", [[0, 1, 1], [1, 1, 0]])
_register("Z", [[1, 1, 0], [0, 1, 1]])


def _rotations(shape: np.ndarray) -> List[np.ndarray]:
    """Devuelve las 4 rotaciones de 90° en sentido horario, sin duplicados."""
    rots = [shape]
    for _ in range(3):
        rots.append(np.rot90(rots[-1], -1))
    unique = []
    seen = set()
    for r in rots:
        key = r.tobytes()
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


PIECE_ROTATIONS = {p: _rotations(s[0]) for p, s in PIECE_SHAPES.items()}


# ---------------------------------------------------------------------------
# Utilidades de tablero
# ---------------------------------------------------------------------------
def empty_board() -> np.ndarray:
    return np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=np.int8)


def column_heights(board: np.ndarray) -> np.ndarray:
    """Altura ocupada de cada columna (0..BOARD_HEIGHT)."""
    heights = np.zeros(BOARD_WIDTH, dtype=np.int32)
    for c in range(BOARD_WIDTH):
        col = board[:, c]
        if col.any():
            heights[c] = BOARD_HEIGHT - np.where(col == 1)[0][0]
    return heights


def is_collision(board: np.ndarray, shape: np.ndarray, col: int, row: int) -> bool:
    """True si la pieza (esquina sup-izq en col,row) colisiona o sale del tablero."""
    h, w = shape.shape
    if col < 0 or col + w > BOARD_WIDTH or row < 0 or row + h > BOARD_HEIGHT:
        return True
    sub = board[row : row + h, col : col + w]
    return bool(np.any(sub & shape))


def place_piece(
    board: np.ndarray, shape: np.ndarray, col: int, row: int
) -> Optional[np.ndarray]:
    """Coloca la pieza si es válida; devuelve nuevo tablero o None."""
    if is_collision(board, shape, col, row):
        return None
    new_board = board.copy()
    h, w = shape.shape
    new_board[row : row + h, col : col + w] |= shape
    return new_board


def landing_row(board: np.ndarray, shape: np.ndarray, col: int) -> int:
    """Fila más baja (mayor índice) donde la pieza cabe sin colisión."""
    max_row = BOARD_HEIGHT - shape.shape[0]
    for row in range(max_row + 1):
        if is_collision(board, shape, col, row):
            return row - 1
    return max_row


def valid_placements(
    board: np.ndarray, piece: str
) -> List[Tuple[np.ndarray, int, int]]:
    """Todas las colocaciones de aterrizaje válidas para una pieza.

    Cada tupla es (shape, col, row).
    """
    placements = []
    for shape in PIECE_ROTATIONS[piece]:
        h, w = shape.shape
        for col in range(BOARD_WIDTH - w + 1):
            row = landing_row(board, shape, col)
            if row < 0:
                continue
            placements.append((shape, col, row))
    return placements


def clear_lines(board: np.ndarray) -> Tuple[np.ndarray, int]:
    """Elimina líneas completas y devuelve (nuevo tablero, líneas eliminadas)."""
    full = np.all(board == 1, axis=1)
    n_cleared = int(np.sum(full))
    if n_cleared == 0:
        return board.copy(), 0
    remaining = board[~full]
    new_board = np.zeros_like(board)
    new_board[-remaining.shape[0] :] = remaining
    return new_board, n_cleared


# ---------------------------------------------------------------------------
# Features de tablero
# ---------------------------------------------------------------------------
def bcts_features(board: np.ndarray, last_landing_col: Optional[int] = None) -> Dict[str, float]:
    """Vector BCTS-like del tablero (lossy, el que usará la regresión)."""
    heights = column_heights(board)
    agg_height = float(np.sum(heights))
    bumpiness = float(np.sum(np.abs(np.diff(heights))))
    well_depth = float(np.min(heights))
    max_height = float(np.max(heights))

    n_holes = 0
    for c in range(BOARD_WIDTH):
        col = board[:, c]
        occupied = np.where(col == 1)[0]
        if len(occupied) == 0:
            continue
        top = occupied[0]
        n_holes += int(np.sum(col[top:] == 0))

    row_transitions = 0
    for r in range(BOARD_HEIGHT):
        row = board[r, :]
        if np.all(row == 0):
            continue
        padded = np.concatenate([[1], row, [1]])
        row_transitions += int(np.sum(padded[:-1] != padded[1:]))

    col_transitions = 0
    for c in range(BOARD_WIDTH):
        col = board[:, c]
        if np.all(col == 0):
            continue
        padded = np.concatenate([[1], col, [1]])
        col_transitions += int(np.sum(padded[:-1] != padded[1:]))

    landing_height = float(heights[last_landing_col]) if last_landing_col is not None else max_height

    return {
        "agg_height": agg_height,
        "n_holes": float(n_holes),
        "bumpiness": bumpiness,
        "well_depth": well_depth,
        "landing_height": landing_height,
        "row_transitions": float(row_transitions),
        "col_transitions": float(col_transitions),
    }


def extended_board_value(board: np.ndarray) -> float:
    """Valor heurístico del tablero para pi_fill base (menor es mejor)."""
    heights = column_heights(board)
    max_h = np.max(heights)
    holes = 0
    for c in range(BOARD_WIDTH):
        col = board[:, c]
        occ = np.where(col == 1)[0]
        if len(occ):
            holes += int(np.sum(col[occ[0]:] == 0))
    bump = np.sum(np.abs(np.diff(heights)))
    var_h = np.var(heights)
    return 2.0 * holes + 0.5 * max_h + 0.3 * bump + 0.2 * var_h


def board_full_features(board: np.ndarray) -> Dict[str, float]:
    """Control oráculo: vector rico del tablero completo."""
    heights = column_heights(board).astype(float)
    holes_per_col = np.zeros(BOARD_WIDTH, dtype=float)
    for c in range(BOARD_WIDTH):
        col = board[:, c]
        occ = np.where(col == 1)[0]
        if len(occ):
            holes_per_col[c] = float(np.sum(col[occ[0]:] == 0))
    feats = {}
    feats["h_max"] = float(np.max(heights))
    feats["h_min"] = float(np.min(heights))
    feats["h_mean"] = float(np.mean(heights))
    feats["h_std"] = float(np.std(heights))
    feats["h_range"] = float(np.max(heights) - np.min(heights))
    feats["total_holes"] = float(np.sum(holes_per_col))
    feats["holes_std"] = float(np.std(holes_per_col))
    feats["h_skew"] = float(pd.Series(heights).skew())
    feats["h_var"] = float(np.var(heights))
    return feats


# ---------------------------------------------------------------------------
# Generador 7-bag con secuencia pre-generada
# ---------------------------------------------------------------------------
class SevenBagGenerator:
    def __init__(self, seed: int, length: int = 5000):
        self.rng = np.random.default_rng(seed)
        self.sequence: List[str] = []
        self.idx = 0
        while len(self.sequence) < length:
            bag = list(BAG)
            self.rng.shuffle(bag)
            self.sequence.extend(bag)

    def advance(self) -> str:
        piece = self.sequence[self.idx]
        self.idx += 1
        return piece

    def peek_n(self, start: int, n: int) -> List[str]:
        """Mira n piezas desde start sin consumirlas."""
        return self.sequence[start : start + n]


def current_bag_state(gen: SevenBagGenerator) -> set:
    """Devuelve el conjunto de piezas no vistas de la bolsa actual."""
    start_of_bag = (gen.idx // 7) * 7
    end_of_bag = start_of_bag + 7
    return set(gen.sequence[gen.idx : end_of_bag])


# ---------------------------------------------------------------------------
# Favorabilidad local del board_resultante para t+2 (clase rica)
# ---------------------------------------------------------------------------
def count_holes(board: np.ndarray) -> int:
    """Número de celdas vacías con al menos una celda ocupada arriba en la columna.

    Vectorizado con numpy.
    """
    # Para columnas vacías, no hay huecos.
    has_occupied = board.any(axis=0)
    first_occ = np.argmax(board, axis=0)
    rows = np.arange(BOARD_HEIGHT)[:, None]
    below_top = rows >= first_occ[None, :]
    empty = board == 0
    holes = below_top & empty & has_occupied[None, :]
    return int(np.sum(holes))


def piece_fits_clean(board: np.ndarray, piece: str) -> bool:
    """True si `piece` puede colocarse en `board` sin aumentar el número de huecos.

    "Limpio" = existe al menos una colocación legal que no incremente n_holes.
    Omitimos clear_lines por velocidad. Optimización: coloca in-place restaurando
    la región afectada para no copiar todo el tablero.
    """
    base_holes = count_holes(board)
    for shape, col, row in valid_placements(board, piece):
        h, w = shape.shape
        # Colocar in-place, guardando la región afectada
        region = board[row : row + h, col : col + w].copy()
        board[row : row + h, col : col + w] |= shape
        new_holes = count_holes(board)
        # Restaurar
        board[row : row + h, col : col + w] = region
        if new_holes <= base_holes:
            return True
    return False


def compatibility_class(board: np.ndarray) -> Tuple[int, List[str]]:
    """Clase de favorabilidad rica: conteo y lista de tipos compatibles con board.

    Devuelve (count, list_of_pieces).
    """
    compatible = [p for p in BAG if piece_fits_clean(board, p)]
    return len(compatible), compatible


# ---------------------------------------------------------------------------
# Distribución de t+2 dado S_t y P_{t+1}
# ---------------------------------------------------------------------------
def p_t2_distribution(gen: SevenBagGenerator) -> Dict[str, float]:
    """Distribución de la pieza en t+2 dado el estado actual del generador.

    El generador está posicionado en t+1 (la próxima pieza a consumir).
    t+2 es la siguiente. Si t+1 es la última del bag, t+2 es del siguiente bag
    y es uniforme sobre las 7 piezas.
    """
    dist = {p: 0.0 for p in BAG}
    if gen.idx % 7 == 6:
        # t+1 es el último del bag; t+2 proviene del siguiente bag (desconocido)
        for p in BAG:
            dist[p] = 1.0 / 7.0
    else:
        # t+2 es uniforme entre las piezas no vistas del bag actual, excluyendo t+1
        S_t = current_bag_state(gen)
        next_piece = gen.sequence[gen.idx]
        candidates = S_t - {next_piece}
        if len(candidates) == 0:
            # fallback improbable: uniforme
            for p in BAG:
                dist[p] = 1.0 / 7.0
        else:
            prob = 1.0 / len(candidates)
            for p in candidates:
                dist[p] = prob
    return dist


def p_favorable_given_class(
    compatible_pieces: List[str], t2_dist: Dict[str, float]
) -> float:
    """P(t+2 es favorable | clase, S_t) = suma de probabilidades de tipos compatibles."""
    return sum(t2_dist[p] for p in compatible_pieces)


def p_stat_favorable_class(compatible_count: int) -> float:
    """P(t+2 es favorable | clase) bajo marginal 7-bag = c/7."""
    return compatible_count / 7.0


# ---------------------------------------------------------------------------
# Agente base bag-ciego (genera partidas naturales)
# ---------------------------------------------------------------------------
def pi_fill_base(board: np.ndarray, piece: str) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
    """Coloca `piece` en `board` eligiendo la colocación con menor extended_board_value."""
    placements = valid_placements(board, piece)
    if not placements:
        return None, None
    best_board = None
    best_val = np.inf
    best_col = None
    for shape, col, row in placements:
        new_board = place_piece(board, shape, col, row)
        if new_board is None:
            continue
        new_board, _ = clear_lines(new_board)
        val = extended_board_value(new_board)
        if val < best_val:
            best_val = val
            best_board = new_board
            best_col = col
    if best_board is None:
        return None, None
    return best_board, {"last_landing_col": best_col}


def get_top_k_placements(
    board: np.ndarray, piece: str, k: int
) -> List[Tuple[np.ndarray, int, int, float]]:
    """Devuelve las top-k colocaciones según pi_fill base, con su valor."""
    placements = valid_placements(board, piece)
    scored = []
    for shape, col, row in placements:
        new_board = place_piece(board, shape, col, row)
        if new_board is None:
            continue
        new_board, _ = clear_lines(new_board)
        val = extended_board_value(new_board)
        scored.append((shape, col, row, val))
    scored.sort(key=lambda x: x[3])
    return scored[:k]


# ---------------------------------------------------------------------------
# Parámetros de agentes y decisión logística
# ---------------------------------------------------------------------------
@dataclass
class AgentParams:
    tau: float = 1.0  # temperatura del término t+2
    board_value_weight: float = 0.5  # peso del valor heurístico local
    local_value_scale: float = 1.0  # escala del valor local para logit
    tracker_prob: float = 1.0  # probabilidad de que el tracker use S_t en cada decisión


def choose_among_placements(
    board: np.ndarray,
    piece: str,
    placements: List[Tuple[np.ndarray, int, int, float]],
    t2_term_fn,
    params: AgentParams,
    rng: np.random.Generator,
) -> Tuple[int, np.ndarray]:
    """Elige una colocación del conjunto de consideración con utilidad logística.

    t2_term_fn(board_resultante) debe devolver el término de t+2 para esa alternativa.
    Devuelve (índice elegido, board_resultante).
    """
    if not placements:
        raise ValueError("No hay colocaciones en el conjunto de consideración")
    utilities = []
    resultants = []
    for shape, col, row, base_val in placements:
        new_board = place_piece(board, shape, col, row)
        # new_board no debería ser None porque placements ya fue filtrado
        new_board, _ = clear_lines(new_board)
        resultants.append(new_board)
        # Utilidad: menor base_val es mejor, así que negamos y escalamos
        local_util = -params.board_value_weight * base_val * params.local_value_scale
        t2_util = params.tau * t2_term_fn(new_board)
        utilities.append(local_util + t2_util)
    utilities = np.array(utilities)
    # Softmax
    max_u = np.max(utilities)
    exp_u = np.exp(utilities - max_u)
    probs = exp_u / np.sum(exp_u)
    chosen = rng.choice(len(placements), p=probs)
    return chosen, resultants[chosen]


# ---------------------------------------------------------------------------
# Simulación de partidas naturales
# ---------------------------------------------------------------------------
def simulate_natural_game(
    seed: int,
    params_tracker: AgentParams,
    params_no_tracker: AgentParams,
    k: int,
    H_min: int,
    H_max: int,
    no_censorship: bool = False,
    max_pieces: int = 500,
) -> Tuple[List[Dict], List[Dict]]:
    """Simula una partida natural y devuelve decisiones logueadas para tracker/no-tracker.

    El agente base genera el tablero. Luego, para cada decisión a H moderado,
    se construye el conjunto de consideración top-k y se hacen elegir a tracker
    y no-tracker. Se loguean ambas elecciones en formato largo para conditional logit.
    """
    gen = SevenBagGenerator(seed)
    rng_tracker = np.random.default_rng(seed + 1)
    rng_no_tracker = np.random.default_rng(seed + 2)

    board = empty_board()

    decisions_tracker: List[Dict] = []
    decisions_no_tracker: List[Dict] = []

    decision_id = 0

    for _ in range(max_pieces):
        piece = gen.advance()
        next_piece = gen.peek_n(gen.idx, 1)[0] if gen.idx < len(gen.sequence) else None
        if next_piece is None:
            break

        # Agente base elige colocación natural
        board_after_base, info_base = pi_fill_base(board, piece)
        if board_after_base is None:
            # Game over
            if no_censorship:
                board = empty_board()
                continue
            else:
                break

        # Altura actual del stack
        heights = column_heights(board)
        H = int(np.max(heights))

        # Solo loguear si H está en rango moderado
        if H_min <= H <= H_max:
            S_t = current_bag_state(gen)
            t2_dist = p_t2_distribution(gen)

            # Conjunto de consideración: top-k por pi_fill base
            top_placements = get_top_k_placements(board, piece, k)
            if len(top_placements) >= 2:
                # Tracker: usa S_t con probabilidad tracker_prob; con probabilidad
                # complementaria usa la creencia estacionaria (simula tracking imperfecto).
                use_tracking = rng_tracker.random() < params_tracker.tracker_prob

                def tracker_t2_term(res_board: np.ndarray) -> float:
                    _, compatible = compatibility_class(res_board)
                    return p_favorable_given_class(compatible, t2_dist)

                def no_tracker_t2_term(res_board: np.ndarray) -> float:
                    count, _ = compatibility_class(res_board)
                    return p_stat_favorable_class(count)

                t2_term_t = tracker_t2_term if use_tracking else no_tracker_t2_term

                chosen_idx_t, _ = choose_among_placements(
                    board, piece, top_placements, t2_term_t, params_tracker, rng_tracker
                )

                chosen_idx_nt, _ = choose_among_placements(
                    board, piece, top_placements, no_tracker_t2_term, params_no_tracker, rng_no_tracker
                )

                # Loguear en formato largo para conditional logit
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
                    # Huecos por columna del board_resultante
                    res_holes_per_col = np.zeros(BOARD_WIDTH, dtype=int)
                    for cc in range(BOARD_WIDTH):
                        col = res_board[:, cc]
                        occ = np.where(col == 1)[0]
                        if len(occ):
                            res_holes_per_col[cc] = int(np.sum(col[occ[0]:] == 0))

                    common = {
                        "game_id": seed,
                        "decision_id": decision_id,
                        "piece_idx": _,
                        "H": H,
                        "piece": piece,
                        "next_piece": next_piece,
                        "S_t_size": len(S_t),
                        "S_t": ",".join(sorted(S_t)),
                        "alternative_id": idx,
                        "base_val": base_val,
                        "compatible_count": count,
                        "p_stat_clase": p_stat,
                        "p_tracker_clase": p_tracker,
                        "p_grad_excess": p_grad_excess,
                    }
                    common.update({f"res_{k}": v for k, v in res_features.items()})
                    common.update({f"res_full_{k}": v for k, v in res_full.items()})
                    for c in range(BOARD_WIDTH):
                        common[f"res_col_h{c}"] = int(res_heights[c])
                        common[f"res_col_holes{c}"] = int(res_holes_per_col[c])

                    decisions_tracker.append({
                        **common,
                        "agent_type": "tracker",
                        "chosen": 1 if idx == chosen_idx_t else 0,
                    })
                    decisions_no_tracker.append({
                        **common,
                        "agent_type": "no_tracker",
                        "chosen": 1 if idx == chosen_idx_nt else 0,
                    })

                decision_id += 1

        # Avanzar tablero con la colocación del agente base
        board = board_after_base

    return decisions_tracker, decisions_no_tracker


# ---------------------------------------------------------------------------
# Conditional logit y regresiones
# ---------------------------------------------------------------------------
def _standardize(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estandariza columnas; devuelve (X_std, mean, std)."""
    mean = X.mean(axis=0)
    std = X.std(axis=0).clip(min=1e-8)
    return (X - mean) / std, mean, std


def fit_conditional_logit_l2(
    df: pd.DataFrame,
    feature_cols: List[str],
    label: str = "",
    lam: float = 1.0,
) -> Optional[Dict]:
    """Conditional logit con penalización L2 sobre las features.

    Útil cuando el oráculo incluye muchas features colineales. Los efectos
    fijos por decisión se eliminan por construcción (conditional likelihood);
    la penalización L2 solo afecta a las features.
    """
    df = df.copy()
    df = df.rename(columns={c: c.replace("-", "_").replace(" ", "_") for c in df.columns})
    feature_cols = [c.replace("-", "_").replace(" ", "_") for c in feature_cols]

    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=feature_cols + ["chosen", "decision_id"])

    n_alts = df.groupby("decision_id").size()
    valid_decisions = n_alts[n_alts >= 2].index
    df = df[df["decision_id"].isin(valid_decisions)].copy()

    chosen_per_dec = df.groupby("decision_id")["chosen"].sum()
    valid_decisions = chosen_per_dec[chosen_per_dec == 1].index
    df = df[df["decision_id"].isin(valid_decisions)].copy()

    if len(df) == 0:
        return {"error": "No hay decisiones válidas"}

    groups = df["decision_id"].values
    X_raw = df[feature_cols].values.astype(float)
    y = df["chosen"].values.astype(float)

    X, mean, std = _standardize(X_raw)
    unique_groups = np.unique(groups)
    group_idx = {g: np.where(groups == g)[0] for g in unique_groups}

    def nll(beta: np.ndarray) -> float:
        ll = 0.0
        for g in unique_groups:
            idx = group_idx[g]
            Xg = X[idx]
            yg = y[idx]
            chosen = np.where(yg == 1)[0]
            if len(chosen) != 1:
                continue
            u = Xg @ beta
            max_u = np.max(u)
            exp_u = np.exp(u - max_u)
            denom = np.sum(exp_u)
            ll += u[chosen[0]] - max_u - np.log(denom)
        return -ll + 0.5 * lam * np.sum(beta**2)

    def grad(beta: np.ndarray) -> np.ndarray:
        g = lam * beta
        for g_id in unique_groups:
            idx = group_idx[g_id]
            Xg = X[idx]
            yg = y[idx]
            chosen = np.where(yg == 1)[0]
            if len(chosen) != 1:
                continue
            u = Xg @ beta
            max_u = np.max(u)
            exp_u = np.exp(u - max_u)
            denom = np.sum(exp_u)
            probs = exp_u / denom
            g -= Xg[chosen[0]] - Xg.T @ probs
        return g

    try:
        result = opt.minimize(nll, np.zeros(X.shape[1]), jac=grad, method="L-BFGS-B")
        if not result.success:
            print(f"[fit_conditional_logit_l2 warning {label}] {result.message}")
        beta_std = result.x
        beta = beta_std / std  # coeficientes en escala original
    except Exception as e:
        print(f"[fit_conditional_logit_l2 error {label}] {e}")
        return {"error": str(e)}

    # Hessiana numérica aproximada para errores estándar e intervalos de confianza
    se = np.full(X.shape[1], np.nan)
    ci = {}
    try:
        eps = 1e-5
        H = np.zeros((X.shape[1], X.shape[1]))
        for i in range(X.shape[1]):
            beta_plus = beta_std.copy()
            beta_minus = beta_std.copy()
            beta_plus[i] += eps
            beta_minus[i] -= eps
            H[i, :] = (grad(beta_plus) - grad(beta_minus)) / (2 * eps)
        # Covarianza en espacio estandarizado; pasar a escala original
        try:
            cov_std = np.linalg.inv(H + lam * np.eye(X.shape[1]))
        except np.linalg.LinAlgError:
            cov_std = np.linalg.pinv(H + lam * np.eye(X.shape[1]))
        cov = np.diag(1.0 / std) @ cov_std @ np.diag(1.0 / std)
        se = np.sqrt(np.diag(cov))
        for i, col in enumerate(feature_cols):
            ci[col] = [float(beta[i] - 1.96 * se[i]), float(beta[i] + 1.96 * se[i])]
    except Exception as e:
        print(f"[fit_conditional_logit_l2 hessian error {label}] {e}")

    z = beta / np.clip(se, a_min=1e-8, a_max=None)
    pvalues = 2 * st.norm.sf(np.abs(z))

    summary = {
        "nobs": int(len(df)),
        "n_decisions": int(len(unique_groups)),
        "converged": bool(result.success),
        "params": {k: float(v) for k, v in zip(feature_cols, beta)},
        "pvalues": {k: float(v) for k, v in zip(feature_cols, pvalues)},
        "ci": ci,
        "loglik": float(-result.fun),
        "lam": lam,
    }
    return summary


def fit_conditional_logit(df: pd.DataFrame, feature_cols: List[str], label: str = "") -> Optional[Dict]:
    """Ajusta un conditional logit sin regularización usando statsmodels."""
    df = df.copy()
    df = df.rename(columns={c: c.replace("-", "_").replace(" ", "_") for c in df.columns})
    feature_cols = [c.replace("-", "_").replace(" ", "_") for c in feature_cols]

    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=feature_cols + ["chosen", "decision_id"])

    n_alts = df.groupby("decision_id").size()
    valid_decisions = n_alts[n_alts >= 2].index
    df = df[df["decision_id"].isin(valid_decisions)].copy()

    chosen_per_dec = df.groupby("decision_id")["chosen"].sum()
    valid_decisions = chosen_per_dec[chosen_per_dec == 1].index
    df = df[df["decision_id"].isin(valid_decisions)].copy()

    if len(df) == 0:
        return {"error": "No hay decisiones válidas"}

    try:
        model = ConditionalLogit(
            endog=df["chosen"].values,
            exog=df[feature_cols].values,
            groups=df["decision_id"].values,
        )
        result = model.fit(disp=0, method="bfgs")
    except Exception as e:
        print(f"[fit_conditional_logit error {label}] {e}")
        return {"error": str(e)}

    summary = {
        "nobs": int(getattr(result, "nobs", len(df))),
        "n_decisions": int(df["decision_id"].nunique()),
        "converged": bool(getattr(result, "mle_retvals", {}).get("converged", True)),
        "params": {k: float(v) for k, v in zip(feature_cols, result.params)},
        "pvalues": {k: float(v) for k, v in zip(feature_cols, result.pvalues)},
        "ci": {k: [float(v[0]), float(v[1])] for k, v in zip(feature_cols, result.conf_int())},
        "loglik": float(getattr(result, "llf", np.nan)),
        "aic": float(getattr(result, "aic", np.nan)),
        "bic": float(getattr(result, "bic", np.nan)),
    }
    return summary


def extract_beta(summary: Dict, var: str) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Devuelve (beta, p, CI_inf, CI_sup) para `var`."""
    if "error" in summary or var not in summary.get("params", {}):
        return None, None, None, None
    beta = summary["params"][var]
    p = summary["pvalues"][var]
    if "ci" in summary and var in summary["ci"]:
        ci = summary["ci"][var]
        return float(beta), float(p), float(ci[0]), float(ci[1])
    return float(beta), float(p), None, None


def run_regressions(df: pd.DataFrame, agent_type: str, feature_cols_bruto: List[str], feature_cols_oracle: List[str]) -> Dict:
    """Corre los modelos bruto (sin regularización) y oráculo (L2) sobre los datos de un agente."""
    df_agent = df[df["agent_type"] == agent_type].copy()

    sum_bruto = fit_conditional_logit(df_agent, feature_cols_bruto, label=f"{agent_type} bruto")
    sum_oracle = fit_conditional_logit_l2(df_agent, feature_cols_oracle, label=f"{agent_type} oracle", lam=1.0)

    return {
        "agent_type": agent_type,
        "model_bruto": sum_bruto,
        "model_oracle": sum_oracle,
    }


def decide_rama(beta: float, ci_low: float, ci_high: float, delta: float = 0.13, eps: float = 0.02) -> str:
    """Aplica la regla de decisión del blueprint."""
    if ci_low <= 0 <= ci_high or abs(beta) < eps:
        return "1_piso_limpio"
    if beta > 0 and beta < delta:
        return "2_piso_moderado"
    return "3_piso_fatal"


# ---------------------------------------------------------------------------
# Residualización del predictor contra el board_resultante
# ---------------------------------------------------------------------------
def residualize_predictor(
    df: pd.DataFrame,
    target: str = "p_tracker_clase",
    controls: List[str] = None,
) -> pd.DataFrame:
    """Residualiza `target` contra `controls` dentro de cada decisión.

    El objetivo es quitar del predictor la componente predecible por la
    geometría del board_resultante, dejando solo la señal de S_t que el
    no-tracker no puede usar.
    """
    if controls is None:
        controls = ["base_val", "p_stat_clase"]
    df = df.copy()
    residuals = np.zeros(len(df))
    for dec_id, grp in df.groupby("decision_id"):
        idx = grp.index
        X = grp[controls].values.astype(float)
        y = grp[target].values.astype(float)
        if len(grp) > len(controls) and np.linalg.matrix_rank(X) >= len(controls):
            Xc = X - X.mean(axis=0)
            yc = y - y.mean()
            beta = np.linalg.lstsq(Xc, yc, rcond=None)[0]
            resid = yc - Xc @ beta
        else:
            resid = y - y.mean()
        residuals[idx] = resid
    df["p_grad_excess_resid"] = residuals
    return df


# ---------------------------------------------------------------------------
# Diagnósticos
# ---------------------------------------------------------------------------
def diagnostic_intra_decision_variance(df: pd.DataFrame) -> Dict:
    """Reporta varianza intra-decisión de p_stat_clase y p_grad_excess."""
    stats = {}
    for var in ["p_stat_clase", "p_grad_excess", "compatible_count"]:
        grp = df.groupby("decision_id")[var]
        stds = grp.std().dropna()
        stats[var] = {
            "mean_intra_std": float(stds.mean()),
            "median_intra_std": float(stds.median()),
            "frac_zero_std": float((stds == 0).mean()),
            "n_decisions": int(len(stds)),
        }
    return stats


def plot_beta_comparison(results_censored: Dict, results_uncensored: Dict, out_dir: Path) -> None:
    """Compara β oráculo del no-tracker entre censura on/off."""
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["censura on", "censura off"]
    betas = []
    lows = []
    highs = []
    for res in [results_censored, results_uncensored]:
        b, _, ci_low, ci_high = extract_beta(res["no_tracker"]["model_oracle"], "p_grad_excess")
        betas.append(b if b is not None else 0)
        lows.append(ci_low if ci_low is not None else b if b is not None else 0)
        highs.append(ci_high if ci_high is not None else b if b is not None else 0)
    x = np.arange(len(labels))
    errs = [[b - l for b, l in zip(betas, lows)], [h - b for b, h in zip(betas, highs)]]
    ax.bar(x, betas, yerr=errs, capsize=5, color=["C0", "C1"])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("β_oráculo (p_grad_excess)")
    ax.set_title("Piso del no-tracker: censura on vs off (test de desacople)")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_censura_desacople.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Git hash
# ---------------------------------------------------------------------------
def get_git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(Path(__file__).resolve().parent.parent.parent)
        ).decode().strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Paralelización de simulación
# ---------------------------------------------------------------------------
def _simulate_one(args_tuple) -> Tuple[List[Dict], List[Dict]]:
    """Wrapper para ProcessPoolExecutor."""
    seed, params_tracker, params_no_tracker, k, H_min, H_max, no_censorship, max_pieces = args_tuple
    return simulate_natural_game(
        seed,
        params_tracker,
        params_no_tracker,
        k=k,
        H_min=H_min,
        H_max=H_max,
        no_censorship=no_censorship,
        max_pieces=max_pieces,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Medición del piso de confound de Fase 1A en acomodación t+2")
    parser.add_argument("--seed", type=int, default=42, help="Semilla base")
    parser.add_argument("--n_games", type=int, default=500, help="Número de partidas")
    parser.add_argument("--tau", type=float, default=2.0, help="Temperatura del término t+2")
    parser.add_argument("--board_value_weight", type=float, default=0.5, help="Peso del valor local")
    parser.add_argument("--tracker_prob", type=float, default=1.0,
                        help="Probabilidad de que el tracker use S_t en cada decisión (1.0=perfecto)")
    parser.add_argument("--k", type=int, default=5, help="Tamaño del conjunto de consideración")
    parser.add_argument("--H_min", type=int, default=4, help="Altura mínima para loguear")
    parser.add_argument("--H_max", type=int, default=8, help="Altura máxima para loguear (H moderado)")
    parser.add_argument("--max_pieces", type=int, default=100, help="Máximo de piezas por partida")
    parser.add_argument("--out", type=str, default="out_t2", help="Directorio de salida")
    parser.add_argument("--no_censorship", action="store_true",
                        help="Desactiva censura por game over (para validar desacople)")
    parser.add_argument("--sweep_k", action="store_true",
                        help="Barrer k=3,5,7 y reportar sensibilidad")
    parser.add_argument("--compare_with", type=str, default=None,
                        help="Directorio de la corrida censurada de referencia para figura de desacople")
    parser.add_argument("--n_workers", type=int, default=None,
                        help="Número de workers para ProcessPoolExecutor (default: min(cpu_count,8))")
    return parser.parse_args()


def main_single_run(args: argparse.Namespace, out_dir: Path) -> Dict:
    """Ejecuta una corrida (posiblemente barrido de k) y guarda resultados."""
    out_dir.mkdir(parents=True, exist_ok=True)
    git_hash = get_git_hash()
    (out_dir / "git_hash.txt").write_text(git_hash + "\n", encoding="utf-8")

    params_tracker = AgentParams(
        tau=args.tau,
        board_value_weight=args.board_value_weight,
        tracker_prob=args.tracker_prob,
    )
    params_no_tracker = AgentParams(
        tau=args.tau,
        board_value_weight=args.board_value_weight,
        tracker_prob=0.0,
    )

    ks = [3, 5, 7] if args.sweep_k else [args.k]
    results_by_k = {}

    for k in ks:
        print(f"[k={k}] Simulando {args.n_games} partidas...")
        all_decisions = []
        n_workers = args.n_workers if args.n_workers is not None else min(os.cpu_count() or 1, 8)
        print(f"    Usando {n_workers} worker(s) paralelo(s)")
        arg_tuples = [
            (
                int(args.seed + g),
                params_tracker,
                params_no_tracker,
                k,
                args.H_min,
                args.H_max,
                args.no_censorship,
                args.max_pieces,
            )
            for g in range(args.n_games)
        ]
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            for dec_t, dec_nt in executor.map(_simulate_one, arg_tuples, chunksize=max(1, args.n_games // n_workers)):
                all_decisions.extend(dec_t)
                all_decisions.extend(dec_nt)

        # Reasignar decision_id de forma global para evitar colisiones entre partidas
        id_map = {}
        next_global_id = 0
        for rec in all_decisions:
            key = (rec["game_id"], rec["decision_id"])
            if key not in id_map:
                id_map[key] = next_global_id
                next_global_id += 1
            rec["decision_id"] = id_map[key]

        df = pd.DataFrame(all_decisions)
        print(f"    Decisiones logueadas: tracker={len(df[df.agent_type=='tracker'])}, "
              f"no_tracker={len(df[df.agent_type=='no_tracker'])}, "
              f"decisiones únicas={df['decision_id'].nunique()}")

        # Variables no lineales de las componentes de utilidad del no-tracker,
        # estandarizadas dentro de cada decisión para estabilidad numérica.
        for c in ["base_val", "p_stat_clase"]:
            mean = df.groupby("decision_id")[c].transform("mean")
            std = df.groupby("decision_id")[c].transform("std").replace(0, 1)
            df[f"{c}_z"] = (df[c] - mean) / std
        df["base_val_z2"] = df["base_val_z"] ** 2
        df["p_stat_z2"] = df["p_stat_clase_z"] ** 2
        df["base_val_x_pstat_z"] = df["base_val_z"] * df["p_stat_clase_z"]

        # Guardar log
        df.to_csv(out_dir / f"decisions_log_k{k}.csv", index=False)
        df.to_parquet(out_dir / f"decisions_log_k{k}.parquet", index=False)

        # Diagnóstico de varianza intra-decisión
        diag = diagnostic_intra_decision_variance(df)
        with open(out_dir / f"diagnostic_intra_decision_k{k}.json", "w", encoding="utf-8") as f:
            json.dump(diag, f, indent=2)
        print(f"    Varianza intra-decisión p_stat_clase: mean_std={diag['p_stat_clase']['mean_intra_std']:.4f}, "
              f"frac_zero_std={diag['p_stat_clase']['frac_zero_std']:.4f}")

        # Regresiones
        # Bruto: p_grad_excess + p_stat_clase (control de geometría resultante)
        feature_cols_bruto = ["p_grad_excess", "p_stat_clase"]
        # Oráculo: bruto + utilidad flexible del no-tracker + featurización rica
        # del board_resultante. Las no-linealidades de base_val y p_stat capturan
        # la utilidad logística real del no-tracker; las features del resultado
        # absorben geometría residual.
        feature_cols_oracle = feature_cols_bruto + [
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

        results = {}
        for agent_type in ["tracker", "no_tracker"]:
            results[agent_type] = run_regressions(df, agent_type, feature_cols_bruto, feature_cols_oracle)

        # Extraer betas
        beta_nt_bruto, p_nt_bruto, ci_nt_bruto_low, ci_nt_bruto_high = extract_beta(
            results["no_tracker"]["model_bruto"], "p_grad_excess"
        )
        beta_nt_oracle, p_nt_oracle, ci_nt_oracle_low, ci_nt_oracle_high = extract_beta(
            results["no_tracker"]["model_oracle"], "p_grad_excess"
        )
        beta_t_bruto, p_t_bruto, ci_t_bruto_low, ci_t_bruto_high = extract_beta(
            results["tracker"]["model_bruto"], "p_grad_excess"
        )
        beta_t_oracle, p_t_oracle, ci_t_oracle_low, ci_t_oracle_high = extract_beta(
            results["tracker"]["model_oracle"], "p_grad_excess"
        )

        rama = decide_rama(beta_nt_bruto, ci_nt_bruto_low, ci_nt_bruto_high) if beta_nt_bruto is not None else "unknown"

        resultados = {
            "software_git_hash": git_hash,
            "params": {
                "seed": args.seed,
                "n_games": args.n_games,
                "tau": args.tau,
                "board_value_weight": args.board_value_weight,
                "k": k,
                "H_min": args.H_min,
                "H_max": args.H_max,
                "no_censorship": args.no_censorship,
            },
            "diagnostic_intra_decision": diag,
            "beta_no_tracker_bruto": {
                "beta": beta_nt_bruto,
                "pvalue": p_nt_bruto,
                "ci_low": ci_nt_bruto_low,
                "ci_high": ci_nt_bruto_high,
            },
            "beta_no_tracker_oracle": {
                "beta": beta_nt_oracle,
                "pvalue": p_nt_oracle,
                "ci_low": ci_nt_oracle_low,
                "ci_high": ci_nt_oracle_high,
            },
            "beta_tracker_bruto": {
                "beta": beta_t_bruto,
                "pvalue": p_t_bruto,
                "ci_low": ci_t_bruto_low,
                "ci_high": ci_t_bruto_high,
            },
            "beta_tracker_oracle": {
                "beta": beta_t_oracle,
                "pvalue": p_t_oracle,
                "ci_low": ci_t_oracle_low,
                "ci_high": ci_t_oracle_high,
            },
            "rama_disparada": rama,
        }

        with open(out_dir / f"resultados_piso_k{k}.json", "w", encoding="utf-8") as f:
            json.dump(resultados, f, indent=2, default=str)

        results_by_k[k] = resultados

        print(f"    no_tracker bruto:  {fmt_beta(beta_nt_bruto, ci_nt_bruto_low, ci_nt_bruto_high, p_nt_bruto)}")
        print(f"    no_tracker oracle: {fmt_beta(beta_nt_oracle, ci_nt_oracle_low, ci_nt_oracle_high, p_nt_oracle)}")
        print(f"    tracker bruto:     {fmt_beta(beta_t_bruto, ci_t_bruto_low, ci_t_bruto_high, p_t_bruto)}")
        print(f"    tracker oracle:    {fmt_beta(beta_t_oracle, ci_t_oracle_low, ci_t_oracle_high, p_t_oracle)}")
        print(f"    rama: {rama}")

    return results_by_k


def fmt_beta(b, ci_low, ci_high, p):
    if b is None:
        return "NA"
    ci_str = f"CI {ci_low:.4f}, {ci_high:.4f}" if ci_low is not None and ci_high is not None else "CI NA"
    return f"{b:.4f} ({ci_str}), p={p:.4g}"


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out)

    if args.sweep_k:
        # Correr barrido de k
        print("=== Barrido de k ===")
        results = main_single_run(args, out_dir)
        print("\n=== Resumen barrido ===")
        for k, res in sorted(results.items()):
            print(f"k={k}: no_tracker_oracle={fmt_beta(res['beta_no_tracker_oracle']['beta'], res['beta_no_tracker_oracle']['ci_low'], res['beta_no_tracker_oracle']['ci_high'], res['beta_no_tracker_oracle']['pvalue'])}, "
                  f"tracker_bruto={fmt_beta(res['beta_tracker_bruto']['beta'], res['beta_tracker_bruto']['ci_low'], res['beta_tracker_bruto']['ci_high'], res['beta_tracker_bruto']['pvalue'])}")
    else:
        results = main_single_run(args, out_dir)
        k = args.k
        res = results[k]
        print("\n--- Resultado ---")
        print(f"no_tracker bruto:  {fmt_beta(res['beta_no_tracker_bruto']['beta'], res['beta_no_tracker_bruto']['ci_low'], res['beta_no_tracker_bruto']['ci_high'], res['beta_no_tracker_bruto']['pvalue'])}")
        print(f"no_tracker oracle: {fmt_beta(res['beta_no_tracker_oracle']['beta'], res['beta_no_tracker_oracle']['ci_low'], res['beta_no_tracker_oracle']['ci_high'], res['beta_no_tracker_oracle']['pvalue'])}")
        print(f"tracker bruto:     {fmt_beta(res['beta_tracker_bruto']['beta'], res['beta_tracker_bruto']['ci_low'], res['beta_tracker_bruto']['ci_high'], res['beta_tracker_bruto']['pvalue'])}")
        print(f"tracker oracle:    {fmt_beta(res['beta_tracker_oracle']['beta'], res['beta_tracker_oracle']['ci_low'], res['beta_tracker_oracle']['ci_high'], res['beta_tracker_oracle']['pvalue'])}")
        print(f"Rama: {res['rama_disparada']}")
        print(f"Salidas en: {out_dir.resolve()}")

        if args.compare_with is not None:
            ref_path = Path(args.compare_with)
            try:
                ref = json.load(open(ref_path / f"resultados_piso_k{k}.json", encoding="utf-8"))
                fig, ax = plt.subplots(figsize=(7, 5))
                labels = ["censura on", "censura off"]
                betas = [ref["beta_no_tracker_oracle"]["beta"], res["beta_no_tracker_oracle"]["beta"]]
                lows = [ref["beta_no_tracker_bruto"]["ci_low"], res["beta_no_tracker_bruto"]["ci_low"]]
                highs = [ref["beta_no_tracker_bruto"]["ci_high"], res["beta_no_tracker_bruto"]["ci_high"]]
                x = np.arange(len(labels))
                errs = [[b - l for b, l in zip(betas, lows)], [h - b for b, h in zip(betas, highs)]]
                ax.bar(x, betas, yerr=errs, capsize=5, color=["C0", "C1"])
                ax.axhline(0, color="black", linewidth=0.8)
                ax.set_xticks(x)
                ax.set_xticklabels(labels)
                ax.set_ylabel("beta_oraculo del no-tracker sobre p_grad_excess")
                ax.set_title(f"Test de desacople: H={args.H_min}-{args.H_max}, n={args.n_games}, tau={args.tau}")
                fig.tight_layout()
                fig.savefig(out_dir / "fig_desacople_censura.png", dpi=150)
                plt.close(fig)
                print(f"Figura de desacople guardada en: {out_dir / 'fig_desacople_censura.png'}")
            except Exception as e:
                print(f"[advertencia] No se pudo generar figura de desacople: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

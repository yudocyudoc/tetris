"""
confound_floor.py — Medición del piso de confound de Fase 1A.

Simula partidas 7-bag en un escenario well-building artificial, compara un tracker
(con modelado del bag) contra un no-tracker (misma política, creencia estacionaria),
y mide cuánto β residual produce el no-tracker bajo el estimador de Fase 1A.

Especificación: BLUEPRINT_piso_confound_fase1A.md
"""

from __future__ import annotations

import argparse
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
import pyarrow.parquet as pq
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod import generalized_linear_model as glm_mod

glm_mod.SET_USE_BIC_LLF(True)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Constantes del dominio
# ---------------------------------------------------------------------------
BOARD_WIDTH = 10
BOARD_HEIGHT = 20
WELL_COL = 0
BAG = ["I", "J", "L", "O", "S", "T", "Z"]
NON_I_PIECES = ["J", "L", "O", "S", "T", "Z"]

# Representación de piezas como matrices binarias (filas × columnas).
# Cada celda es 0/1. La forma se coloca con su esquina superior-izquierda.
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
    # Dedup por forma
    unique = []
    seen = set()
    for r in rots:
        key = r.tobytes()
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


# Precalcular todas las rotaciones posibles por pieza.
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
            # row=0 es arriba; altura = BOARD_HEIGHT - primera fila ocupada
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
    # Buscamos la fila más baja válida. Como row=0 es arriba, la pieza "cae"
    # hacia filas mayores. El aterrizaje es el máximo row tal que no colisiona.
    max_row = BOARD_HEIGHT - shape.shape[0]
    for row in range(max_row + 1):
        if is_collision(board, shape, col, row):
            return row - 1
    return max_row


def valid_placements(
    board: np.ndarray, piece: str, allow_well: bool = True
) -> List[Tuple[np.ndarray, int, int]]:
    """Todas las colocaciones de aterrizaje válidas para una pieza.

    Cada tupla es (shape, col, row). Si allow_well=False, se descartan las que
    ocupen la columna-pozo WELL_COL.
    """
    placements = []
    for shape in PIECE_ROTATIONS[piece]:
        h, w = shape.shape
        for col in range(BOARD_WIDTH - w + 1):
            if not allow_well and col == WELL_COL:
                # Solo descartamos si la pieza ocupa realmente la columna pozo
                # (si w==1 y col==WELL_COL, sí la ocupa; si es más ancha y col<WELL_COL<col+w, también)
                if col <= WELL_COL < col + w:
                    continue
            row = landing_row(board, shape, col)
            if row < 0:
                continue
            # Verificar explícitamente que no ocupe pozo si allow_well=False
            if not allow_well:
                if _shape_covers_well(shape, col):
                    continue
            placements.append((shape, col, row))
    return placements


def _shape_covers_well(shape: np.ndarray, col: int) -> bool:
    """True si la pieza, colocada con esquina en col, ocupa la columna pozo."""
    w = shape.shape[1]
    return col <= WELL_COL < col + w


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
    """Vector BCTS-like del tablero actual (lossy, el que usará la regresión)."""
    heights = column_heights(board)
    agg_height = float(np.sum(heights))
    bumpiness = float(np.sum(np.abs(np.diff(heights))))
    well_depth = float(np.min(heights))  # pozo como la columna más baja
    max_height = float(np.max(heights))

    # Huecos: celdas vacías con al menos una celda ocupada arriba en la misma columna.
    n_holes = 0
    for c in range(BOARD_WIDTH):
        col = board[:, c]
        occupied = np.where(col == 1)[0]
        if len(occupied) == 0:
            continue
        top = occupied[0]
        n_holes += int(np.sum(col[top:] == 0))

    # Transiciones de fila (0↔1) y columna.
    row_transitions = 0
    for r in range(BOARD_HEIGHT):
        row = board[r, :]
        if np.all(row == 0):
            continue
        # contar transiciones en la fila, incluyendo bordes con paredes
        padded = np.concatenate([[1], row, [1]])
        row_transitions += int(np.sum(padded[:-1] != padded[1:]))

    col_transitions = 0
    for c in range(BOARD_WIDTH):
        col = board[:, c]
        if np.all(col == 0):
            continue
        padded = np.concatenate([[1], col, [1]])
        col_transitions += int(np.sum(padded[:-1] != padded[1:]))

    # landing_height: altura de la columna donde aterrizó la última pieza.
    # Si no hay dato, usamos max_height como aproximación conservadora.
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
    """Valor heurístico del tablero para pi_fill.

    No es idéntico a BCTS: usa el perfil completo de alturas + huecos de forma
    no lineal, de modo que el canal del confound A permanezca abierto.
    """
    heights = column_heights(board)
    max_h = np.max(heights)
    holes = 0
    for c in range(BOARD_WIDTH):
        col = board[:, c]
        occ = np.where(col == 1)[0]
        if len(occ):
            holes += int(np.sum(col[occ[0] :] == 0))
    bump = np.sum(np.abs(np.diff(heights)))
    var_h = np.var(heights)
    # Penalizar huecos, altura, irregularidad.
    return 2.0 * holes + 0.5 * max_h + 0.3 * bump + 0.2 * var_h


def board_full_features(board: np.ndarray) -> Dict[str, float]:
    """Control oráculo: vector rico del tablero completo.

    Usamos momentos + resumen de huecos. No incluimos h_i individuales porque
    en el escenario bien-formado las columnas de relleno son casi iguales y
    producen colinealidad perfecta; los momentos capturan la forma del perfil.
    """
    heights = column_heights(board).astype(float)
    holes_per_col = np.zeros(BOARD_WIDTH, dtype=float)
    for c in range(BOARD_WIDTH):
        col = board[:, c]
        occ = np.where(col == 1)[0]
        if len(occ):
            holes_per_col[c] = float(np.sum(col[occ[0] :] == 0))
    feats = {}
    # Alturas de relleno (excluyendo el pozo WELL_COL, que siempre es 0).
    fill_heights = np.delete(heights, WELL_COL)
    feats["h_max"] = float(np.max(heights))
    feats["h_std"] = float(np.std(fill_heights))
    feats["h_range"] = float(np.max(fill_heights) - np.min(fill_heights))
    feats["total_holes"] = float(np.sum(holes_per_col))
    feats["holes_std"] = float(np.std(holes_per_col))
    # Añadir algunas alturas relativas para capturar asimetría sin colinealidad.
    feats["h_skew"] = float(pd.Series(fill_heights).skew())
    return feats



# ---------------------------------------------------------------------------
# pi_fill — política de relleno compartida (bag-ciega)
# ---------------------------------------------------------------------------
def pi_fill(
    board: np.ndarray, piece: str, action: str
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, any]]]:
    """Coloca `piece` en `board` siguiendo la acción 'leave' o 'close'.

    Devuelve (nuevo_tablero, info) o (None, None) si no hay colocación válida.
    info incluye 'last_landing_col' para calcular landing_height posteriormente.
    """
    if action == "leave":
        placements = valid_placements(board, piece, allow_well=False)
        if not placements:
            return None, None
        best = None
        best_val = np.inf
        for shape, col, row in placements:
            new_board = place_piece(board, shape, col, row)
            if new_board is None:
                continue
            new_board, _ = clear_lines(new_board)
            val = extended_board_value(new_board)
            if val < best_val:
                best_val = val
                best = (new_board, col)
        if best is None:
            return None, None
        new_board, landing_col = best
        return new_board, {"last_landing_col": landing_col}

    elif action == "close":
        # Preferimos colocaciones que cubran la columna pozo.
        all_placements = valid_placements(board, piece, allow_well=True)
        if not all_placements:
            return None, None
        well_placements = [
            (shape, col, row)
            for shape, col, row in all_placements
            if _shape_covers_well(shape, col)
        ]
        candidates = well_placements if well_placements else all_placements
        best = None
        best_val = np.inf
        for shape, col, row in candidates:
            new_board = place_piece(board, shape, col, row)
            if new_board is None:
                continue
            new_board, _ = clear_lines(new_board)
            val = extended_board_value(new_board)
            if val < best_val:
                best_val = val
                best = (new_board, col)
        if best is None:
            return None, None
        new_board, landing_col = best
        return new_board, {"last_landing_col": landing_col}
    else:
        raise ValueError(f"Acción desconocida: {action}")


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


# ---------------------------------------------------------------------------
# Estado well-building
# ---------------------------------------------------------------------------
def is_well_building(board: np.ndarray) -> Tuple[bool, int]:
    """Devuelve (True/False, H) si el tablero tiene un pozo de ≥4 en WELL_COL.

    H es la altura máxima de las columnas de relleno (1-9).
    """
    heights = column_heights(board)
    fill_heights = np.delete(heights, WELL_COL)
    well_h = heights[WELL_COL]
    H = int(np.max(fill_heights))
    min_fill = int(np.min(fill_heights))
    # Pozo de al menos 4 de profundidad relativa a la superficie de relleno
    has_well = H - well_h >= 4
    # Las columnas de relleno no deben estar demasiado dispersas
    uniform = (np.max(fill_heights) - np.min(fill_heights)) <= 4
    return has_well and uniform and H > 0, H


def make_well_board(H: int) -> np.ndarray:
    """Tablero inicial bien-formado: 9 columnas a altura H, pozo vacío."""
    board = empty_board()
    # row=0 arriba; filas inferiores tienen índices altos.
    # Altura H significa que las últimas H filas están ocupadas.
    start_row = BOARD_HEIGHT - H
    for c in range(BOARD_WIDTH):
        if c == WELL_COL:
            continue
        board[start_row:, c] = 1
    return board


# ---------------------------------------------------------------------------
# N(H) empírico
# ---------------------------------------------------------------------------
def estimate_N_for_H(
    H: int, n_sims: int = 200, base_seed: int = 0
) -> Tuple[float, float]:
    """Estima E[N|H] simulando pi_fill desde un tablero bien-formado de altura H."""
    counts = []
    for s in range(n_sims):
        board = make_well_board(H)
        gen = SevenBagGenerator(base_seed + s)
        count = 0
        while True:
            piece = gen.advance()
            if piece == "I":
                # N(H) es horizonte bajo piezas no-I: ignorar I para relleno
                continue
            new_board, _ = pi_fill(board, piece, "leave")
            if new_board is None:
                break
            count += 1
            # Si topamos (alcanzamos fila 0), terminamos
            if column_heights(new_board).max() >= BOARD_HEIGHT:
                break
            board = new_board
        counts.append(count)
    return float(np.mean(counts)), float(np.std(counts))


def build_N_lookup(
    H_min: int = 4, H_max: int = 19, n_sims: int = 200, base_seed: int = 0
) -> Dict[int, Tuple[float, float]]:
    lookup = {}
    for H in range(H_min, H_max + 1):
        mean, std = estimate_N_for_H(H, n_sims=n_sims, base_seed=base_seed + H * 1000)
        lookup[H] = (mean, std)
    return lookup


# ---------------------------------------------------------------------------
# Modelo de llegada de la I
# ---------------------------------------------------------------------------
def p_grad(S_t: set, H: int, N_lookup: Dict[int, Tuple[float, float]]) -> float:
    """Probabilidad graduada de que la I llegue antes de topar, dado S_t y H."""
    N = N_lookup[H][0]
    if "I" in S_t:
        m = len(S_t)
        return min(N / max(m, 1.0), 1.0)
    else:
        m = len(S_t)
        return float(np.clip((N - m) / 7.0, 0.0, 1.0))


def p_stat(H: int, N_lookup: Dict[int, Tuple[float, float]]) -> float:
    """Creencia estacionaria 7-bag: I uniforme en ventana de 7, sin tracking."""
    N = N_lookup[H][0]
    return min(N / 7.0, 1.0)


# ---------------------------------------------------------------------------
# Utilidades y decisión logística
# ---------------------------------------------------------------------------
@dataclass
class AgentParams:
    v_tetris: float = 10.0
    c_topar: float = 15.0
    v_estable: float = 3.0
    tau: float = 0.5
    board_value_weight: float = 0.5


def agent_decision(
    board: np.ndarray,
    piece: str,
    next_piece: str,
    p_arrival: float,
    params: AgentParams,
) -> Tuple[str, Optional[np.ndarray], Optional[np.ndarray], Optional[int]]:
    """Decide 'leave' o 'close' y devuelve tableros resultantes."""
    board_leave, info_leave = pi_fill(board, piece, "leave")
    board_close, info_close = pi_fill(board, piece, "close")

    # Si no se puede dejar (no cabe sin tocar pozo), forzar cierre.
    if board_leave is None:
        if board_close is None:
            return "close", None, None, None
        return "close", None, board_close, info_close.get("last_landing_col") if info_close else None

    # Valores heurísticos del tablero resultante.
    val_leave = extended_board_value(board_leave)
    val_close = extended_board_value(board_close) if board_close is not None else np.inf

    U_leave = p_arrival * params.v_tetris - (1 - p_arrival) * params.c_topar
    U_leave += params.board_value_weight * val_leave

    if board_close is not None:
        U_close = params.v_estable + params.board_value_weight * val_close
    else:
        U_close = -np.inf

    # Decisión logística con temperatura tau.
    diff = (U_leave - U_close) / max(params.tau, 1e-6)
    p_leave = 1.0 / (1.0 + np.exp(-diff))
    action = "leave" if np.random.rand() < p_leave else "close"

    landing_col = None
    if action == "leave" and info_leave:
        landing_col = info_leave.get("last_landing_col")
    elif action == "close" and info_close:
        landing_col = info_close.get("last_landing_col")

    return action, board_leave, board_close, landing_col


# ---------------------------------------------------------------------------
# Simulación de una partida
# ---------------------------------------------------------------------------
def current_bag_state(gen: SevenBagGenerator) -> set:
    """Devuelve el conjunto de piezas no vistas de la bolsa actual.

    El bag actual es el bloque de 7 piezas que contiene gen.idx.
    S_t son las piezas desde gen.idx hasta el final de ese bloque.
    """
    start_of_bag = (gen.idx // 7) * 7
    end_of_bag = start_of_bag + 7
    return set(gen.sequence[gen.idx : end_of_bag])


def simulate_game(
    seed: int,
    agent_type: str,
    params: AgentParams,
    N_lookup: Dict[int, Tuple[float, float]],
    H0: int = 8,
    no_censorship: bool = False,
) -> List[Dict]:
    """Simula una partida y devuelve lista de decisiones logueadas."""
    gen = SevenBagGenerator(seed)
    board = make_well_board(H0)
    piece = gen.advance()
    next_piece = gen.advance()

    decisions = []
    alive = True
    step = 0
    max_steps = 1000

    while alive and step < max_steps:
        ok, H = is_well_building(board)
        if not ok:
            break

        S_t = current_bag_state(gen)
        m = len(S_t)
        I_in_S = "I" in S_t

        if agent_type == "tracker":
            p = p_grad(S_t, H, N_lookup)
        elif agent_type == "no_tracker":
            p = p_stat(H, N_lookup)
        else:
            raise ValueError(agent_type)

        action, board_leave, board_close, landing_col = agent_decision(
            board, piece, next_piece, p, params
        )

        # Loguear decisión actual
        bcts = bcts_features(board, last_landing_col=landing_col)
        full = board_full_features(board)
        record = {
            "game_id": seed,
            "agent_type": agent_type,
            "piece_idx": step,
            "H": H,
            "m": m,
            "I_in_S": I_in_S,
            "p_grad": p_grad(S_t, H, N_lookup),
            "p_stat": p_stat(H, N_lookup),
            "action": 1 if action == "leave" else 0,
            "board_full": board.copy(),
            "last_landing_col": landing_col,
            **bcts,
            **full,
        }

        # Ejecutar acción
        if action == "leave":
            new_board = board_leave
            if new_board is None:
                record["survived_next"] = False
                decisions.append(record)
                alive = False
                break
            # Censura: ¿llega la I dentro de N piezas futuras?
            N = int(round(N_lookup[H][0]))
            future = gen.peek_n(gen.idx, N)
            survived = "I" in future
            record["survived_next"] = survived
            decisions.append(record)
            if not no_censorship and not survived:
                alive = False
                break
            board = new_board
        else:  # close
            new_board = board_close
            record["survived_next"] = True
            decisions.append(record)
            if new_board is None:
                alive = False
                break
            board = new_board
            # Al cerrar, la partida ya no está en well-building; terminamos.
            break

        piece = next_piece
        next_piece = gen.advance()
        step += 1

    return decisions



# ---------------------------------------------------------------------------
# Regresiones y descomposición
# ---------------------------------------------------------------------------
def fit_logit(formula: str, df: pd.DataFrame, label: str = "", weights: Optional[pd.Series] = None) -> Optional[Dict]:
    """Ajusta un logit y devuelve resultados resumidos."""
    try:
        if weights is not None:
            model = smf.glm(formula, data=df, family=sm.families.Binomial(), freq_weights=np.asarray(weights)).fit(disp=0)
        else:
            model = smf.logit(formula, data=df).fit(disp=0, maxiter=200)
    except Exception as e:
        print(f"[fit_logit error{label}] {e}")
        return {"error": str(e)}
    summary = {
        "nobs": int(model.nobs),
        "pseudo_r2": float(getattr(model, "prsquared", np.nan)),
        "converged": bool(getattr(model, "mle_retvals", {}).get("converged", True)),
        "params": {k: float(v) for k, v in model.params.items()},
        "pvalues": {k: float(v) for k, v in model.pvalues.items()},
        "ci": {k: [float(v[0]), float(v[1])] for k, v in model.conf_int().iterrows()},
        "aic": float(getattr(model, "aic", np.nan)),
        "bic": float(getattr(model, "bic", np.nan)),
    }
    return summary


def extract_beta(summary: Dict, var: str = "p_grad") -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Devuelve (beta, p, CI_inf, CI_sup) para `var`."""
    if "error" in summary or var not in summary.get("params", {}):
        return None, None, None, None
    beta = summary["params"][var]
    p = summary["pvalues"][var]
    ci = summary["ci"].get(var, [None, None])
    return float(beta), float(p), float(ci[0]), float(ci[1])


def run_regressions(df: pd.DataFrame, agent_type: str) -> Dict:
    """Corre los tres modelos sobre los datos de un agente."""
    df_agent = df[df["agent_type"] == agent_type].copy()
    # Evitar nombres problemáticos
    df_agent = df_agent.rename(columns={c: c.replace("-", "_") for c in df_agent.columns})
    df_agent["I_in_S"] = df_agent["I_in_S"].astype(int)

    features_bcts = [
        "agg_height",
        "n_holes",
        "bumpiness",
        # "well_depth" se excluye: en el escenario bien-formado el pozo está vacío (constante 0).
        # "landing_height" se excluye: en tableros bien-formados y regulares es función lineal de agg_height.
        "row_transitions",
        # "col_transitions" se excluye: en este sustrato es casi lineal con n_holes (r≈0.99).
    ]

    # Usamos p_grad_excess = p_grad - p_stat como regresor principal.
    # p_grad crudo contiene la componente estacionaria p_stat(H) que el no-tracker
    # usa legítimamente; el contraste tracker/no-tracker vive en el exceso.
    formula1 = "action ~ p_grad_excess + " + " + ".join(features_bcts)
    sum1 = fit_logit(formula1, df_agent, label=f" {agent_type} m1")

    # Control oráculo: board_full features
    full_cols = ["h_max", "h_std", "h_range", "total_holes", "holes_std", "h_skew"]
    full_cols = [c for c in full_cols if c in df_agent.columns]
    formula2 = "action ~ p_grad_excess + " + " + ".join(full_cols)
    sum2 = fit_logit(formula2, df_agent, label=f" {agent_type} m2")

    # Sin survivorship (confound B): IPW por probabilidad de supervivencia.
    # Estimamos P(survived_next=1 | H, action=leave) con un logit simple y
    # reponderamos para recuperar la población que habría existido sin censura.
    H_threshold = 12
    df_ipw = df_agent.copy()
    leave_df = df_ipw[df_ipw["action"] == 1].copy()
    leave_df["survived_next_int"] = leave_df["survived_next"].astype(int)
    if len(leave_df) > 0 and leave_df["survived_next_int"].nunique() > 1:
        surv_model = smf.logit("survived_next_int ~ H", data=leave_df).fit(disp=0)
        leave_df = leave_df.copy()
        leave_df["psurv"] = surv_model.predict(leave_df)
    else:
        leave_df = leave_df.copy()
        leave_df["psurv"] = 1.0
    close_df = df_ipw[df_ipw["action"] == 0].copy()
    close_df["psurv"] = 1.0
    df_ipw = pd.concat([leave_df, close_df], ignore_index=True)
    df_ipw["ipw_weight"] = 1.0 / df_ipw["psurv"].clip(lower=0.01)

    sum3 = fit_logit(formula1, df_ipw, label=f" {agent_type} m3", weights=df_ipw["ipw_weight"])

    return {
        "agent_type": agent_type,
        "model1_bruto": sum1,
        "model2_oracle": sum2,
        "model3_lowH": sum3,
        "H_threshold": H_threshold,
    }


def decide_rama(beta: float, ci_low: float, ci_high: float, delta: float = 0.13, eps: float = 0.02) -> str:
    """Aplica la regla de decisión del §6."""
    if ci_low <= 0 <= ci_high or abs(beta) < eps:
        return "1_piso_limpio"
    if beta > 0 and beta < delta:
        return "2_piso_moderado"
    return "3_piso_fatal"


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------
def plot_signal_by_H(df: pd.DataFrame, out_dir: Path) -> None:
    """P(dejar|I∈S) − P(dejar|I∉S) por H para tracker y no-tracker."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for agent in ["tracker", "no_tracker"]:
        dfa = df[df["agent_type"] == agent]
        signals = []
        hs = []
        for H, grp in dfa.groupby("H"):
            if len(grp) < 30:
                continue
            p_in = grp[grp["I_in_S"] == True]["action"].mean()
            p_out = grp[grp["I_in_S"] == False]["action"].mean()
            signals.append(p_in - p_out)
            hs.append(H)
        label = "tracker" if agent == "tracker" else "no-tracker (piso)"
        ax.plot(hs, signals, marker="o", label=label)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Stack height H")
    ax.set_ylabel("P(leave | I ∈ S) − P(leave | I ∉ S)")
    ax.set_title("Señal por stack height — ¿se concentra en H alto? (survivorship)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fig_piso_por_H.png", dpi=150)
    plt.close(fig)


def plot_decomposition(results: Dict, out_dir: Path) -> None:
    """Barras de β(1), β(2), β(3) para el no-tracker."""
    m1 = results["no_tracker"]["model1_bruto"]
    m2 = results["no_tracker"]["model2_oracle"]
    m3 = results["no_tracker"]["model3_lowH"]

    labels = ["β(1) bruto\n(A+B)", "β(2) oráculo\n(aísla A)", "β(3) low-H\n(aísla B)"]
    betas = []
    lows = []
    highs = []
    for m in [m1, m2, m3]:
        b, _, ci_low, ci_high = extract_beta(m, "p_grad")
        betas.append(b if b is not None else 0)
        lows.append(ci_low if ci_low is not None else b if b is not None else 0)
        highs.append(ci_high if ci_high is not None else b if b is not None else 0)

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(labels))
    errs = [[b - l for b, l in zip(betas, lows)], [h - b for b, h in zip(betas, highs)]]
    ax.bar(x, betas, yerr=errs, capsize=5, color=["C0", "C1", "C2"])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Coeficiente β sobre p_grad")
    ax.set_title("Descomposición del piso de confound (no-tracker)")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_descomposicion.png", dpi=150)
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
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Medición del piso de confound de Fase 1A")
    parser.add_argument("--seed", type=int, default=42, help="Semilla base")
    parser.add_argument("--n_games", type=int, default=2000, help="Número de partidas por agente")
    parser.add_argument("--tau", type=float, default=1.0, help="Temperatura de decisión logística")
    parser.add_argument("--v_tetris", type=float, default=10.0)
    parser.add_argument("--c_topar", type=float, default=15.0)
    parser.add_argument("--v_estable", type=float, default=3.0)
    parser.add_argument("--board_value_weight", type=float, default=0.5,
                        help="Peso del valor heurístico del tablero en la utilidad")
    parser.add_argument("--n_sims_N", type=int, default=200,
                        help="Simulaciones por H para estimar N(H)")
    parser.add_argument("--out", type=str, default="out", help="Directorio de salida")
    parser.add_argument("--no_censorship", action="store_true",
                        help="Desactiva la censura por topar (para validar confound B)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    params = AgentParams(
        v_tetris=args.v_tetris,
        c_topar=args.c_topar,
        v_estable=args.v_estable,
        tau=args.tau,
        board_value_weight=args.board_value_weight,
    )

    git_hash = get_git_hash()
    (out_dir / "git_hash.txt").write_text(git_hash + "\n", encoding="utf-8")

    print("[1/5] Estimando N(H) empiricamente bajo pi_fill...")
    N_lookup = build_N_lookup(H_min=4, H_max=19, n_sims=args.n_sims_N, base_seed=args.seed)
    print({H: f"{N_lookup[H][0]:.2f} +/- {N_lookup[H][1]:.2f}" for H in sorted(N_lookup)})

    print("[2/5] Simulando partidas (tracker y no-tracker)...")
    all_decisions = []
    H0_values = list(range(4, 16))
    rng = np.random.default_rng(args.seed)
    for agent_type in ["tracker", "no_tracker"]:
        for g in range(args.n_games):
            seed = int(args.seed + g + (100_000 if agent_type == "no_tracker" else 0))
            H0 = int(rng.choice(H0_values))
            decisions = simulate_game(seed, agent_type, params, N_lookup, H0=H0,
                                       no_censorship=args.no_censorship)
            all_decisions.extend(decisions)

    df = pd.DataFrame(all_decisions)
    print(f"    Decisiones logueadas: tracker={len(df[df.agent_type=='tracker'])}, "
          f"no_tracker={len(df[df.agent_type=='no_tracker'])}")

    # Guardar log crudo
    # Regresor corregido: p_grad - p_stat (la parte de p_grad que excede el saber estacionario).
    # p_grad crudo contiene p_stat, que el no-tracker usa legítimamente; restarla evita
    # atribuirle al tracking la respuesta a H/N.
    df["p_grad_excess"] = df["p_grad"] - df["p_stat"]

    df_save = df.drop(columns=["board_full"])
    df_save.to_csv(out_dir / "decisions_log.csv", index=False)
    df_save.to_parquet(out_dir / "decisions_log.parquet", index=False)

    print("[3/5] Ajustando regresiones...")
    results = {}
    for agent_type in ["tracker", "no_tracker"]:
        results[agent_type] = run_regressions(df, agent_type)

    print("[4/5] Calculando descomposición A/B y rama...")
    beta1, p1, ci1_low, ci1_high = extract_beta(results["no_tracker"]["model1_bruto"], "p_grad_excess")
    beta2, p2, ci2_low, ci2_high = extract_beta(results["no_tracker"]["model2_oracle"], "p_grad_excess")
    beta3, p3, ci3_low, ci3_high = extract_beta(results["no_tracker"]["model3_lowH"], "p_grad_excess")

    contrib_A = beta1 - beta2 if beta1 is not None and beta2 is not None else None
    contrib_B = beta1 - beta3 if beta1 is not None and beta3 is not None else None

    rama = decide_rama(beta1, ci1_low, ci1_high) if beta1 is not None else "unknown"

    resultados = {
        "software_git_hash": git_hash,
        "params": {
            "seed": args.seed,
            "n_games": args.n_games,
            "tau": args.tau,
            "v_tetris": args.v_tetris,
            "c_topar": args.c_topar,
            "v_estable": args.v_estable,
            "board_value_weight": args.board_value_weight,
            "n_sims_N": args.n_sims_N,
        },
        "N_lookup": {str(H): {"mean": N_lookup[H][0], "std": N_lookup[H][1]} for H in sorted(N_lookup)},
        "beta_piso_bruto": {
            "regressor": "p_grad_excess",
            "beta": beta1,
            "pvalue": p1,
            "ci_low": ci1_low,
            "ci_high": ci1_high,
        },
        "beta_oracle": {
            "regressor": "p_grad_excess",
            "beta": beta2,
            "pvalue": p2,
            "ci_low": ci2_low,
            "ci_high": ci2_high,
        },
        "beta_lowH": {
            "regressor": "p_grad_excess",
            "beta": beta3,
            "pvalue": p3,
            "ci_low": ci3_low,
            "ci_high": ci3_high,
        },
        "contrib_A": contrib_A,
        "contrib_B": contrib_B,
        "rama_disparada": rama,
    }

    with open(out_dir / "resultados_piso.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, default=str)

    # Tabla de regresiones
    rows = []
    for agent_type, res in results.items():
        for model_name, summary in [
            ("model1_bruto", res["model1_bruto"]),
            ("model2_oracle", res["model2_oracle"]),
            ("model3_lowH", res["model3_lowH"]),
        ]:
            b, p, ci_low, ci_high = extract_beta(summary, "p_grad")
            rows.append({
                "agent_type": agent_type,
                "model": model_name,
                "N": summary.get("nobs"),
                "beta_p_grad_excess": b,
                "pvalue": p,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "pseudo_r2": summary.get("pseudo_r2"),
                "converged": summary.get("converged"),
            })
    tabla = pd.DataFrame(rows)
    tabla.to_csv(out_dir / "tabla_regresiones.csv", index=False)

    print("[5/5] Generando figuras...")
    plot_signal_by_H(df, out_dir)
    plot_decomposition(results, out_dir)

    print("\n--- Resultado ---")
    def fmt_beta(b, ci_low, ci_high, p):
        if b is None:
            return "NA"
        return f"{b:.4f} (CI {ci_low:.4f}, {ci_high:.4f}), p={p:.4g}"
    print(f"beta_piso_bruto (p_grad_excess) = {fmt_beta(beta1, ci1_low, ci1_high, p1)}")
    print(f"beta_oracle     (p_grad_excess) = {fmt_beta(beta2, ci2_low, ci2_high, p2)}")
    print(f"beta_lowH       (p_grad_excess) = {fmt_beta(beta3, ci3_low, ci3_high, p3)}")
    print(f"contrib_A       = {contrib_A:.4f}" if contrib_A is not None else "contrib_A       = NA")
    print(f"contrib_B       = {contrib_B:.4f}" if contrib_B is not None else "contrib_B       = NA")
    print(f"Rama            = {rama}")
    print(f"Salidas en      = {out_dir.resolve()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

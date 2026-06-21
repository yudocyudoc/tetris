"""Métricas conductuales derivadas de los streams finos.

`decision_time_ms = t_lock - t_spawn` está contaminado por la gravedad: mide
cuánto tardó la pieza en caer, no cuánto deliberó el jugador. Este módulo
calcula insumos más limpios para el σ a partir de `actions.csv`:

- n_inputs: número de acciones de juego sobre la pieza.
- time_to_first_input_ms: tiempo desde spawn hasta la primera acción de juego.
- active_time_ms: tiempo entre la primera y la última acción de juego.
- hard_drop_used: si la pieza se terminó con hard drop.
- hard_drop_ratio: proporción de piezas terminadas con hard drop por ventana.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional


def _float(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compute_piece_metrics(
    pieces_path: Path,
    actions_path: Path,
) -> List[Dict[str, float]]:
    """Devuelve, para cada pieza, métricas conductuales derivadas.

    El resultado preserva el orden de pieces.csv.
    """
    if not pieces_path.exists() or not actions_path.exists():
        return []

    # Cargar acciones de juego (excluyendo eventos raw de teclado).
    game_actions: Dict[str, Dict[int, List[Dict]]] = defaultdict(lambda: defaultdict(list))
    with open(actions_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            action = row["action"]
            if action in ("key_down", "key_up"):
                continue
            game_id = row["game_id"]
            piece_idx = int(row["piece_idx"])
            t_ms = _float(row["t_ms"])
            if t_ms is None:
                continue
            game_actions[game_id][piece_idx].append(
                {"action": action, "t_ms": t_ms}
            )

    # Ordenar acciones por tiempo dentro de cada pieza.
    for game_id in game_actions:
        for piece_idx in game_actions[game_id]:
            game_actions[game_id][piece_idx].sort(key=lambda a: a["t_ms"])

    metrics = []
    with open(pieces_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            game_id = row["game_id"]
            piece_idx = int(row["piece_idx"])
            t_spawn = _float(row["t_spawn_ms"])
            t_lock = _float(row["t_lock_ms"])
            n_inputs = _float(row["n_inputs"])

            actions = game_actions.get(game_id, {}).get(piece_idx, [])
            game_action_times = [a["t_ms"] for a in actions]
            hard_drop_used = any(a["action"] == "hard_drop" for a in actions)

            time_to_first_input: Optional[float] = None
            active_time: Optional[float] = None
            if game_action_times:
                time_to_first_input = game_action_times[0] - t_spawn
                active_time = game_action_times[-1] - game_action_times[0]

            metrics.append(
                {
                    "game_id": game_id,
                    "piece_idx": piece_idx,
                    "piece_type": row["piece_type"],
                    "t_spawn_ms": t_spawn,
                    "t_lock_ms": t_lock,
                    "decision_time_ms": t_lock - t_spawn if t_lock is not None and t_spawn is not None else None,
                    "n_inputs": int(n_inputs) if n_inputs is not None else None,
                    "time_to_first_input_ms": time_to_first_input,
                    "active_time_ms": active_time,
                    "hard_drop_used": hard_drop_used,
                }
            )

    return metrics


def summarize_piece_metrics(pieces_path: Path, actions_path: Path) -> Dict[str, Any]:
    """Resumen agregado de métricas conductuales por sesión."""
    metrics = compute_piece_metrics(pieces_path, actions_path)
    if not metrics:
        return {}

    def stats(values: List[float]) -> Dict[str, Optional[float]]:
        if not values:
            return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
        n = len(values)
        mean = sum(values) / n
        std = None
        if n >= 2:
            std = (sum((x - mean) ** 2 for x in values) / (n - 1)) ** 0.5
        return {
            "n": n,
            "mean": round(mean, 2),
            "std": round(std, 2) if std is not None else None,
            "min": round(min(values), 2),
            "max": round(max(values), 2),
        }

    n_inputs = [m["n_inputs"] for m in metrics if m["n_inputs"] is not None]
    first = [m["time_to_first_input_ms"] for m in metrics if m["time_to_first_input_ms"] is not None]
    active = [m["active_time_ms"] for m in metrics if m["active_time_ms"] is not None]
    hard_drops = [1.0 if m["hard_drop_used"] else 0.0 for m in metrics]

    return {
        "n_inputs": stats(n_inputs),
        "time_to_first_input_ms": stats(first),
        "active_time_ms": stats(active),
        "hard_drop_ratio": round(sum(hard_drops) / len(hard_drops), 4) if hard_drops else None,
        "total_pieces": len(metrics),
    }

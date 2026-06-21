"""Validación de integridad de una sesión de Tetris instrumentado.

Comprueba que los streams de datos existen, que los timestamps son monotónicos,
que la secuencia de piezas coincide con la semilla, y que las referencias
entre actions y pieces son coherentes.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd

from .tetris_core import SevenBagGenerator


REQUIRED_FILES = [
    "session_meta.json",
    "pieces.csv",
    "actions.csv",
    "board_snapshots.parquet",
    "piece_sequence.json",
    "game_events.csv",
    "games_summary.csv",
]


def regenerate_sequence(seed: int, n: int) -> List[str]:
    gen = SevenBagGenerator(seed)
    return [gen.next()[0] for _ in range(n)]


def check_files(session_dir: Path) -> None:
    missing = [f for f in REQUIRED_FILES if not (session_dir / f).exists()]
    if missing:
        raise AssertionError(f"Faltan archivos requeridos: {missing}")


def load_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def check_monotonicity(rows: List[Dict[str, str]], key: str = "t_ms") -> None:
    by_game: Dict[str, List[float]] = {}
    for row in rows:
        by_game.setdefault(row["game_id"], []).append(float(row[key]))
    for game_id, times in by_game.items():
        for i in range(1, len(times)):
            if times[i] < times[i - 1]:
                raise AssertionError(
                    f"t_ms no monotónico en {game_id}: {times[i-1]} -> {times[i]}"
                )


def check_piece_sequence(session_dir: Path) -> None:
    sequences: List[Dict] = []
    with open(session_dir / "piece_sequence.json", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                sequences.append(json.loads(line))

    for entry in sequences:
        seed = entry["seed"]
        recorded = entry["sequence"]
        regenerated = regenerate_sequence(seed, len(recorded))
        if recorded != regenerated:
            raise AssertionError(
                f"Secuencia inconsistente con seed {seed} en {entry['game_id']}"
            )


def check_actions_vs_pieces(session_dir: Path) -> None:
    pieces = load_csv(session_dir / "pieces.csv")
    actions = load_csv(session_dir / "actions.csv")

    valid: Dict[str, Set[int]] = {}
    for row in pieces:
        valid.setdefault(row["game_id"], set()).add(int(row["piece_idx"]))

    for row in actions:
        game_id = row["game_id"]
        piece_idx = int(row["piece_idx"])
        # Los eventos raw de teclado usan piece_idx = -1 como placeholder.
        if piece_idx < 0:
            continue
        if game_id not in valid or piece_idx not in valid[game_id]:
            raise AssertionError(
                f"Action refiere a piece_idx inexistente: {game_id}/{piece_idx}"
            )


def check_decision_times(session_dir: Path) -> None:
    pieces = load_csv(session_dir / "pieces.csv")
    for row in pieces:
        dt = float(row["decision_time_ms"])
        if dt < 0:
            raise AssertionError(
                f"decision_time_ms negativo en {row['game_id']}/piece {row['piece_idx']}: {dt}"
            )


def validate_session(session_dir: Path) -> None:
    session_dir = Path(session_dir)
    print(f"Validando sesión: {session_dir}")
    check_files(session_dir)
    check_monotonicity(load_csv(session_dir / "actions.csv"))
    check_monotonicity(load_csv(session_dir / "game_events.csv"))
    check_piece_sequence(session_dir)
    check_actions_vs_pieces(session_dir)
    check_decision_times(session_dir)
    print("OK: todos los checks pasaron.")


def main() -> None:
    if len(sys.argv) < 2:
        print("Uso: python -m src.validate_session <ruta_a_sesion>")
        sys.exit(1)
    validate_session(Path(sys.argv[1]))


if __name__ == "__main__":
    main()

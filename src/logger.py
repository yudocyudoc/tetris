"""Logging de eventos del Tetris instrumentado.

Todo se bufferiza en memoria durante la partida y se vuelca a disco al final
de cada partida (nunca por frame). Los archivos siguen el esquema del blueprint.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def _iso_with_tz() -> str:
    """Devuelve la hora local en formato ISO 8601 con zona horaria (-06:00)."""
    now = time.localtime()
    tz = time.strftime("%z", now)
    if len(tz) == 5:
        tz = tz[:3] + ":" + tz[3:]
    return time.strftime("%Y-%m-%dT%H:%M:%S", now) + tz


@dataclass
class GameBuffers:
    pieces: List[Dict[str, Any]] = field(default_factory=list)
    actions: List[Dict[str, Any]] = field(default_factory=list)
    snapshots: List[Dict[str, Any]] = field(default_factory=list)
    events: List[Dict[str, Any]] = field(default_factory=list)
    piece_sequence: List[str] = field(default_factory=list)
    game_meta: Dict[str, Any] = field(default_factory=dict)


class SessionLogger:
    """Gestiona los buffers y volcado de una sesión de juego."""

    def __init__(
        self,
        session_id: Optional[str] = None,
        data_root: str = "data",
        condition: str = "easy",
        config: Optional[Dict[str, Any]] = None,
        state_covariates: Optional[Dict[str, Any]] = None,
    ):
        self.session_id = session_id or str(uuid.uuid4())
        self.session_dir = Path(data_root) / self.session_id
        self.condition = condition
        self.config = config or {}
        self.state_covariates = state_covariates or {}
        self.wall_clock_start = _iso_with_tz()
        self.software_git_hash = self._git_hash()
        self.perceived_effort: Optional[int] = None

        self._buffers: Dict[str, GameBuffers] = {}
        self._current_game_id: Optional[str] = None
        self._games_summary: List[Dict[str, Any]] = []

    def _git_hash(self) -> str:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except Exception:
            return "unknown"

    def _warn_if_no_git(self) -> None:
        if self.software_git_hash == "unknown":
            print(
                "\n[ADVERTENCIA] No se pudo obtener el git hash. "
                "Inicializa un repositorio git en la raíz del proyecto "
                "para garantizar trazabilidad de versión."
            )

    def start_game(
        self,
        game_id: str,
        seed: int,
        condition: str,
        wall_clock_start: str,
    ) -> None:
        self._warn_if_no_git()
        self._current_game_id = game_id
        self._buffers[game_id] = GameBuffers()
        self._buffers[game_id].game_meta = {
            "game_id": game_id,
            "seed": seed,
            "condition": condition,
            "wall_clock_start": wall_clock_start,
        }
        self.log_event(game_id, t_ms=0.0, event="game_start", detail="")

    def log_piece(
        self,
        game_id: str,
        piece_idx: int,
        piece_type: str,
        bag_idx: int,
        preview_at_spawn: str,
        t_spawn_ms: float,
        t_lock_ms: float,
        n_inputs: int,
        final_x: int,
        final_y: float,
        final_rot: int,
        gravity_at_spawn: float,
        lines_cleared_by_lock: int,
    ) -> None:
        buf = self._buffers[game_id]
        buf.pieces.append(
            {
                "game_id": game_id,
                "piece_idx": piece_idx,
                "piece_type": piece_type,
                "bag_idx": bag_idx,
                "preview_at_spawn": preview_at_spawn,
                "t_spawn_ms": t_spawn_ms,
                "t_lock_ms": t_lock_ms,
                "decision_time_ms": t_lock_ms - t_spawn_ms,
                "n_inputs": n_inputs,
                "final_x": final_x,
                "final_y": final_y,
                "final_rot": final_rot,
                "gravity_at_spawn": gravity_at_spawn,
                "lines_cleared_by_lock": lines_cleared_by_lock,
            }
        )

    def log_action(
        self,
        game_id: str,
        t_ms: float,
        piece_idx: int,
        action: str,
        x: int,
        y: float,
        rot: int,
        raw_key: Optional[str] = None,
        key_event: Optional[str] = None,
    ) -> None:
        buf = self._buffers[game_id]
        entry: Dict[str, Any] = {
            "game_id": game_id,
            "t_ms": t_ms,
            "piece_idx": piece_idx,
            "action": action,
            "x": x,
            "y": y,
            "rot": rot,
        }
        if raw_key is not None:
            entry["raw_key"] = raw_key
        if key_event is not None:
            entry["key_event"] = key_event
        buf.actions.append(entry)

    def log_snapshot(
        self,
        game_id: str,
        piece_idx: int,
        t_ms: float,
        board: List[List[Optional[str]]],
    ) -> None:
        buf = self._buffers[game_id]
        # Serializamos el tablero como lista de celdas ocupadas (col, row, type).
        occupied = [
            {"x": x, "y": y, "type": cell}
            for y, row in enumerate(board)
            for x, cell in enumerate(row)
            if cell is not None
        ]
        buf.snapshots.append(
            {
                "game_id": game_id,
                "piece_idx": piece_idx,
                "t_ms": t_ms,
                "board": json.dumps(occupied),
            }
        )

    def log_event(
        self,
        game_id: str,
        t_ms: float,
        event: str,
        detail: str,
    ) -> None:
        buf = self._buffers[game_id]
        buf.events.append(
            {
                "game_id": game_id,
                "t_ms": t_ms,
                "event": event,
                "detail": detail,
            }
        )

    def set_piece_sequence(self, game_id: str, seed: int, sequence: List[str]) -> None:
        self._buffers[game_id].piece_sequence = sequence

    def end_game(
        self,
        game_id: str,
        duration_ms: float,
        total_pieces: int,
        total_lines: int,
        score: int,
        game_over_reason: str,
    ) -> None:
        meta = self._buffers[game_id].game_meta
        self._games_summary.append(
            {
                "game_id": game_id,
                "seed": meta["seed"],
                "condition": meta["condition"],
                "wall_clock_start": meta["wall_clock_start"],
                "duration_ms": duration_ms,
                "total_pieces": total_pieces,
                "total_lines": total_lines,
                "score": score,
                "game_over_reason": game_over_reason,
            }
        )
        self.flush_game(game_id)

    def flush_game(self, game_id: str) -> None:
        """Vuelca los buffers de una partida a disco."""
        buf = self._buffers[game_id]
        out_dir = self.session_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        # pieces.csv
        if buf.pieces:
            with open(out_dir / "pieces.csv", "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(buf.pieces[0].keys()))
                if f.tell() == 0:
                    writer.writeheader()
                writer.writerows(buf.pieces)

        # actions.csv
        if buf.actions:
            with open(out_dir / "actions.csv", "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(buf.actions[0].keys()))
                if f.tell() == 0:
                    writer.writeheader()
                writer.writerows(buf.actions)

        # game_events.csv
        if buf.events:
            with open(out_dir / "game_events.csv", "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(buf.events[0].keys()))
                if f.tell() == 0:
                    writer.writeheader()
                writer.writerows(buf.events)

        # board_snapshots.parquet
        if buf.snapshots:
            df = pd.DataFrame(buf.snapshots)
            path = out_dir / "board_snapshots.parquet"
            if path.exists():
                existing = pd.read_parquet(path)
                df = pd.concat([existing, df], ignore_index=True)
            df.to_parquet(path, index=False)

        # piece_sequence.json (append como JSONL para múltiples partidas)
        seq_path = out_dir / "piece_sequence.json"
        with open(seq_path, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "game_id": game_id,
                        "seed": buf.game_meta["seed"],
                        "sequence": buf.piece_sequence,
                    }
                )
                + "\n"
            )

        # games_summary.csv
        summary_path = out_dir / "games_summary.csv"
        with open(summary_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "game_id",
                    "seed",
                    "condition",
                    "wall_clock_start",
                    "duration_ms",
                    "total_pieces",
                    "total_lines",
                    "score",
                    "game_over_reason",
                ],
            )
            if f.tell() == 0:
                writer.writeheader()
            for row in self._games_summary:
                if row["game_id"] == game_id:
                    writer.writerow(row)

        # Liberar memoria de esta partida.
        del self._buffers[game_id]

    def close_session(self, perceived_effort: Optional[int] = None) -> None:
        self.perceived_effort = perceived_effort
        session_meta = {
            "session_id": self.session_id,
            "wall_clock_start": self.wall_clock_start,
            "condition": self.condition,
            "software_git_hash": self.software_git_hash,
            "config": self.config,
            "state_covariates": self.state_covariates,
            "perceived_effort_1_10": self.perceived_effort,
        }
        self.session_dir.mkdir(parents=True, exist_ok=True)
        with open(self.session_dir / "session_meta.json", "w", encoding="utf-8") as f:
            json.dump(session_meta, f, indent=2, ensure_ascii=False)
